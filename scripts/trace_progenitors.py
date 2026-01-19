# Owen Gonzales
# Last modfied: 12 Nov 2025

# This script performs progenitor-tracing on halos in the FIRE-2 simulations by tracking the location of star particles across snapshots.
# It returns one .hdf5 file per simulation, organized into groups based on the halo tracked, and then datasets for various properties.
# Properties of halos are organized as a function of simulation snapshot, starting at <startsnap> (default 67) and ending at <endsnap> (default 11).
# Properties include: best-progenitor ID, position in ckpc, virial radius, virial mass, inner 100% stellar mass, inner 50% stellar mass, and the "relaxation" of the algorithm.
# *****   This program REQUIRES that the generate_halostar_catalog.py file is run beforehand as the resulting catalogs are necessary to core funcitonality   *****

import numpy as np
import parallelization as pl
import h5py
import sys
import os
from math import ceil

# # # # # # # # # # #
# Halo Chain Object # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #

class HaloChain():

    ### Set variable staring values and inital star particle weights.
    def __init__(self, halo_id, params, mainhalo_progenitors=None, extra_cores=0):
        '''
        Initializes HaloChain object with given params and calculates starting values

        Parameters:
            halo_id:                catalog ID of halo to be tracked
            params:                 (sim, z0, startsnap, endsnap, path, finder, nmem, pcut)
            mainhalo_progenitors:   list of all progenitors for central host halo. Avoids mis-assigning for 
                                    satellite halos. None if halo_id is the host halo
            extra_cores:            default value of 0. If not 0, indicates how many other cores are available
                                    for speeding up calculations (particularly for host halos)
        Output:
            None
        '''

        # Expand input parameters
        sim, z0, startsnap, endsnap, path, finder, nmem, pcut = params

        # Open halo-star catalog and extract initial star IDs, weights, and halo properties
        with h5py.File(f'../data/HaloStarCatalogs/halostarcatalog_{sim}.hdf5', 'r') as catalog:

            decoder = np.vectorize(lambda x: x.decode('utf-8'))
            init_starids = decoder(catalog[str(startsnap)][str(halo_id)]['star.ids'][:])

            distances = catalog[str(startsnap)][halo_id]['star.distance'][:]
            halo_pos = catalog[str(startsnap)][halo_id]['halo.pos'][:]
            halo_rvir = catalog[str(startsnap)][halo_id]['halo.rvir'][()]
            halo_mvir = np.log10(catalog[str(startsnap)][halo_id]['halo.mvir'][()])
            halo_mstell50 = np.log10(np.sum(catalog[str(startsnap)][halo_id]['star.mass'][:][distances < 0.5*halo_rvir]))
            halo_mstell100 = np.log10(np.sum(catalog[str(startsnap)][halo_id]['star.mass'][:]))
            
            init_starids = init_starids[distances < 0.5*halo_rvir]
            init_starweights = 1 / distances[distances < 0.5*halo_rvir]**2
            init_starweights /= sum(init_starweights)

        # Assign attributes (non-looped variables)
        self.name = halo_id                                                         # Halo catalog ID
        self.sim = sim                                                              # Simulation
        self.z0 = z0                                                                # Starting redshift
        self.path = path                                                            # Path to snapshot data
        self.finder = finder                                                        # Halo finder used (either AHF or rockstar)
        self.nmem = nmem                                                            # Number of snapshots included in star particle memory
        self.pcut = pcut                                                            # Minimum number of particles necessary for halo to be considered
        self.startsnap = startsnap                                                  # Starting snapshot
        self.endsnap = endsnap                                                      # Ending snapshot
        self.extracores = extra_cores                                               # Number of extra cores available for speeding up calculations
        self.state = 'CandidateHalos'                                               # State of tracking: (CandidateHalos, NoHalos, NoStarParticles)
        self.nskipped = 0                                                           # Number of adjacent snapshots without an identified progenitor
        self.mainhalo_progenitors = mainhalo_progenitors                            # Progenitors of main halo in simulation -> should not be identified as progenitors of another descendant halo
        self.initstarids = init_starids                                             # Star particles IDs at final snapshot -> anchor to these
        
        # Assign attributes (looped variables)
        # Flagging convention:  1: progenitor identified, 0: no progenitor identified, progenitor star particles present, -1: no progenitor halo or stars
        self.relaxation = -np.ones(startsnap - endsnap + 1)             # Flags to reflect how "relaxed" the progenitor identifiaction is
                                                                        #       0: within 1 dex virial mass of descendent, at least 75% of constituent star particles were already in memory
                                                                        #       1: within 2 dex virial mass of descendent, at least 75% of constituent star particles were already in memory
                                                                        #       2: within 2 dex virial mass of descendent, at least 50% of constituent star particles were already in memory
                                                                        #       3: star particles present, no good candidate halos
                                                                        #       4: no remaining star particles, algorithm terminated
        self.progenitor_id = -np.ones(startsnap - endsnap + 1)          # Array containing the catalog IDs of each of the progenitor halos
        self.rvir = -np.ones(startsnap - endsnap + 1)                   # Array of virial radii of progenitor halos
        self.mvir = -np.ones(startsnap - endsnap + 1)                   # Array of virial mass of progenitor halos
        self.mstell50 = -np.ones(startsnap - endsnap + 1)               # Array of stellar mass contained within the inner 50% of the virial radius
        self.mstell100 = -np.ones(startsnap - endsnap + 1)              # Array of stellar mass contained within the entire virial radius
        self.position = -np.ones((startsnap - endsnap + 1, 3))          # (x, y, z) position of progenitor in comoving kpc
        
        self.relaxation[0] = 0
        self.progenitor_id[0] = halo_id
        self.rvir[0] = halo_rvir
        self.mvir[0] = halo_mvir
        self.mstell50[0] = halo_mstell50
        self.mstell100[0] = halo_mstell100
        self.position[0, :] = halo_pos
        
        # Create dictionary of star particle weights to be updated each snapshot. Populate all nmem slots with the starting weight upon initialization
        self.starids = init_starids                                                 # Memory of all star particle IDs -> this array is dynamic
        self.starweights = np.tile(init_starweights[:, np.newaxis], (1, 5))         # Memory of all star particle weights (shape: (nstars, 5)) -> this array is also dynamic
        self.starids_nohalo = None                                                  # If a progenitor cannot be identified, the star particles of the most recent progenitor are saved here
        self.id_to_idx = {}                                                         # Allows matching star particle IDs to positions in the weight array
        for i, starid in enumerate(init_starids):
            self.id_to_idx[starid] = i
        return
    
    ### For a given halo, finds how many and what fraction of its star particles are stored in star particle memory
    def match_starparticles(self, catalog, snap, halo, star_ids):
        '''
        For a given candidate progenitor, calculates how many of its constituent star particles are found in star particle
        memory, and what fraction of the total number of star particles are in memory.

        Parameters:
            catalog:    catalog (opened hdf5) of halos and which star particles lie inside them
            halo:       halo ID whose chi score is to be calculated
            snap:       snapshot at which progenitor is being calculated
            star_ids:   star particle IDs for all star particles in memory with nonzero weights
        Output:
            array containing nmatched and fmatched
            nmatched:   number of star particles found in memory
            fmatched:   fraction of star particles found in memory
        '''

        # Read in star particle IDs for the halo
        decoder = np.vectorize(lambda x: x.decode('utf-8'))

        # Read in star particle distances to halo and halo virial radius
        # Unless there are less than <pcut> (default 10) star particles, read in star particle IDs within 50% of virial radius
        distances = catalog[str(snap)][halo]['star.distance'][:]
        rvir = catalog[str(snap)][halo]['halo.rvir'][()]
        in_half = distances <= 0.5*rvir
        if sum(in_half) < self.pcut:
            starids_thishalo = decoder(catalog[str(snap)][halo]['star.ids'][:])
        else:
            starids_thishalo = decoder(catalog[str(snap)][halo]['star.ids'][in_half])
        
        # Calculate number of star particle in memory and fraction of star particles in memory
        nstars = len(starids_thishalo)
        nmatched = sum(np.isin(starids_thishalo, star_ids))
        fmatched = nmatched / nstars if nstars != 0 else 0.

        return np.array([nmatched, fmatched])
    
    ### Get chi score of a single halo at a given snapshot
    def get_chi(self, catalog, snap, halo, star_ids):
        '''
        Given a candidate progenitor and an array of all star particle IDs found in memory, calculate the chi score of the candidate progenitor.

        Parameters:
            catalog:    catalog (opened hdf5) of halos and which star particles lie inside them (generate_halostar_catalog.py output)
            halo:       halo ID whose chi score is to be calculated
            snap:       snapshot at which progenitor is being calculated
            star_ids:   star particle IDs for all star particles in memory with nonzero weights
        Output:
            float representing the chi score of the halo in question
        '''
        
        # Read in star particle IDs and halo virial radius
        decoder = np.vectorize(lambda x: x.decode('utf-8'))
        starids_thishalo = decoder(catalog[str(snap)][halo]['star.ids'][:])
        halo_rvir = catalog[str(snap)][halo]['halo.rvir'][()]

        # Sort star particles to allow matching between lists using array indexing
        star_ids = np.sort(star_ids)
        sort_catalog = np.argsort(starids_thishalo)
        starids_thishalo = starids_thishalo[sort_catalog]

        # Boolean arrays allows us to determine which star particles are in both
        in_thishalo = np.isin(star_ids, starids_thishalo)
        in_dict = np.isin(starids_thishalo, star_ids)

        # Create sorted array of star particles found both in the halo and in the star particle memory
        star_distances_indict = catalog[str(snap)][halo]['star.distance'][sort_catalog][in_dict]
        if len(star_distances_indict) == 0:
            return 0.
        else:
            pass
        
        # Weight adjustment coefficients are multiplied to the snapshot-summed weights of each star particle and are 
        # based off of the star particle's distance to the center of the halo.
        # Particles at the edge have their weights multiplied by 0.5, ranging to 1 at the center.
        # This helps avoid star particles at the outskirts having undue influence on the best progenitor, especially if they
        # have large weights from previous snapshots.
        weight_adjustment = 1 - (catalog[str(snap)][halo]['star.distance'][:][sort_catalog][in_dict] / (2*halo_rvir))
        weight_idxs = np.array([self.id_to_idx[star] for star in star_ids[in_thishalo]])
        weights_snapsummed = np.sum(self.starweights[weight_idxs, :], axis=1)
        weights_snapsummed *= weight_adjustment

        # The chi score is the sum of the adjusted snapshot-summed weights
        chi = np.sum(weights_snapsummed)
        
        return chi
    
    ### Identifies progenitor at previous snapshot and updates memory. Main functionality of progenitor-identifiaction
    def step(self, snap):
        '''
        Main functionality of progenitor identification code. Searches for candidate progenitors at previous snapshot, finds the best one
        using an adaptive criteria for relaxation of match strictness if necessary, and updates system state

        Parameters:
            snap:       snapshot at which progenitor is to be identified
        Output:
            If self.state = 'CandidateHalos':   Catalog ID, (x, y, z) position in ckpc, virial radius, virial mass, inner 50% stellar mass, 100% stellar mass,
                                                and consitituent star particles and their weights for the identified progenitor halo
            If self.state = 'NoHalos':          Returns the IDs of the star particles with nonzero weights in the previous snapshot, and uniform, normalized weights
            If self.state = 'NoStarParticles:   Returns None
        '''

        # Index of master arrays to update this step
        idx = self.startsnap - snap
        decoder = np.vectorize(lambda x: x.decode('utf-8'))

        # Progenitor identification can continue for up to 5 steps without and identified progenitor so long as enough star
        # particles are present. Terminates automatically after 5 consecutive steps without a best-progenitor being found
        if self.nskipped == 5:
            print('*** More than five snapshots without an identified progenitor. Terminating algorithm... ***')
            return
        else:
            pass

        # Boolean arrays for star particles in all memory and in recent memory
        # Terminate program if there are fewer than <pcut> (default 10) star particles in all memory or fewer than <pcut>/2 in recent memory
        in_mem = np.sum(self.starweights, axis=1) > 0.
        in_recent = self.starweights[:, 0] > 0.   # Used for tracking haloless star particles
        print(sum(in_mem), sum(in_recent))
        if (sum(in_mem) < self.pcut) or (sum(in_recent) < self.pcut/2):
            self.state = 'NoStarParticles'
            self.relaxation[idx] = 4
            print('*** Insufficient number of star particles for tracking. Terminating program... ***')
            return
        else:
            pass

        ### Match star IDs in memory with star IDs present in each halo in the previous snapshot
        with h5py.File(f'../data/HaloStarCatalogs/halostarcatalog_{self.sim}.hdf5', 'r') as catalog:
            
            ## Read in halo IDs and their virial masses and radii from catalog
            ## Exclude the main halo progenitor at the current snapshot, if available, for ease of calculation
            halos_id = np.array(list(catalog[str(snap)].keys()))
            if self.mainhalo_progenitors is not None:
                halos_id = halos_id[halos_id != self.mainhalo_progenitors[idx]]
            else:
                pass
            halos_rvir = np.array([catalog[str(snap)][ID]['halo.rvir'][()] for ID in halos_id])
            halos_mvir = np.array([np.log10(catalog[str(snap)][ID]['halo.mvir'][()]) for ID in halos_id])
            halos_nmatched = np.zeros(len(halos_id))   # Will hold number of constituent star particles found in memory
            halos_fmatched = np.zeros(len(halos_id))   # Will hold fraction of constituent star particles found in memory

            ## Identify candidate progenitors in the standard case (best-progenitor identified in the snapshot in previous step, next snapshot chronologically)
            if self.state == 'CandidateHalos':
                
                # Strictest check (relaxation = 0): 
                # Halo is within 0.6 dex of previous virial mass, 30% of previous virial radius, and at least 55% of constituent 
                # star particles are found in the descendant halo
                relaxation = 0
                rad_filt = np.abs(np.log10(halos_rvir / self.rvir[idx-1])) < np.log10(2)
                mass_filt = np.abs(halos_mvir - self.mvir[idx-1]) < 0.6
                filt = mass_filt & rad_filt

                # Set relaxation to 1 if no halos meet these criteria
                # Otherwise save the most massive halos (up to five) to have their chi scores calculated
                if sum(filt) == 0:
                    relaxation = 1
                else:
                    match_stats = np.array(list(map(lambda halo: self.match_starparticles(catalog, snap, halo, self.starids[in_mem]), halos_id[filt])))
                    halos_nmatched[filt] = match_stats[:, 0]
                    halos_fmatched[filt] = match_stats[:, 1]
                    print(halos_id[filt])
                    print(halos_nmatched[filt])
                    print(halos_fmatched[filt])
                    fraction_filt = (halos_fmatched >= 0.55) & (halos_nmatched >= self.pcut)

                    # Only check the most massive 5 (or fewer depending on the number that pass the filter)
                    if sum(fraction_filt) > 5:
                        halos_check = halos_id[np.argsort(halos_fmatched)[::-1][:5]]
                    elif (sum(fraction_filt) > 0) and (sum(fraction_filt) <= 5):
                        halos_check = halos_id[fraction_filt]
                    else:
                        relaxation = 1

                # Relaxed match fraction check (relaxation = 1)
                # At least 5% of the constituent star particles are found in the descendant halo (virial mass and raidus are not relaxed)
                # If the match fraction seems low seems low, remember that star particles can belong to multiple halos simultaneously at this stage,
                # so halos momentarily overlapping can drive the match fraction down dramatically.
                if relaxation == 1:

                    print('*** Relaxing fractional match filter ***')
                    fraction_filt = (halos_fmatched >= 0.05) & (halos_nmatched >= self.pcut/2)

                    # Save the most massive 5 halo (or fewer depending on the number that pass the filter) to have their chi scores calculated
                    # If no halos meet these criteria, set relaxation to 2
                    if sum(fraction_filt) > 5:
                        halos_check = halos_id[np.argsort(halos_fmatched)[::-1][:5]]
                        print(halos_check)
                    elif (sum(fraction_filt) > 0) and (sum(fraction_filt) <= 5):
                        halos_check = halos_id[fraction_filt]
                        print(halos_check)
                    else:
                        relaxation = 2
                else:
                    pass
                
                # Relaxed mass and match fraction check (relaxation = 2)
                # Mass is between 0.6 and 1.0 dex of previous virial mass and at least 10% of the constituent star particles are found
                # in the descendent halo (virial radius is not relaxed).
                if relaxation == 2:

                    print('*** Relaxing mass filter ***')
                    mass_filt_relaxed = (np.abs(halos_mvir - self.mvir[idx-1]) >= 0.6) & (np.abs(halos_mvir - self.mvir[idx-1]) < 1.)
                    filt = mass_filt_relaxed & rad_filt

                    # If no halos meet this criteria, set relaxation to 3
                    # Otherwise save the most massive halos (up to five) to have their chi scores calculated
                    if sum(filt) == 0:
                        relaxation = 3
                    else:
                        match_stats = np.array(list(map(lambda halo: self.match_starparticles(catalog, snap, halo, self.starids[in_mem]), halos_id[filt])))
                        halos_nmatched[filt] = match_stats[:, 0]
                        halos_fmatched[filt] = match_stats[:, 1]
                        fraction_filt = (halos_fmatched >= 0.10) & (halos_nmatched >= self.pcut/2)

                    # Only check the most massive 5 (or fewer depending on the number that pass the filter)
                    if sum(fraction_filt) > 5:
                        halos_check = halos_id[np.argsort(halos_fmatched)[::-1][:5]]
                    elif (sum(fraction_filt) > 0) and (sum(fraction_filt) <= 5):
                        halos_check = halos_id[fraction_filt]
                    else:
                        relaxation = 3
                else:
                    pass
                
                # No identified best progenitor (relaxation = 3)
                if relaxation == 3:
                    self.state = 'NoHalos'
                    self.mvir_last = self.mvir[idx-1]
                    self.rvir_last = self.rvir[idx-1]
                    self.starids_nohalo = self.starids[in_recent]
                else:
                    pass

            ## Check for candidate progenitors in the special case (no descendant halo identified in the subsequent snapshot but more than pcut/2 star particles in memory present)
            else:   # self.state = 'NoHalos'
                
                # Slightly relaxed filters for re-identifying a progenitor
                match_stats = np.array(list(map(lambda halo: self.match_starparticles(catalog, snap, halo, self.starids_nohalo), halos_id)))
                halos_nmatched = match_stats[:, 0]
                halos_fmatched = match_stats[:, 1]
                fraction_filt = (halos_fmatched >= 0.03) & (halos_nmatched >= self.pcut/2)

                rad_filt = np.abs(np.log10(halos_rvir / self.rvir_last)) < 0.5
                mass_filt = np.abs(halos_mvir - self.mvir_last) < 1
                filt = fraction_filt & rad_filt & mass_filt

                # If a halo does meet these criteria, return to the algorithm with a relaxation of 0
                if sum(filt) > 0:
                    relaxation = 2
                    self.state = 'CandidateHalos'
                    halos_check = halos_id[filt]
                else:
                    relaxation = 3

            ## Identify best progenitor (candidate progenitor(s) have been identified)
            if self.state == 'CandidateHalos':
                
                print(f'*** Calculating progenitor chi values... ***')
                
                # Determine best progenitor by maximum chi score
                chi = np.zeros(len(halos_check))
                for i, halo in enumerate(halos_check):
                    chi[i] = self.get_chi(catalog, snap, halo, self.starids[in_mem])
                    print(halo, chi[i])

                # If the maximum chi score is greater than 0, save properties of this progenitor to temporary variables
                # to be returned by this function
                if max(chi) > 0.:

                    progenitor_id = halos_check[np.argmax(chi)]

                    print(f'*** Progenitor identified (Halo ID: {progenitor_id}) ***')

                    # Get progenitor properties from generate_halostar_catalog.py output file
                    distances = catalog[str(snap)][progenitor_id]['star.distance'][:]
                    progenitor_position = catalog[str(snap)][progenitor_id]['halo.pos'][:]
                    progenitor_rvir = catalog[str(snap)][progenitor_id]['halo.rvir'][()]
                    progenitor_mvir = np.log10(catalog[str(snap)][progenitor_id]['halo.mvir'][()])

                    # Star particle IDs and weights to be saved to the star particle dictionary
                    # Use only weights of star particles within 50% of virial radius if there are at least <pcut>/2 such star particles
                    print(sum(distances <= 0.5*progenitor_rvir))
                    if sum(distances <= 0.5*progenitor_rvir) <= self.pcut/2:
                        progenitor_mstell50 = np.log10(np.sum(catalog[str(snap)][progenitor_id]['star.mass'][distances <= 0.5*progenitor_rvir]))
                        progenitor_mstell100 = np.log10(np.sum(catalog[str(snap)][progenitor_id]['star.mass'][:]))
                        progenitor_starids = decoder(catalog[str(snap)][progenitor_id]['star.ids'])
                        progenitor_weights = 1 / distances**2
                    else:
                        progenitor_mstell50 = np.log10(np.sum(catalog[str(snap)][progenitor_id]['star.mass'][distances <= 0.5*progenitor_rvir]))
                        progenitor_mstell100 = np.log10(np.sum(catalog[str(snap)][progenitor_id]['star.mass'][:]))
                        progenitor_starids = decoder(catalog[str(snap)][progenitor_id]['star.ids'][distances <= 0.5*progenitor_rvir])
                        progenitor_weights = 1 / distances[distances <= 0.5*progenitor_rvir]**2
                    progenitor_weights /= sum(progenitor_weights)

                    # Update relaxation, package data, and return
                    self.relaxation[idx] = relaxation
                    progenitor_data = (progenitor_id, progenitor_position, progenitor_rvir, progenitor_mvir, progenitor_mstell50, 
                                    progenitor_mstell100, progenitor_starids, progenitor_weights)
                    
                    print(f'*** Progenitor data recorded ***')
                    return progenitor_data
                
                # If the max chi score is 0, there is no identified best progenitor
                else:
                    self.relaxation[idx] = 4
                    self.state = 'NoHalos'
            else:
                pass
            
            ## If no candidate progenitors have been identified, continue to track what remains of the previously tracked group of star particles. 
            ## Assign uniform, normalized weights. These will be passed to the next step.
            if self.state == 'NoHalos':
                
                print('*** No candidate progenitors. Setting tracking to star particles only ***')
                
                # Create list of all star particles present during the snapshot of interest that lie within one of the tracked halos
                starids_thissnap = []
                for halo in halos_id:
                    starids_thissnap += list(decoder(catalog[str(snap)][halo]['star.ids'][:]))
                starids_thissnap = np.unique(np.array(starids_thissnap))

                # Of those, take the star particles which were tracked in the previous snapshot and assign uniform, normalized weights
                starids_totrack = self.starids[np.isin(self.starids, starids_thissnap)]
                if len(starids_totrack) != 0:
                    weights = np.ones(len(starids_totrack)) / len(starids_totrack)
                else:
                    weights = np.array([])

                # Update relaxation, package data, and return
                self.relaxation[idx] = relaxation
                progenitor_data = (starids_totrack, weights)

                return progenitor_data
            
            else:
                print('Weird Error! This should never happen and Im just leaving this were because I dont like leaving if statements without an else.')
                print('Writing it this way is necessary because the a statement in the if self.state == CandidateProgenitor block can trigger self.state = NoHalos')
                return

    ### Updates master arrays of progenitor data and star particle dictionary each interation
    def update(self, snap, progenitor_data):
        '''
        Updates memory with properties of the identified best-progenitor halo

        Parameters:
            snap:       snapshot at which progenitor was identified
            progenitor data:
                        If self.state = 'CandidateHalos':   Catalog ID, (x, y, z) position in ckpc, virial radius, virial mass, inner 50% stellar mass, 100% stellar mass,
                                                            and consitituent star particles and their weights for the identified progenitor halo
                        If self.state = 'NoHalos':          Returns the IDs of the star particles with nonzero weights in the previous snapshot, and uniform, normalized weights
                        If self.state = 'NoStarParticles:   Returns None
        Output:
            None
        '''

        # Index of master arrays to update this step
        idx = self.startsnap - snap

        # If there is progenitor data (self.state != 'NoStarParticles')
        if progenitor_data is not None:
            
            # If there was a indentified progenitor in the last step
            if self.state == 'CandidateHalos':
                
                # Expand data and save if candidate halos identified
                progenitor_id, progenitor_position, progenitor_rvir, progenitor_mvir, progenitor_mstell50, progenitor_mstell100, progenitor_starids, progenitor_weights = progenitor_data

                # Save data to master arrays
                self.progenitor_id[idx] = progenitor_id
                self.rvir[idx] = progenitor_rvir
                self.mvir[idx] = progenitor_mvir
                self.mstell50[idx] = progenitor_mstell50
                self.mstell100[idx] = progenitor_mstell100
                self.position[idx] = progenitor_position

                print('*** Progenitor halo properties recorded ***')

                # Update and cycles star particle weights in memory
                nstars = len(self.starids)                                                                      # Total number of star particles currently in memory
                in_mem = np.isin(progenitor_starids, self.starids)                                              # Which of the progenitors star particles are already in memory
                dict_update_idxs = np.array([self.id_to_idx[star] for star in progenitor_starids[in_mem]])      # Get their corresponding indices in memory
                self.starids = np.append(self.starids, progenitor_starids[~in_mem])                             # Extend memory to include progenitor star particles not already seen
                for (i, star) in enumerate(progenitor_starids[~in_mem]):                                        # Add new entries to the index dictionary, and their corresponding indices
                    self.id_to_idx[star] = nstars+i

                # Create temporary weight array, cycle that, and update with progenitor weights
                temp_weights = np.zeros((nstars+sum(~in_mem), self.nmem))
                temp_weights[:nstars, 1:] = self.starweights[:, :-1]
                if sum(in_mem) != 0:
                    temp_weights[dict_update_idxs, 0] = progenitor_weights[in_mem]
                else:
                    pass
                temp_weights[nstars:, 0] = progenitor_weights[~in_mem]
                self.starweights = temp_weights
                self.nskipped = 0
                    
                print('*** Star particle dictionary updated ***')

            else:   # self.state = 'NoHalos'
                
                # Update and cycle weights for group of star particles tracked
                starids_tracked, weights = progenitor_data

                temp_weights = np.zeros_like(self.starweights)
                temp_weights[:, 1:] = self.starweights[:, :-1]
                if len(starids_tracked) != 0:
                    dict_update_idxs = np.array([self.id_to_idx[star] for star in starids_tracked])
                    temp_weights[dict_update_idxs, 0] = weights
                else:
                    pass
                self.starweights = temp_weights
                self.nskipped += 1
                    
                print('*** Star particle IDs tracked ***')

        else:
            pass
        
        return

    ### Saves data to a temporary hdf5 file. These will all be merged into a master file in the final step
    def save(self):
        '''
        Saves HaloChain object to temporary hdf5 file (will be merged into master file with all halos from the simulation later)
        Parameters:
            None
        Output:
            None
        '''

        print('*** Saving to hdf5 file... ***')
        with h5py.File(f'../data/ProgenitorTracks/temp_{self.sim}_{self.name}_{self.finder.lower()}_progenitortracks.hdf5', 'w') as file:
            file.create_dataset('prog.id', data=self.progenitor_id.astype(int))
            file.create_dataset('prog.position', data=self.position)
            file.create_dataset('prog.rvir', data=self.rvir)
            file.create_dataset('prog.logmvir', data=self.mvir)
            file.create_dataset('prog.logmstell50', data=self.mstell50)
            file.create_dataset('prog.logmstell100', data=self.mstell100)
            file.create_dataset('prog.relaxation', data=self.relaxation)   
        print('*** Data saved ***')
        
        return
    
# # # # # # #
# Functions # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #

def print_error_message():
    print()
    print('! Error: This program requires the following eight arguments:\n')
    print('         sim: str -> name of simulation (e.g. \'z5m11a\')')
    print('         finder_name: str -> name of halo finder (e.g. \'rockstar\')')
    print('         z0: float -> starting redshift of simulation (e.g. 5.000)')
    print('         nhalos: int -> number of halos to calculate progenitor tracks for (e.g. 50) or list of IDs (e.g. [104,225,17]) or 0 for all halos')
    print('         pcut: int -> minimum number of star particles to track (e.g. 50)')
    print('         endsnap: int -> snapshot past which progenitor identification will be terminated (e.g. 11)')
    print('         nmem: int -> desired number of memory slots (e.g. 5)')
    print('         ncores: int -> number of desired cores (e.g. 20)')
    print()


def get_args():
    '''
    Interprets arguments passed in the command line, checks for errors, and, if successful, returns them as variables to use in the program.
    This mainly screens for variable types to ensure that arguments were input correctly.

    Parameters:
        None
    Output:
        See parameters in print_error_message() above
    '''

    try:
        print('*** Unpacking arguments... ***')
        sim, finder, z0, nhalos, pcut, endsnap, nmem, ncores = sys.argv[1:]
        z0 = float(z0)
        pcut = int(pcut)
        endsnap = int(endsnap)
        nmem = int(nmem)
        ncores = int(ncores)
        if (nhalos[0] == '[') & (nhalos[-1] == ']'):
            nhalos = np.array(nhalos[1:-1].split(',')).astype(int)
        else:
            nhalos = int(nhalos)
            if nhalos < 0:
                raise Exception('! Error: When specifying a number of halos, nhalos must be a positive integer or 0 (for all halos)')
            else:
                pass
    except ValueError:
        print_error_message()
        sys.exit()
    else:
        if not (isinstance(nhalos, int) or isinstance(nhalos, np.ndarray)):
            print_error_message()
            sys.exit()
        if not (isinstance(sim, str) and isinstance(finder, str) and isinstance(z0, float) and isinstance(pcut, int) and\
                isinstance(endsnap, int) and isinstance(nmem, int) and isinstance(ncores, int)):
            print_error_message()
            sys.exit()
        if (finder.lower() != 'rockstar') and (finder.lower() != 'ahf'):
            print_error_message()
            raise Exception('! Error: Please select a valid halo finder\n')
        if (z0 < 0) or (pcut < 0) or (endsnap < 0) or (nmem < 1):
            raise Exception('! Error: z0, mcut, pcut, and endsnap must be positive numbers. nmem must be an integer 1 or greater\n')
        if ncores > pl.allcores:
            print(f'! Warning: cannot use more cores than available. Defaulting to the maximum number of available cores ({pl.allcores})\n')
            ncores = pl.allcores
        return sim, finder, z0, nhalos, pcut, endsnap, nmem, ncores
    

def calculate_mainhalo(nhalos, params, snaps, halo_arr=None):
    '''
    Calculates the complete HaloChain object for the central host halo and saves it
    (separating this step ensures that one core isn't bogged down with the larges halo during parallelization)

    Parameters:
        nhalos:     number of halos from this simulation to be tracked
        params:     (sim, z0, startsnap, endsnap, path, finder, nmem, pcut, ncores)
        snaps:      array of snapshots to loop over (flipped so highest snapshot is first -> loops backwards in time)
        halo_arr:   If not None, this is an array of halo IDs to be calculated
    Output:
        nhalos:     has 1 subtraced from nhalos input parameters
        halo_arr:   If not None, returns the halo_arr input with the host halo ID removed
    '''

    sim, _, startsnap, endsnap, _, _, _, _, ncores = params                     # Expand function parameters

    # Identify the main halo (largest number of star particles)
    with h5py.File(f'../data/HaloStarCatalogs/halostarcatalog_{sim}.hdf5', 'r') as catalog:
        halo_ids = [halo for halo in catalog[str(startsnap)].keys()]
        mainhalo_idx = np.argmax(np.array([len(catalog[str(startsnap)][halo]['star.ids'][:]) for halo in halo_ids]))
        mainhalo_id = halo_ids[mainhalo_idx]

    # Can only calculate as many halos as are present
    if nhalos > len(halo_ids):
        nhalos = len(halo_ids)
    elif nhalos == 0:
        nhalos = len(halo_ids)
    else:
        pass

    # Return an array of -1s for the host halo progenitors if halo_arr is specified and the host halo is not included
    if (halo_arr is not None) and (mainhalo_id not in halo_arr):
        return nhalos, halo_arr, -np.ones(startsnap-endsnap+1)
    else:
        pass
    
    # Calculate main halo
    print('*** Executing progenitor tracking for host halo... ***')
    MainHaloChain = track_progenitors(mainhalo_id, params[:-1], snaps, extra_cores=ncores-1, return_object=True)
    print('*** Host halo progenitors successfully idenitified ***')
    
    nhalos = nhalos - 1

    if halo_arr is None:
        return nhalos, MainHaloChain.progenitor_id
    else:
        return nhalos, halo_arr[halo_arr != mainhalo_id], MainHaloChain.progenitor_id
    

def calculate_halos(nhalos, params, snaps, mainhalo_progenitors, halo_arr=None):
    '''
    Loops over the calculation of progenitor tracks for all halos that are not the host halo

    Parameters:
        nhalos:                 number of halos from this simulation to be tracked
        ncores:                 the number of cores to be used during parallelization
        params:                 (sim, z0, startsnap, endsnap, path, finder, nmem, pcut, ncores)
        snaps:                  array of snapshots to loop over (flipped so highest snapshot is first -> loops backwards in time)
        mainhalo_progenitors:   progenitors of the host halo (included to avoid other halos identifying one of its progenitors as their own)
        halo_arr:               If not None, this is an array of halo IDs to be calculated
    Output:
        None
    '''
    sim, _, startsnap, _, _, _, _, _, ncores = params                     # Expand function parameters
    
    # Retrieve halo IDs and virial masses from the catalog file
    with h5py.File(f'../data/HaloStarCatalogs/halostarcatalog_{sim}.hdf5', 'r') as catalog:
        halo_ids = np.array([halo for halo in catalog[str(startsnap)].keys()])
        halo_mvir = np.array([np.log10(catalog[str(startsnap)][halo]['halo.mvir'][()]) for halo in halo_ids])

    with h5py.File('../data/ProgenitorTracks/z5m11c_progenitortracks.hdf5', 'r') as catalog:
        completed_halos = np.array(list(catalog.keys()))
        halo_mvir = halo_mvir[~np.isin(halo_ids, completed_halos)]
        halo_ids = halo_ids[~np.isin(halo_ids, completed_halos)]

    # We want to distribute the job of calcualting halo tracks across multiple different cores so that each core calculates one track
    # To do this, we sort the halos by virial mass. On the first loop, we take the first ncores most massive ones, and on the next loop we take the next ncores most massive ones, etc.
    necessary_loops = ceil(nhalos / ncores)

    for j in range(necessary_loops):

        if j == necessary_loops - 1:
            lower = j * ncores
            upper = nhalos
        else:
            lower = j * ncores
            upper = (j+1) * ncores

        if halo_arr is not None:
            which_halos = np.where(np.isin(halo_ids, halo_arr))[0]
            which_halos = which_halos[lower:upper]
        else:
            which_halos = np.argsort(halo_mvir)[::-1][1:][lower:upper]

        # Halo IDs to be calculated this loop
        ID_arr = halo_ids[which_halos]

        # Calculate halo tracks in parallel for each halo in ID_arr
        print()
        print(f'*** Loop {j+1} / {necessary_loops}: beginning progenitor tracking... ***')
        pl.Map(track_progenitors, ID_arr, ncores, params[:-1], snaps, mainhalo_progenitors=mainhalo_progenitors)

    return


def track_progenitors(halo_id, params, snaps, mainhalo_progenitors=None, extra_cores=None, return_object=False):
    '''
    Core looping logic for progenitor identification. Continues to track until too few tracked star particles remain, at which point the algorithm terminates

    Parameters:
        halo_id:                halo ID for which progenitor tracks are to be calculated
        params:                 (sim, z0, startsnap, endsnap, path, finder, nmem, pcut)
        snaps:                  array of snapshots to loop over (flipped so highest snapshot is first -> loops backwards in time)
        mainhalo_progenitors:   If specified, excludes from consideration any candidate progenitor that is the progenitor of the host halo
        extra_cores:            If not None, indicates the number of extra available cores for calculation
        return_object:          If True, returns the HaloChain object calculated
    '''

    # Initialize halo chain
    if extra_cores is not None:
        ThisHaloChain = HaloChain(halo_id, params, mainhalo_progenitors=mainhalo_progenitors, extra_cores=extra_cores)
    else:
        ThisHaloChain = HaloChain(halo_id, params, mainhalo_progenitors=mainhalo_progenitors)

    # Loop over snapshots (backwards in time)
    for snap in snaps:
        
        # Each loop, identify progenitor halos or star particles and save to HaloChain object
        print()
        print(f'Halo id: {halo_id}, Snapshot: {snap}')
        progenitor_data = ThisHaloChain.step(snap)
        ThisHaloChain.update(snap, progenitor_data)

        # Terminate if no candidate progenitors and too few star particles
        if progenitor_data is not None:
            pass
        else:
            break
    
    # Save data to hdf5 file
    ThisHaloChain.save()

    if return_object:
        return ThisHaloChain
    else:
        return


def merge_files(sim, finder, threshold=20, window=25):
    '''
    Merges temporary save files into one large master file. Temporary files for each halo -> group in master hdf5 file. Halo tracks with
    too few identified progenitors (less than 20 of the 25 most recent snapshots) are removed

    Parameters:
        sim:        simulation for which progenitor tracks have been calculated
        finder:     halo finder used
        threshold:  minimum number of identified progenitors in the first <window> snapshots, starting with the most recent
        window:     number of snapshots to search for progenitors
    Output:
        None
    '''

    print('*** Merging files... ***')

    # Delete file if it already exists
    if f'{sim}_{finder.lower()}_progenitortracks.hdf5' in os.listdir('../data/ProgenitorTracks/'):
        os.system(f'rm ../data/ProgenitorTracks/{sim}_{finder.lower()}_progenitortracks.hdf5')
    else:
        pass

    # Extract file names of temporary save files
    tempfile_names = [name for name in os.listdir('../data/ProgenitorTracks') if name[:11] == f'temp_{sim}']

    # Save to new master hdf5 file
    with h5py.File(f'../data/ProgenitorTracks/{sim}_progenitortracks.hdf5', 'w') as master_file:
        for tempfile in tempfile_names:
            with h5py.File(f'../data/ProgenitorTracks/{tempfile}', 'r') as halo_data:

                # Check for a progenitor track of sufficient length. Ignore if insufficient. Otherwise, add to the master file.
                if sum(halo_data['prog.id'][:][:window] != -1) < threshold:
                    continue
                else:
                    halo = str(int(halo_data['prog.id'][:][0]))
                    master_file.create_group(halo)
                    master_file[halo].create_dataset('prog.id', data=halo_data['prog.id'][:])
                    master_file[halo].create_dataset('prog.position', data=halo_data['prog.position'][:])
                    master_file[halo].create_dataset('prog.rvir', data=halo_data['prog.rvir'][:])
                    master_file[halo].create_dataset('prog.logmvir', data=halo_data['prog.logmvir'][:])
                    master_file[halo].create_dataset('prog.logmstell50.unmatched', data=halo_data['prog.logmstell50'][:])
                    master_file[halo].create_dataset('prog.logmstell100.unmatched', data=halo_data['prog.logmstell100'][:])
                    master_file[halo].create_dataset('prog.relaxation', data=halo_data['prog.relaxation'][:])

    # Delete temporary save files
    os.system(f'rm ../data/ProgenitorTracks/temp_{sim}*')

    print('*** Files merged ***')

    return

# # # # # # # # # # #
# Main body of code # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #

# Retrieve and expand arguments
#args = ('z5m11e', 'rockstar', 5., 0, 10, 11, 5, 1)
args = get_args()
sim, finder, z0, nhalos, pcut, endsnap, nmem, ncores = args
print('*** Arguments received successfully ***')

# Open snapshot times data and package parameters for passing into HaloChain functions
print('*** Accessing data from snapshot_times file... ***')
path = '/projects/b1026/gjsun/high_redshift/'                                   # Path to simulation data
snaptimes = np.loadtxt(path+sim+'/snapshot_times.txt')                          # Snapshot information read in as a numpy array
z0_idx = np.argmin(np.abs(snaptimes[:, 2] - z0))                                # Index of the snapshot that matches z0
startsnap = int(snaptimes[z0_idx, 0])                                           # Number of the snapshot that matches z0
snaps = snaptimes[endsnap:z0_idx+1, 0].astype(int)                              # These are the numbers of each snapshot
snaps = snaps[::-1][1:]                                                         # Reverse and remove the first snapshot for iterating backwards in time
params = (sim, z0, startsnap, endsnap, path, finder, nmem, pcut, ncores)        # Parameters to pass into main() function

# Check for input type
if isinstance(nhalos, np.ndarray):

    # Save array to separate variable
    halo_arr = nhalos
    nhalos = len(halo_arr)

    # If included in the list, calculate and save progenitors for the host halo
    nhalos, halo_arr, mainhalo_progenitors = calculate_mainhalo(nhalos, params, snaps, halo_arr=halo_arr)

    # Calculate and save progenitors for the rest of the halos
    print(mainhalo_progenitors)
    calculate_halos(nhalos, params, snaps, mainhalo_progenitors=mainhalo_progenitors, halo_arr=halo_arr)
else:

    # Calculate and save progenitors for the host halo
    nhalos, mainhalo_progenitors = calculate_mainhalo(nhalos, params, snaps)

    # Calculate and save progenitors forthe rest of the halos
    calculate_halos(nhalos, params, snaps, mainhalo_progenitors=mainhalo_progenitors)

# Merge files for individual halos into one master file for the entire simulation
merge_files(sim, finder)