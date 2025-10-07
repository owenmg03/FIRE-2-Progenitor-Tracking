# Owen Gonzales
# Last modfied: 6 Aug 2024

# This script performs progenitor-tracing on halos in the FIRE-2 simulations by tracking the location of star particles across snapshots
# It returns one .hdf5 file per simulation, organized into groups based on the halo tracked, and then datasets for various properties

import numpy as np
import gizmo_analysis as gizmo
import halo_tools as ht
import parallelization as pl
import h5py
import sys
import os
from math import ceil
from importlib import reload
reload(ht)

# # # # # # #
# Functions #
# # # # # # #

class HaloChain():

    def __init__(self, halo_data, initstar_data, params, mainhalo_progenitors):

        halo_id, halo_rad, halo_mvir, halo_mstell50, halo_mstell100, halo_pos = halo_data
        initstar_ids, initstar_weights = initstar_data
        sim, z0, startsnap, endsnap, path, finder, nmem, mcut = params 

        self.name = str(halo_id)                            # Halo catalog ID
        self.sim = str(sim)                                 # Simulation
        self.z0 = z0                                        # Starting redshift
        self.path = path                                    # Path to snapshot data
        self.finder = finder                                # Halo finder used (either AHF or rockstar)
        self.nmem = nmem                                    # Number of snapshot for star particle memory
        self.mcut = mcut                                    # Initial virial mass cut for candidate halos
        self.startsnap = startsnap                          # Starting snapshot
        self.endsnap = endsnap                              # Ending snapshot
        self.state = 'CandidateHalos'                       # State of tracking: (CandidateHalos, NoHalos, NoStarParticles)
        self.mainhalo_progenitors = mainhalo_progenitors    # Progenitors of main halo in simulation -> should not be identified as progenitors of another descendant halo

        # Flagging convention:  1: progenitor identified, 0: no progenitor identified, progenitor star particles present, -1: no progenitor halo or stars
        self.flag = -np.ones(startsnap - endsnap + 1)                   # Flags to reflect state of progenitor tracking
        self.progenitor_id = -np.ones(startsnap - endsnap + 1)          # Array containing the catalog IDs of each of the progenitor halos
        self.rvir = -np.ones(startsnap - endsnap + 1)                   # Array of virial radii of progenitor halos
        self.mvir = -np.ones(startsnap - endsnap + 1)                   # Array of virial mass of progenitor halos
        self.mstell50 = -np.ones(startsnap - endsnap + 1)               # Array of stellar mass contained within the inner 50% of the virial radius
        self.mstell100 = -np.ones(startsnap - endsnap + 1)               # Array of stellar mass contained within the entire virial radius
        self.position = -np.ones((startsnap - endsnap + 1, 3))          # (x, y, z) position of progenitor in comoving kpc

        self.flag[0] = 1
        self.progenitor_id[0] = halo_id
        self.rvir[0] = halo_rad
        self.mvir[0] = halo_mvir
        self.mstell50[0] = halo_mstell50
        self.mstell100[0] = halo_mstell100
        self.position[0, :] = halo_pos
        
        # Create dictionary of star particle weights to be updated each snapshot
        init_dictionary = {}
        for (star, weight) in zip(initstar_ids, initstar_weights):
            init_dictionary[star] = np.array([weight] + (nmem-1)*[0.])
        self.stardictionary = init_dictionary

        return
    
    # Define function to get chi score of a single halo and apply to each unique halo
    def get_chi(self, halo, halos, starids_ts):
        '''
        Parameters:
            halo:       halo ID whose chi score is to be calculated
            halos:      array of host halos for each star particle
            starids_ts: IDs of star particles present in the snapshot at which we are determining a progenitor
        Output:
            float representing the chi score of the halo in question
        '''
        idxs = np.where(halos == halo)[0]
        weights = [np.sum(self.stardictionary[star]) for star in starids_ts[idxs]]
        return np.sum(weights)

    # Helper function to cycle weights in the star particle dictionary each snapshot. This works for a single star particle and is mapped across all
    def cycle_weights(self, weight_arr, weight):
        '''
        Parameters:
            weight_arr: array of weights for a single star particle
            weight:     new weight calculated this snapshot for that star particle
        Output:
            cycled weight array where the variable "weight" is at index 0 and the old values are cycled back one index
        '''
        weight_arr[1:] = weight_arr[:-1]
        weight_arr[0] = weight
        return weight_arr

    # This function contains the majority of the functionality and is called each iteration -> identifies progenitor halo
    def step(self, starids_desc, deschalo_rad, deschalo_pos_phys, z, snap, mc, snaptimes, isparallel, nterminate=5):
        '''
        Parameters:
            starids_desc:       star particle IDs present in every descendent halo (just the keys of the star particle dictionary). If all weights are 0, star particle will not factor into chi
            deschalo_rad:       virial radius of the immediate descendent halo. This is used to prevent major jumps in virial radii that would happen when 
                                the host halo is assigned to a subhalo's progenitor
            deschalo_pos_phys:  (x, y, z) position of the immediate descendent halo in physical kpc. This is used to filter out halos that cannot physically be progenitors
            z:                  redshift of current snapshot
            snap:               current shapshot number
            mc:                 current virial mass cut at snapshot
            snaptimes:          array of snapshot time data
            isparallel:         1 if unparallelized, >1 if parallelized. Indicates desired number of cores
            nterminate:         Minimum number of star particles to track. Algorithm terminates if the number of trackable star particles falls below this value
        Output:
            If self.state = 'CandidateHalos':   Catalog ID, virial radius, virial mass, inner 50% stellar mass, (x, y, z) position in ckpc, and consitituent star particles and their weights
                                                for the identified progenitor halo
            If self.state = 'NoHalos':          Just the IDs of the star particles are returned and all of their weights are normalized to 1
            If self.state = 'NoStarParticles:   Returns nothing
        '''

        idx = self.startsnap - snap
        if self.mainhalo_progenitors is not None:
            mainhalo_progenitor = self.mainhalo_progenitors[idx]
        else:
            mainhalo_progenitor = -1
        print(snap, mainhalo_progenitor)
        
        print('*** Reading in particle dictionary... ***\n')
        part = gizmo.io.Read.read_snapshots('star', 'redshift', z, self.path+self.sim)
        starids_ts, starpos_ts, starmass_ts = ht.unpackAndSortPartDict(part, starID_form='string')[:3]
        h = part.Cosmology['hubble']

        # Read in the halo finder data of previous snapshot
        print('\n*** Reading in halo finder data... ***')
        hal = ht.getData(self.sim, snap, self.finder)
        haloids_ts, halorad_ts, halomvir_ts, halopos_ts = ht.unpackData(hal, self.sim, snap, z, h, self.finder, mcut=mc)

        # Filter for star particles that exists and this snapshot and hich are also present in the descendent halo
        print('\n*** Applying filters... ***')
        in_descendant = np.isin(starids_ts, starids_desc)
        starids_ts_desc = starids_ts[in_descendant] 
        starpos_ts_desc = starpos_ts[in_descendant, :]

        # If there are fewer star particles identified than the nterminate threshold (default 5) then we cannot reliably track -> terminate loop
        if sum(in_descendant) < nterminate:
            self.state = 'NoStarParticles'
            print('*** Too few star particles to continue tracking. Terminating algorithm... ***')
            return
        else:
            pass
        
        # Apply a speed of light criterion to potential halo centers. Using physical kpc to determine if it is possible for each galaxy to actually be a progenitor
        # 1 kpc = 3261.564 ly
        if self.state == 'CandidateHalos':
            halopos_ts_phys = halopos_ts / (1 + z)
            delta_p_ly = np.sqrt(np.sum((halopos_ts_phys - deschalo_pos_phys)**2, axis=1)) * 3261.546
            delta_t_yr = snaptimes[snap + 1, 3] - snaptimes[snap, 3] * 1e9
            is_physical = (delta_p_ly / delta_t_yr) < 1
            haloids_ts, halorad_ts, halomvir_ts, halopos_ts = haloids_ts[is_physical], halorad_ts[is_physical], halomvir_ts[is_physical], halopos_ts[is_physical]
        else:
            pass

        # For each star particle that is found in the descendent halo, find the halo that it is in at this snapshot
        print('*** Identifying halo of origin for each star particle... ***')
        if isparallel == 1:
            host_halos = ht.whichHalo(starpos_ts_desc, haloids_ts, halorad_ts, halopos_ts, deschalo_rad, fullhalo=True)
        else:
            host_halos = ht.whichHaloPL(starpos_ts_desc, haloids_ts, halorad_ts, halopos_ts, ncores, deschalo_rad, fullhalo=True)
            print('parallel')
        
        unique_halos = np.unique(host_halos)
        print(unique_halos)
        print(mainhalo_progenitor)
        if mainhalo_progenitor != -1:
            unique_halos = unique_halos[unique_halos != int(mainhalo_progenitor)]   # Remove the main halo's progenitor from consideration
        else:
            pass

        # If there is no available halo we may have a case of a disappearing subhalo
        # Keep tracking the set of star particles from the last identified progenitor in case one appears again
        # Weights are unified at 1 and no progenitor information is recorded. We can rectify this with a second algorithm after the fact to identify boundries of structure
        print(set(unique_halos))
        if set(unique_halos) == {-1}:
            self.state = 'NoHalos'
            self.flag[idx] = 0

            print('*** No viable progenitors. Tracking star particles only ***')
            nparticles = len(starids_ts_desc)
            progenitor_data = (starids_ts_desc, np.ones(nparticles)/nparticles, -1, -1, -1, -1, -1, [-1, -1, -1])
            return progenitor_data
            
        # If there is at least one candidate progenitor, keep looping as normal
        else:
            self.state = 'CandidateHalos'
            self.flag[idx] = 1
            unique_halos = unique_halos[(unique_halos != -1)]

            print('*** Identifying most likely progenitor... ***')
            chis = np.array(list(map(lambda unique_halo: self.get_chi(unique_halo, host_halos, starids_ts_desc), unique_halos)))

            # The progenitor halo identified is the one with the highest chi score
            prog_id = int(unique_halos[np.argmax(chis)])
            progidx = np.where(haloids_ts.astype(int) == prog_id)[0][0]

            prog_rad = halorad_ts[progidx]
            prog_mvir = halomvir_ts[progidx]
            prog_pos = halopos_ts[progidx, :]

            # Now we find all of the star particles which are identified to belong to the progenitor galaxy
            # We use the distances of these stars to the halo to generate weights for the next loop
            # For the stellar mass, we are interested in the stars contained in the *entire* virial radius, but we only track stars that
            # fall within the inner half of the virial radius, so these are the only ones which get assigned weights
            isin_prog50 = ht.inHalo(starpos_ts, prog_rad, prog_pos, distances=False, fullhalo=False)
            isin_prog100, distances = ht.inHalo(starpos_ts, prog_rad, prog_pos, distances=True, fullhalo=True)
            progstar_ids = starids_ts[isin_prog50]
            prog_mstell50 = np.sum(starmass_ts[isin_prog50])
            if (snap > 25) or (snap < 35):
                print('**********')
                print(starmass_ts)
                print(prog_mstell50)
                print('**********')
            prog_mstell100 = np.sum(starmass_ts[isin_prog100])
            #ranks = np.argsort(distances[isin_prog50]) + 1
            #progstar_weights = 1 / ranks**2
            #progstar_weights /= np.sum(progstar_weights)
            progstar_weights = 1 / distances[isin_prog50]**2
            progstar_weights /= np.sum(progstar_weights)

            print('*** Progenitor successfully identified! ***')
            progenitor_data = (progstar_ids, progstar_weights, prog_id, prog_rad, prog_mvir, prog_mstell50, prog_mstell100, prog_pos)
            return progenitor_data

    # Updates arrays of progenitor data and star particle dictionary each interation
    def update(self, progenitor_data, snapshot):

        idx = self.startsnap - snapshot
        if progenitor_data is not None:

            progstar_ids, progstar_weights, prog_id, prog_rad, prog_mvir, prog_mstell50, prog_mstell100, prog_pos = progenitor_data

            self.progenitor_id[idx] = int(prog_id)
            self.rvir[idx] = prog_rad
            self.mvir[idx] = prog_mvir
            self.mstell50[idx] = prog_mstell50
            self.mstell100[idx] = prog_mstell100
            self.position[idx, :] = prog_pos

            print('1')
            # Update and cycle star particle weight dictionary
            current_starids = np.array(list(self.stardictionary.keys()))
            in_stardict = np.isin(progstar_ids, current_starids)
            notin_prog = ~np.isin(current_starids, progstar_ids)
            print('2')

            print(len(progstar_ids))
            print(len(progstar_weights))
            print(len(in_stardict))

            keyvals_proginstardict = [(star, self.cycle_weights(self.stardictionary[star], weight)) for (star, weight) in \
                                      zip(progstar_ids[in_stardict], progstar_weights[in_stardict])]
            keyvals_prognotinstardict = [(star, np.array([weight] + (nmem-1)*[0.])) for (star, weight) in \
                                         zip(progstar_ids[in_stardict], progstar_weights[in_stardict])]
            keyvals_stardictnotinprog = [(star, self.cycle_weights(self.stardictionary[star], 0.)) for star in \
                                         current_starids[notin_prog] if np.sum(self.stardictionary[star]) > 0.]
            self.stardictionary = dict(keyvals_proginstardict + keyvals_prognotinstardict + keyvals_stardictnotinprog)

        else:
            pass
        
        return

    # Saves data to a temporary hdf5 file. These will all be merged into a master file in the final step
    def save(self):

        # Save these to hdf5 files
        # Check to see if the file already exists. If so, append. If not, create the file and write to it.
        print('*** Saving to hdf5 file... ***')
        with h5py.File(f'../data/ProgenitorTracks/temp_{sim}_{self.name}_{self.finder.lower()}_progenitortracks.hdf5', 'w') as file:
            file.create_dataset('prog.id', data=self.progenitor_id)
            file.create_dataset('prog.radius', data=self.rvir)
            file.create_dataset('prog.mvir', data=self.mvir)
            file.create_dataset('prog.mstell50', data=self.mstell50)
            file.create_dataset('prog.mstell100', data=self.mstell100)
            file.create_dataset('prog.position', data=self.position)
            file.create_dataset('prog.flag', data=self.flag)
        print('*** Data saved ***')
        
        return


def printErrorMessage():
    print()
    print('! Error: This program requires the following seven arguments:\n')
    print('         sim: str -> name of simulation (e.g. \'z5m11a\')')
    print('         finder_name: str -> name of halo finder (e.g. \'rockstar\')')
    print('         z0: float -> starting redshift of simulation (e.g. 5.000)')
    print('         nhalos: int -> number of halos to calculate progenitor tracks for (e.g. 50) or list of IDs (e.g. [104,225,17]) or 0 for all halos')
    print('         mcut: float -> lower mass cutoff in solar masses for halos at the snapshot corresponding to z0 (e.g. 1e9)')
    print('         pcut: int -> minimum number of star particles to track (e.g. 50)')
    print('         endsnap: int -> snapshot past which progenitor identification will be terminated (e.g. 11)')
    print('         nmem: int -> desired number of memory slots (e.g. 5)')
    print('         ncores: int -> number of desired cores (e.g. 20)')
    print()


def get_args():
    '''
    Interprets arguments passed in the command line, checks for errors, and, if successful, returns them as variables to use in the program.

    Output: see printErrorMessage() above
    '''

    try:
        print('*** Attempting to unpack arguments... ***')
        sim, finder, z0, nhalos, mcut, pcut, endsnap, nmem, ncores = sys.argv[1:]
        z0 = float(z0)
        mcut = float(mcut)
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
        printErrorMessage()
        sys.exit()
    else:
        if not (isinstance(nhalos, int) or isinstance(nhalos, np.ndarray)):
            printErrorMessage()
            sys.exit()
        if not (isinstance(sim, str) and isinstance(finder, str) and isinstance(z0, float) and isinstance(mcut, float) and\
                isinstance(pcut, int) and isinstance(endsnap, int) and isinstance(nmem, int) and isinstance(ncores, int)):
            printErrorMessage()
            sys.exit()
        if (finder.lower() != 'rockstar') and (finder.lower() != 'ahf'):
            printErrorMessage()
            raise Exception('! Error: Please select a valid halo finder\n')
        if (mcut < 0.0) or (z0 < 0) or (pcut < 0) or (endsnap < 0) or (nmem < 1):
            raise Exception('! Error: z0, mcut, pcut, and endsnap must be positive numbers. nmem must be an integer 1 or greater\n')
        if ncores > pl.allcores:
            print(f'! Warning: cannot use more cores than available. Defaulting to the maximum number of available cores ({pl.allcores})\n')
            ncores = pl.allcores
        return sim, finder, z0, nhalos, mcut, pcut, endsnap, nmem, ncores


def get_starting_data(args):

    sim, finder, z0, _, mcut, _, endsnap, nmem, _ = args

    path = '/projects/b1026/gjsun/high_redshift/'                   # Path to simulation data

    # Data from the snapshot times file
    print('*** Accessing data from snapshot_times file... ***')
    snaptimes = np.loadtxt(path+sim+'/snapshot_times.txt')          # Snapshot information read in as a numpy array
    z0_idx = np.argmin(np.abs(snaptimes[:, 2] - z0))                # Index of the snapshot that matches z0
    snap0 = int(snaptimes[z0_idx, 0])                               # Number of the snapshot that matches z0
    params = (sim, z0, snap0, endsnap, path, finder, nmem, mcut)    # Parameters to pass into main() function

    # Read in the particle dictionary
    print('***Reading in particle dictionary... ***\n')
    part = gizmo.io.Read.read_snapshots('star', 'redshift', z0, path+sim)
    star_ids, star_pos, star_mass = ht.unpackAndSortPartDict(part, starID_form='string')[:3]
    h = part.Cosmology['hubble']   # Hubble parameter

    # Read in the halo finder data
    print('\n*** Reading in halo finder data... ***')
    hal0 = ht.getData(sim, snap0, finder)
    halo_ids, halo_rad, halo_mvir, halo_pos = ht.unpackData(hal0, sim, snap0, z0, h, finder, mcut=mcut)

    particle_dictionary_data = (star_ids, star_pos, star_mass)
    halo_finder_data = (halo_ids, halo_rad, halo_mvir, halo_pos)

    return params, particle_dictionary_data, halo_finder_data, snaptimes


def star_particle_filter(pcut, nhalos, particle_dictionary_data, halo_finder_data, halo_arr=None):

    _, star_pos, _ = particle_dictionary_data
    halo_ids, halo_rad, halo_mvir, halo_pos = halo_finder_data

    print('\n*** Identifying halos with star particles and applying particle filter... ***')
    # Perform an initial calculation to determine which halos have star particles and filter out the ones that don't
    # For the purposes of this algorithm, here we define a halo as having star particles if there is at least one star particle within
    # half of the virial radius of the halo (since we only track the locations of star particles within the inner half)
    # There is no need to calculate tracks for halos with no star particles as this will result in a track of only -1
    nstars = np.array(list(map(lambda pos, rad: ht.hasStars(pos, rad, star_pos, return_nstars=True), halo_pos, halo_rad)))
    is_pcut = nstars >= pcut
    halo_ids, halo_rad, halo_mvir, halo_pos = halo_ids[is_pcut], halo_rad[is_pcut], halo_mvir[is_pcut], halo_pos[is_pcut, :]
    nstars = nstars[is_pcut]   # For correspondence with the arrays in the line above

    # For 'list' input: handle cases where at least one of the specified halos does not pass the above mass and particle count filters
    # for 'number' input: handle cases where more halos are requested than halos that passed the above mass and particle count filters
    # Passing '0' for nhalos tells the code to calculate tracks for all halos
    if halo_arr is not None:
        failed_filters = ~np.isin(halo_arr, halo_ids)
        if (sum(failed_filters) > 0) and (sum(failed_filters) != len(halo_arr)):
            print(f'\n! Warning: Halos {halo_ids[failed_filters]} failed either the virial mass or particle count filter.')
            print(f'           Calculating progenitor tracks for remaining halos: {halo_arr[~failed_filters]}\n')
            halo_arr = halo_arr[~failed_filters]
            nhalos = len(halo_arr)
        elif (sum(failed_filters) > 0) and (sum(failed_filters) == len(halo_arr)):
            print(f'\n! Error: None of the selected halos pass the star particle threshold. Terminating script... ')
            sys.exit()
    elif (halo_arr is None) and (len(halo_ids) < nhalos):
        print('\n! Warning: Fewer valid halos than number of halos specified.')
        print(f'           Calculating progenitor tracks for {len(halo_ids)} instead of {nhalos}.\n')
        nhalos = len(halo_ids)
    elif nhalos == 0:
        nhalos = len(halo_ids)
    else:
        pass

    halo_finder_data = (halo_ids, halo_rad, halo_mvir, halo_pos)

    return nhalos, nstars, halo_finder_data


def calculate_mainhalo(nstars, nhalos, ncores, halo_finder_data, particle_dictionary_data, params, snaptimes, halo_arr=None):

    halo_ids, halo_rad, halo_mvir, halo_pos = halo_finder_data
    star_ids, star_pos, star_mass = particle_dictionary_data
    sim, _, snap0, endsnap, _, _, _, _ = params

    # The zoom in halo is much larger than any of the others, often by multiple orders of magnitude and takes up the most 
    # computational resources -> parallelizing this individually before running the algorithm on the others should result in a speed up
    # This section of code computes the track for the largest halo, if it is specified, and removes it from the halos to be tracked later
    mainhalo_idx = np.argmax(nstars)
    mainhalo_id = halo_ids[mainhalo_idx]
    if (halo_arr is not None) and (mainhalo_id not in halo_arr):
        #return nhalos, halo_arr, -np.ones(snap0-endsnap+1)
        return nhalos, halo_arr, [2004, 4375, 1981, 1878, 2012, 1901, 2117, 3227, 3282, 2283, 2252, 4314, 5576, 4421, 9918, 9911, 5957, 9989, 6143, 
                                  3709, 3726, 3753, 3769, 3749, 10278, 10288, 10276, 10294, 10376, 4135, 10302, 10130, 4748, 6251, 10047, 9942, 3543, 
                                  10011, 3560, 3505, 3498, 3465, 75, 351, 418, 3609, 4611, 1174, 146, 172, 3178, 188, 3144, 3044, 2991, 2879, 2737]
    else:
        pass
        
    print(f'\n *** Executing progenitor tracking in parallel for central halo: {halo_ids[mainhalo_idx]} ***')
    in_mainhalo50 = ht.inHalo(star_pos, halo_rad[mainhalo_idx], halo_pos[mainhalo_idx, :], fullhalo=False)
    in_mainhalo100 = ht.inHalo(star_pos, halo_rad[mainhalo_idx], halo_pos[mainhalo_idx, :], fullhalo=True)
    mainhalo_mstell50 = np.sum(star_mass[in_mainhalo50])
    mainhalo_mstell100 = np.sum(star_mass[in_mainhalo100])
    mainhalo_data = [halo_ids[mainhalo_idx], halo_rad[mainhalo_idx], halo_mvir[mainhalo_idx], mainhalo_mstell50, mainhalo_mstell100, halo_pos[mainhalo_idx, :]]
    
    create_progenitor_file(mainhalo_data, star_ids, star_pos, snaptimes, params, isparallel=ncores)
    
    #with h5py.File(f'../data/ProgenitorTracks/temp_{sim}_{mainhalo_id}_rockstar_progenitortracks.hdf5', 'r') as mainhalo_file:
    #    mainhalo_progenitors = mainhalo_file['prog.id'][:]
    mainhalo_progenitors = np.array([2004, 4375, 1981, 1878, 2012, 1901, 2117, 3227, 3282, 2283, 2252, 4314, 5576, 4421, 9918, 9911, 5957, 9989, 6143, 3709, 3726, 3753, 
                                     3769, 3749, 10278, 10288, 10276, 10294, 10376, 4135, 10302, 10130, 4748, 6251, 10047, 9942, 3543, 10011, 3560, 3505, 3498, 3465, 75, 
                                     351, 418, 3609, 4611, 1174, 146, 172, 3178, 188, 3144, 3044, 2991, 2879, 2737])
    
    nhalos = nhalos - 1

    if halo_arr is None:
        return nhalos, mainhalo_progenitors
    else:
        mainhalo_idx_haloarr = np.where(halo_arr == halo_ids[mainhalo_idx])[0][0]
        np.delete(halo_arr, mainhalo_idx_haloarr)
        return nhalos, halo_arr, mainhalo_progenitors


def main_loop(nhalos, ncores, halo_finder_data, particle_dictionary_data, params, mainhalo_progenitors, snaptimes, halo_arr=None):

    halo_ids, halo_rad, halo_mvir, halo_pos = halo_finder_data
    star_ids, star_pos, star_mass = particle_dictionary_data

    print('* * * * * * * * * * *')
    print(halo_ids == 179)
    print(halo_mvir[halo_ids == 179])

    # We want to distribute the job of calcualting halo tracks across multiple different cores so that each core calculates one track
    # To do this, we sort the halos by virial mass. On the first loop, we take the first ncores most massive ones, and on the next loop we take the next ncores most massive ones, etc.
    # The data necessary to start the tracking is repackaged into a list of lists, where each element of the list contains [ID, radius, virial_mass, [(x, y, z) position]]
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

        ID_arr = halo_ids[which_halos].astype(int)
        print(ID_arr)
        rad_arr = halo_rad[which_halos]
        mvir_arr = halo_mvir[which_halos]
        pos_arr = halo_pos[which_halos, :]

        if len(ID_arr) == 0:
            print('! Warning: no halos matching the specified criteria were identified. Terminating program... ')
            sys.exit()
        else:
            pass

        # Calculate the initial stellar mass of each halo and repackage halo properties
        print('*** Calculating stellar masses and repackaging halo properties... ***')
        isin_halo_list50 = list(map(lambda rad, pos: ht.inHalo(star_pos, rad, pos, fullhalo=False), rad_arr, pos_arr))
        isin_halo_list100 = list(map(lambda rad, pos: ht.inHalo(star_pos, rad, pos, fullhalo=True), rad_arr, pos_arr))
        mstell_arr50 = np.array(list(map(lambda isin_halo: np.sum(star_mass[isin_halo]), isin_halo_list50)))
        mstell_arr100 = np.array(list(map(lambda isin_halo: np.sum(star_mass[isin_halo]), isin_halo_list100)))
        halodata_block = list(zip(ID_arr, rad_arr, mvir_arr, mstell_arr50, mstell_arr100, pos_arr))

        # Return a list of dictionaries of progenitor tracks for each halo
        print('*** Beginning progenitor tracking... ***')
        pl.mapProcesses(create_progenitor_file, halodata_block, ncores, star_ids, star_pos, snaptimes, params, mainhalo_progenitors=mainhalo_progenitors)


def merge_files(sim, finder):

    print('*** Merging files... ***')

    if f'{sim}_{finder.lower()}_progenitortracks_v3.hdf5' in os.listdir('../data/ProgenitorTracks/'):
        os.system(f'rm ../data/ProgenitorTracks/{sim}_{finder.lower()}_progenitortracks_v3.hdf5')
    else:
        pass

    tempfile_names = [name for name in os.listdir('../data/ProgenitorTracks') if name[:11] == f'temp_{sim}']

    with h5py.File(f'../data/ProgenitorTracks/{sim}_{finder.lower()}_progenitortracks_v3.hdf5', 'w') as master_file:
        for tempfile in tempfile_names:
            with h5py.File(f'../data/ProgenitorTracks/{tempfile}', 'r') as halo_data:

                halo = str(int(halo_data['prog.id'][:][0]))
                master_file.create_group(halo)
                master_file[halo].create_dataset('prog.id', data=halo_data['prog.id'][:])
                master_file[halo].create_dataset('prog.radius', data=halo_data['prog.radius'][:])
                master_file[halo].create_dataset('prog.mvir', data=halo_data['prog.mvir'][:])
                master_file[halo].create_dataset('prog.mstell50', data=halo_data['prog.mstell50'][:])
                master_file[halo].create_dataset('prog.mstell100', data=halo_data['prog.mstell100'][:])
                master_file[halo].create_dataset('prog.position', data=halo_data['prog.position'][:])
                master_file[halo].create_dataset('prog.flag', data=halo_data['prog.flag'][:])

    os.system(f'rm ../data/ProgenitorTracks/temp_{sim}*')

    print('*** Files merged ***')

    return
    

def create_progenitor_file(halo_data, star_ids, star_pos, snaptimes, params, isparallel=1, mainhalo_progenitors=None):
    '''
    Main body of the progenitor-tracing algorithm. This will take one halo and identify its progenitors as far back as it is able to

    Parameters:
        halo_info: list containing [ID, radius, virial_mass, stellar_mass, [(x, y, z) position]]

    Output:
        track_dict: dictionary containing the progenitor track of the halo of interest. The dictionary includes the progenitor IDs,
                    virial masses, stellar masses, and positions
    '''

    _, z0, _, endsnap, _, _, _, mcut = params                           # Expand function parameters
    z0_idx = np.argmin(np.abs(snaptimes[:, 2] - z0))                    # Index of the snapshot that matches z0
    snaps = snaptimes[endsnap:z0_idx+1, 0].astype(int)                  # These are the numbers of each snapshot
    zsnaps = snaptimes[endsnap:z0_idx+1, 2]                             # These are the redshifts of each snapshot
    mcut_arr = ht.getMcutArr(mcut, z0, snaptimes, endsnap)              # These are the cutoff values for the rest of the simulation snapshots

    # Unpack halo info
    _, halrad, _, _, _, halpos = halo_data

    # Find which star particles lie in our starting halo, and their associated distances
    # Use the distances to determine the associated weights
    print('*** Selecting starting set of star particles and calculating weights... ***')
    isin_halo, distances = ht.inHalo(star_pos, halrad, halpos, distances=True, fullhalo=True)
    initstar_ids = star_ids[isin_halo]
    #ranks = np.argsort(distances[isin_halo]) + 1           # Rank by proximity to center of halo -> more gravitationally bound particles are favored
    #initstar_weights = 1 / ranks[isin_halo]**2             # Define weights
    #initstar_weights /= np.sum(initstar_weights)           # Normalize weights
    #initstar_data = [initstar_ids, initstar_weights]
    initstar_weights = 1 / distances[isin_halo]**2
    initstar_weights /= np.sum(initstar_weights)
    initstar_data = [initstar_ids, initstar_weights]

    ThisHaloChain = HaloChain(halo_data, initstar_data, params, mainhalo_progenitors=mainhalo_progenitors)

    # Rename initstar_data at this step to keep naming convention consistent and clear
    starids_desc = initstar_ids
    deschalo_rad = halrad
    deschalo_pos_phys = halpos / (1 + z0)

    # Loop over our arrays of redshift, snapshot, and masscut
    # Each loop will identify our progenitor and some of its properties, then save them to the above arrays
    # The loop will automatically terminate if a progenitor cannot be identified

    print('*** Executing loop... ***')
    for (z, snap, mc) in zip(zsnaps[::-1][1:], snaps[::-1][1:], mcut_arr[::-1][1:]):   # Loops backwards in time

        progenitor_data = ThisHaloChain.step(starids_desc, deschalo_rad, deschalo_pos_phys, z, snap, mc, snaptimes, isparallel)
        ThisHaloChain.update(progenitor_data, snap)

        if progenitor_data is not None:
            _, _, _, prog_rad, _, _, _, prog_pos = progenitor_data
            deschalo_rad = prog_rad
            deschalo_pos_phys = prog_pos / (1+z)   # Physical kpc position of progenitor halo
            starids_desc = np.array(list(ThisHaloChain.stardictionary.keys()))
        else:
            break

    ThisHaloChain.save()

    return


# # # # # # # # # # # # # # # # # # #
# Start of progenitor-tracing code  #
# # # # # # # # # # # # # # # # # # #

# Get initial variables and check if an array or integer was assigned to nhalos
#   -->   Note that z0 and snap0 refer to the snapshot that the algorithm begins tracing at, not snapshot 0 of the simulation   <--
args = get_args()
sim, finder, z0, nhalos, mcut, pcut, endsnap, nmem, ncores = args
if isinstance(nhalos, np.ndarray):
    halo_arr = nhalos
    nhalos = len(halo_arr)
    halo_input = 'list'
else:
    halo_input = 'number'
print('*** Arguments received successfully ***')

params, particle_dictionary_data, halo_finder_data, snaptimes = get_starting_data(args)

if halo_input == 'list':
    nhalos, nstars, halo_finder_data = star_particle_filter(pcut, nhalos, particle_dictionary_data, halo_finder_data, halo_arr=halo_arr)
    nhalos, halo_arr, mainhalo_progenitors = calculate_mainhalo(nstars, nhalos, ncores, halo_finder_data, particle_dictionary_data, params, snaptimes, halo_arr=halo_arr)
    main_loop(nhalos, ncores, halo_finder_data, particle_dictionary_data, params, mainhalo_progenitors, snaptimes, halo_arr=halo_arr)
else:
    nhalos, nstars, halo_finder_data = star_particle_filter(pcut, nhalos, particle_dictionary_data, halo_finder_data)
    print(halo_finder_data[0])
    nhalos, mainhalo_progenitors = calculate_mainhalo(nstars, nhalos, ncores, halo_finder_data, particle_dictionary_data, params, snaptimes)
    main_loop(nhalos, ncores, halo_finder_data, particle_dictionary_data, params, mainhalo_progenitors, snaptimes)

#merge_files(sim, finder)