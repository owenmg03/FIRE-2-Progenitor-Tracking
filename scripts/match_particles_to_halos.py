# Owen Gonzales
# Last modfied: 12 Jan 2026

# This file is run following the completion of trace_progenitors.py. It generates a new star particle catalog where
# each star particle is only matched to a single host halo per snapshot. Data is saved in numpy arrays, making later
# comparisons and indexing much easier. New quantities (formation snapshot and fractional distances) are calculated as well.

import numpy as np
import h5py
import sys
from collections import Counter
from parallelization import Map

# # # # # # #
# Functions # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #

def formtime_to_formsnap(formtime, startsnap=67):
    '''
    Takes an array of star particle formation times and returns the snapshot in which they first appear ("formation" snapshot).

    Parameters:
        formtime:   array of star particle formation times in Gyr
        startsnap:  snapshot at which progenitor tracking began (defaults to 67)
    Output:
        formsnap:   list of formation snapshots for each formtime
    '''

    # Open snapshot times file and get the time (in Gyr) at which each snapshot was taken
    snaptimes = np.loadtxt(f'/projects/b1026/gjsun/high_redshift/{sim}/snapshot_times.txt')
    snapshot_time = snaptimes[:startsnap+1, 3]
    
    # Tile both formtime and snapshot_time. Subtracting these allows us to calculate all particles simulaneously
    # Any negative numbers indicate a snapshot before the existance of the star particle. Summing all of these negatives along the
    # snapshot axis gives us the number of snapshots before particle existance. This is also the index of the "formation" snapshot
    snapshot_time_block = np.tile(snapshot_time, (len(formtime), 1))
    formtime_block = np.tile(formtime, (len(snapshot_time), 1)).T
    formsnap = np.sum(snapshot_time_block - formtime_block < 0., axis=1)

    return list(formsnap)


def starhalodata_to_arrays(snap, halostar_catalog, allprogenitors_block, startsnap=67):
    '''
    This function extracts all data from the halostar catalog at a given snapshot. This data is packaged into one dimensional
    numpy arrays for later use.

    Parameters:
        snap:                   snapshot of interest
        halostar_catalog:       opened hdf5 catalog containing halo and star particle data
        allprogenitors_block:   2D array (nhalos, nsnaps) containing the progenitor tracks of each starting halo
        startsnap:              snapshot at which progenitor tracking begins (defaults to 67)
    Output:
        star_data:              tuple of arrays of star particle ids, formation times, formation snapshots, formation masses
                                hosthalos, star particle masses, and fractional distances. May contain star particles which
                                have been identified to more than one halo
    '''

    # Decoder to decode star particle ids from utf-8 format
    decoder = np.vectorize(lambda x: x.decode('utf-8'))

    # Get progenitor halos at snapshot of interest to loop over
    idx = startsnap - snap
    progenitors = allprogenitors_block[:, idx]
    progenitors = progenitors[progenitors != -1]

    # Initialize empty lists for all quantities
    starids = []
    starformtime = []
    starformsnap = []
    starformmass = []
    hosthalos = []
    starmass = []
    fractional_distances = []
    weights = []

    # Loop through each progenitor halo and flatten/append all data to lists
    # Starformsnap is calculated using the formtime_to_formsnap() function
    for halo in progenitors:
        rvir = halostar_catalog[str(snap)][str(halo)]['halo.rvir'][()]
        starids_thishalo = list(decoder(halostar_catalog[str(snap)][str(halo)]['star.ids'][:]))
        starids += starids_thishalo
        starformtime += list(halostar_catalog[str(snap)][str(halo)]['star.formtime'][:])
        starformsnap += formtime_to_formsnap(halostar_catalog[str(snap)][str(halo)]['star.formtime'][:], startsnap=startsnap)
        starformmass += list(halostar_catalog[str(snap)][str(halo)]['star.formmass'][:])
        hosthalos += [halo] * len(starids_thishalo)
        starmass += list(halostar_catalog[str(snap)][str(halo)]['star.mass'][:])
        fractional_distances += list(halostar_catalog[str(snap)][str(halo)]['star.distance'][:] / rvir)
        inv_sq_dists = 1 / halostar_catalog[str(snap)][str(halo)]['star.distance'][:]**2
        weights += list(inv_sq_dists / sum(inv_sq_dists))

    # Change all lists to arrays
    starids = np.array(starids)
    starformtime = np.array(starformtime)
    starformsnap = np.array(starformsnap)
    starformmass = np.array(starformmass)
    hosthalos = np.array(hosthalos)
    starmass = np.array(starmass)
    fractional_distances = np.array(fractional_distances)
    weights = np.array(weights)

    # Package data into tuple
    star_data = (starids, starformtime, starformsnap, starformmass, hosthalos, starmass, fractional_distances, weights)

    return star_data


def identify_bestfithalo(star, star_data):
    '''
    Given a star particle that is known to be ambiguous, find its best fit host halo and associated data.

    Parameters:
        star:                   ambiguous star particle ID
        star_data:              tuple of data arrays returned by the starhalodata_to_arrays() function (see documentation for more detail)
    Output:
        bestfit_data:           2-D array containing best-fit formation time, formation snapshot, formation masse, host halo, 
                                mass, and fractional distance
    '''

    # Unpack data
    starids, starformtime, starformsnap, starformmass, hosthalos, starmass, fractional_distances, weights = star_data

    # Locate where overlap occurs
    overlap = starids == star
    hosthalo_overlap = hosthalos[overlap]
    frac_dist_overlap = fractional_distances[overlap]
    weight_overlap = weights[overlap]

    # Assign best fit host halo by fraction of virial radius
    #bestfit_idx = np.argmax(weight_overlap)
    bestfit_idx = np.argmin(frac_dist_overlap)
    hosthalo_bestfit = hosthalo_overlap[bestfit_idx]
    fractional_distance_bestfit = frac_dist_overlap[bestfit_idx]

    # This data is not dependent on choice of host halo so each overlap value is identical -> taking the first entry suffices
    starformtime_bestfit = starformtime[overlap][0]
    starformsnap_bestfit = starformsnap[overlap][0]
    starformmass_bestfit = starformmass[overlap][0]
    starmass_bestfit = starmass[overlap][0]

    # Package data into 2-D array
    bestfit_data = np.array([starformtime_bestfit, starformsnap_bestfit, starformmass_bestfit, \
                            hosthalo_bestfit, starmass_bestfit, fractional_distance_bestfit])

    return bestfit_data


def resolve_ambiguity(star_data, starids_ambiguous, ncores):
    '''
    Given a list of ambiguous (identified to more than one host halo) star particles, this function assigns a "best" halo to
    each ambiguous star particle and manipulates the arrays accordingly so that each star particle gets only one entry.

    Parameters:
        sim:                simulation in question (string)
        star_data:          tuple of data arrays returned by the starhalodata_to_arrays() function (see documentation for more detail)
        starids_ambiguous:  array of ids of star particles found inside more than one halo
        ncores:             number of cores used to execute this function
    Output:
        star_data_unique:   contains the same data as star_data but each duplicated star particle has been matched with a best fit host halo
    '''

    # Unpack tuple
    starids, starformtime, starformsnap, starformmass, hosthalos, starmass, fractional_distances, weights = star_data
    
    # Index for unambiguous (only one host halo) star particles
    idx_unambiguous = ~np.isin(starids, starids_ambiguous)

    # Save these to their own arrays
    starids_unambiguous = starids[idx_unambiguous]
    starformtime_unambiguous = starformtime[idx_unambiguous]
    starformsnap_unambiguous = starformsnap[idx_unambiguous]
    starformmass_unambiguous = starformmass[idx_unambiguous]
    hosthalos_unambiguous = hosthalos[idx_unambiguous]
    starmass_unambiguous = starmass[idx_unambiguous]
    fractional_distances_unambiguous = fractional_distances[idx_unambiguous]

    # Loop through each ambiguous star particle and find best fit host halo
    # Parallelized for faster execution (this is the bottleneck of this program)
    #bestfit_data = -np.ones((len(starids_ambiguous), 6))
    #for i, star in enumerate(starids_ambiguous):
    #    print(f'{i+1}, {len(starids_ambiguous)}')
    #    bestfit_data[i, :] = identify_bestfithalo(star, star_data)
    bestfit_data = Map(identify_bestfithalo, starids_ambiguous, ncores, star_data)
    bestfit_data = np.array(bestfit_data).T

    # Split data into individual arrays
    starformtime_ambiguous = bestfit_data[0, :]
    starformsnap_ambiguous = bestfit_data[1, :]
    starformmass_ambiguous = bestfit_data[2, :]
    hosthalos_ambiguous = bestfit_data[3, :]
    starmass_ambiguous = bestfit_data[4, :]
    fractional_distances_ambiguous = bestfit_data[5, :]

    # Concatenate both arrays
    starids_unique = np.concatenate((starids_unambiguous, starids_ambiguous))
    starformtime_unique = np.concatenate((starformtime_unambiguous, starformtime_ambiguous))
    starformsnap_unique = np.concatenate((starformsnap_unambiguous, starformsnap_ambiguous))
    starformmass_unique = np.concatenate((starformmass_unambiguous, starformmass_ambiguous))
    hosthalos_unique = np.concatenate((hosthalos_unambiguous, hosthalos_ambiguous))
    starmass_unique = np.concatenate((starmass_unambiguous, starmass_ambiguous))
    fractional_distances_unique = np.concatenate((fractional_distances_unambiguous, fractional_distances_ambiguous))

    # Package data into tuple
    star_data_unique = (starids_unique, starformtime_unique, starformsnap_unique, starformmass_unique, \
                        hosthalos_unique, starmass_unique, fractional_distances_unique)
    
    return star_data_unique


def expand_arrays(arrays, id_to_idx):
    '''
    Dynamically expands the size of arrays holding star particle data as new star particles are encountered

    Parameters:
        arrays:             arrays to be resized. Contains star particle ids, formation times, formation snapshots, formation masses, host halos, masses, and fractional distances
        id_to_idx:          dictionary for indexing the location of star particle ids in the starids_seen array
    Output:
        expanded_arrays     the same arrays as input but expanded to account for new star particles encountered
        id_to_idx:          dictionary with new indices for new star particles added
    '''

    # Unpack data
    starids_seen, starformtime_seen, starformsnap_seen, starformmass_seen, hosthalos_block, starmass_block, fractional_distances_block = arrays

    # Create temporary arrays of the correct dimensions
    starformtime_seen_temp = -np.ones(len(starids_seen)+len(starids_new))
    starformsnap_seen_temp = -np.ones(len(starids_seen)+len(starids_new))
    starformmass_seen_temp = -np.ones(len(starids_seen)+len(starids_new))
    hosthalos_block_temp = -np.ones((len(starids_seen)+len(starids_new), len(snaps)))
    starmass_block_temp = -np.ones((len(starids_seen)+len(starids_new), len(snaps)))
    fractional_distances_block_temp = -np.ones((len(starids_seen)+len(starids_new), len(snaps)))

    # Populate with old data
    starformtime_seen_temp[:len(starids_seen)] = starformtime_seen
    starformsnap_seen_temp[:len(starids_seen)] = starformsnap_seen
    starformmass_seen_temp[:len(starids_seen)] = starformmass_seen
    hosthalos_block_temp[:len(starids_seen), :] = hosthalos_block
    starmass_block_temp[:len(starids_seen), :] = starmass_block
    fractional_distances_block_temp[:len(starids_seen), :] = fractional_distances_block

    # Populate 1-D arrays with the snapshot-independent data for new star particles
    starformtime_seen_temp[len(starids_seen):] = starformtime_unique[new_stars]
    starformsnap_seen_temp[len(starids_seen):] = starformsnap_unique[new_stars]
    starformmass_seen_temp[len(starids_seen):] = starformmass_unique[new_stars]

    # Add new indices to id_to_idx dictionary
    for i, star in enumerate(starids_new):
        id_to_idx[star] = len(starids_seen) + i

    # Re-assign temporary arrays to main arrays
    starids_seen = np.concatenate((starids_seen, starids_new))
    starformtime_seen = starformtime_seen_temp
    starformsnap_seen = starformsnap_seen_temp
    starformmass_seen = starformmass_seen_temp
    hosthalos_block = hosthalos_block_temp
    starmass_block = starmass_block_temp
    fractional_distances_block = fractional_distances_block_temp

    # Repack data
    expanded_arrays = (starids_seen, starformtime_seen, starformsnap_seen, starformmass_seen, hosthalos_block, starmass_block, fractional_distances_block)

    return expanded_arrays, id_to_idx


def find_earliest_halo(allprogenitors_block, hosthalos_block, startsnap, endsnap, particle_threshold=10):
    '''
    Given a track of best-progenitors and the snapshot at which it starts, returns the snapshot of the earliest identified progenitor.

    Parameters:
        progenitor_track:           array containing the rockstar halo catalog IDs of each progenitor, in reverse chronological order
        hosthalos_block:            array containing the halo to which each star particle belongs at each snapshot, in reverse chronological order
        startsnap:                  snapshot at which progenitor tracking begins
        endsnap:                    snapshot at which progenitor tracking ends
    Output:
        earlist_snapid_pairs:       snapshot at which the earliest progenitor is identified
    '''
    
    # Initialize array of all snapshots looped and list that will hold 
    snaps = np.arange(endsnap, startsnap+1)
    earliest_snapids = []

    # Loop through progenitor tracks
    for progenitor_track in allprogenitors_block:

        # Reorder to chronological and find snapshots with no identified progenitor
        progenitor_track = progenitor_track[::-1]
        has_halo = progenitor_track != -1

        # Loop through snapshots with identified progenitors
        for (snap, prog) in zip(snaps[has_halo], progenitor_track[has_halo]):

            idx = startsnap - snap
            nstars = sum(hosthalos_block[:, idx] == prog)

            # Find the earliest snapshot where the progenitor halo crosses the particle threshold
            if nstars >= particle_threshold:
                earliest_snapids.append([snap, prog, progenitor_track[-1]])
                break
            else:
                pass
            
    return np.array(earliest_snapids)


def assign_insitu(starids_seen, starformsnap_seen, hosthalos_block, allprogenitors_block, startsnap, endsnap):
    '''
    This function determines the in-situ status of each star particle in the simulation encountered by our algorithm. It is run after the star particle 
    and host halo matching has occurred and all other arrays of interest have been calculated.

    Parameters:
        starids_seen:           array of all star particle IDs encountered by the algorithm (meaning they belong to at least one progenitor at some point between starting and ending snapshots)
        starformsnap_seen:      array of formation snapshots for each star particle, corresponding with starids_seen
        hosthalos_block:        array of (nstars, nsnapshots) containing the hosthalo of each star particle at every snapshot, corresponding with starids_seen
        allprogenitors_block:   array of (nhalos, nsnapshots) containing the progenitor halos of each halo identified at startsnap at every snapshot
        startsnap:              snapshot at which progenitor tracking begins
        endsnap:                snapshot at which progenitor tracking ends
    Output:
        star_insitu             array of all star particle in-situ flags, corresponding with starids_seen
    '''

    # Initialize empty array -> will contain in-situ flags
    # -1:   In-situ status cannot be determined (due to not belonging to a halo of sufficient mass at snapshot of origin)
    #  0:   Ex-situ (formed inside of an identified halo but it is not the progenitor)
    #  1:   In-situ (formed inside a progenitor of the halo to which it belongs at the starting snapshot)
    star_insitu = -np.ones(len(starids_seen))

    # Loop through the rest of the snapshots to determine if the star particles that formed during that snapshot formed in an identified progenitor
    snaps = np.arange(endsnap, startsnap)   # Purposefully leaving out last snapshot
    for i, snap in enumerate(snaps[::-1]):

        # Progenitor and final halo pairs
        insitu_pairs = allprogenitors_block[:, np.array([i, 0])]
        insitu_pairs = insitu_pairs[insitu_pairs[:, 0] != -1]

        # Host halo, final halo pairs for each star particle formed during this snapshot
        if snap != endsnap:
            newstars_idxs = starformsnap_seen == snap
        else:
            newstars_idxs = starformsnap_seen <= snap
        newstars_hosthalo = hosthalos_block[newstars_idxs, i]
        newstars_finalhalo = hosthalos_block[newstars_idxs, 0]
        newstars_halopairs = np.array([newstars_hosthalo, newstars_finalhalo]).T   # Dimensions (sum(newstars_idxs), 2)

        # Remove star particles which are missing an identified halo -> these insitu flags will be left as -1
        newstars_nohalo = (newstars_halopairs[:, 0] == -1) | (newstars_halopairs[:, 1] == -1)

        # Convert this to a string format for ease of comparison
        insitu_pairs = np.array([f'{prog}.{desc}' for (prog, desc) in insitu_pairs])
        newstars_halopairs = np.array([f'{prog}.{desc}' for (prog, desc) in newstars_halopairs])

        # Determine which final halo, formation halo pairs match the identified progenitors
        newstars_insitu = np.isin(newstars_halopairs, insitu_pairs)
        isnot_insitu = np.in1d(starids_seen, starids_seen[newstars_idxs][~newstars_nohalo & ~newstars_insitu])
        is_insitu = np.in1d(starids_seen, starids_seen[newstars_idxs][~newstars_nohalo & newstars_insitu])

        # Assign in-situ flags
        star_insitu[isnot_insitu] = 0   # This star particle did not form in one of the identified progenitors of the halo it ended up in
        star_insitu[is_insitu] = 1   # This star particle formed in one of the identified progenitors of the halo it ended up in
        
    # Any star that is present in the earliest identified progenitor with particle_threshold star particles is automatically considered to have formed in-situ
    earliest_snaphalo = find_earliest_halo(allprogenitors_block, hosthalos_block, startsnap, endsnap)
    for (snap, earliest_progenitor, descendant) in earliest_snaphalo:
        earliest_progenitor_stars = hosthalos_block[:, startsnap-snap] == earliest_progenitor
        in_descendant = hosthalos_block[:, 0] == descendant
        star_insitu[earliest_progenitor_stars & in_descendant] == 1
        
    # Define "in-situ" for star particles formed by starting snapshot or before ending snapshot
    formed_finalsnap = starformsnap_seen == startsnap   # Any star particle formed in the starting snapshot is guarenteed to be in-situ by our definition
    formed_early = starformsnap_seen < endsnap   # We assume any star particle formed before ending snapshot was formed in-situ for simplicity
    has_descendant = hosthalos_block[:, -1] != -1   # Must also be identified with a halo at ending snapshot
    star_insitu[formed_finalsnap | (formed_early & has_descendant)] = 1


    return star_insitu

# # # # # # # # # # #
# Main body of code # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #

# Data needed to execute particle matching
sim = sys.argv[1]
ncores = int(sys.argv[2])
#sim = 'z5m11d'
#ncores = 1
startsnap = 67
endsnap = 11

# Initialize empty dictionary for preserving indices
# (this ensures that star particles don't get mixed up between snapshots when the number of star particles is dynamic)
id_to_idx = {}

# Create (nhalos, nsnaps) array containing the each progenitor halo at each snapshot
with h5py.File(f'../data/ProgenitorTracks/{sim}_progenitortracks.hdf5', 'r') as progenitor_catalog:
    descendants = np.array(list(progenitor_catalog.keys()))
    descendants_startsnap = np.array([progenitor_catalog[halo]['prog.id'][67-startsnap].astype(int) for halo in descendants])
    descendants_startsnap = descendants_startsnap[descendants_startsnap != -1]
    
    allprogenitors_block = np.array([progenitor_catalog[halo]['prog.id'][67-startsnap:].astype(int) for halo in descendants[descendants_startsnap != -1]])

# Loop over snapshots (backwards in time)
snaps = np.arange(endsnap, startsnap+1)
for i, snap in enumerate(snaps[::-1]):

    # Open halostar catalog
    with h5py.File(f'../data/HaloStarCatalogs/halostarcatalog_{sim}.hdf5', 'r') as halostar_catalog:
        
        # Extract data from halostar catalog
        star_data = starhalodata_to_arrays(snap, halostar_catalog, allprogenitors_block, startsnap=startsnap)
        starids, starformtime, starformsnap, starformmass, hosthalos, starmass, fractional_distances, weights = star_data

        # Remove any assignments to halos not seen in progenitor lineages at this snapshot
        available_progenitors = allprogenitors_block[:, i][allprogenitors_block[:, i] != -1]
        is_progenitor = np.isin(hosthalos, available_progenitors)
        
        starids = starids[is_progenitor]
        starformtime = starformtime[is_progenitor]
        starformsnap = starformsnap[is_progenitor]
        starformmass = starformmass[is_progenitor]
        hosthalos = hosthalos[is_progenitor]
        starmass = starmass[is_progenitor]
        fractional_distances = fractional_distances[is_progenitor]
        weights = weights[is_progenitor]

        # Identify any star particles included more than once (more than one identified host halo)
        starids_count = Counter(starids)
        starids_ambiguous = np.array([star for star, count in starids_count.items() if count > 1])

        # Resolve any ambiguity in host halos and create new arrays in which star particles are unique (only one host halo)
        if len(starids_ambiguous) != 0:
            print()
            print(snap)
            star_data_unique = resolve_ambiguity(star_data, starids_ambiguous, ncores)
            starids_unique, starformtime_unique, starformsnap_unique, starformmass_unique, hosthalos_unique, starmass_unique, fractional_distances_unique = star_data_unique
        else:
            starids_unique = starids
            starformtime_unique = starformtime
            starformsnap_unique = starformsnap
            starformmass_unique = starformmass
            hosthalos_unique = hosthalos
            starmass_unique = starmass
            fractional_distances_unique = fractional_distances

        # Initialize final arrays (maintained across snapshot loop) if this is the first interation. Otherwise add data to arrays
        if i == 0:
            
            # 1-D arrays that contain data that does not change between snapshots
            starids_seen = np.copy(starids_unique)
            starformtime_seen = np.copy(starformtime_unique)
            starformsnap_seen = np.copy(starformsnap_unique)
            starformmass_seen = np.copy(starformmass_unique)

            # 2-D arrays that contain data on a snapshot-by-snapshot basis
            hosthalos_block = -np.ones((len(hosthalos_unique), len(snaps)))
            starmass_block = -np.ones((len(hosthalos_unique), len(snaps)))
            fractional_distances_block = -np.ones((len(hosthalos_unique), len(snaps)))

            hosthalos_block[:, 0] = hosthalos_unique
            starmass_block[:, 0] = starmass_unique
            fractional_distances_block[:, 0] = fractional_distances_unique

            # Add indices to id_to_idx dictionary
            for j, star in enumerate(starids_seen):
                id_to_idx[star] = j

        else:
            
            # Find any previously unsees star particles
            new_stars = ~np.isin(starids_unique, starids_seen)
            starids_new = starids_unique[new_stars]

            # Resize arrays to accomodate any new particles
            if len(starids_new) != 0:
                arrays = starids_seen, starformtime_seen, starformsnap_seen, starformmass_seen, hosthalos_block, starmass_block, fractional_distances_block
                expanded_arrays, id_to_idx = expand_arrays(arrays, id_to_idx)
                starids_seen, starformtime_seen, starformsnap_seen, starformmass_seen, hosthalos_block, starmass_block, fractional_distances_block = expanded_arrays
            else:
                pass
            
            # Find the main array indices of all star particles seen this snapshot
            # Use these indices to place snapshot-dependent data for the stars into the 2-D arrays
            update_idxs = np.array([id_to_idx[star] for star in starids_unique])
            hosthalos_block[update_idxs, i] = hosthalos_unique
            starmass_block[update_idxs, i] = starmass_unique
            fractional_distances_block[update_idxs, i] = fractional_distances_unique

# Determine the in-situ status of each star particle
# (In the snapshot that the star particle is first seen, is it inside the best progenitor of its host halo at <startsnap>)
hosthalos_block = hosthalos_block.astype(int)
starformsnap_seen = starformsnap_seen.astype(int)
star_insitu = assign_insitu(starids_seen, starformsnap_seen, hosthalos_block, allprogenitors_block, startsnap, endsnap)

# Save data to hdf5 file
with h5py.File(f'../data/ParticleHaloMatchCatalogs/uniquematchcatalog_{sim}.hdf5', 'w') as catalog:

    # Encoder to encode star particle ids in utf-8 format
    encoder = np.vectorize(lambda x: x.encode('utf-8'))

    catalog.create_dataset('star.ids', data=encoder(starids_seen), dtype=h5py.string_dtype(encoding='utf-8'))
    catalog.create_dataset('star.formtime', data=starformtime_seen)
    catalog.create_dataset('star.formsnap', data=starformsnap_seen)
    catalog.create_dataset('star.formmass', data=starformmass_seen)
    catalog.create_dataset('star.insitu', data=star_insitu)
    catalog.create_dataset('star.hosthalo', data=hosthalos_block)
    catalog.create_dataset('star.mass', data=starmass_block)
    catalog.create_dataset('star.fractionaldistance', data=fractional_distances_block)