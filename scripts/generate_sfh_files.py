# Owen Gonzales
# Last modfied: 12 Jan 2026

# This file calculates the total archaeological and in-situ archaeological SFHs and sSFHs for every halo in each simulation
# Data for each simulation is saved to an hdf5 file

import numpy as np
import h5py
from parallelization import Map

# # # # # # #
# Functions # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #

def generate_halo_sfh(sim, halo, stardata, bwidth, startsnap=67):
    '''
    This function returns a total archaeological and in-situ archaeological SFH for a given simulation and halo.

    Parameters:
        sim:                simulation that halo of interest belongs to
        halo:               halo to have SFHs calculated
        stardata:           tuple containing:
            starhosthalo:   2-D array of the host halo of each star particle at every snapshot
            starfracdist:   2-D array of the distance as a fraction of virial radius from each star particle to its host halo at every snapshot
            starformtime:   array of formation times for each star particle
            starfrommass:   array of formation masses for each star particle
            starinsitu:     array of in-situ flags for each star particle
        bwidth:             bin width for SFH in Myr
        startsnap:          snapshot at which progenitor tracking begins (defaults to 67)
    Output:
        sfh_time:           time points corresponding to SFR values (histogram bin midpoints)
        sfh_total:          array of SFR values for the total archaeological SFH
        sfh_insitu:         array of SFR values for the in-situ archaeological SFH
    '''
    
    starhosthalo, starfracdist, _, starformtime, starformmass, starinsitu = stardata

    snaptimes = np.loadtxt(f'/projects/b1026/gjsun/high_redshift/{sim}/snapshot_times.txt')
    starttime = snaptimes[startsnap, 3]

    # Total archaeological SFH
    infinal50 = (starhosthalo[:, 0] == halo) & (starfracdist[:, 0] <= 0.5)
    infinal100 = starhosthalo[:, 0] == halo
    formmass_total50 = starformmass[infinal50]
    formtime_total50 = starformtime[infinal50]
    formmass_total100 = starformmass[infinal100]
    formtime_total100 = starformtime[infinal100]
        
    # In-situ archaeological SFH
    formmass_insitu50 = starformmass[infinal50 & (starinsitu==1)]
    formtime_insitu50 = starformtime[infinal50 & (starinsitu==1)]
    formmass_insitu100 = starformmass[infinal100 & (starinsitu==1)]
    formtime_insitu100 = starformtime[infinal100 & (starinsitu==1)]

    # Define evenly spaced bins based on bwidth parameter (in Myr)
    bwidth_gyr = bwidth/1e3
    bins = np.arange(0, starttime + bwidth_gyr, bwidth_gyr)
    bins = bwidth_gyr * np.arange(np.round(starttime/bwidth_gyr)+1)
    sfh_time = (bins[:-1] + bins[1:]) / 2

    # Histograms for each type of SFH
    sfh50_total, _ = np.histogram(formtime_total50, bins=bins, weights=formmass_total50/(bwidth*1e6))
    sfh50_insitu, _ = np.histogram(formtime_insitu50, bins=bins, weights=formmass_insitu50/(bwidth*1e6))
    sfh100_total, _ = np.histogram(formtime_total100, bins=bins, weights=formmass_total100/(bwidth*1e6))
    sfh100_insitu, _ = np.histogram(formtime_insitu100, bins=bins, weights=formmass_insitu100/(bwidth*1e6))

    return sfh_time, sfh50_total, sfh50_insitu, sfh100_total, sfh100_insitu


def check_for_flag(sim, sfhdata, mstell50, threshtime100, bwidth, particle_threshold=10, lower=0.1, upper=10):
    '''
    Checks for problematic lineages by checking if the interpolated mass falls outside of bounds set by the integrated
    total (upper bound) and in-situ (lower bound) archaeological SFHs
    
    Parameters:
        sim:                        simulation that halo of interest belongs to
        sfhdata:                    see calculate_interpolated_mass()
        mstell50:                   interpolated mass (50% of virial radius)
        threshtime100:              time at which number of star particles within the virial radius crosses the particle threshold
        bwidth:                     bin width for SFH in Myr
        particle_threshold:         minimum number of star particles that a halo can have before progenitor tracking terminates (defaults to 10)
        lower:                      coefficient multiplied to integrated in-situ archaeological SFH to get lower mass bound
        upper:                      coefficient multiplied to integrated total archaeological SFH to get lower mass bound
    Output:
        flag:                       True if lineage is problematic, otherwise False
    '''
    
    # Upack data
    sfh_time, sfh50_total, sfh50_insitu, _, _ = sfhdata
    
    mass_res_dict = {'z5m11a': 954.,
                     'z5m11b': 890.,
                     'z5m11c': 813.,
                     'z5m11d': 7162.,
                     'z5m11e': 5557.,
                     'z5m11f': 7162.,
                     'z5m11g': 7162.,
                     'z5m11h': 7162.,
                     'z5m11i': 890,
                     'z5m12a': 7126.,
                     'z5m12b': 7126.,
                     'z5m12c': 0,
                     'z5m12d': 0,
                     'z5m12e': 0}
    mass_res = mass_res_dict[sim]
    
    # Integrate the total archaeological SFH to get total archaeological interpolated mass
    # Ignore points before the particle threshold
    mstell50_interp_t = (bwidth*1e6) * np.cumsum(sfh50_total)
    mstell50_interp_t[sfh_time < threshtime100] = np.nan
    
    # Integrate the in-situ archaeological SFH to get in-situ archaeological interpolated mass
    # Ignore points before the particle threshold
    mstell50_interp_i = (bwidth*1e6) * np.cumsum(sfh50_insitu)
    mstell50_interp_i[sfh_time < threshtime100] = np.nan
    
    # Set upper and lower mass bounds
    lower_mass_bound_50 = lower * mstell50_interp_i
    upper_mass_bound_50 = upper * mstell50_interp_t
    
    # Cannot be lower than particle resolution
    lower_mass_bound_50[lower_mass_bound_50 < particle_threshold*mass_res/2] = particle_threshold*mass_res/2
    
    # Check for flagged times
    filt50 = ~np.isnan(mstell50)
    flag50 = (mstell50[filt50] < lower_mass_bound_50[filt50]) | (mstell50[filt50] > upper_mass_bound_50[filt50])
    
    flag = sum(flag50) > 0
    
    return flag


def calculate_interpolated_mass(sim, halo, sfhdata, stardata, bwidth, startsnap=67, endsnap=11, particle_threshold=10):
    '''
    Performs linear interpolation of progenitor stellar masses in log-space (50% and 100% of the virial radius).
    Also flags for problematic lineages (where the progenitor halo is more than upper*integrated_total_sfh or 
    less than lower*integrated_insitu_sfh)

    Parameters:
        sim:                    simulation that halo of interest belongs to
        halo:                   halo to have SFHs calculated
        sfhdata:                tuple containing:
            sfh_time:           time points for SFH array
            sfh50_total:        total archaeological SFH, 50% of virial radius
            sfh50_insitu        insitu archaeological SFH, 50% of virial radius
            sfh100_total:       total archaeological SFH, 100% of virial radius
            sfh100_insitu       insitu archaeological SFH, 100% of virial radius
        stardata:               tuple containing:
            starhosthalo:       2-D array of the host halo of each star particle at every snapshot
            starfracdist:       2-D array of the distance as a fraction of virial radius from each star particle to its host halo at every snapshot
            starmass:           2-D array of snapshot-by-snapshot masses of each star particle
            starformtime:       array of formation times for each star particle
            starfrommass:       array of formation masses for each star particle
            starinsitu:         array of in-situ flags for each star particle
        bwidth:                 bin width for SFH in Myr
        startsnap:              snapshot at which progenitor tracking begins (defaults to 67)
        endsnap:                snapshot at whcih progenitor tracking ends (defaults to 11)
        particle_threshold:     minimum number of star particles that a halo can have before progenitor tracking terminates (defaults to 10)
    Output:
        mstell50_time:          time points corresponding to mstell50_interp values (histogram bin midpoints)
        mstell50_interp:        linearly interpolated mstell50 values
        flag:                   True if lineage is problematic, otherwise False
    '''
    
    # Get cosmic time of snapshots in Gyr
    snaptimes = np.loadtxt(f'/projects/b1026/gjsun/high_redshift/{sim}/snapshot_times.txt')
    snaps = snaptimes[endsnap:startsnap+1, 0].astype(int)
    snaptimes = snaptimes[endsnap:startsnap+1, 3]   # These correspond to the data that we get from the progenitor tracks file
    
    # Unpack data
    sfh_time, sfh50_total, sfh50_insitu, sfh100_total, sfh100_insitu = sfhdata
    starhosthalo, starfracdist, starmass, starformtime, starformmass, starinsitu = stardata
    starhosthalo = np.flip(starhosthalo, 1)   # Flip snapshot axis for less confusing indexing (chronological)
    starfracdist = np.flip(starfracdist, 1)
    starmass = np.flip(starmass, 1)
    
    # Get snapshot-by-snapshot progenitors
    with h5py.File(f'../data/ProgenitorTracks/{sim}_progenitortracks.hdf5', 'r') as catalog:
        progenitors = catalog[str(halo)]['prog.id'][:][::-1]   # Also flipped to chronological order
        first_prog_time = snaptimes[progenitors != -1][0]
        first_prog_idx = sum(snaptimes < first_prog_time)
    
    # Filter arrays to include only points with an identified progenitor
    mstell_times_bysnap = snaptimes[first_prog_idx:]
    snaps = snaps[first_prog_idx:]
    progenitors = progenitors[first_prog_idx:]
    
    mstell_times_bysnap = mstell_times_bysnap[progenitors != -1]
    snaps = snaps[progenitors != -1]
    progenitors = progenitors[progenitors != -1]
    
    # Initialize arrays
    mstell50_bysnap = np.zeros(len(mstell_times_bysnap))
    mstell100_bysnap = np.zeros(len(mstell_times_bysnap))
    nstars50_bysnap = np.zeros(len(mstell_times_bysnap))
    nstars100_bysnap = np.zeros(len(mstell_times_bysnap))
    
    # Loop through progenitors, finding the stellar mass (inner 50% and full virial radius) at each snapshot where
    # there is an identified progenitor and at least particle_threshold star particles
    for i, (prog, snap) in enumerate(zip(progenitors, snaps)):
        
        idx = snap - endsnap
        in_prog = starhosthalo[:, idx] == prog
        in_50 = starfracdist[:, idx] <= 0.5
        
        if i == 0:
            in_prog_thresh = np.copy(in_prog)   # Will be used later
            in_50_thresh = np.copy(in_50)   # Will be used later
            threshtime100 = np.sort(starformtime[in_prog_thresh])[particle_threshold-1]
            if sum(in_prog_thresh & in_50) >= particle_threshold:
                threshtime50 = np.sort(starformtime[in_prog_thresh & in_50_thresh])[particle_threshold-1]
            else:
                threshtime50 = first_prog_time
        else:
            pass
        
        mstell50_bysnap[i] = np.log10(np.sum(starmass[(in_prog & in_50), idx]))
        mstell100_bysnap[i] = np.log10(np.sum(starmass[in_prog, idx]))
        nstars50_bysnap[i] = sum(in_prog & in_50)
        nstars100_bysnap[i] = sum(in_prog)
    
    # Interpolate the snapshot-spaced mass points to the SFH time points
    mstell_interp_time = sfh_time[sfh_time >= first_prog_time]
    mstell50_interp = np.interp(mstell_interp_time, mstell_times_bysnap, mstell50_bysnap)
    mstell100_interp = np.interp(mstell_interp_time, mstell_times_bysnap, mstell100_bysnap)
    
    # Initialize bins for the mini-archaeological SFH
    # (this is pointless if the progenitor tracker terminates because of too few star particles, but it is easier to do it anyway and not build a switch)
    bwidth_gyr = bwidth/1e3
    bins = np.arange(0, np.round(mstell_interp_time[0]/bwidth_gyr + bwidth_gyr/2))
    bins *= bwidth_gyr
    
    # Create a mini-archaeological SFH with just the star particles present at the earliest snapshot with an identified
    # progenitor and sufficient star particles. Integrate to get stellar mass
    thresh_sfh50, _ = np.histogram(starformtime[in_prog_thresh & in_50_thresh], bins=bins, weights=starmass[in_prog_thresh & in_50_thresh, first_prog_idx]/(bwidth*1e6))
    thresh_sfh100, _ = np.histogram(starformtime[in_prog_thresh], bins=bins, weights=starmass[in_prog_thresh, first_prog_idx]/(bwidth*1e6))
    thresh_sfh50_int = (bwidth*1e6) * np.cumsum(thresh_sfh50)
    thresh_sfh100_int = (bwidth*1e6) * np.cumsum(thresh_sfh100)
    
    # Concatenate both arrays together and remove points before the particle threshold is crossed
    mstell50_interp = np.concatenate((np.log10(thresh_sfh50_int), mstell50_interp))
    mstell100_interp = np.concatenate((np.log10(thresh_sfh100_int), mstell100_interp))
    
    # Identify the points at which the halo has not yet crossed the particle threshold for both inner 50% and 100% of virial radius
    # Save all data here in terms of <pcut50> for ease of future use.
    pcut50 = sfh_time < threshtime50
    pcut100 = sfh_time < threshtime100
    sfh50_total[pcut50] = np.nan
    sfh50_insitu[pcut50] = np.nan
    sfh100_total[pcut50] = np.nan
    sfh100_insitu[pcut50] = np.nan
    mstell50_interp[pcut50] = np.nan
    mstell100_interp[pcut100] = np.nan
    
    # Interpolate over points where mass is 0 --> these are errors
    mstell50_interp[~pcut50] = np.interp(sfh_time[~pcut50], sfh_time[np.isfinite(mstell50_interp)], mstell50_interp[np.isfinite(mstell50_interp)])
    mstell100_interp[~pcut100] = np.interp(sfh_time[~pcut100], sfh_time[np.isfinite(mstell100_interp)], mstell100_interp[np.isfinite(mstell100_interp)])
    
    # Check for problematic lineages
    flag = check_for_flag(sim, sfhdata, mstell50_interp, threshtime50, bwidth)
    
    return (mstell50_interp, mstell100_interp), sfhdata, flag


def generate_sfh_file(sim, bwidth):
    '''
    Calculates total archaeological and in-situ archaeological SFHs and sSFHs for every halo in a given simulation. Saves data to an hdf5 file.

    Parameters:
        sim:        simulation of interest
        bwidth:     bin width for SFH in Myr
    Output:
        None
    '''

    # Read in star particle data from match_particles_to_halos.py output catalog
    with h5py.File(f'../data/ParticleHaloMatchCatalogs/uniquematchcatalog_{sim}.hdf5', 'r') as catalog:
         
        starhosthalo = catalog['star.hosthalo'][:]
        starfracdist = catalog['star.fractionaldistance'][:]
        starmass = catalog['star.mass'][:]
        starformtime = catalog['star.formtime'][:]
        starformmass = catalog['star.formmass'][:]
        starinsitu = catalog['star.insitu'][:]
        stardata = (starhosthalo, starfracdist, starmass, starformtime, starformmass, starinsitu)
    
    # Save IDs of halos at final snapshot
    finalhalos = np.unique(starhosthalo[:, 0])
    finalhalos = finalhalos[finalhalos != -1]   # Remove errors
    
    # Remove halos with an in-situ fraction of 0. These are errors.
    bad_tracks = []
    for halo in finalhalos:
        idxs = starhosthalo[:, 0] == halo
        if not sum(starinsitu[idxs] == 1) > 0:
            bad_tracks.append(halo)
        else:
            pass
    bad_tracks = np.array(bad_tracks)
    finalhalos = finalhalos[~np.isin(finalhalos, bad_tracks)]
    
    # Loop through each halo and calculate SFHs
    with h5py.File(f'../data/SFHs/sfh_{str(bwidth)}bw_{sim}.hdf5', 'w') as sfhfile:
        
        for i, halo in enumerate(finalhalos):

            print(f'{i+1}/{len(finalhalos)}: {halo}')

            # Calcualte SFHs
            sfhdata = generate_halo_sfh(sim, halo, stardata, bwidth)
            sfh_time, sfh50_total, sfh50_insitu, sfh100_total, sfh100_insitu = sfhdata

            # Calculate interpolated mass and add nan points to SFHs
            return_val = calculate_interpolated_mass(sim, halo, sfhdata, stardata, bwidth)
            interpolated_masses, sfhdata, flag = return_val
            mstell50_interp, mstell100_interp = interpolated_masses
            sfh_time, sfh50_total, sfh50_insitu, sfh100_total, sfh100_insitu = sfhdata

            if flag:
                print(f'Halo mass flag warning: {halo}')
            else:
                pass
                
            # Calculate sSFHs using interpolated mass
            ssfh50_total = sfh50_total / 10**mstell50_interp
            ssfh50_insitu = sfh50_insitu / 10**mstell50_interp
            ssfh100_total = sfh100_total / 10**mstell100_interp
            ssfh100_insitu = sfh100_insitu / 10**mstell100_interp

            # Save data to .hdf5 file
            sfhfile.create_group(str(halo))
            sfhfile[str(halo)].create_dataset('sfh.time', data=np.round(sfh_time, 3))
            sfhfile[str(halo)].create_dataset('sfh.total50', data=sfh50_total)
            sfhfile[str(halo)].create_dataset('sfh.insitu50', data=sfh50_insitu)
            sfhfile[str(halo)].create_dataset('sfh.total100', data=sfh100_total)
            sfhfile[str(halo)].create_dataset('sfh.insitu100', data=sfh100_insitu)
            sfhfile[str(halo)].create_dataset('ssfh.total50', data=ssfh50_total)
            sfhfile[str(halo)].create_dataset('ssfh.insitu50', data=ssfh50_insitu)
            sfhfile[str(halo)].create_dataset('ssfh.total100', data=ssfh100_total)
            sfhfile[str(halo)].create_dataset('ssfh.insitu100', data=ssfh100_insitu)
            sfhfile[str(halo)].create_dataset('mstell.interp50', data=mstell50_interp)
            sfhfile[str(halo)].create_dataset('mstell.interp100', data=mstell100_interp)

    return

# # # # # # # # # # #
# Main body of code # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #

# Parameters
sims = ['z5m11a', 'z5m11b', 'z5m11c', 'z5m11d', 'z5m11e', 'z5m11f', 'z5m11g', 
        'z5m11h', 'z5m11i', 'z5m12a', 'z5m12b', 'z5m12c', 'z5m12d', 'z5m12e']
bwidth = 5   # Myr
particle_threshold = 10

#Map(generate_sfh_file, sims, 14, concat=False, bwidth=bwidth)
for sim in sims[2:3]:
    print(sim)
    generate_sfh_file(sim, bwidth)