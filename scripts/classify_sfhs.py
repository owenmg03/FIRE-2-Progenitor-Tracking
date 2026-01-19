# Owen Gonzales
# Last modfied: 12 Jan 2026

# This file is run on the SFHs output by generate_sfh_files.py and classifies them based on their deltaSFR value. This file
# also identifies bursts and quenches in the SFHs and saves this data to a new .hdf5 file.

import numpy as np
import parallelization as pl
import h5py
from math import ceil


def get_namestring(bwidth, avgwidth, bt_lower, bt_upper, qt_string, use_smoothed, particle_threshold):
    '''
    Uses the parameters of the makeSigmaSFRPlot function to create a file-naming scheme to keep track of parameters

    Parameters:
        bwidth: binning timescale for SFH (in Myr)
        avgwidth: averaging timescale for SFH (in Myr)
        bt_lower: lower threshold for identification of bursts (in dex)
        bt_upper: upper threshold for identification of bursts (in dex)
        quench_threshold: minimum length (in Myr) of a mini-quenching event
        particle_theshold: minimum required number of star particles

    Output:
        namestring: string which records parameters used in the making of the plot
    '''
    
    s1 = f'bin{int(bwidth)}'
    s2 = f'avg{int(avgwidth)}'
    s3 = f'bt0{int(bt_lower*10)}0{int(bt_upper*10)}'
    s4 = f'qt{qt_string}'
    s5 = 'smoothed' if use_smoothed else ''
    s6  = f'pt{int(particle_threshold)}'
    
    return s1+s2+s3+s4+s5+s6


def smooth_array(arr, width, avgwidth):
    
    # Initialize smoothed array
    arr_smoothed = np.zeros(len(arr))
    
    # Remove nan values (writing it this way is because all nan values come at the beginning in this data)
    nnan = sum(np.isnan(arr))
    arr = arr[nnan:]

    # Roll and sum array
    for i in range(int(avgwidth / width)):

        arr_rolled = np.roll(arr, i)
        arr_rolled[:i] = 0
        arr_smoothed[nnan:] += arr_rolled

    # Divide out the number of nonzero numbers summed to get averages
    n_averaged = np.arange(len(arr)) + 1
    n_averaged[int(avgwidth / width):] = int(avgwidth / width)
    arr_smoothed[nnan:] /= n_averaged
    arr_smoothed[:nnan] = np.nan

    return arr_smoothed


def calculate_and_classify_deltasfr(sfh, sfh_smoothed, bt_lower, bt_upper):

    # Code the SFH points based on burstiness, flagging potentially problematic datapoints
    # Code:
    # -1: Halo hasn't crossed particle threshold
    # 0: sfh = 0 and sfh_smoothed = 0
    # 1: sfh = 0 and sfh_smoothed != 0
    # 2: 0 < sfh <= bt_lower
    # 3: bt_lower < sfh <= bt_upper
    # 4: sfh > bt_upper
    
    # Keep track of nan values -> these will be included in deltasfr as nan and in deltasfr_classification as -1
    deltasfr = np.log10(sfh / sfh_smoothed)
    deltasfr_classification = -np.ones(len(sfh))
    
    # Keep track of various values of interest
    isnan = np.isnan(sfh)
    is0 = sfh == 0.
    is0_smoothed = sfh_smoothed == 0.
    
    deltasfr[is0 & is0_smoothed] = -np.inf
    deltasfr[is0 & ~is0_smoothed] = -2
    deltasfr[~(is0 & is0_smoothed) & (deltasfr < -2)] = -2   # Set a floor value for all points below deltasfr = -2
    
    deltasfr_classification[is0 & is0_smoothed] = 0
    deltasfr_classification[is0 & ~is0_smoothed] = 1
    deltasfr_classification[~is0 & ~isnan & (deltasfr <= bt_lower)] = 2
    deltasfr_classification[(deltasfr > bt_lower) & (deltasfr < bt_upper)] = 3
    deltasfr_classification[deltasfr >= bt_upper] = 4
    
    return deltasfr, deltasfr_classification


def find_bursts(time, deltasfr, deltasfr_classification, bwidth):

    # This block identifies the locations of bursts in our SFHs using the SFH code
    # We start by identifying all points where sfhcode = 4, meaning deltaSFH lies above the burst threshold
    # We then widen the boolean to include points on either side of already identified bursts by sliding it 
    # one block to either side a performing an | operation. This necessitates the dummy elements
    # We repeat this until diff = 0, where diff measures how many new points we gained doing the sliding operation
    classification_dummy = np.zeros(len(deltasfr_classification)+2)
    classification_dummy[1:-1] = deltasfr_classification
    isburst = np.zeros(len(deltasfr_classification)+2).astype(bool)
    isburst[1:-1] = deltasfr_classification == 4

    # <diff> indicates whether or not we have gained burst points by widening the window one time bin in each direction
    diff = 1
    while diff > 0:
        widen_isburst = (isburst | np.roll(isburst, 1) | np.roll(isburst, -1))
        isburst_new = (widen_isburst & ((classification_dummy == 3) | (classification_dummy == 4)))
        diff = sum(isburst_new) - sum(isburst)
        isburst = np.copy(isburst_new)
    isburst = isburst[1:-1]
    
    # We would like to change this boolean array into a list of tuples, representing (tstart, tend, magnitude) of the bursts
    # This means we need to identify the edges of the sections of True values
    # <burst_where> indicates what part of the burst we are actively looking for
    bwidth_gyr = bwidth / 1e3
    burst_timesmags = []
    n_burst_timesteps = 0
    burst_loc = [-1, -1, -1]
    burst_where = 'start'

    for i, isb in enumerate(isburst):

        # If the time bin is a burst
        if isb:
            # If we are looking for a start, save data and change burst_where
            if burst_where == 'start':
                n_burst_timesteps = 1
                burst_loc[0] = time[i] - bwidth_gyr/2
                burst_loc[2] = deltasfr[i]
                burst_where = 'end'
            # Otherwise update final burst time and add magnitude (final burst time will be overwritten)
            else:
                n_burst_timesteps += 1
                burst_loc[1] = time[i] + bwidth_gyr/2
                burst_loc[2] += deltasfr[i]

        # If the time bin is not a burst
        else:
            # If we are looking for an end, update final burst time and average burst magnitude. Set <burst_where> to 'start'
            if burst_where == 'end':
                if n_burst_timesteps == 1:
                    burst_loc[1] = burst_loc[0] + bwidth_gyr
                else:
                    burst_loc[2] /= n_burst_timesteps
                burst_timesmags.append(burst_loc)
                n_burst_timesteps = 0
                burst_where = 'start'
                burst_loc = [-1, -1, -1]
            # If we are looking for a start, do nothing
            else:
                pass

    return isburst, burst_timesmags


def find_quenches(time, ssfr_smoothed, bwidth, quench_threshold, quench_duration):

    # Minimum number of timesteps to be considered a burst
    bwidth_gyr = bwidth / 1e3
    quench_mintimesteps = ceil(quench_duration / bwidth)

    # Points beneath the quench threshold are quenched
    isquench_prelim = ssfr_smoothed < quench_threshold
    isquench = np.zeros(len(isquench_prelim)).astype(bool)
    quench_times = []

    # We would like to change this boolean array into a list of tuples, representing (tstart, tend) of the quenches
    # <quench_where> indicates what part of the quench we are actively looking for
    n_quench_timesteps = 0
    quench_loc = [-1, -1]
    quench_where = 'start'

    for i, (time, isq) in enumerate(zip(time, isquench_prelim)):
        
        # If the time bin is a quench
        if isq:
            # If we are looking for a start, save data and change quench_where
            if quench_where == 'start':
                n_quench_timesteps = 1
                quench_loc[0] = time - bwidth_gyr/2
                quench_where = 'end'
            # Otherwise update final quench time (final quench time will be overwritten)
            else:
                n_quench_timesteps += 1
                quench_loc[1] = time + bwidth_gyr/2

        # If the time bin is not a burst
        else:
            # If we are looking for an end and the quench is sufficiently long, update final quench time. Set <quench_where> to 'start'
            if (quench_where == 'end') and (n_quench_timesteps >= quench_mintimesteps):
                quench_times.append(quench_loc)
                isquench[i-n_quench_timesteps:i] = True
            else:
                # If we are looking for a start, do nothing
                pass
            # Reset values for next quench
            n_quench_timesteps = 0
            quench_where = 'start'
            quench_loc = [-1, -1]

    return isquench, quench_times


def create_bq_file(sim, params):

    # Unpack parameters and get namestring
    bwidth, avgwidth, particle_threshold, bt_lower, bt_upper, (quench_threshold, qt_string), quench_duration, use_smoothed = params
    namestring = get_namestring(bwidth, avgwidth, bt_lower, bt_upper, qt_string, use_smoothed, particle_threshold)
    
    # Read in halos at snapshot <startsnap>
    with h5py.File(f'../data/SFHs/sfh_{bwidth}bw_{sim}.hdf5', 'r') as sfhfile:
        halos = np.array(list(sfhfile.keys()))

    with h5py.File(f'../data/SFHsClassified/{sim}_{namestring}.hdf5', 'w') as statfile:

        # Loop through halos and classify SFHs
        for i, halo in enumerate(halos):

            print(f'{i+1} / {len(halos)}: {halo}')

            # Read in SFH data output from generate_sfh_files.py
            with h5py.File(f'../data/SFHs/sfh_{bwidth}bw_{sim}.hdf5', 'r') as sfhfile:
                time = sfhfile[halo]['sfh.time'][:]
                sfh_total = sfhfile[halo]['sfh.total50'][:]
                sfh_insitu = sfhfile[halo]['sfh.insitu50'][:]
                mstell_interp = sfhfile[halo]['mstell.interp50'][:]

            # Smooth SFH
            sfh_total_smoothed = smooth_array(sfh_total, bwidth, avgwidth)
            sfh_insitu_smoothed = smooth_array(sfh_insitu, bwidth, avgwidth)
            
            # The smoothed sSFR will be used to assign quenched values to points
            ssfh_total = sfh_total / mstell_interp
            ssfh_insitu = sfh_insitu / mstell_interp
            ssfh_total_smoothed = sfh_total_smoothed / mstell_interp
            ssfh_insitu_smoothed = sfh_insitu_smoothed / mstell_interp
            
            # Classify deltasfr points 
            deltasfr_total, deltasfr_total_classification = calculate_and_classify_deltasfr(sfh_total, sfh_total_smoothed, bt_lower, bt_upper)
            deltasfr_insitu, deltasfr_insitu_classification = calculate_and_classify_deltasfr(sfh_insitu, sfh_insitu_smoothed, bt_lower, bt_upper)

            # Identify bursts, their durations, and average magnitudes
            isburst_total, burst_timesmags_total = find_bursts(time, deltasfr_total, deltasfr_total_classification, bwidth)
            isburst_insitu, burst_timesmags_insitu = find_bursts(time, deltasfr_insitu, deltasfr_insitu_classification, bwidth)

            # Identify mini-quenches and their durations
            if use_smoothed:
                isquench_total, quench_times_total = find_quenches(time, ssfh_total_smoothed, bwidth, quench_threshold, quench_duration)
                isquench_insitu, quench_times_insitu = find_quenches(time, ssfh_insitu_smoothed, bwidth, quench_threshold, quench_duration)
            else:
                isquench_total, quench_times_total = find_quenches(time, ssfh_total, bwidth, quench_threshold, quench_duration)
                isquench_insitu, quench_times_insitu = find_quenches(time, ssfh_insitu, bwidth, quench_threshold, quench_duration)

            # Save to hdf5 file
            statfile.create_group(halo)

            statfile[halo].create_dataset('mstell', data=mstell_interp)
            statfile[halo].create_dataset('time', data=np.round(time, 3))

            statfile[halo].create_dataset('sfh.total', data=sfh_total)
            statfile[halo].create_dataset('sfh.total.smoothed', data=sfh_total_smoothed)
            statfile[halo].create_dataset('ssfh.total', data=ssfh_total)
            statfile[halo].create_dataset('ssfh.total.smoothed', data=ssfh_total_smoothed)
            statfile[halo].create_dataset('deltasfr.total', data=deltasfr_total)
            statfile[halo].create_dataset('deltasfr.total.classification', data=deltasfr_total_classification)
            statfile[halo].create_dataset('isburst.total', data=isburst_total)
            statfile[halo].create_dataset('burst.locations.total', data=burst_timesmags_total)
            statfile[halo].create_dataset('isquench.total', data=isquench_total)
            statfile[halo].create_dataset('quench.locations.total', data= quench_times_total)

            statfile[halo].create_dataset('sfh.insitu', data=sfh_insitu)
            statfile[halo].create_dataset('sfh.insitu.smoothed', data=sfh_insitu_smoothed)
            statfile[halo].create_dataset('ssfh.insitu', data=ssfh_insitu)
            statfile[halo].create_dataset('ssfh.insitu.smoothed', data=ssfh_insitu_smoothed)
            statfile[halo].create_dataset('deltasfr.insitu', data=deltasfr_insitu)
            statfile[halo].create_dataset('deltasfr.insitu.classification', data=deltasfr_insitu_classification)
            statfile[halo].create_dataset('isburst.insitu', data=isburst_insitu)
            statfile[halo].create_dataset('burst.locations.insitu', data=burst_timesmags_insitu)
            statfile[halo].create_dataset('isquench.insitu', data=isquench_insitu)
            statfile[halo].create_dataset('quench.locations.insitu', data= quench_times_insitu)

    return

# Read in snapshot times
snaptimes = np.loadtxt(f'/projects/b1026/gjsun/high_redshift/z5m11a/snapshot_times.txt')[:68, 2:4]
sims = ['z5m11a', 'z5m11b', 'z5m11c', 'z5m11d', 'z5m11e', 'z5m11f', 'z5m11g', \
        'z5m11h', 'z5m11i', 'z5m12a', 'z5m12b', 'z5m12c', 'z5m12d', 'z5m12e']

# Parameters
bwidth = 10
avgwidth = 100
particle_threshold = 10
bt_lower = 0.2
bt_upper = 0.5
quench_threshold = (3e-10, '3e-10')
quench_duration = 30
use_smoothed = False
params = (bwidth, avgwidth, particle_threshold, bt_lower, bt_upper, quench_threshold, quench_duration, use_smoothed)

#tbins = [0.268996271, 0.544257711, 0.835479993, 1.169487934]   # zbins = [15, 9, 6.5, 5]

#pl.mapProcesses(buildStatsfile, sims, 14, tbins, params, concat=False)

for sim in sims[0:9]:
    
    print(sim)
    create_bq_file(sim, params)
    print()