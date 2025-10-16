# Owen Gonzales
# Last modfied: 15 Oct 2025

# For each simulation, this script loops through each snapshot. For each halo in a given snapshot, this program finds all star particles that lie within its
# virial radius, their masses and distances, and the catalog ID, (x, y, z) position in ckpc, virial radius in ckpc, and virial mass in solar masses. This
# information is saved to an hdf5 file (one per simulation, structured with a group for each snapshot, and each snapshot with groups for each halo)

import numpy as np
import gizmo_analysis as gizmo
import halo_tools as ht
import parallelization as pl
import h5py
import sys
import os


def hasStarsWrap(halo_data, starpos, return_type='bool', full_halo=False):
    '''
    Wrapper function for hasStars() function from halo_tools.py. Allows for parallelization with Map() function from parallelization.py over multiple looped inputs

    Parameters:
        halo_data:      (x, y, z) position and virial radius of halo
        starpos:        (x, y, z) position of all star particles present at the simulation snapshot
        return_type:    determines the type of data returned by the hasStars() function (see documentation in halo_tools.py)
        full_halo:      If False, considers only the inner 0.5*Rvir. If True, considers entire Rvir
    Output:
        identical to hasStars function, but can loop over two input variables (hpos and hrad)
    '''

    hpos, hrad = halo_data[:3], halo_data[3]
    return ht.hasStars(hpos, hrad, starpos, return_type=return_type, full_halo=full_halo)


# Make HaloStarCatalogs directory if it does not already exist
if not os.path.isdir('../data/HaloStarCatalogs'):
    os.system('mkdir ../data/HaloStarCatalogs')
else:
    pass

# Simulations for which to calculate catalogs
sims = ['z5m11a', 'z5m11b', 'z5m11c', 'z5m11d', 'z5m11e', 'z5m11f', 'z5m11g', 
        'z5m11h', 'z5m11i', 'z5m12a', 'z5m12b', 'z5m12c', 'z5m12d', 'z5m12e']
ncores = int(sys.argv[1])   # Number of cores used in parallelization
nparticles = 10             # Minimum number of star particles that a halo must have in order to be tracked and recorded

# Loop over simulations (unparallelized)
for sim in sims:

    # Read in snapshot times data
    snaptimes = np.loadtxt(f'/projects/b1026/gjsun/high_redshift/{sim}/snapshot_times.txt')
    snaps = snaptimes[11:68, 0][::-1].astype(int)
    zs = snaptimes[11:68, 2][::-1]
    mcut = 1e5 if sim[1] == '5' else 1e6

    # Bitwise encoder for saving string variables to hdf5 file
    encoder = np.vectorize(lambda x: x.encode('utf-8'))
    dt = h5py.string_dtype(encoding='utf-8')

    # Create catalog file
    with h5py.File(f'../data/HaloStarCatalogs/halostarcatalog_{sim}.hdf5', 'w') as file:

        # Loop over snapshots and redshifts (unparallelized)
        for (snap, z) in zip(snaps, zs):
            
            # Read in particle dictionary data
            print('*** Reading in particle dictionary... ***\n')
            part = gizmo.io.Read.read_snapshots('star', 'redshift', z, f'/projects/b1026/gjsun/high_redshift/{sim}')
            starids, starpos, starmass = ht.unpackAndSortPartDict(part, starID_form='string')[:3]
            starids_utf8 = encoder(starids)
            h = part.Cosmology['hubble']

            # Read in the halo finder data
            print('\n*** Reading in halo finder data... ***')
            hal = ht.getData(sim, snap, 'rockstar')
            haloids, halorad, halomvir, halopos = ht.unpackData(hal, sim, snap, z, h, 'rockstar', mcut=mcut)
            
            # Loop over halos (parallelized)
            # Returns an array of dimension (nhalos, nstars) where each column is a boolean array denoting whether or not each star particle belongs to the corresponding halo
            nstars = pl.Map(hasStarsWrap, np.concatenate((halopos, halorad[:, np.newaxis]), axis=1), ncores, starpos, return_type='arr', full_halo=True)
            nstars = np.array(nstars)
            pcut = np.sum(nstars, axis=1) > nparticles

            # Save to hdf5 file
            # Remove any halos that do not pass the particle threshold
            file.create_group(str(snap))
            for (ID, rad, mvir, pos, nstar_bool) in zip(haloids[pcut], halorad[pcut], halomvir[pcut], halopos[pcut], nstars[pcut, :]):
                distances = np.linalg.norm(starpos[nstar_bool, :] - pos, axis=1)
                file[str(snap)].create_group(str(ID))
                file[str(snap)][str(ID)].create_dataset('halo.rvir', data=rad)
                file[str(snap)][str(ID)].create_dataset('halo.mvir', data=mvir)
                file[str(snap)][str(ID)].create_dataset('halo.pos', data=pos)
                file[str(snap)][str(ID)].create_dataset('star.ids', data=starids_utf8[nstar_bool], dtype=dt)
                file[str(snap)][str(ID)].create_dataset('star.mass', data=starmass[nstar_bool])
                file[str(snap)][str(ID)].create_dataset('star.distance', data=distances)