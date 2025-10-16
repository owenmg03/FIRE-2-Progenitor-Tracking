
# Owen Gonzales
# Last updated: 14 Aug 2024

# This file contains functions that may be useful when working with the FIRE-2 simulations

import numpy as np
#import numexpr as ne
import gizmo_analysis as gizmo
import halo_analysis as halo
import parallelization as pl
import sys
from os import listdir


def getData(sim: str, snap: int, which_finder: str):
    '''
    Retreives halo finder data for either Rockstar or AHF

    Parameters:
        sim: simulation of interest (e.g. 'z5m11a')
        snap: snapshot of interest
        which_finder: which halo finder to retrieve the data from (either Rockstar or AHF)

    Output:
        data: halo catalog data for either Rockstar (dictionary) or AHF (numpy array)
    '''

    if which_finder.lower() == 'rockstar':
        return halo.io.IO.read_catalogs('index', int(snap), '/projects/b1026/gjsun/high_redshift/'+sim)
    elif which_finder.lower() == 'ahf':
        try:
            path = '/projects/b1026/gjsun/high_redshift/' + sim + '/halo/AHF/AHFHalos/'
            files = np.array([name[:7] for name in listdir(path)])
            filename = listdir(path)[np.where(files == f'snap0{int(snap)}')[0][0]]
            data = np.loadtxt(path+filename)
            return data
        except FileNotFoundError:
            path = '/projects/b1026/gjsun/high_redshift/' + sim + '/halo/AHF/'
            files = np.array([name[:7] for name in listdir(path)])
            filename = listdir(path)[np.where(files == f'snap0{int(snap)}')[0][0]]
            data = np.loadtxt(path+filename)
            return data
    else:
        raise Exception('! Error: Invalid halo finder. Try either \'rockstar\' or \'ahf\'')


def unpackData(data, sim, snap, z, h, which_finder: str, mcut=0.0):
    '''
    Retrieves useful quantities (IDs, radii, virial masses, and positions) from halo finder catalog for either Rockstar or AHF.
    Returns a tuple of this information. Filters based on the cutoff mass if specified.

    Parameters:
        data: data object from loading in and reading halo catalog (this will be a dictionary for Rockstar and a numpy array for AHF)
        z: redshift corresponding to halo finder data
        h: hubble parameter (from particle dictionary)
        which_finder: which halo finder to retrieve the data from (either Rockstar or AHF)
        mcut: mass cut for halos. Will remove any halo with a virial mass below mcut
    
    Output:
        halo_ids: catalog IDs of each halo (filtered by applied mass cut)
        halo_rad: virial radii of halos (filtered by applied mass cut)
        halo_mvir: virial masses of halos (filtered by applied mass cut)
        halo_pos: (x, y, z) positions of halos in comoving kpc (filtered by applied mass cut)
    '''

    if which_finder.lower() == 'rockstar':
        halo_ids = data['id'].astype(int)
        halo_rad = data['radius'] * (1+z)   # Comoving kpc
        halo_mvir = data['mass.vir']   # Solar masses
        halo_pos = data['position']   # Comoving kpc

        filt = halo_mvir > mcut
        return halo_ids[filt], halo_rad[filt], halo_mvir[filt], halo_pos[filt]
    
    elif which_finder.lower() == 'ahf':
        halo_ids = data[:, 0].astype(int)
        halo_rad = data[:, 11] / h   # Comoving kpc
        halo_mvir = data[:, 3] / h   # Solar masses
        halo_pos = data[:, 5:8] / h  # Comoving kpc
        mass_filt = halo_mvir > mcut

        # Because so many AHF galaxies fall outside of the range of Rockstar galaxies, we can get many massive halos that do
        # not actually have star particles associated with them. To resolve this issue, I open up the Rockstar halo catalog
        # and filter for halos that are within the range set out by the rockstar halos.
        try:
            rockstar_data = getData(sim, snap, 'rockstar')
        except OSError:
            filt = mass_filt
            print('\n! Warning: no data found associated to Rockstar. This could result in many halos without identified star particles\n')
        else:
            rockstar_halo_pos = rockstar_data['position'] # Comoving kpc
            minx, maxx = np.min(rockstar_halo_pos[:, 0]), np.max(rockstar_halo_pos[:, 0])
            miny, maxy = np.min(rockstar_halo_pos[:, 1]), np.max(rockstar_halo_pos[:, 1])
            minz, maxz = np.min(rockstar_halo_pos[:, 2]), np.max(rockstar_halo_pos[:, 2])
            pos_filt_x = (halo_pos[:, 0] >= minx) & (halo_pos[:, 0] <= maxx)
            pos_filt_y = (halo_pos[:, 1] >= miny) & (halo_pos[:, 1] <= maxy)
            pos_filt_z = (halo_pos[:, 2] >= minz) & (halo_pos[:, 2] <= maxz)
            pos_filt = pos_filt_x * pos_filt_y * pos_filt_z
            filt = mass_filt * pos_filt

        return halo_ids[filt], halo_rad[filt], halo_mvir[filt], halo_pos[filt]
    
    else:
        raise Exception('! Error: Invalid halo finder. Try either \'rockstar\' or \'ahf\'')


def unpackAndSortPartDict(part, starID_form='tuple'):
    '''
    This function retreives relevant data from the particle dictionary (IDs of star particles, their positions, masses, times of
        formation, and masses at formation). It then removes any duplicate star particle IDs and sorts the remaining star particles
        by their ID number. Returns a tuple of this information.

    Parameters:
        part: particle dictionary (opened using the gizmo_analysis package)

    Output:
        star_ids: IDs of star particles (unique across snapshots)
        star_pos: (x, y, z) positions of star particles in comoving kpc
        star_mass: masses of star particles at the snapshot from which the data comes
        star_formtime: time at which each star particle formed in Gyr
        star_formmass: mass of each star particle at formation
    '''

    # Turn each child ID into a decimal. This allows us to add them to the star IDs and create unique IDs where
    # star IDs might be repeated due to particle splitting
    if starID_form == 'tuple':
        star_ids = part['star']['id'].astype(int)
        child_ids = part['star']['id.child'].astype(int)
        star_ids = np.array(list(zip(star_ids, child_ids)), dtype="i,i")
    elif starID_form == 'string':
        star_ids = part['star']['id'].astype(str)
        child_ids = part['star']['id.child'].astype(str)
        star_ids = np.char.add(np.char.add(star_ids.astype(str), '.'), child_ids.astype(str))
    elif starID_form == 'float':
        star_ids = part['star']['id'].astype(float)
        child_ids = part['star']['id.child'].astype(float)
        child_ids = np.array([x / 10**len(str(x)) for x in child_ids])
        star_ids += child_ids
    else:
        raise Exception('! Error: Not a valid argument for starID_form. Valid choices are \'tuple\', \'string\', or \'int\'')

    # There are rare edge cases where even the star_id + child_id is repeated. To deal with this, we remove any
    # remaining duplicate particle IDs. Without this step, the code will get confused when it tries to compare
    # particle IDs between snapshots
    # We will also sort the star particles to ensure consistency in their order between snapshots
    unique_stars, unique_idxs = np.unique(star_ids, return_index=True)
    starsort = np.argsort(unique_stars)

    star_ids = unique_stars[starsort]
    star_pos = part['star']['position'][unique_idxs, :][starsort, :]   # Comoving kpc
    star_mass = part['star'].prop('mass')[unique_idxs][starsort]   # Solar masses
    star_formtime = part['star'].prop('form.time')[unique_idxs][starsort]   # Gyr
    star_formmass = part['star'].prop('form.mass')[unique_idxs][starsort]   # Solar masses

    return star_ids, star_pos, star_mass, star_formtime, star_formmass


def hasStars(halo_pos, halo_rad, star_pos, return_type='bool', full_halo=False):
    '''
    This function is used at the beginning of progenitor tracing in order to determine which halos have the minimum 10 star particles contained 
        within their virial radii. Progenitor tracing utilizes the star particles in the inner half of the virial radius, so a halo is defined 
        as containing star particles if there is at least one star particle with a distance <= 0.5 R_vir from the center of the halo.

    Parameters:
        halo_pos: position of the halo of interest [comoving kpc]
        halo_rad: virial radius of the halo of interest [comoving kpc]
        star_pos: array containing the (x, y, z) positions of all star particles in comoving kpc
        return_type: type of data returned (see Output)
        full_halo: default bounds are 1/2 of the virial radius. Setting full_halo=True counts all star particles in the entire virial radius

    Output:
        return_type == 'bool': outputs a binary True/False where True indicates that the halo has at least 1 star particle
        return_type == 'int': outputs integer representing the number of star particles found within 1/2 of the virial radius (the
                              entire virial radius if full_halo=True)
        return_type == 'arr': returns a boolean array of len(star_pos) where True represents a star particle found within 1/2 of the
                              virial radius (entire virial radius if full_halo=True)
    '''
    
    halo_pos = np.tile(halo_pos, (len(star_pos[:, 0]), 1))   # Has dimensions (nstarparticles, 3)
    relative_pos = star_pos - halo_pos
    distances = np.sqrt(np.sum(relative_pos**2, axis=1))
    if full_halo:
        inhalo = distances <= halo_rad
    else:
        inhalo = distances <= 0.5*halo_rad

    if return_type == 'bool':
        return sum(inhalo) > 0
    elif return_type == 'int':
        return sum(inhalo)
    elif return_type == 'arr':
        return inhalo
    else:
        print('! Error: incorrect return type')
        sys.exit()