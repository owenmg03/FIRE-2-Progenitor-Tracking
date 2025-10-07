# Owen Gonzales
# Last updated: 14 Aug 2024

# This file contains functions that may be useful when working with the FIRE-2 simulations

import numpy as np
#import numexpr as ne
import gizmo_analysis as gizmo
import halo_analysis as halo
import parallelization as pl
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


def getMcutArr(mcut, z0, snaptimes, endsnap, mmin=1e5):
    '''
    Generates an array of mass cutoff values at each snapshot, assuming a linear slope in time

    Parameters:
        mcut: mass cutoff value at z0
        z0: redshift of final simulation snapshot
        snaptimes: The snapshot_times.txt file for the simulation, read in as a numpy file
        mmin: mass cutoff value at the last snapshot
    
    Output:
        mcut_arr: array of masscuts for each snapshot
    '''

    idx = np.argmin(np.abs(snaptimes[:, 2] - z0))
    nsnaps = idx-(endsnap-1)

    if mcut == 0.0:
        return np.zeros(nsnaps)
    else:
        return np.logspace(np.log10(mmin), np.log10(mcut), num=nsnaps)


def inHalo(star_pos, halo_rad, halo_pos, distances=False, adaptive=False, fullhalo=False):
    '''
    This function evaluates whether or not each star particle in array of star particles lies inside of some halo.
    Returns a boolean array of length (nstars)

    Parameters:
        star_pos: an array of the (x, y, z) positions of the star particles with dimensions (nstars, 3)
        halo_rad: the radius of the halo of interest
        halo_pos: the (x, y, z) position of the halo of interest

    Output:
        isin_halo: boolean array containing whether each star particle is in the halo of interest
        distance_arr: array of distances of each star particle to the halo of interest
    '''

    if len(star_pos) == 0:
        raise Exception('! Error: No identified star particles')
    else:
        nstars = len(star_pos)

    halopos_block = np.tile(halo_pos, (nstars, 1))

    distance_arr = np.sqrt(np.sum((star_pos - halopos_block)**2, 1))
    
    #isin_halo = distance_arr <= 0.2*halo_rad
    #factors = [0.5, 0.75, 1.]

    if fullhalo:
        isin_halo = distance_arr <= halo_rad
    else:
        isin_halo = distance_arr <= 0.5*halo_rad
        factors = [0.75, 0.9, 1.]
        
        if sum(isin_halo) == 0 and adaptive:
            for n in factors:
                isin_halo = distance_arr <= n*halo_rad
                if sum(isin_halo) > 0:
                    break
                else:
                    pass
        else:
            pass

    if distances:
        return isin_halo, distance_arr
    else:
        return isin_halo

'''
def inHaloFast(star_pos, halo_rad, halo_pos, distances=False, adaptive=False, fullhalo=False):
    
    This function evaluates whether or not each star particle in array of star particles lies inside of some halo.
    Returns a boolean array of length (nstars)

    Parameters:
        star_pos: an array of the (x, y, z) positions of the star particles with dimensions (nstars, 3)
        halo_rad: the radius of the halo of interest
        halo_pos: the (x, y, z) position of the halo of interest

    Output:
        isin_halo: boolean array containing whether each star particle is in the halo of interest
        distance_arr: array of distances of each star particle to the halo of interest
    

    if len(star_pos) == 0:
        raise Exception('! Error: No identified star particles')
    else:
        nstars = len(star_pos)

    halopos_block = np.tile(halo_pos, (nstars, 1))

    differences = ne.evaluate('star_pos - halopos_block')
    sumofsquares = ne.evaluate('sum(differences**2, axis=1)')
    distance_arr = ne.evaluate('sqrt(sumofsquares)')

    if fullhalo:
        isin_halo = ne.evaluate('distance_arr <= halo_rad')
    else:
        isin_halo = ne.evaluate('distance_arr <= 0.5*halo_rad')
        factors = [0.75, 0.9, 1.]
        
        if sum(isin_halo) == 0 and adaptive:
            for n in factors:
                isin_halo = ne.evaluate('distance_arr <= n*halo_rad')
                if sum(isin_halo) > 0:
                    break
                else:
                    pass
        else:
            pass

    if distances:
        return isin_halo, distance_arr
    else:
        return isin_halo
'''


def whichHalo(star_pos, halo_id, halo_rad, halo_pos, *subseqhalorad, fullhalo=False):
    '''
    This function evaluates to which halo each star particle belongs.
    When a star particle lies within the virial radii of two or more halos, I first eliminate halos which have virial radii which differ from
        that of the descendent halo by a factor of three or greater. From the remaining halos, the distance of the star particle to the center
        of the halo is calculated in terms of fraction of the virial radius. The halo with the smallest fraction is chosen. 
    Returns an array containing the ID of the halo to which each star particle belongs.

    Parameters:
        star_pos: array of the positions of the star particles with dimensions (nstars, 3)
        halo_id: array of the IDs (Rockstar or AHF) of the halos with length (nhalos)
        halo_rad: array of the radii of the halos with length (nhalos)
        halo_pos: array of the positions of the centers of the halos with dimensions (nhalos, 3)
        subseqhalorad: optional argument which contains the virial radius of the descendent halo
        fullhalo: if True, a star particle is within the halo if it falls within its virial radius. If False, a star particle is within
                  the halo if it falls within half of the virial radius

    Output:
        halos: array containing the halo to which each star particle belongs (-1 if host halo could not be identified)
    '''

    # Execution of whichHalo() function -> specific helper function depends on number of star particles
    if len(star_pos[:, 0]) > 1e5:
        star_chunks = np.array_split(star_pos, int(len(star_pos[:, 0]) / 5e4), axis=0)
        halos = []
        for starchunk in star_chunks:
            halos.extend(whichHaloHelpChunk(starchunk, halo_id, halo_rad, halo_pos, fullhalo, *subseqhalorad))
        halos = np.array(halos).astype(int)
        return halos
    else:
        halos = []
        for pos in star_pos:
            halos.append(whichHaloHelp(pos, halo_id, halo_rad, halo_pos, fullhalo, *subseqhalorad))
        halos = np.array(halos).astype(int)
        return halos
    

def whichHaloPL(star_pos, halo_id, halo_rad, halo_pos, ncores, *subseqhalorad, fullhalo=False):

    # Execution of whichHalo() function -> specific helper function depends on number of star particles
    halos = pl.splitProcesses(whichHaloHelpChunk, star_pos, ncores, halo_id, halo_rad, halo_pos, *subseqhalorad)
    halos = np.array(halos).astype(int)
    return halos


def whichHaloHelpChunk(star_pos, halo_ids, halo_rad, halo_pos, fullhalo, *subseqhalorad):
    '''
    Helper function for whichHalo(). This function is the case for a chunk of star particles simultaneously and is used when the number of total star
        particles in a halo is greater than 100,000. Star particle chunks all consist of roughly 50,000 star particles.

    Parameters:
        see whichHalo()

    Output:
        hosthalo: the best identified halos to which each of the star particle in the chunk belong
    '''

    nstars = len(star_pos[:, 0])
    nhalos = len(halo_ids)
    print(nstars)

    # Create 3-dimensional arrays so that multiple star particles can be considered at once
    starpos_block = np.tile(star_pos, (nhalos, 1, 1))
    starpos_block = np.transpose(starpos_block, (1, 0, 2))   # dimensions are (nstars, nhalos, 3)
    halopos_block = np.tile(halo_pos, (nstars, 1, 1))   # dimensions are (nstars, nhalos, 3)
    halorad_block = np.tile(halo_rad, (nstars, 1))   # dimensions are (nstars, nhalos)
    haloid_block = np.tile(halo_ids, (nstars, 1))   # dimensions are (nstars, nhalos)

    relative_pos = starpos_block - halopos_block
    distances = np.sqrt(np.sum(relative_pos**2, 2))   # Dimensions are (nstars, nhalos)

    if fullhalo:
        isin_halo = distances <= halorad_block
    else:
        isin_halo = distances <= 0.5*halorad_block

    # Initialize array of length nstars filled with -1. This will contain the identified halos of each star particle. This is done to preserve order throughout this function
    halos = -np.ones(len(star_pos[:, 0]))

    # Star particles which only have one identified halo of origin in previous snapshot can be directly substituted into this array
    single_halo = np.sum(isin_halo, axis=1) == 1
    temp_haloidblock = haloid_block[single_halo, :]   # contains only the star particles which unambiguously lie in one halo at the previous snapshot
    halos[single_halo] = temp_haloidblock[isin_halo[single_halo, :]]   # index the isin_halo array accordingly to preserve order of halos
        
    # Treatment of star particles which fall "inside" of multiple halos
    ambiguous_stars = np.sum(isin_halo, axis=1) >= 2
    ambiguous_halos = -np.ones(sum(ambiguous_stars))

    for i, star_bool in enumerate(isin_halo[ambiguous_stars, :]):

        # The subseqhalorad optional argument allows us to check for large jumps in virial radius
        # If passed, we filter out candidate progenitor halos that have virial radii which are outside of a factor of two of the descendent halo
        if subseqhalorad:
            bad_identifications = (halo_rad >= 2*float(subseqhalorad[0])) | (halo_rad <= (1/2)*float(subseqhalorad[0]))
            star_bool = np.array(~bad_identifications * star_bool).astype(bool)   # For some reason & operator does not work properly here   ##### new line
            if sum(star_bool) == 0:
                ambiguous_halos[i] = -1   # Assign value of -1 if the only available halos are either too large or too small
            else:
                pass
            #if sum(~bad_identifications & star_bool) != 0:
            #    star_bool = np.array(~bad_identifications * star_bool).astype(bool)   # For some reason & operator does not work properly here
            #else:
            #    pass
        else:
            pass

        rad_frac = distances[i, :] / halo_rad
        rad_frac[~star_bool] = np.inf
        hosthalo = halo_ids[np.argmin(rad_frac)]

        ambiguous_halos[i] = hosthalo

    # Fill in halo array with halos of origin identified for ambiguous cases
    halos[ambiguous_stars] = ambiguous_halos

    return halos
    

def whichHaloHelp(star_pos, halo_id, halo_rad, halo_pos, fullhalo, *subseqhalorad):
    '''
    Helper function for whichHalo(). This function is the case for a single star particle and is used when the number of total star
        particles in a halo is 100,000 or less or when parallelizing

    Parameters:
        see whichHalo()

    Output:
        hosthalo: the best identified halo to which the star particle belongs
    '''

    # Determine inside of which halos the star particle falls
    star_pos = np.tile(star_pos, (len(halo_id), 1))
    relative_pos = star_pos - halo_pos
    distances = np.sqrt(np.sum(relative_pos**2, axis=1))
    if fullhalo:
        isin_halo = distances <= halo_rad
    else:
        isin_halo = distances <= 0.5*halo_rad

    # Treatment of star particles which fall "inside" of multiple halos
    if sum(isin_halo) >= 2:

        # The subseqhalorad optional argument allows us to check for large jumps in virial radius
        # If passed, we filter out candidate progenitor halos that have virial radii which are outside of a factor of three of the descendent halo
        if subseqhalorad:
            bad_identifications = (halo_rad >= 2*float(subseqhalorad[0])) | (halo_rad <= (1/2)*float(subseqhalorad[0]))
            isin_halo = np.array(~bad_identifications * isin_halo).astype(bool)   # For some reason & operator does not work properly here   ##### new line
            if sum(isin_halo) == 0:
                return -1   # Assign value of -1 if the only available halos are either too large or too small
            else:
                pass
            #if sum(~bad_identifications & isin_halo) != 0:
            #    isin_halo = ~bad_identifications & isin_halo
            #else:
            #    pass
        else:
            pass

        # In ambiguous cases, the chosen progenitor is the one which minimizes the fraction of the virial radius at which the star particle falls
        rad_frac = distances / halo_rad
        rad_frac[~isin_halo] = np.inf

        print('********')
        print(distances[isin_halo])
        print(rad_frac[isin_halo])
        print(halo_rad[isin_halo])
        print('********')

        hosthalo = halo_id[np.argmin(rad_frac)]
        print(hosthalo, halo_rad[np.argmin(rad_frac)])
        
    else:
        hosthalo = halo_id[isin_halo]

    if isinstance(hosthalo, np.ndarray):
        if len(hosthalo) == 0:
            return -1
        else:
            hosthalo = hosthalo[0]
    else:
        pass


    return hosthalo


def hasStars(halo_pos, halo_rad, star_pos, return_nstars=False):
    '''
    This function is used at the beginning of progenitor tracing in order to determine which halos have the minimum 10 star particles contained 
        within their virial radii. Progenitor tracing utilizes the star particles in the inner half of the virial radius, so a halo is defined 
        as containing star particles if there is at least one star particle with a distance <= 0.5 R_vir from the center of the halo.

    Parameters:
        halo_pos: position of the halo of interest [comoving kpc]
        halo_rad: virial radius of the halo of interest [comoving kpc]
        star_pos: array containing the (x, y, z) positions of all star particles in comoving kpc
        return_nstars: if True, this function also returns the number of star particles contained in the halo of interest

    Output:
        has_star: boolean which is True if the halo contains at least one star particle and False otherwise
        nstars: number of star particles in the halo
    '''
    
    halo_pos = np.tile(halo_pos, (len(star_pos[:, 0]), 1))   # Has dimensions (nstarparticles, 3)
    relative_pos = star_pos - halo_pos
    distances = np.sqrt(np.sum(relative_pos**2, axis=1))
    nstars = np.sum(distances <= 0.5*halo_rad)

    if return_nstars:
        return nstars
    else:
        return nstars > 0


def findOriginSnap(tform: float, snaptimes) -> int:
    '''
    Finds the snapshot of origin for a star particle of interest. t_form and t_snaps must have the same units.

    Parameters:
        tform: formation time of the star particles
        snaptimes: The snapshot_times.txt file for the simulation, read in as a numpy file
    
    Output:
        origin_snaps: snapshot at which each of the star particle formed
    '''

    nsnaps = len(snaptimes[:, 0])
    tform_arr = np.tile(tform, (nsnaps, 1)).T   # Shape (nstars, nsnaps)
    snaptimes_arr = np.tile(snaptimes[:, 3], (len(tform_arr), 1))   # Shape (nstars, nsnaps)

    delta_t = snaptimes_arr - tform_arr
    delta_t[delta_t < 0] = np.inf

    origin_snaps = np.argmin(delta_t, axis=1)

    return origin_snaps


def findOriginHalo(star_ids, star_originsnap, sim, h, snaptimes, which_finder, ncores, mcut_arr, fullhalo=True):
    '''
    Finds the halo of origin for star particles born "during" some snapshot of interest. A star particle is born "during"
        snapshot i if its time of creation is after the ending time of snapshot i-1 and before that of snapshot i.
    This function returns one array of length len(star_ids) containing the halo of origin for each star particle. In addition,
        this function returns a second array of dimension (len(star_ids), 3) containing the position of each star particle
        relative to its halo of origin at its snapshot of origin (in comoving coordinates).

    Parameters:
        snaps: array containing snapshots of interest
        zsnaps: corresponding redshift for snapshots above
        sim: simulation of interest (e.g. 'z5m11a')
        star_ids: array containing the IDs of all of the star particles found at the latest snapshot (at z0)
        star_originsnap: the snapshot during which each star particle was born
        which_finder: which halo finder to retrieve the data from (either Rockstar or AHF)
        ncores: number of cores to split dataset between
        mcut_arr: array containing minimum halo mass for each snapshot
        fullhalo: if True, looks for star particles inside the entire virial radius. If False, looks for star particles inside 0.5*R_vir

    Output:
        star_originhalo: halo of origin for each star particle
    '''
    # Initialize arrays filled with the value -1
    # The first will store the halo of origin of each star particle
    # The second will store the position of each star particle relative to its halo of origin
    star_originhalo = -np.ones(len(star_ids))
    star_relpos = -np.ones((len(star_ids), 3))

    # Iterate through each snapshot and find the origin halos of each star particle born "during" that snapshot
    for (snap, zsnap, mc) in zip(snaptimes[11:, 0], snaptimes[11:, 2], mcut_arr):

        # Get halo finder catalog data
        hal_i = getData(sim, snap, which_finder)
        halo_ids_i, halo_rad_i, _, halo_pos_i = unpackData(hal_i, sim, int(snap), zsnap, h, which_finder, mcut=mc)

        # Get star particle dictionary data
        part = gizmo.io.Read.read_snapshots('star', 'index', int(snap), '/projects/b1026/gjsun/high_redshift/'+sim)
        star_ids_i, star_pos_i, _, star_formtime_i, _ = unpackAndSortPartDict(part, starID_form='tuple')
        star_originsnap_i = findOriginSnap(star_formtime_i, snaptimes)

        # Filter for star particles which were created during the snapshot of interest
        these_stars = star_originsnap == snap
        these_star_ids = star_ids[these_stars]

        # Find the indices of the star particles of interest so we can isolate them and their properties
        # Because we have already removed duplicates and sorted by ID, there should be no missing particles and they should be
        # in the same order.
        shared_stars = np.isin(star_ids_i, these_star_ids)
        star_ids_i_thissnap = star_ids_i[shared_stars]
        star_pos_i_thissnap = star_pos_i[shared_stars, :]

        # We run the which_halo() function on the subset of star particles born at snapshot {snap}. This will return the origin halo of each
        # If there are enough shared stars, we can parallelize
        if sum(shared_stars) >= ncores:
            origin_halos = pl.splitProcesses(whichHalo, star_pos_i_thissnap, ncores, halo_ids_i, halo_rad_i, halo_pos_i, fullhalo=fullhalo)
        else:
            origin_halos = whichHalo(star_pos_i_thissnap, halo_ids_i, halo_rad_i, halo_pos_i, fullhalo=fullhalo)

        # This section calculates the relative position of each star particle to the origin halo in which it was identified
        has_halo = origin_halos != -1
        origin_pos = -np.ones((len(origin_halos), 3))   # Initialize array of -1
        if sum(has_halo) > 0:
            origin_pos[has_halo, :] = np.array(list(map(lambda hal: halo_pos_i[np.where(halo_ids_i == hal)[0]][0], origin_halos[has_halo])))
        else:
            pass
        pos_rel_thissnap = -np.ones_like(star_pos_i_thissnap)
        pos_rel_thissnap[has_halo, :] = star_pos_i_thissnap[has_halo, :] - origin_pos[has_halo, :]   # Position of star particles relative to origin halo

        stars_thissnap = np.isin(star_ids, star_ids_i_thissnap)
        star_originhalo[stars_thissnap] = origin_halos.astype(int)
        star_relpos[stars_thissnap, :] = pos_rel_thissnap

        if snap == 11:
            early_stars = star_originsnap_i < 11
            star_ids_i_early = star_ids_i[early_stars]
            star_pos_i_early = star_pos_i[early_stars]

            early_halos = whichHalo(star_pos_i_early, halo_ids_i, halo_rad_i, halo_pos_i, fullhalo=fullhalo)

            has_halo = early_halos != -1
            early_pos = -np.ones((len(early_halos), 3))   # Initialize array of -1
            if sum(has_halo) > 0:
                early_pos[has_halo, :] = np.array(list(map(lambda hal: halo_pos_i[np.where(halo_ids_i == hal)[0]][0], early_halos[has_halo])))
            else:
                pass
            pos_rel_early = -np.ones_like(star_pos_i_early)
            pos_rel_early[has_halo, :] = star_pos_i_early[has_halo, :] - early_pos[has_halo, :]   # Position of star particles relative to origin halo

            earlystar_locs = np.isin(star_ids, star_ids_i_early)
            star_originhalo[earlystar_locs] = early_halos
            star_relpos[earlystar_locs, :] = pos_rel_early
        
        else:
            pass

    return star_originhalo, star_relpos

def isInSitu(origin_snap, origin_halo, final_halo, prog_tracks):
    '''
    Determines whether or not a given star particle was formed in-situ or not
    It does this by opening the progenitor track for the halo to which the star particle belongs at the final snapshot. It then compares
        the star particle's halo of origin to the halo identified along the progenitor track at the star particle's snapshot of origin.
    If the two halos match up, this function returns 1, indicating that the star particle was formed in-situ. If the two halos do not
        match up, this function returns 0 and the star particle is considered to have formed ex-situ. If the progenitor tracks file
        does not contain a track for the final halo, this function returns a -1, indicating an error.

    Parameters:
        origin_snap: snapshot of origin for a given star particle
        origin_halo: halo of origin for a given star particle
        final_halo: halo the star particle is inside of at the last snapshot
        prog_tracks: the opened hdf5 file containing the progenitor tracks
    
    Output:
        is_insitu: array containing 1 for star particles that were formed in-situ, 0 for those that didn't, and -1 for those
                   for which it could not be determined
    '''

    progenitors = -np.ones(len(origin_snap))
    is_insitu = -np.ones(len(origin_snap))

    can_determine = (origin_halo != -1) & (final_halo != -1)
    regularstars = (origin_snap > 11) & can_determine
    earlystars = (origin_snap <= 11) & can_determine
    progenitors_reg = [prog_tracks[str(finhal)]['prog.id'][:][::-1][snap-11] for (snap, finhal) in \
                   zip(origin_snap[regularstars], final_halo[regularstars])]
    progenitors_erl = [prog_tracks[str(finhal)]['prog.id'][:][-1] for finhal in final_halo[earlystars]]

    progenitors[regularstars] = progenitors_reg
    progenitors[earlystars] = progenitors_erl

    is_insitu[progenitors == origin_halo] = 1
    is_insitu[progenitors != origin_halo] = 0
    is_insitu[progenitors == -1] = -1

    return is_insitu