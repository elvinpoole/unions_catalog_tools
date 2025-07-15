"""
This script show you an example of how to match the photometry and shear catalogs

The file directory should contain 4 files (and their "chunked" version for easier random indexing/memory use)

MAIN, PHOT, SHEAR, MISSINGTILES

MAIN
  This is the "master" catalog and is the only one you probably want to access
  It is in hdf5 format and contains groups that reference the other files

  groups:
    photometry/ -- external link to the photometry catalog. has ra,dec, flags and magnitude infomation
    shear/ -- external link to the shapepipe catalog
    index/ -- contains infomation on matching between the photometry and shear catalogs

PHOT
  The photometry catalog (eahc tile concatenated together)

SHEAR
  The ShapePipe catalog (converted from fits to hdf5)

MISSINGTILES
  txt file containing a list of tile with no photometry catalog

"""

import h5py
import numpy as np

max_sep_arcsec = 0.4

with h5py.File('CATALOGP_MAIN_SP_v1.5.4_UNIONS5000_2025-05-15_SP_ugriz_photoz_ext.cat.h5', 'r') as f:

    # For each shear catalog object, how far was the closest photometry object (in deg)
    dis = f['index/d2d_s2p'][:] 

    #get index of the shear objects with a nearby match
    shear_index_matches = np.where(dis*60.*60. < max_sep_arcsec)[0]
    print(f'Found {len(shear_index_matches)}/{len(dis)} ({100*len(shear_index_matches)/len(dis)}%) shear catalog objects with a photometry object within {max_sep_arcsec} arcsec')
    
    del dis #these catalogs are big so we'll clean up as we go

    #get some shear catalog info for these objects
    e1 = f['shear/e1'][:][shear_index_matches]
    e2 = f['shear/e2'][:][shear_index_matches]

    # For each shear catalog match, what was the closest photometry object's index
    id_s2p = f['index/index_shear_to_phot'][:][shear_index_matches]

    # Get some photometry catalog info for these objects
    ra = f['photometry/ALPHA_J2000'][:][id_s2p]
    dec = f['photometry/DELTA_J2000'][:][id_s2p]

    # Use shear_index_matches to access any columns from the shear catalog
    # and id_s2p to access any columns from the photometry catalog