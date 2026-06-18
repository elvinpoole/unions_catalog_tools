# Some tools to work with the UNIONS catalogs on CANFAR

Details on the directories:

## calibrate

Contains scripts for converting comprehensive catalogs (ShapePipe+PhotoPipe) into a calibrated catalog cuts to a given selection



## catalog_tests

Some tools to work with the comprehensive catalogs. It makes histograms and maps, and counts objects that pass given cuts. Designed to loop over chunks of the catalog to keep memory use low.

## concat

Tools to concatenate the individual PHOTOPIPE photometry files from each tile into one hdf5 file
(This was developed before we had a comprehensive ShapePipe+PhotoPipe catalog, kept here for reference)

