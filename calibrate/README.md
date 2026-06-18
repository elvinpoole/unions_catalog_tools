# Running the ShapePipe calibration on the comprehensive catalog

These scripts will each produce a calibrated catalog using the cuts specified in the yaml file

I assume you have cloned Martin's SP validation repo https://github.com/martinkilbinger/sp_validation/tree/develop to your homespace

terminal output goes to `logs.txt`

## Version numbering

File names have the structure:

`calibrate_<SP version number (including cuts on shapepipe quantities)>_<PhotoPipe cuts version>_<PhotoPipe version>.sh`

e.g. 

`1.6.6` SP version 

`ppv1` valid ugriz and 0<Z_B<3

`UNIONS5000` The Photopipe run
