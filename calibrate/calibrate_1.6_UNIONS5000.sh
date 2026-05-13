#!/bin/bash
mkdir -p v1.6.6
cd v1.6.6
ln -sf ~/sp_validation/config/calibration/mask_v1.X.6_phot_noext.yaml config_mask.yaml
ln -sf /arc/projects/unions/lensing/ShapePipe/v1.6.x/unions_shapepipe_comprehensive_struc_ugriz_2024_v1.6.c.1.hdf5 unions_shapepipe_comprehensive_struc_2024_v1.X.c.hdf5
python ~/sp_validation/notebooks/calibrate_comprehensive_cat.py > logs.txt