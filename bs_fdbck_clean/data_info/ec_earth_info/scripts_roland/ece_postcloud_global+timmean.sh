#!/bin/bash 

# Author:
# -------
# Roland Schrödner, TROPOS, 07/2018
#

# Exit in case of any error. Do not ignore errors.
set -e

cd /cfs/klemming/nobackup/r/rolandsc/Runs_Moa
expt=noIsop
cd $expt
cd ifs
cdo mergetime ${expt}*ifs_monmean.nc merge_ifs.nc
cdo timmean merge_ifs.nc ${expt}_timmean_2000_2009_ifs.nc
cdo fldmean merge_ifs.nc ${expt}_fldmean_2000_2009_ifs.nc
cdo fldmean ${expt}_timmean_2000_2009_ifs.nc ${expt}_timmean+fldmean_2000_2009_ifs.nc
rm merge_ifs.nc

cd ../tm5
cdo mergetime ${expt}*tm5_monmean.nc merge_tm5.nc
cdo timmean merge_tm5.nc ${expt}_timmean_2000_2009_tm5.nc
cdo fldmean merge_tm5.nc ${expt}_fldmean_2000_2009_tm5.nc
cdo fldmean ${expt}_timmean_2000_2009_tm5.nc ${expt}_timmean+fldmean_2000_2009_tm5.nc
rm merge_tm5.nc
