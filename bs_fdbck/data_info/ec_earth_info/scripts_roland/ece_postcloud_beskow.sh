#!/bin/bash 

# Author:
# -------
# Declan O'Donnell, FMI, 02/2017
#

# cloud properties: IFS runs with LACI_DIAG set to T give output  
# files that contain accumulated values of CDNC, ICNC, effective radius
# for liquid droplets and ice crystals, and liquid water and ice cloud time.
# This script gives mean 6-hourly and monthly values of CDNC, ICNC
# and effective radii.

set -e

if [ -z $1 ] 
  then
   echo 'Usage:' `basename $0` 'experiment_name,  e.g.' `basename $0` 'ECE3'
   exit 1
fi

echo `basename $0` 'started at' `date`
scratch=/cfs/scratch/r/rolandsc/ece-run
expt=$1

origdir=`pwd`

cd ${scratch}/${expt}/output/ifs

for dir in `ls`
do
    if [ -d $dir ]; then
	cd $dir
	echo 'Processing' $dir
	for infn in `ls ICMGG*`
	do
	    if [ ! -z $infn ]; then
                # IFS gridpoint file names follow a fixed format: ICMGGexpt+YYYYMM
		yymm=${infn:10:6}
		if [ "$yymm" != "000000" ]; then
		    pfx=${expt}_${yymm}
		    outfn=${pfx}_cloud.nc
		    outavg=${pfx}_cloud_mean.nc

                    # define missing value and set 0 to missing value in time values to avoid divide by zero
		    aprun -n 1 cdo -R -f nc chname,var105,wat_cloud_time -setctomiss,0 -setmissval,-9e33 \
                       -selname,var105 $infn ${pfx}_wat_cloud_time.nc
		    aprun -n 1 cdo -R -f nc chname,var106,ice_cloud_time -setctomiss,0 -setmissval,-9e33 \
                       -selname,var106 $infn ${pfx}_ice_cloud_time.nc

                    # warm cloud properties
		    aprun -n 1 cdo -R -f nc chname,var101,cdnc -setmisstoc,0 -div -selname,var101 $infn \
                       ${pfx}_wat_cloud_time.nc ${pfx}_cdnc.nc
		    aprun -n 1 cdo -R -f nc chname,var103,re_liq -setmisstoc,0 -div -selname,var103 $infn \
                       ${pfx}_wat_cloud_time.nc ${pfx}_reffl.nc

                    # ice cloud properties
		    aprun -n 1 cdo -R -f nc chname,var102,icnc -setmisstoc,0 -div -selname,var102 $infn \
                       ${pfx}_ice_cloud_time.nc ${pfx}_icnc.nc
		    aprun -n 1 cdo -R -f nc chname,var104,re_ice -setmisstoc,0 -div -selname,var104 $infn \
                       ${pfx}_wat_cloud_time.nc ${pfx}_reffi.nc

		    # merge results, time average and remove temporary files
		    aprun -n 1 cdo merge ${pfx}_cdnc.nc ${pfx}_icnc.nc ${pfx}_reffl.nc ${pfx}_reffi.nc $outfn 
                    aprun -n 1 cdo timmean $outfn $outavg
		    rm ${pfx}_cdnc.nc ${pfx}_icnc.nc ${pfx}_reffl.nc ${pfx}_reffi.nc

		    ncatted -a units,cdnc,a,c,"cm-3"  $outfn
		    ncatted -a long_name,cdnc,a,c,"cloud time averaged CDNC" $outfn
		    ncatted -a units,cdnc,a,c,"cm-3"  $outavg
		    ncatted -a long_name,cdnc,a,c,"cloud time averaged CDNC" $outavg
		    ncatted -a units,icnc,a,c,"cm-3"  $outfn
		    ncatted -a long_name,icnc,a,c,"cloud time averaged ICNC" $outfn
		    ncatted -a units,icnc,a,c,"cm-3"  $outavg
		    ncatted -a long_name,icnc,a,c,"cloud time averaged ICNC" $outavg
		    ncatted -a units,re_liq,a,c,"um"  $outfn
		    ncatted -a long_name,re_liq,a,c,"cloud time averaged liquid water effective radius" $outfn
		    ncatted -a units,re_liq,a,c,"um"  $outavg
		    ncatted -a long_name,re_liq,a,c,"cloud time averaged liquid water effective radius" $outavg
		    ncatted -a units,re_ice,a,c,"um"  $outfn
		    ncatted -a long_name,re_ice,a,c,"cloud time averaged ice crystal effective radius" $outfn
		    ncatted -a units,re_ice,a,c,"um"  $outavg
		    ncatted -a long_name,re_ice,a,c,"cloud time averaged ice crystal effective radius" $outavg
		fi
	    else
		echo 'No IFS gridpoint output found in' $dir
	    fi
	done
	cd ..
    fi
done

cd $origdir
echo `basename $0` 'finished at' `date` 

exit



