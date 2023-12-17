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

# Exit in case of any error. Do not ignore errors.
set -e 

# -z: checks if string is empty -> if so: exit and post manual message
if [ -z $1 ] 
  then
   echo 'Usage:' `basename $0` 'experiment_name,  e.g.' `basename $0` 'ECE3'
   exit 1
fi

echo `basename $0` 'started at' `date`
scratch=/cfs/klemming/nobackup/r/rolandsc/Runs_Moa
expt=$1

origdir=`pwd`

#cd ${scratch}/noTerp/ifs
#cd ${scratch}/noIsop/ifs
#cd ${scratch}/noLVSOA/ifs
#cd ${scratch}/CTRL/ifs
cd ${scratch}/ALLp50/ifs
#cd ${scratch}/ifs

for dir in 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120
#for dir in 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120
do
    cd $dir
    echo 'Processing' $dir
    infn_gg=`ls ICMGG${expt}+2*`
    infn_sh=ICMSH${expt}+2*
    echo $infn_gg
    if [ ! -z $infn_gg ]; then
        # IFS gridpoint file names follow a fixed format: ICMGGexpt+YYYYMM
	yyyymm=${infn_gg:10:6}
	if [ "$yyyymm" != "000000" ]; then
#	    pfx=${expt}_${yymm}
#	    pfx=noTerp_${yyyymm}
#	    pfx=noIsop_${yyyymm}
#	    pfx=noLVSOA_${yyyymm}
#	    pfx=CTRL_${yyyymm}
	    pfx=ALLp50_${yyyymm}
	    outfn=../${pfx}_ifs_monmean_CLCV.nc

            # -- time average of ifs necessary fields and merge with CDNC and reff
            # gg
            cdo -R -f nc chname,var164,total_cloud_cover,var186,low_cloud_cover,var187,medium_cloud_cover,var188,high_cloud_cover \
                -timmean -selname,var164,var186,var187,var188 $infn_gg $outfn

            # add long name and units
	    ncatted -a units,total_cloud_cover,a,c,"0-1"  $outfn
	    ncatted -a long_name,total_cloud_cover,a,c,"Total cloud cover" $outfn
	    ncatted -a units,low_cloud_cover,a,c,"0-1"  $outfn
	    ncatted -a long_name,low_cloud_cover,a,c,"Cloud cover low clouds" $outfn
	    ncatted -a units,medium_cloud_cover,a,c,"0-1"  $outfn
	    ncatted -a long_name,medium_cloud_cover,a,c,"Cloud cover medium clouds" $outfn
	    ncatted -a units,high_cloud_cover,a,c,"0-1"  $outfn
	    ncatted -a long_name,high_cloud_cover,a,c,"Cloud cover high clouds" $outfn
	fi
    else
	echo 'No IFS gridpoint output found in' $dir
    fi
    cd ..
done

cd $origdir
echo `basename $0` 'finished at' `date` 

exit
