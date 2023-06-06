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
cd ${scratch}/noIsop/ifs
#cd ${scratch}/noLVSOA/ifs
#cd ${scratch}/CTRL/ifs
#cd ${scratch}/ALLm50/ifs
#cd ${scratch}/ifs

#for dir in 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120
for dir in 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120
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
	    pfx=noIsop_${yyyymm}
#	    pfx=noLVSOA_${yyyymm}
#	    pfx=CTRL_${yyyymm}
#	    pfx=ALLm50_${yyyymm}
	    outfn=../${pfx}_ifs_monmean.nc
	    outfn2=../${pfx}_ifs_monmean_additonal.nc

            # -- cloud time averaged CDNC and effective cloud droplet radius
            # define missing value and set 0 and negative values to missing value in time values to avoid divide by zero
	    cdo -R -f nc chname,var105,wat_cloud_time -setrtomiss,-1e33,0 -setmissval,-9e33 \
                -selname,var105 $infn_gg wat_cloud_time.nc
            cdo timsum wat_cloud_time.nc wat_cloud_time_timsum.nc

            # warm cloud properties
	    cdo -R -f nc chname,var102,cdnc -timmean -setmisstoc,0 -div -selname,var102 $infn_gg \
                wat_cloud_time.nc cdnc_wat_cloud.nc
	    cdo -R -f nc chname,var101,re_liq -setmisstoc,0 -div -timsum -mul -selname,var101 $infn_gg \
                wat_cloud_time.nc wat_cloud_time_timsum.nc reff_wat_cloud.nc

            # -- time average of ifs necessary fields and merge with CDNC and reff
            # gg
            cdo -R -f nc chname,var169,surf_SW_down,var175,surf_LW_down,var176,surf_SW_net,var177,surf_LW_net,var178,TOA_SW_net,var179,TOA_LW_net,var208,TOA_SW_net_ClSky,var209,TOA_LW_net_ClSky,var210,surf_SW_net_ClSky,var211,surf_LW_net_ClSky \
                -divc,10800 -timmean \
                -selname,var169,var175,var176,var177,var178,var179,var208,var209,var210,var211 \
                $infn_gg gg_rad_timmean.nc
            cdo -R -f nc chname,var78,LWP,var79,IWP,var34,SST,var134,Press_surf,var167,T2m,var151,MSLP \
                -timmean -selname,var78,var79,var34,var134,var167,var151 \
                $infn_gg gg_other_timmean.nc
            # spectral
            cdo -f nc -sp2gpl -chname,var130,Temp3D,var54,Press3D,var129,Geopot \
                -timmean -selname,var130,var54,var129 $infn_sh sh_timmean.nc
            # SWCF and LWCF
            cdo chname,TOA_SW_net,SWCF -sub -selname,TOA_SW_net gg_rad_timmean.nc \
                -selname,TOA_SW_net_ClSky gg_rad_timmean.nc SWCF.nc
            cdo chname,TOA_LW_net,LWCF -sub -selname,TOA_LW_net gg_rad_timmean.nc \
                -selname,TOA_LW_net_ClSky gg_rad_timmean.nc LWCF.nc
            # merge
            cdo merge gg_rad_timmean.nc gg_other_timmean.nc sh_timmean.nc cdnc_wat_cloud.nc \
                reff_wat_cloud.nc SWCF.nc LWCF.nc $outfn
            # remove temporary files
            rm gg_rad_timmean.nc gg_other_timmean.nc sh_timmean.nc cdnc_wat_cloud.nc \
                reff_wat_cloud.nc wat_cloud_time_timsum.nc wat_cloud_time.nc SWCF.nc LWCF.nc
            # add long name and units
	    ncatted -a units,cdnc,a,c,"cm-3"  $outfn
	    ncatted -a long_name,cdnc,a,c,"cloud time averaged CDNC" $outfn
	    ncatted -a units,re_liq,a,c,"um"  $outfn
	    ncatted -a long_name,re_liq,a,c,"cloud time averaged liquid water effective radius" $outfn
            ncatted -a units,LWP,a,c,"kg m-2"  $outfn
            ncatted -a long_name,LWP,a,c,"total column liquid water" $outfn
            ncatted -a units,IWP,a,c,"kg m-2"  $outfn
            ncatted -a long_name,IWP,a,c,"total column ice water" $outfn
            ncatted -a units,surf_SW_down,a,c,"Wm-2"  $outfn
            ncatted -a long_name,surf_SW_down,a,c,"downward SW radiation at surface" $outfn
            ncatted -a units,surf_LW_down,a,c,"Wm-2"  $outfn
            ncatted -a long_name,surf_LW_down,a,c,"downward LW radiation at surface" $outfn
            ncatted -a units,surf_SW_net,a,c,"Wm-2"  $outfn
            ncatted -a long_name,surf_SW_net,a,c,"net SW radiation at surface" $outfn
            ncatted -a units,surf_LW_net,a,c,"Wm-2"  $outfn
            ncatted -a long_name,surf_LW_net,a,c,"net LW radiation at surface" $outfn
            ncatted -a units,TOA_SW_net,a,c,"Wm-2"  $outfn
            ncatted -a long_name,TOA_SW_net,a,c,"net SW radiation at TOA" $outfn
            ncatted -a units,TOA_LW_net,a,c,"Wm-2"  $outfn
            ncatted -a long_name,TOA_LW_net,a,c,"net LW radiation at TOA" $outfn
            ncatted -a units,TOA_SW_net_ClSky,a,c,"Wm-2"  $outfn
            ncatted -a long_name,TOA_SW_net_ClSky,a,c,"clear sky SW radiation at TOA" $outfn
            ncatted -a units,TOA_LW_net_ClSky,a,c,"Wm-2"  $outfn
            ncatted -a long_name,TOA_LW_net_ClSky,a,c,"clear sky LW radiation at TOA" $outfn
            ncatted -a units,surf_SW_net_ClSky,a,c,"Wm-2"  $outfn
            ncatted -a long_name,surf_SW_net_ClSky,a,c,"clear sky SW radiation at surface" $outfn
            ncatted -a units,surf_LW_net_ClSky,a,c,"Wm-2"  $outfn
            ncatted -a long_name,surf_LW_net_ClSky,a,c,"clear sky LW radiation at surface" $outfn
            ncatted -a units,SST,a,c,"K"  $outfn
            ncatted -a long_name,SST,a,c,"sea surface temperature" $outfn
            ncatted -a units,T2m,a,c,"K"  $outfn
            ncatted -a long_name,T2m,a,c,"air temperature 2m" $outfn
            ncatted -a units,MSLP,a,c,"Pa"  $outfn
            ncatted -a long_name,MSLP,a,c,"mean sea level pressure" $outfn
            ncatted -a units,SWCF,a,c,"Wm-2"  $outfn
            ncatted -a long_name,SWCF,a,c,"short wave cloud forcing" $outfn
            ncatted -a units,LWCF,a,c,"Wm-2"  $outfn
            ncatted -a long_name,LWCF,a,c,"long wave cloud forcing" $outfn
            ncatted -a units,Temp3D,a,c,"K"  $outfn
            ncatted -a long_name,LWCF,a,c,"air temperature 3D" $outfn
            ncatted -a units,Press3D,a,c,"Pa"  $outfn
            ncatted -a long_name,LWCF,a,c,"pressure 3D" $outfn
            ncatted -a units,Geopot,a,c,"m2 s-2"  $outfn
            ncatted -a long_name,LWCF,a,c,"geopotential" $outfn

            # -- additional ifs output (from double call radition with and without aerosols)
            cdo -R -f nc chname,var107,ClearSky_TOA_SW_net,var108,Total_TOA_SW_net,var109,ClearSky_surf_SW_net,var110,Total_surf_SW_net,var111,ClearSky_TOA_LW_net,var112,Total_TOA_LW_net,var113,ClearSky_surf_LW_net,var114,Total_surf_LW_net,var115,ClearSky_TOA_SW_net_woAer,var116,Total_TOA_SW_net_woAer,var91,ClearSky_surf_SW_net_woAer,var92,Total_surf_SW_net_woAer,var93,ClearSky_TOA_LW_net_woAer,var94,Total_TOA_LW_net_woAer,var95,ClearSky_surf_LW_net_woAer,var96,Total_surf_LW_net_woAer \
                -divc,10800 -timmean \
                -selname,var107,var108,var109,var110,var111,var112,var113,var114,var115,var116,var91,var92,var93,var94,var95,var96 \
                $infn_gg $outfn2

            # with aerosols
            ncatted -a units,ClearSky_TOA_SW_net,a,c,"Wm-2"  $outfn2
            ncatted -a long_name,ClearSky_TOA_SW_net,a,c,"clear Sky SW radiation at TOA" $outfn2
            ncatted -a units,Total_TOA_SW_net,a,c,"Wm-2"  $outfn2
            ncatted -a long_name,Total_TOA_SW_net,a,c,"net total SW radiation at TOA" $outfn2
            ncatted -a units,ClearSky_surf_SW_net,a,c,"Wm-2"  $outfn2
            ncatted -a long_name,ClearSky_surf_SW_net,a,c,"clear Sky SW radiation at surf" $outfn2
            ncatted -a units,Total_surf_SW_net,a,c,"Wm-2"  $outfn2
            ncatted -a long_name,Total_surf_SW_net,a,c,"net total SW radiation at surf" $outfn2
            ncatted -a units,ClearSky_TOA_LW_net,a,c,"Wm-2"  $outfn2
            ncatted -a long_name,ClearSky_TOA_LW_net,a,c,"clear Sky LW radiation at TOA" $outfn2
            ncatted -a units,Total_TOA_LW_net,a,c,"Wm-2"  $outfn2
            ncatted -a long_name,Total_TOA_LW_net,a,c,"net total LW radiation at TOA" $outfn2
            ncatted -a units,ClearSky_surf_LW_net,a,c,"Wm-2"  $outfn2
            ncatted -a long_name,ClearSky_surf_LW_net,a,c,"clear Sky LW radiation at surf" $outfn2
            ncatted -a units,Total_surf_LW_net,a,c,"Wm-2"  $outfn2
            ncatted -a long_name,Total_surf_LW_net,a,c,"net total LW radiation at surf" $outfn2
            # with aerosols
            ncatted -a units,ClearSky_TOA_SW_net_woAer,a,c,"Wm-2"  $outfn2
            ncatted -a long_name,ClearSky_TOA_SW_net_woAer,a,c,"clear Sky SW radiation at TOA without aerosols" $outfn2
            ncatted -a units,Total_TOA_SW_net_woAer,a,c,"Wm-2"  $outfn2
            ncatted -a long_name,Total_TOA_SW_net_woAer,a,c,"net total SW radiation at TOA without aerosols" $outfn2
            ncatted -a units,ClearSky_surf_SW_net_woAer,a,c,"Wm-2"  $outfn2
            ncatted -a long_name,ClearSky_surf_SW_net_woAer,a,c,"clear Sky SW radiation at surf without aerosols" $outfn2
            ncatted -a units,Total_surf_SW_net_woAer,a,c,"Wm-2"  $outfn2
            ncatted -a long_name,Total_surf_SW_net_woAer,a,c,"net total SW radiation at surf without aerosols" $outfn2
            ncatted -a units,ClearSky_TOA_LW_net_woAer,a,c,"Wm-2"  $outfn2
            ncatted -a long_name,ClearSky_TOA_LW_net_woAer,a,c,"clear Sky LW radiation at TOA without aerosols" $outfn2
            ncatted -a units,Total_TOA_LW_net_woAer,a,c,"Wm-2"  $outfn2
            ncatted -a long_name,Total_TOA_LW_net_woAer,a,c,"net total LW radiation at TOA without aerosols" $outfn2
            ncatted -a units,ClearSky_surf_LW_net_woAer,a,c,"Wm-2"  $outfn2
            ncatted -a long_name,ClearSky_surf_LW_net_woAer,a,c,"clear Sky LW radiation at surf without aerosols" $outfn2
            ncatted -a units,Total_surf_LW_net_woAer,a,c,"Wm-2"  $outfn2
            ncatted -a long_name,Total_surf_LW_net_woAer,a,c,"net total LW radiation at surf without aerosols" $outfn2

	fi
    else
	echo 'No IFS gridpoint output found in' $dir
    fi
    cd ..
done

cd $origdir
echo `basename $0` 'finished at' `date` 

exit
