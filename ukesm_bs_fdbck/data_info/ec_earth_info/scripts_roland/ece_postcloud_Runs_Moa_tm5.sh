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

#cd ${scratch}/${expt}/tm5
#cd ${scratch}/noTerp/tm5
cd ${scratch}/noIsop/tm5
#cd ${scratch}/noLVSOA/tm5
#cd ${scratch}/CTRL/tm5
#cd ${scratch}/ALLm50/tm5

#for infn in `ls general_TM5_*`
for infn in `ls general_TM5_*2007* general_TM5_*2008* general_TM5_*2009*` 
#for infn in general_TM5_ACI1_200001_monthly.nc
do
    if [ ! -z $infn ]; then
        # TM5 file names follow a fixed format: general_TM5_expt_YYYYMM_monthly.nc
	yyyymm=${infn:17:6}
        echo $yyyymm
        echo $infn
	if [ "$yyyymm" != "000000" ]; then
#	    pfx=${expt}_${yymm}
#	    pfx=noTerp_${yyyymm}
	    pfx=noIsop${yyyymm}
#	    pfx=noLVSOA_${yyyymm}
#	    pfx=CTRL_${yyyymm}
#	    pfx=ALLm50_${yyyymm}
	    outfn=${pfx}_tm5_monmean.nc
	    outfn2=${pfx}_tm5_monmean_additonal.nc

            # -- required fields
            cdo selname,CCN0.20,CCN1.00,emiterp,emiisop,loadterp,loadisop,loadsoa,p_elvoc2D,p_svoc2D,od550aer $infn temp.nc

            # SOA mass fraction
            cdo expr,'mass_frac_SOA=loadsoa/(loadbc+loaddust+loadno3+loadoa+loadso4+loadsoa+loadss);' $infn fracSOA.nc

            # total aerosol number concentration
            cdo expr,'N_tot=N_NUS+N_AIS+N_ACS+N_COS+N_AII+N_ACI+N_COI;' $infn Ntot.nc

            # merge
            cdo merge temp.nc fracSOA.nc Ntot.nc $outfn

            # add unit and long name for new calculated fields
	    ncatted -a units,mass_frac_SOA,a,c,"unity"  $outfn
	    ncatted -a long_name,mass_frac_SOA,a,c,"column burden mass fraction SOA" $outfn
	    ncatted -a units,N_tot,a,c,"m-3"  $outfn
	    ncatted -a long_name,N_tot,a,c,"total aerosol number concentration" $outfn

            # remove temporary files
            rm fracSOA.nc Ntot.nc temp.nc

            # -- additional fields
            cdo selname,loadbc,loaddust,loadno3,loadoa,loadso4,loadsoa,loadss,prod_elvoc,prod_svoc,p_el_o3isop,p_el_o3terp,p_el_ohisop,p_el_ohterp,p_sv_o3isop,p_sv_o3terp,p_sv_ohisop,p_sv_ohterp,N_NUS,N_AIS,N_ACS,N_COS,N_AII,N_ACI,N_COI,M_SOANUS,M_SOAAIS,M_SOAACS,M_SOACOS,M_SOAAII,GAS_TERP,GAS_ISOP,GAS_ELVOC,GAS_SVOC,GAS_OH,GAS_O3,od550aerh2o,od550bc,od550dust,od550lt1aer,od550lt1dust,od550lt1ss,od550no3,od550oa,od550soa,od550so4,od550ss $infn $outfn2
	fi
    else
	echo 'No TM5 output found in' $dir
    fi
done

echo `basename $0` 'finished at' `date` 

exit
