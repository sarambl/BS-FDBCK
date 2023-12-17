import xarray as xr

from pathlib import Path
from bs_fdbck_clean.constants import path_outdata

path_out = path_outdata /'NorESM_clm'

path_out.mkdir(exists_ok=True)
vs = [
    'PCT_NAT_PFT',
    'PCT_LANDUNIT',
    'PCT_CFT',
    'PFT_FIRE_CLOSS',
    'PFT_FIRE_NLOSS',
    'TLAI',
    'LAISUN',
    'LAISHA',
    'ELAI',
    'MEG_limonene',
    'MEG_sabinene',
    'MEG_pinene_b',
    'MEG_pinene_a',
    'MEG_ocimene_t_b',
    'MEG_myrcene',
    'MEG_isoprene',
    'MEG_carene_3',
    'H2OSOI',
    'QSOIL',
    'SOILLIQ',
    'SOILRESIS',
    'TOTSOILLIQ',

]

case_name= 'OsloAero_intBVOC_f09_f09_mg17_full'
_pa_out = path_out/case_name
_pa_out.mkdir(exists_ok=True)
for y in range(2012,2015):
    print(y)
    f = f'/proj/bolinc/users/x_sarbl/noresm_archive/{case_name}/lnd/hist/{case_name}.clm2*{y}*'
    print(f)
    _ds = xr.open_mfdataset(f)

    f_out =   _pa_out / f'{case_name}.clm2.concat.{y}.nc'
    print(f'writing to {f_out}')
    _ds[vs].to_netcdf(f_out)


case_name= 'OsloAero_intBVOC_f09_f09_mg17_ssp245'
_pa_out = path_out/case_name
_pa_out.mkdir(exists_ok=True)
for y in range(2016,2019):
    print(y)
    f = f'/proj/bolinc/users/x_sarbl/noresm_archive/{case_name}/lnd/hist/{case_name}.clm2*{y}*'
    print(f)
    _ds = xr.open_mfdataset(f)
    f_out =   _pa_out/ f'{case_name}.clm2.concat.{y}.nc'
    print(f'writing to {f_out}')
    _ds[vs].to_netcdf(f_out)
