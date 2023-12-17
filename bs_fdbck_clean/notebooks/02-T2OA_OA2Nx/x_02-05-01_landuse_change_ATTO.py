# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt

# %%
from bs_fdbck_clean.constants import get_input_datapath

# %%
noresm_raw_data_path = get_input_datapath(model='NorESM')
noresm_raw_data_path

# %% tags=[]
import xarray as xr

from pathlib import Path
from bs_fdbck_clean.constants import path_outdata

path_out = path_outdata /'NorESM_clm'

path_out.mkdir(exist_ok=True)
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
_pa_out.mkdir(exist_ok=True)
for y in range(2012,2015):
    print(y)
    f = noresm_raw_data_pathf /case_name/'lnd'/'hist'/f'{case_name}.clm2*{y}*'
    print(f)
    _ds = xr.open_mfdataset(f)

    f_out =   _pa_out / f'{case_name}.clm2.concat.{y}.nc'
    print(f'writing to {f_out}')
    _ds[vs].to_netcdf(f_out)


case_name= 'OsloAero_intBVOC_f09_f09_mg17_ssp245'
_pa_out = path_out/case_name 
_pa_out.mkdir(exist_ok=True)
for y in range(2015,2019):
    print(y)
    f = noresm_raw_data_pathf /case_name/'lnd'/'hist'/f'{case_name}.clm2*{y}*'
    print(f)
    _ds = xr.open_mfdataset(f)
    f_out =   _pa_out/ f'{case_name}.clm2.concat.{y}.nc'
    print(f'writing to {f_out}')
    _ds[vs].to_netcdf(f_out)


# %%

# %%
case_name= 'OsloAero_intBVOC_f09_f09_mg17_full'
_pa_out = path_out/case_name 
ds_clm_hist= xr.open_mfdataset(str(_pa_out)+'/*.nc')#/proj/bolinc/users/x_sarbl/noresm_archive/OsloAero_intBVOC_f09_f09_mg17_full/lnd/*.nc')

# %%
case_name= 'OsloAero_intBVOC_f09_f09_mg17_ssp245'

_pa_out = path_out/case_name 
ds_clm_ssp= xr.open_mfdataset(str(_pa_out)+'/*.nc')#/proj/bolinc/users/x_sarbl/noresm_archive/OsloAero_intBVOC_f09_f09_mg17_full/lnd/*.nc')

# %%
ds_clm_ssp

# %%

ds_clm_hist['PCT_NAT_PFT'].sum('natpft').sel(lon=300.9, lat=-2.15, method='nearest').plot()
ds_clm_ssp['PCT_NAT_PFT'].sum('natpft').sel(lon=300.9, lat=-2.15, method='nearest').plot()

# %%
for i in range(15):
    ds_clm_hist['PCT_NAT_PFT'].isel(natpft=i).sel(lon=300.9, lat=-2.15, method='nearest').plot()
    ds_clm_ssp['PCT_NAT_PFT'].isel(natpft=i).sel(lon=300.9, lat=-2.15, method='nearest').plot()
    plt.title(i)
    plt.show()

# %%
ds_clm_hist

# %%
ds_clm_hist['TLAI'].sel(lon=300.9, lat=-2.15, method='nearest').plot()
ds_clm_ssp['TLAI'].sel(lon=300.9, lat=-2.15, method='nearest').plot()
plt.show()

# %% tags=[]
ds_clm_hist['QSOIL'].sel(lon=300.9, lat=-2.15, method='nearest').plot()
ds_clm_ssp['QSOIL'].sel(lon=300.9, lat=-2.15, method='nearest').plot()
#plt.title(i)
plt.show()

# %% tags=[]
vl = ['SOILRESIS','PFT_FIRE_CLOSS','PFT_FIRE_NLOSS','LAISUN','TLAI','ELAI','TOTSOILLIQ']
for v in vl:
    ds_clm_hist[v].sel(lon=300.9, lat=-2.15, method='nearest').plot()
    ds_clm_ssp[v].sel(lon=300.9, lat=-2.15, method='nearest').plot()
    plt.title(v)
    plt.show()

# %%
ds_clm_hist

# %%
for i in range(9):
    ds_clm_hist['PCT_LANDUNIT'].isel(ltype=i).sel(lon=300.9, lat=-2.15, method='nearest').plot()
    ds_clm_ssp['PCT_LANDUNIT'].isel(ltype=i).sel(lon=300.9, lat=-2.15, method='nearest').plot()
    plt.title(i)
    plt.show()

# %%

# %%

# %%

# %%
f = '/proj/bolinc/users/x_sarbl/noresm_input_data/ems_OsloAero_intBVOC_f09_f09_mg17_fssp_2012-2018_SFisoprene_addleapyear.nc'

# %%
_ds = xr.open_dataset(f)

# %%
_ds['datesec'].plot(linewidth=0, marker='.')

# %%
f2 = '/proj/bolinc/users/x_sarbl/noresm_input_data/ems_OsloAero_intBVOC_f19_f19_mg17_ssp245_2015-2016_SFmonoterp_addleapyear.nc'

# %%
_ds2 = xr.open_dataset(f2)

# %%
_ds2['datesec'].plot(linewidth=0, marker='.')
_ds['datesec'].plot(linewidth=0, marker='.')
plt.ylim([4000,8000])

# %%
_ds2['datesec'].drop_duplicates

# %%
