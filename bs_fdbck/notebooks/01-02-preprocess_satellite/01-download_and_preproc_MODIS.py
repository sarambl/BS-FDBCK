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

# %% [markdown]
# # Postprocesses MODIS data 

# %% [markdown]
# Downloaded data is in hd5 and we convert it to NetCDF for easy use with xarray. We use rioxarray to open the hd5 files. 

# %%

# %%
import pandas as pd
import numpy as np
import rioxarray as rxr
import xarray as xr
from pathlib import Path 

import time


import netCDF4


f1 = Path('/proj/bolinc/users/x_sarbl/analysis/BS-FDBCK/Data/satellite/Not_collocated_T_AOD_and_CER_with_wind_and_FRP_all_cases_all_JJA_agg_for_Sara.nc')

f2 = '/proj/bolinc/users/x_sarbl/analysis/BS-FDBCK/Data/satellite/Not_collocated_T_AOD_and_CER_with_wind_and_FRP_all_cases_all_JJA_agg_for_Sara_mean_time.nc'




default_varl = [
    'Cloud_Water_Path_Liquid_Mean',
    'Cloud_Water_Path_Liquid_Mean_Uncertainty',
    'Cloud_Water_Path_Liquid_Standard_Deviation',
    'Cloud_Water_Path_Liquid_Maximum',
    'Cloud_Water_Path_Liquid_Minimum',
    'Cloud_Water_Path_Liquid_Histogram_Counts',
    'Cloud_Top_Temperature_Day_Maximum',
    'Cloud_Top_Temperature_Day_Mean',
    'Cloud_Top_Temperature_Day_Histogram_Counts',
    'Cloud_Top_Temperature_Day_Minimum',
    'Cloud_Top_Pressure_Day_Mean',
    'Cloud_Top_Pressure_Day_Maximum',
    'Cloud_Top_Pressure_Day_Minimum',
    'Cloud_Top_Height_Day_Mean',
    'Cloud_Top_Height_Day_Maximum',
    'Cloud_Top_Height_Day_Minimum',
    'Cloud_Fraction_Day_Mean',
    'Cloud_Fraction_Day_Maximum',
    'Cloud_Fraction_Day_Minimum',
    'Cloud_Optical_Thickness_Liquid_Maximum',
    'Cloud_Optical_Thickness_Liquid_Mean',
    'Cloud_Optical_Thickness_Liquid_Liquid_Histogram_Counts',
    'Cloud_Optical_Thickness_Liquid_Mean_Uncertainty',
    'Cloud_Optical_Thickness_Liquid_Minimum',
    'Cloud_Optical_Thickness_Liquid_Standard_Deviation',
    'Cloud_Optical_Thickness_Combined_Mean',
    'Cloud_Optical_Thickness_Combined_Maximum',
    'Cloud_Optical_Thickness_Combined_Minimum',
    'Cloud_Optical_Thickness_Liquid_Histogram_Counts',
    'Cloud_Optical_Thickness_Combined_Standard_Deviation',
    'Cloud_Effective_Radius_Liquid_Mean_Uncertainty',
    'Cloud_Effective_Radius_Liquid_Mean',
    'Cloud_Effective_Radius_Liquid_Histogram_Counts',
]

def convert_hd5(fn_in, fn_out, varl = None):
    """
    Converts to netcdf and only variables in varl
    """
    start = time.time()

    if varl is None: 
        varl = default_varl
        
    modis_pre = rxr.open_rasterio(str(fn_in),
                              variable=varl,
                              #parse_coordinates=False,
                              # cache=True,
                              #chunks = True, 
                              #masked=True,
                              from_disk=True,
                              #export_grid_mapping=False
                             )
    modis_pre = fix_time_coord(modis_pre)
    # print(modis_pre)
    
    for v in modis_pre.data_vars:
        atts = modis_pre[v].attrs
        if 'add_offset' in atts:
            old_add_offset = atts['add_offset']
            scale_factor = atts['scale_factor']
            new_add_offset = -1*scale_factor*old_add_offset
            atts['modis_add_offset'] = old_add_offset
            atts['add_offset'] = new_add_offset
            
    modis_pre.to_netcdf(fn_out)
    end = time.time()
    print(f'Took {end - start} seconds to run')
    return modis_pre

def fix_time_coord(modis_pre):
    """
    Fixes the time stamps.
    """
    timestamp_beg = modis_pre.attrs['RANGEBEGINNINGDATE']+' '+ modis_pre.attrs['RANGEBEGINNINGTIME']
    timestamp_end = modis_pre.attrs['RANGEENDINGDATE']+' '+ modis_pre.attrs['RANGEENDINGTIME']
    time_beg = pd.to_datetime(timestamp_beg)
    time_end = pd.to_datetime(timestamp_end)

    modis_pre.attrs['timestamp_beginning'] = timestamp_beg
    modis_pre.attrs['timestamp_end'] = timestamp_end



    modis_pre['time'] = time_beg

    modis_pre = modis_pre.set_coords('time').expand_dims('time')
    return modis_pre

# %%
modis_pre = rxr.open_rasterio(
    '/proj/bolinc/users/x_sarbl/analysis/BS-FDBCK/Data/satellite/MODIS_raw/MYD08_D3.A2012001.061.2018037013520.hdf',
    #variable=varl,
    #parse_coordinates=False,
    # cache=True,
    # chunks = True,
    # masked=True,
    from_disk=True,
    # export_grid_mapping=False
)

# %%
modis_pre

# %%

# %% [markdown]
# ## Download 

# %% [markdown]
# Data originally downloaded from:
# https://ladsweb.modaps.eosdis.nasa.gov/search/order/2/MYD08_D3--61
#
# Only day values. 

# %% [markdown] tags=[]
# ## Paths: 

# %%
from bs_fdbck.constants import path_measurement_data

# %%
path_raw_data = path_measurement_data /'satellite' / 'MODIS_raw'

path_out_netcdf = path_measurement_data /'satellite' / 'MODIS_netcdf'

path_out_postproc = path_measurement_data /'satellite' / 'MODIS_postproc'

path_out_netcdf.mkdir(exist_ok=True)

fl = list(path_raw_data.glob('*.hdf'))
fl.sort()

# %% [markdown]
# ## Convert files to nc

# %% tags=[]
for f in fl:
    print(f'Processing {f.stem}...')
    
    fn_out = path_out_netcdf / f'{f.stem}_subset_vars.nc'
    if fn_out.exists():
        print(f'Skipping {fn_out.stem} because file already exists')
        continue
    #print(fn_out)
    modis_pre = convert_hd5(f,fn_out)


# %%

# %%

# %%

# %% [markdown]
# ## Produce file for each variable per year: 

# %%
produce_files = [
    'Cloud_Optical_Thickness_Liquid_Mean', 
    'Cloud_Effective_Radius_Liquid_Mean',
    'Cloud_Water_Path_Liquid_Mean',
    'Cloud_Water_Path_Liquid_Mean_Uncertainty',
    'Cloud_Water_Path_Liquid_Standard_Deviation',
    'Cloud_Water_Path_Liquid_Maximum',
    'Cloud_Water_Path_Liquid_Minimum',
    'Cloud_Top_Temperature_Day_Maximum',
    'Cloud_Top_Temperature_Day_Mean',
    'Cloud_Top_Temperature_Day_Minimum',

]
producw_files_extra=[
    'Cloud_Top_Temperature_Day_Maximum',
    'Cloud_Top_Temperature_Day_Mean',
    'Cloud_Top_Temperature_Day_Minimum',
    'Cloud_Top_Pressure_Day_Mean',
    'Cloud_Top_Pressure_Day_Maximum',
    'Cloud_Top_Pressure_Day_Minimum',
    'Cloud_Top_Height_Day_Mean',
    'Cloud_Top_Height_Day_Maximum',
    'Cloud_Top_Height_Day_Minimum',
    'Cloud_Fraction_Day_Mean',
    'Cloud_Fraction_Day_Maximum',
    'Cloud_Fraction_Day_Minimum',
    'Cloud_Optical_Thickness_Liquid_Maximum',
    #'Cloud_Optical_Thickness_Liquid_Mean',
    'Cloud_Optical_Thickness_Liquid_Mean_Uncertainty',
    'Cloud_Optical_Thickness_Liquid_Minimum',
    'Cloud_Optical_Thickness_Liquid_Standard_Deviation',
    'Cloud_Optical_Thickness_Combined_Mean',
    'Cloud_Optical_Thickness_Combined_Maximum',
    'Cloud_Optical_Thickness_Combined_Minimum',
    'Cloud_Optical_Thickness_Combined_Standard_Deviation',
    'Cloud_Effective_Radius_Liquid_Mean_Uncertainty',
    #'Cloud_Effective_Radius_Liquid_Mean',
]

# %% [markdown]
# #### Get list of input files:

# %% tags=[]
fl = list(path_out_netcdf.glob('*.nc'))
fl.sort()
print(fl) 


# %%
fl[0]

# %%

# %%
fl_df = pd.DataFrame(fl, columns = ['path'])

fl_df['name'] = fl_df['path'].apply(lambda x:str(x.name))

fl_df['year'] = fl_df['name'].apply(lambda x: x.split('.')[1][1:5])

fl_df['ym'] = fl_df['name'].apply(lambda x: x.split('.')[1][1:7])

fl_df['ym']

# %%
fl_df['year'].unique()

# %%
unique_years = fl_df['year'].unique()

# %% [markdown]
# ## Open one year at a time and produce one file per variable

# %%
for yr in unique_years[:-1]:
    print(yr)
    fldf_sub = fl_df[fl_df['year'] == yr]
    fl_sub = list(fldf_sub['path'])
    #print(list(fldf_sub['name']))
    ds_sub = xr.open_mfdataset(fl_sub)
    
    for v in produce_files:
        fn_out = path_out_postproc / f'MYD08_D3_{v}_{yr}.nc'
        if not fn_out.exists():
            ds_sub[v].to_netcdf(fn_out)


# %%
fl_sub[:10]

# %%
xr.open_mfdataset(fl_sub[:10])#,decode_cf=False )

# %%
fl

# %% [markdown]
# ## Extra:

# %%

# %%
ft = '/proj/bolinc/users/x_sarbl/analysis/BS-FDBCK/Data/satellite/MODIS_netcdf/MYD08_D3.A2014001.061.2018051055620_subset_vars.nc'

# %%
dst = xr.open_dataset(ft,decode_cf=False )

# %%
v = 'Cloud_Top_Temperature_Day_Mean'

# %%
dst[v] = dst[v].where(dst[v]!=-9999)

# %%
dst#[v]#.plot()

# %%
(((dst[v]- dst[v].attrs['add_offset'])*dst[v].attrs['scale_factor'])-273.15).plot()

# %%
(((dst[v]- dst[v].attrs['add_offset'])*dst[v].attrs['scale_factor'])-273.15).plot()

# %%
dst[v].plot()

# %%
(dst[v]*0.00999999977648258 - 15000.0).plot()

# %%
(((dst[v]- dst[v].attrs['modis_add_offset'])*dst[v].attrs['scale_factor'])-273.15).plot()

# %%
da_t1 =((dst[v]- dst[v].attrs['modis_add_offset'])*dst[v].attrs['scale_factor'])-273.15

# %%
dst2 = xr.open_dataset(ft,decode_cf=True )

# %%
dst2

# %%
dst2[v].plot()

# %%
(dst2[v]-273.15).plot()

# %%
da_t2 = dst2[v]-273.15

# %%
da_t1

# %%
da_t2.squeeze()

# %%
(((dst2[v]+ 15000.0)*0.00999999977648258)-273.15).plot()

# %%
dst2[v].plot()

# %%
((dst[v]+ dst[v].attrs['add_offset'])*dst[v].attrs['scale_factor']).plot()

# %%
dst['Cloud_Top_Temperature_Day_Mean'].where(dst[v]!=-9999).plot()

# %%
fl = list(path_out_netcdf.glob('*.nc'))[:100]

ds_conc = xr.open_mfdataset(fl, concat_dim='time').squeeze('band')

# %%
ds_conc

# %%
ds_conc['Cloud_Top_Temperature_Day_Mean'].mean('time').plot()

# %%
import matplotlib.pyplot as plt

# %%
_da = ds_conc['Cloud_Top_Temperature_Day_Mean']

# %%
_da.attrs

# %%
fig, ax = plt.subplots(dpi=200)
(ds_conc['Cloud_Top_Temperature_Day_Mean'].mean('time')-273.15).plot(robust=True, ax=ax)#={'dpi':200})


# %%
fig, ax = plt.subplots(dpi=200)
ds_conc['Cloud_Optical_Thickness_Combined_Mean'].mean('time').plot(robust=True, ax=ax)#={'dpi':200})

# %%
fig, ax = plt.subplots(dpi = 200)
ds_conc['Cloud_Effective_Radius_Liquid_Mean'].mean('time').plot(robust=True, ax=ax, cmap='cividis')#={'dpi':200})

# %%
fig, ax = plt.subplots(dpi = 200)
ds_conc['Cloud_Optical_Thickness_Combined_Mean'].mean('time').plot(robust=True, ax=ax, cmap='cividis')#={'dpi':200})

# %%
fig, ax = plt.subplots(dpi = 200)
ds_conc['Cloud_Fraction_Day_Mean'].mean('time').plot(robust=True, ax=ax, cmap='cividis')#={'dpi':200})

# %%
fig, ax = plt.subplots(dpi = 200)
ds_conc['Cloud_Optical_Thickness_Liquid_Mean'].mean('time').plot(robust=True, ax=ax, cmap='cividis', vmax=35)#={'dpi':200})

# %%
fig, ax = plt.subplots(dpi = 200)
ds_conc['Cloud_Top_Temperature_Day_Mean'].mean('time').plot(robust=True, ax=ax, cmap='cividis')#={'dpi':200})

# %%
ds_conc['Cloud_Optical_Thickness_Combined_Mean'].mean('time').plot(robust=True)

# %%
import matplotlib.pyplot as plt

# %%
for i in range(4):
    ds_conc['Cloud_Optical_Thickness_Combined_Mean'].isel(time=i).plot(robust=True)
    plt.show()

# %%
