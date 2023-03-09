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
import pandas as pd
import numpy as np
import rioxarray as rxr
import xarray as xr
from pathlib import Path 

import time
import netCDF4

# %%

# %%
rn_dic = {
    'Cloud_Optical_Thickness_Liquid_Mean':'COT',
    'Cloud_Effective_Radius_Liquid_Mean': 'r_eff',
    'Cloud_Water_Path_Liquid_Mean': 'CWP',
}


# %%
produce_files = [
    'Cloud_Optical_Thickness_Liquid_Mean', 
    'Cloud_Effective_Radius_Liquid_Mean',
    'Cloud_Water_Path_Liquid_Mean',
    'Cloud_Water_Path_Liquid_Mean_Uncertainty',
    'Cloud_Water_Path_Liquid_Standard_Deviation',
    'Cloud_Water_Path_Liquid_Maximum',
    'Cloud_Water_Path_Liquid_Minimum',
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

# %%

# %%
from bs_fdbck.constants import path_measurement_data

# %% [markdown]
# ## Settings: 

# %%
path_measurement_data

# %%
# path_raw_data = path_measurement_data /'satellite' / 'MODIS_raw'

# path_out_netcdf = path_measurement_data /'satellite' / 'MODIS_netcdf'

path_out_postproc = path_measurement_data /'satellite' / 'MODIS_postproc'
path_out_postproc_lev2 = path_measurement_data /'satellite' / 'MODIS_postproc_lev2'

station = 'ATTO'
outfilename = path_out_postproc_lev2 / f'MODIS_date_{station}.nc'


fl = list(path_out_postproc.glob('*.nc'))
fl.sort()

# %%
tau_lims = [5,50]
r_eff_lim = 5
cloud_top_temp_above = -15 
cld_water_path_above = 50
#include_months = [7,8]

from_year = '2012'
to_year = '2018'
daytime_from = 9
daytime_to = daytime_from + 7


# %% [markdown] tags=[]
# ## Set station specifics;

# %%
high_perc_OA = 3.02
low_perc_OA = 1.59

# %%
(293 + 180) % 360 - 180

# %%
[a for a in [1,2]]


# %%
def lon_sh(l):
    return ((l+180)%360-180) 


# %%
#lat_lims = [60, 66]
#lon_lims = [22,30]
lon_lims = [293.,308.]
lon_lims =[lon_sh(l) for l in lon_lims]
lat_lims = [-8.,-1.]


# %%
lon_lims

# %% tags=[]
for f in fl:
    print(f)
 


# %% [markdown]
# ## Extract relevant data:

# %%
ds_satellite = xr.open_mfdataset(fl).squeeze('band')

# %%
ds_satellite['Cloud_Top_Temperature_Day_Mean'].mean('time').plot()

# %%
ds_satellite['month'] = ds_satellite['time.month']
#ds_satellite_sum = ds_satellite.where(ds_satellite['month'].isin([7,8]), drop=True)

# %% [markdown]
# ## Mask values by r_eff, tau, cloud top temperature and cloud water path: 

# %%
ds_satellite_mask = ds_satellite.where(ds_satellite['Cloud_Effective_Radius_Liquid_Mean']>=r_eff_lim)
ds_satellite_mask = ds_satellite_mask.where(ds_satellite_mask['Cloud_Optical_Thickness_Liquid_Mean']>tau_lims[0])
ds_satellite_mask = ds_satellite_mask.where(ds_satellite_mask['Cloud_Optical_Thickness_Liquid_Mean']<tau_lims[1])
ds_satellite_mask = ds_satellite_mask.where(ds_satellite_mask['Cloud_Top_Temperature_Day_Mean']>(cloud_top_temp_above+273.15))
ds_satellite_mask = ds_satellite_mask.where(ds_satellite_mask['Cloud_Water_Path_Liquid_Mean']>=cld_water_path_above)

# %%
(ds_satellite_mask['Cloud_Top_Temperature_Day_Mean']-273.15).plot.hist(alpha=.5)

(ds_satellite['Cloud_Top_Temperature_Day_Mean']-273.15).plot.hist(alpha=.5)

# %% [markdown]
# ## Mask values by time

# %%
ds_satellite_mask['month'] = ds_satellite_mask['time.month']
ds_satellite_mask['is_JA'] = ds_satellite_mask['month'].isin([7,8])
ds_satellite_mask['is_JJA'] = ds_satellite_mask['month'].isin([6,7,8])
#df_sum = ds_satellite_mask_sum[['Cloud_Effective_Radius_Liquid_Mean']].squeeze().sel(x = slice(*lon_lims), y = slice(*lat_lims[::-1])).to_dataframe()#'band')
#df_sum = df_sum.dropna().reset_index()
#df_sum_cnt = df_sum.groupby([df_sum['y'],df_sum['x']]).count()

# %% [markdown]
# ## Get correct lat lon:

# %%
ds_sat_hyy = ds_satellite_mask.squeeze().sel(x = slice(*lon_lims), y = slice(*lat_lims[::-1]))#.to_dataframe()#'band')


# %% [markdown]
# ## Get OA data: 

# %%
import datetime

# %%
from bs_fdbck.constants import path_measurement_data
import pandas as pd

# %%
#fn_ATTO_data = path_measurement_data / 'ACSM_DEFAULT.mat'
postproc_data = path_measurement_data /'ATTO'/'processed'

path_comb_data_full_time =postproc_data /'ATTO_data_comb_hourly.nc'


# %%
ds_observations = xr.open_dataset(path_comb_data_full_time).sel(time_traj=0)

# %%
ds_observations = ds_observations[['N50-500','N100-500','N200-500','OA']]

# %%
ds_observations['day_of_year'] = ds_observations['time.dayofyear']
ds_observations['month'] = ds_observations['time.month']

# %%
import matplotlib.pyplot as plt

# %% [markdown] tags=[]
# ## Daytime OA: 

# %%
ds_observations['hour'] = ds_observations['time.hour']

hour = ds_observations['hour']

ds_observations_maskh = ds_observations.where((hour >daytime_from) & (hour<daytime_to))


# %%
ds_observations_med_mh = ds_observations_maskh.resample(time='1D').median()
ds_observations_med_mean = ds_observations_maskh.resample(time='1D').mean()

# %%
ds_observations_med_mh = ds_observations_med_mh.drop('time_traj')

# %%

# %%
ds_sat_hyy['OA'] = ds_observations_med_mh['OA']
for v in ds_observations_med_mh.data_vars:
    ds_sat_hyy[v] = ds_observations_med_mh[v]

# %%
#df_sat_nonan = ds_sat_hyy.to_dataframe().dropna()

# %%
ds_sat_hyy['Cloud_Optical_Thickness_Liquid_Mean'].plot.hist()

# %%
#da = df_sat_nonan[['Cloud_Top_Temperature_Day_Maximum']].reset_index().groupby(['y','x']).count().to_xarray()['Cloud_Top_Temperature_Day_Maximum']#.plot()
#da.where(da>110).plot()

# %%
#ds_sat_hyy['number_of_values_in_pixel'] = da#

# %%
ds_sat_hyy = ds_sat_hyy.rename({'y':'LAT','x':'LON'}).squeeze()#.to_dataframe().dropna()

# %% [markdown]
# ## Rename vars

# %%
ds_sat_hyy_rn = ds_sat_hyy.rename({'Cloud_Effective_Radius_Liquid_Mean':'CER (micrometer)',
                           'Cloud_Optical_Thickness_Liquid_Mean':'COT',
                           'Cloud_Water_Path_Liquid_Mean':'CWP (g m^-2)',
                          'OA':'OA (microgram m^-3)',
                          })

# %% [markdown]
# ## Save dataset: 

# %%
outfilename

# %%

# %%
ds_sat_hyy_rn.to_netcdf(outfilename)

# %%
outfilename

# %% [markdown] tags=[]
# # Extra analysis

# %%
df_sat = ds_sat_hyy.to_dataframe().dropna()

# %%
df_sat_rn = df_sat.rename({'Cloud_Effective_Radius_Liquid_Mean':'CER (micrometer)',
                           'Cloud_Optical_Thickness_Liquid_Mean':'COT',
                           'Cloud_Water_Path_Liquid_Mean':'CWP (g m^-2)',
                          'OA':'OA (microgram m^-3)',
                          }, axis=1)

# %%
ds_sat_rn = df_sat_rn.to_xarray()

# %%

# %%
ds_sat_rn

# %% [markdown]
# ## Div diagnostics

# %%
df_sat['OA_cat'] = np.nan

# %%
high_OA_dic = {1:'high OA', 0:'low OA'}

# %%
