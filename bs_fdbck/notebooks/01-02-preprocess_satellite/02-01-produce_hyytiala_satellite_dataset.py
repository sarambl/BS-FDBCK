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
# path_raw_data = path_measurement_data /'satellite' / 'MODIS_raw'

# path_out_netcdf = path_measurement_data /'satellite' / 'MODIS_netcdf'

path_out_postproc = path_measurement_data /'satellite' / 'MODIS_postproc'
path_out_postproc_lev2 = path_measurement_data /'satellite' / 'MODIS_postproc_lev2'

station = 'SMR'
outfilename = path_out_postproc_lev2 / f'MODIS_date_{station}.nc'


fl = list(path_out_postproc.glob('*.nc'))
fl.sort()

# %%
postproc_data = path_measurement_data /'SMR'/'processed'
path_station_dataset =postproc_data /'SMEAR_data_comb_hourly.csv'

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


# %%
daytime_from

# %%
daytime_from
daytime_to

# %% [markdown] tags=[]
# ## Set station specifics;

# %%
high_perc_OA = 3.02
low_perc_OA = 1.59

# %%
lat_lims = [60, 66]
lon_lims = [22,30]

# %% tags=[]
for f in fl:
    print(f)
 


# %% [markdown]
# ## Set station specifics;

# %%
lat_lims = [60, 66]
lon_lims = [22,30]

# %%
tau_lims = [5,50]
r_eff_lim = 5
cloud_top_temp_above = -15 
cld_water_path_above = 60
include_months = [7,8]

# %% [markdown]
# ## Extract relevant data:

# %%
ds_satellite = xr.open_mfdataset(fl).squeeze('band')

# %%
ds_satellite['Cloud_Effective_Radius_Liquid_Mean'].mean('time').plot()

# %%
ds_satellite['Cloud_Top_Temperature_Day_Mean'].mean('time').plot()

# %%
ds_satellite['month'] = ds_satellite['time.month']
ds_satellite_sum = ds_satellite.where(ds_satellite['month'].isin([7,8]), drop=True)

# %%
df_sum = ds_satellite_sum[['Cloud_Effective_Radius_Liquid_Mean']].squeeze().sel(x = slice(*lon_lims), y = slice(*lat_lims[::-1])).to_dataframe()#'band')

# %%
df_sum = df_sum.dropna().reset_index()
df_sum.groupby([df_sum['y'],df_sum['x']]).count()

# %% [markdown]
# ## Mask values by r_eff, tau, cloud top temperature and cloud water path: 

# %%
ds_satellite_mask = ds_satellite.where(ds_satellite['Cloud_Effective_Radius_Liquid_Mean']>=r_eff_lim)
ds_satellite_mask = ds_satellite_mask.where(ds_satellite_mask['Cloud_Optical_Thickness_Liquid_Mean']>tau_lims[0])
ds_satellite_mask = ds_satellite_mask.where(ds_satellite_mask['Cloud_Optical_Thickness_Liquid_Mean']<tau_lims[1])
ds_satellite_mask = ds_satellite_mask.where(ds_satellite_mask['Cloud_Top_Temperature_Day_Mean']>(cloud_top_temp_above+273.15))
ds_satellite_mask = ds_satellite_mask.where(ds_satellite_mask['Cloud_Water_Path_Liquid_Mean']>=cld_water_path_above)

# %%
ds_satellite_mask['Cloud_Effective_Radius_Liquid_Mean'].mean('time').plot()

# %% [markdown] tags=[]
# ## Mask values by 

# %%
ds_satellite_mask['month'] = ds_satellite_mask['time.month']
ds_satellite_mask_sum = ds_satellite_mask.where(ds_satellite_mask['month'].isin([7,8]), drop=True)
df_sum = ds_satellite_mask_sum[['Cloud_Effective_Radius_Liquid_Mean']].squeeze().sel(x = slice(*lon_lims), y = slice(*lat_lims[::-1])).to_dataframe()#'band')
df_sum = df_sum.dropna().reset_index()
df_sum_cnt = df_sum.groupby([df_sum['y'],df_sum['x']]).count()

# %%
ds_sat_hyy = ds_satellite_mask.squeeze().sel(x = slice(*lon_lims), y = slice(*lat_lims[::-1]))#.to_dataframe()#'band')


# %% [markdown]
# ## Get station data: 

# %%
path_station_dataset

# %%
df_station = pd.read_csv(path_station_dataset, index_col=0)
df_station.index = pd.to_datetime(df_station.index)
df_station

# %%
import datetime

# %%
from bs_fdbck.constants import path_measurement_data
import pandas as pd


# %%
    
def timeround10(dt):
    a, b = divmod(round(dt.minute, -1), 60)
    tdelta = datetime.timedelta(hours = (dt.hour+a), minutes=b)
    nh = (dt.hour+a)%24
    ndt = datetime.datetime(dt.year,dt.month, dt.day,) + tdelta
    #dt_o = datetime.datetime(dt.year,dt.month, dt.day, (dt.hour + a) % 24,b)
    return ndt



def fix_matlabtime(t):
    ind = pd.to_datetime(t-719529, unit='D')
    ind_s = pd.Series(ind)
    return ind_s.apply(timeround10)
    
    


# %%
fn_liine = path_measurement_data / 'ACSM_DEFAULT.mat'

# %%
columns = ['time', 'Org','SO4','NO3','NH4','Chl']

# %%
import scipy.io as sio
test = sio.loadmat(fn_liine)

df_lii = pd.DataFrame(test['ACSM_DEFAULT'], columns=columns)#.set_index('time')

df_lii['time'] = fix_matlabtime(df_lii['time']) + datetime.timedelta(hours=1)

df_lii = df_lii.set_index('time')

df_lii['Org'].plot()

# %%
df_lii['day_of_year'] = df_lii.index.dayofyear
df_lii['month'] = df_lii.index.month

# %%
df_lii['Org'][df_lii.index.month.isin(range(0,13))].groupby(df_lii['month']).mean().plot()


# %%
df_lii['Org'][df_lii.index.month.isin([6,7,8])].quantile([.32,.64])


# %%
obs_hyy_s = df_lii

# %%

# %%
obs_hyy_s.loc['2012':'2018']#.quantile([.33,.66])

# %%
fn = path_measurement_data / 'SourceData_Yli_Juuti2021.xls'

df_hyy_1 = pd.read_excel(fn, sheet_name=0, header=2, usecols=range(6))

df_hyy_1.head()

df_hyy_1['date'] = df_hyy_1.apply(lambda x: f'{x.year:.0f}-{x.month:02.0f}-{x.day:02.0f}', axis=1)

df_hyy_1['date'] = pd.to_datetime(df_hyy_1['date'] )



# %%
import matplotlib.pyplot as plt

# %%
df_hyy_1.set_index('date')['OA (microgram m^-3)'].plot()
df_lii['Org'].resample('1D').median().plot(alpha=.4, marker='o')
df_station['Org_amb'].resample('1D').median().plot(marker='*', alpha=.5)
plt.xlim(['2017-07','2017-09'])

# %%
df_lii['hour'] = df_lii.index.hour

# %%
df_station['hour'] = df_station.index.hour

# %%
df_lii_mh = df_lii[(df_lii['hour']>9) & (df_lii['hour']<19)]
df_station_mh2 = df_station[(df_station['hour']>9) & (df_station['hour']<19)]
df_station_mh = df_station[(df_station['hour']>daytime_from) & (df_station['hour']<daytime_to)]


# %%
df_lii_mh_med = df_lii_mh.resample('1D').median()#.loc[df_hyy_1['date']]
#df_lii_msk_mh['Org'].loc[isna['date']] = np.nan

# %%
df_lii_msk = df_lii.resample('1D').median()

# %%
is_JA = df_lii['month'].isin([7,8])
df_lii[is_JA]['Org'].quantile([0.33, .66])

# %%
df_station['month'] = df_station.index.month
is_JA = df_station['month'].isin([7,8])
df_station[is_JA]['Org_amb'].quantile([0.33, .66])

# %%
is_JA = df_station['month'].isin([7,8])
df_station[is_JA]['Org_amb'].plot.hist(alpha=.5, bins = np.linspace(0,15), label = 'pre-proc')

is_JA = df_lii['month'].isin([7,8])
df_lii[is_JA]['Org'].plot.hist(alpha=.5, bins = np.linspace(0,15), label='orig')

# %% [markdown]
# ## Read in the data from Yli-Juuti cloud figures

# %%
fn = path_measurement_data / 'SourceData_Yli_Juuti2021.xls'

df_cld_yli = pd.read_excel(fn, sheet_name=4, header=1,)# usecols=range(7,12),nrows=7)

df_cld_yli.head()



# %%
import pandas as pd

# %%
df_cld_yli['date'] = df_cld_yli.apply(lambda x: f'{x.year:.0f}-{x.month:02.0f}-{x.day:02.0f}', axis=1)

df_cld_yli['date'] = pd.to_datetime(df_cld_yli['date'] )


# %%
df_cld_yli = df_cld_yli.set_index(['date','LAT','LON'])

# %%
timelist = ['2015-07','2015-08']
df_station_mh2 = df_station[(df_station['hour']>daytime_from) & (df_station['hour']<daytime_to)]



fig = plt.figure(dpi=150)
#df_hyy_1.set_index('date')['OA (microgram m^-3)'].plot(label='Yli-Juuti, fig 1', marker='*', alpha=.1)
df_cld_yli['OA (microgram m^-3)'].to_xarray().sortby('date').isel(LAT=3,LON=3).sel(date=slice(*timelist)).plot(marker='o', alpha=.7, label='Yli-Juti, fig 3')
#df_lii_msk_mh['Org'].plot(alpha=.4, label='daytime', marker='o')
df_station_mh.resample('1D').median()['Org_amb'].plot(alpha=.4, label='daytime new', marker='*')
#df_lii_msk_mh_last['Org'].plot(alpha=.4, label='daytime', marker='o')

plt.xlim(timelist)
plt.legend()

# %%
timelist = ['2018-07','2018-08']
df_station_mh2 = df_station[(df_station['hour']>daytime_from) & (df_station['hour']<daytime_to)]



fig = plt.figure(dpi=150)
#df_hyy_1.set_index('date')['OA (microgram m^-3)'].plot(label='Yli-Juuti, fig 1', marker='*', alpha=.1)
df_cld_yli['OA (microgram m^-3)'].to_xarray().sortby('date').isel(LAT=3,LON=3).sel(date=slice(*timelist)).plot(marker='o', alpha=.7, label='Yli-Juti, fig 3')
#df_lii_msk_mh['Org'].plot(alpha=.4, label='daytime', marker='o')
df_station_mh.resample('1D').median()['Org_amb'].plot(alpha=.4, label='daytime new', marker='*')
#df_lii_msk_mh_last['Org'].plot(alpha=.4, label='daytime', marker='o')

plt.xlim(timelist)
plt.legend()

# %% [markdown]
# ## Mask by high or low OA: 

# %%
df_station_mh['Org_amb'].plot()

# %%
ds_sat_hyy['Cloud_Effective_Radius_Liquid_Mean'].mean('time').plot()

# %%
df_station_mh.groupby('hour').mean()['Org_amb'].plot()

# %% [markdown]
# ### Daily median of station values between 7 and 14 UCT. 

# %%
df_station_mh_med = df_station_mh.resample('1D').median()
df_station_mh_med_sum = df_station_mh_med[df_station_mh_med.index.month.isin([7,8])]


# %%
df_station_mh_med_sum['Org_amb'].quantile([0.33,0.67])

# %%
df_station_mh_sum = df_station_mh_med[df_station_mh_med.index.month.isin([7,8])]
df_station_mh_sum['Org_amb'].quantile([0.33333,0.66666])

# %%
df_station_mh_sum = df_station_mh_med[df_station_mh_med.index.month.isin([7,8])]
df_station_mh_sum['Org_STP'].quantile([0.33333,0.66666])

# %% [markdown]
# ## Add station values to satellite dataset: 

# %%
ds_sat_hyy['OA_STP'] = df_station_mh_med_sum['Org_STP']
ds_sat_hyy['OA_amb'] = df_station_mh_med_sum['Org_amb']

for v in ['N50','N100','N200']:
    ds_sat_hyy[v] = df_station_mh_med_sum[v]

# %%
df_sat_nonan = ds_sat_hyy.to_dataframe().dropna()

# %%

# %%
df_cld_yli['OA (microgram m^-3)'].to_xarray().sortby('date').isel(LAT=3,LON=3).sel(date=slice(*timelist)).plot(marker='o', alpha=.7, label='Yli-Juti, fig 3')


# %%
timelist = ['2015-07-02','2015-09-01']
fig = plt.figure(dpi=150)
#df_hyy_1.set_index('date')['OA (microgram m^-3)'].plot(label='Yli-Juuti, fig 1', marker='*', alpha=.1)
df_cld_yli['OA (microgram m^-3)'].to_xarray().sortby('date').isel(LAT=3,LON=3).sel(date=slice(*timelist)).plot(marker='o', alpha=.7, label='Yli-Juti, fig 3')
ds_sat_hyy['OA_amb'].sel(time=slice(*timelist)).plot(label='omg')
#df_lii_msk_mh_last['Org'].plot(alpha=.4, label='daytime', marker='o')
#
#ax.set_xlim(['2015-07-01','2015-09-01'])

plt.legend()

# %%
ds_sat_hyy

# %%
ds_sat_hyy = ds_sat_hyy.rename({'y':'LAT','x':'LON'}).squeeze()#.to_dataframe().dropna()

# %% [markdown]
# ## Calculate ACSM data in STP:

# %% [markdown] tags=[]
# ## Rename vars

# %%
ds_sat_hyy_rn = ds_sat_hyy.rename({'Cloud_Effective_Radius_Liquid_Mean':'CER (micrometer)',
                           'Cloud_Optical_Thickness_Liquid_Mean':'COT',
                           'Cloud_Water_Path_Liquid_Mean':'CWP (g m^-2)',
                          'OA_STP':'OA (microgram m^-3)',
                          })

# %% [markdown]
# ## Save dataset: 

# %% tags=[]
ds_sat_hyy_rn.to_netcdf(outfilename)

# %%
outfilename

# %% tags=[]
df_sat_rn = ds_sat_hyy_rn.squeeze().to_dataframe().dropna()

# %% [markdown]
# # Extra analysis

# %%
ds_sat_rn = df_sat_rn.to_xarray()

# %%
