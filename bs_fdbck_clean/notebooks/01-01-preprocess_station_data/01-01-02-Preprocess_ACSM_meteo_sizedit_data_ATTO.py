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

# %% [markdown] tags=[]
# # Combine data from different sources: 

# %%
from bs_fdbck_clean.constants import path_measurement_data

# %%
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# %% [markdown]
# ### Input data:
#

# %%
path_sizedist_ATTO = path_measurement_data /'ATTO'#'sizedistrib' 

# %%
list(path_sizedist_ATTO.glob('*'))

# %%
path_acsm = path_measurement_data / 'ATTO'/ 'QACSM_time_series_C4_60m_2014_2016STP_v3.xlsx'
path_acsm2017 = path_measurement_data / 'ATTO'/ 'acsm_data_for_sara_2017.txt'

# %%
path_bc = path_measurement_data / 'ATTO'/'ATTO_BC_Sara.xlsx'

# %%
fn_theo = path_measurement_data / 'ATTO'/ 'ds_atto_2014_2019_4Sara.nc'

# %%
fn_meteo = path_measurement_data / 'ATTO'/ 'meteodataComplete.dat'
fn_meteo_dir = path_measurement_data / 'ATTO'/'meteo'# 'meteodataComplete.dat'

# %% [markdown]
# ### Output data

# %% tags=[]
postproc_data = path_measurement_data /'ATTO'/'processed'
postproc_data.mkdir( exist_ok=True)

path_acsm_daily_median = postproc_data /'daily_median_QACSM_time_series_C4_60m_2014_2016STP_v3.csv'

path_comb_data =postproc_data /'ATTO_data_comb_daily.nc'
path_comb_data_full_time =postproc_data /'ATTO_data_comb_hourly.nc'

# %%
path_comb_data_full_time

# %% [markdown]
# ### Read in acsm data

# %%
df_ATTO = pd.read_excel(path_acsm, sheet_name=0, index_col=0, parse_dates=[0])#'QACSM 60m v3')

# %%
df_ATTO.index = df_ATTO.index.rename('time')

# %%
df_ATTO

# %% [markdown]
# ### Read in acsm data 2017

# %%
rn_acsm_names = {
    'Org':'org (ug m-3)',
    'SO4':'sul',
    'NO3':'nit',
    'N4' :'am',
    'Chl':'chl',
}
    

# %%
df_acsm2017 = pd.read_csv(path_acsm2017,parse_dates=[0], index_col=0 )#'QACSM 60m v3')

# %%
df_acsm2017 = df_acsm2017.rename(rn_acsm_names, axis=1)

# %%
df_ATTO = pd.concat([df_ATTO, df_acsm2017], axis=0)

# %% [markdown]
# ## ATTO is at UTC - 4: convert to local time 

# %%
import datetime

# %%
df_ATTO.index

# %%
if 'time_utc' not in df_ATTO.columns:
    time_ind = pd.to_datetime(df_ATTO.index)# -datetime.timedelta(hours=4)
    df_ATTO['time_utc'] =time_ind
    time_ind_local_time = time_ind - datetime.timedelta(hours=4)
    time_ind_local_time
    df_ATTO.index = time_ind_local_time
    print('converted time to local time')

# %%
df_ATTO_daily_med = df_ATTO.resample('1D').median()

# %%
df_ATTO_daily_med['org (ug m-3)'].plot()

# %%
df_ATTO_daily_med['org (ug m-3)']

# %%
df_ATTO_daily_med = df_ATTO_daily_med.rename({'org (ug m-3)':'Org'}, axis=1)

# %%
df_ATTO_daily_med.to_csv(path_acsm_daily_median)

# %% [markdown]
# ## BC data

# %%
df_ATTO_bc = pd.read_excel(path_bc,parse_dates=[0], index_col=0, header=None, )
df_ATTO_bc

# %%
df_ATTO_bc = df_ATTO_bc.rename({1:'BC_conc',},axis=1)

# %%
df_ATTO_bc.index.name ='time'

# %%
if 'time_utc' not in df_ATTO_bc.columns:
    time_ind = pd.to_datetime(df_ATTO_bc.index)# -datetime.timedelta(hours=4)
    df_ATTO_bc['time_utc'] =time_ind
    time_ind_local_time = time_ind - datetime.timedelta(hours=4)
    df_ATTO_bc.index = time_ind_local_time
    print('converted time to local time')

# %%
df_ATTO_bc['BC_conc'].plot()

df_ATTO['BC MAAP'].plot()

# %%
df_ATTO_bc_hourly_mean = df_ATTO_bc.resample('h').mean()


# %%
df_ATTO_bc_hourly_mean

# %%
df_ATTO_bc_hourly_mean['BC_conc'].loc['2014':'2017'].plot()
df_ATTO['BC MAAP'].plot()

# %%
df_ATTO['BC MAAP'].groupby(df_ATTO.index.hour).median().plot()
df_ATTO_bc_hourly_mean['BC_conc'].groupby(df_ATTO_bc_hourly_mean.index.hour).median().plot()

# %% [markdown] tags=[]
# ## Data from Theodore (all data will be added to this dataset)

# %%
ds = xr.open_dataset(fn_theo, engine='netcdf4')
ds['timeUTC-3'] = ds['time'].copy()

# %%
ds


# %% [markdown] tags=[]
# ### Seems like Theos data is UTC-3, so we shift all to -4

# %%
if 'timeUTC-4' not in ds:
    ds['timeUTC'] = pd.to_datetime(ds['timeUTC-3']) + datetime.timedelta(hours=3)
    ds['timeUTC-4'] = pd.to_datetime(ds['timeUTC']) - datetime.timedelta(hours=4)
    ds['time'] = ds['timeUTC-4'].values
    print('shifted timezone from UTC-3 to UTC-4')

# %% [markdown]
# ### Recalculate N50, N100 etc

# %%
from bs_fdbck_clean.util.EBAS_data.sizedistribution_integration import calc_Nx_interpolate_first

# %%
for x in [50,100,200]:
    ds[f'N{x}_new']=calc_Nx_interpolate_first(ds, x=x,
                              var_diam='D',
                              v_dNdlog10D='pnsd')



# %%
ds['N100_new'].isel(time=slice(0,200)).plot()


ds['N100'].isel(time=slice(0,200)).plot()


# %%
ds['N50_new'].isel(time=slice(0,200)).plot()


ds['N50'].isel(time=slice(0,200)).plot()


# %%
fig, axs = plt.subplots(1,3, figsize=[13,4])

for i,v in enumerate(['N50','N100','N200']):
    ax = axs[i]
    ds.plot.scatter(x=f'{v}_new', y=v,  ax = ax)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
fig.tight_layout()

# %% [markdown]
# ### Replace old values (even though the same) 

# %%
for i,v in enumerate(['N50','N100','N200']):
    ds[v] = ds[f'{v}_new']


# %%
import matplotlib.pyplot as plt


# %% [markdown] tags=[]
# ## Read and fix meteo data 2 (not from re analysis):

# %%
def replace_no(x):
    if len(x)==0:
        return '0'
    else:
        return x

def make_datetime(_df):
    _df['hour'] = _df['Time'].apply(lambda x:replace_no(str(x)[:-2])).astype(int)

    _df['minute'] = _df['Time'].apply(lambda x: replace_no(str(x)[-2:])).astype(int)
    _df['year'] = _df['Yr']
    _df['day'] = _df['Timestamps'].apply(lambda x: int(x.split('.')[0]))
    _df['month'] = _df['Timestamps'].apply(lambda x: int(x.split('.')[1]))

    return pd.to_datetime(_df[['minute', 'hour', 'month','day','year']])


# %%
fl = list(fn_meteo_dir.glob('*.txt'))
fl.sort()
fl

# %%
f = fl[0]
ls_met_df = list()
for f in fl:
    _df = pd.read_csv(f, sep='\t', decimal=",")
    _df['TimeUTC'] =make_datetime(_df) 
    _df = _df.set_index('TimeUTC')
    ls_met_df.append(_df)

# %%
df_meteo2 = pd.concat(ls_met_df)

# %%
df_meteo2 = df_meteo2.replace(-9999.,np.nan)

# %%
df_meteo2['T_81m'].plot()

# %% [markdown] tags=[]
# ### Read old meteo meteo data:

# %%
df_met = pd.read_csv(fn_meteo, sep='\t', index_col=0)

# %%
df_met = df_met.replace(9999,np.nan).rename({' temperature':'temperature'}, axis=1)

# %%
df_met.index = pd.to_datetime(df_met.index)

# %%
df_met['temperature'].plot()

# %%
df_meteo2['T_81m'].plot()

# %%
vars_to_add_meteo = ['pressure', 'temperature', 'humidity', 'wind_dir', 'precip',
       'Solar_inc_Wm2', 'Solar_out_Wm2', 'wind_speed', 'wind_speed_v']


# %%
df_meteo2.columns

# %%
vars_to_add_meteo2 = ['AirPress_81m', 'T_81m', 'RH_81m', 'WSp_73m','WDir_73m' ,'Rainfall',
       'PAR_in', 'PAR_out',  'LW_atm', 'LW_terr',]

dic_old2new_metfile=dict(
    AirPress_81m = 'pressure',
    T_81m = 'temperature',
    RH_81m='humidity',
    Rainfall='precip',
    
    
)

# %%

for v in vars_to_add_meteo:
    df_met[v].plot()
    plt.title(v)
    plt.show()

# %%

for v in vars_to_add_meteo2:
    df_meteo2[v].plot()
    plt.title(v)
    plt.show()

# %%
df_met['wind_speed'].loc['2018-06':'2019-01'].plot()

# %%
df_met['wind_speed'].loc['2018-06':'2018-12'].plot()

# %%
ts = df_met['temperature']
ma = ts.resample('1d').max()#.plot(alpha=0.5, label='ECHAM-SALSA', c='r')
mi = ts.resample('1d').min()#.plot(alpha=0.5, label='ECHAM-SALSA', c='r')
plt.fill_between(ma.index, mi, ma,alpha=0.5, label='ECHAM-SALSA')
ts.resample('7d').mean().plot()

# %%
ts = df_meteo2['T_81m']
ma = ts.resample('1d').max()#.plot(alpha=0.5, label='ECHAM-SALSA', c='r')
mi = ts.resample('1d').min()#.plot(alpha=0.5, label='ECHAM-SALSA', c='r')
plt.fill_between(ma.index, mi, ma,alpha=0.5, label='ECHAM-SALSA')
ts.resample('7d').mean().plot()

# %% [markdown]
# ### Due to the change in the position of the instrument, we only use data up until end of 2018 (which is also when the model runs stop)

# %%
df_met = df_met.loc['2012-01':'2018-12']

# %%
df_met.loc[df_met['wind_speed']>100, 'wind_speed'] = np.nan
df_met.loc[df_met['wind_speed_v']>100,'wind_speed_v'] = np.nan

# %%
df_met.loc[df_met['precip']>4000,'precip'] = np.nan
#.plot()

# %%

for v in vars_to_add_meteo:
    df_met[v].plot()
    plt.title(v)
    plt.show()

# %% [markdown]
# ### Check timezone meteo2:

# %% [markdown]
# Only difference in left or right labeled. 

# %%
fig, ax = plt.subplots()
df_meteo2.groupby('hour').mean()['PAR_in'].plot()
df_met.loc['2014-01':'2019-01'].groupby(df_met.loc['2014-01':'2019-01'].index.hour).mean()['Solar_inc_Wm2'].plot(ax = ax.twinx(), c='r')


# %%
df_meteo2.loc['2014-01-01':'2014-01-03']['T_81m'].plot()
df_met.loc['2014-01-01':'2014-01-03']['temperature'].plot()

# %%
fig, ax = plt.subplots()

df_meteo2.loc['2014-01-01':'2014-01-03']['PAR_in'].plot()
df_met.loc['2014-01-01':'2014-01-03']['Solar_inc_Wm2'].plot(ax=ax.twinx(), c='r')

# %% [markdown]
# Half an hour difference due to right verus left labeling of timestep

# %%
df_met

# %% [markdown] tags=[]
# #### Go from UTC time to local time

# %%
if 'TimeLocal' not in df_met.columns:
    df_met = df_met.reset_index()
    df_met['TimeLocal'] = pd.to_datetime(df_met['TimeUTC']) - datetime.timedelta(hours=4)

    df_met = df_met.set_index('TimeLocal')
    print('shifted timezone from UTC to UTC-4')


# %%
if 'TimeLocal' not in df_met.columns: 
    df_meteo2 = df_meteo2.reset_index()
    df_meteo2['TimeLocal'] = pd.to_datetime(df_meteo2['TimeUTC']) - datetime.timedelta(hours=4)

    df_meteo2 = df_meteo2.set_index('TimeLocal')
    print('shifted timezone from UTC to UTC-4')


# %%
df_met['hour'] = df_met.index.hour

# %%
df_meteo2['hour'] = df_meteo2.index.hour

# %%
df_met['temperature']

# %%
df_meteo2['T_81m']

# %%
df_met.columns

# %%
df_met_daily_med = df_met.resample('D').median()

df_met_daily_cycle = df_met.groupby(df_met['hour']).mean()

df_met_daily_cycle['temperature'].plot()

df_met_daily_med = df_meteo2.resample('D').median()

df_met_daily_cycle = df_meteo2.groupby(df_meteo2['hour']).mean()

df_met_daily_cycle['T_81m'].plot()

# %% tags=[]
ds

# %%
_ds = ds.groupby('time.hour').mean()

# %%
ds['hour'] = ds['time.hour']

# %%
f,ax = plt.subplots()
ds['Temperature'].isel(time_traj=0).groupby(ds['hour']).mean().plot()

df_met_daily_cycle['T_81m'].plot(ax = ax.twinx(), c='r')

ds['Solar_Radiation'].isel(time_traj=0).groupby(ds['hour']).mean().plot(ax = ax.twinx(), c='k')


# %% [markdown]
# ### Add meteo variables to final dataset and rename reanalysis variables

# %%
ds = ds.rename(dict(
    Pressure='Pressure_reanalysis',
    Temperature='Temperature_reanalysis',
    Potential_Temperature='Potential_Temperature_reanalysis',
    Specific_Humidity = 'Specific_Humidity_reanalysis',
    
    
))
ds = ds.rename(dict(
    N50='N50-500',
    N100 ='N100-500',
    N200 ='N200-500',
))

# %% [markdown]
# **Next one not used anymore**

# %% [markdown]
# from_dt = ds['time'].isel(time=0).values
# to_dt = ds['time'].isel(time=-1).values
#
#
# for v in vars_to_add_meteo:
#     xa = xr.DataArray(df_met[v]).rename(dict(TimeLocal='time'))
#     ds[v] = xa.sel(time=slice(from_dt, to_dt))

# %% [markdown]
# ### *Adding meteo data from dataset 2

# %%
from_dt = ds['time'].isel(time=0).values
to_dt = ds['time'].isel(time=-1).values


for v in vars_to_add_meteo2:
    xa = xr.DataArray(df_meteo2[v]).rename(dict(TimeLocal='time'))
    if v in dic_old2new_metfile:
        vo = dic_old2new_metfile[v]
        print(f'renaming {v} to {vo}')
        ds[vo] = xa.sel(time=slice(from_dt, to_dt))
    else:
        ds[v] = xa.sel(time=slice(from_dt, to_dt))
        

# %% [markdown]
# ### Check how well reanalysis temperature fits

# %%
f,ax = plt.subplots()
ds['Temperature_reanalysis'].isel(time_traj=0).groupby(ds['hour']).mean().plot()
ds['temperature'].groupby(ds['hour']).mean().plot(ax = ax.twinx(), c='r')


# %%
ds['temperature'].resample(time='14d').mean().plot()

# %%
ds.isel(time_traj=0).sel(time=slice('2014','2014')).plot.scatter(x='Temperature_reanalysis', y = 'temperature')

# %%
ds.isel(time_traj=0).sel(time=slice('2016-01','2016')).plot.scatter(x='Temperature_reanalysis', y = 'temperature')

# %%
ds['pnsd'].mean('time').plot(xscale='log')

# %%
ds_nx = ds.resample(time='D').median()

# %%
ds_nx['N50-100'] = ds_nx['N50-500']-ds_nx['N100-500']
ds_nx['N100-200'] = ds_nx['N100-500']-ds_nx['N200-500']
_df = ds_nx[['N50-100','N100-200','N200-500']].to_dataframe()

_df.groupby(_df.index.month).median().plot(kind='area', stacked='true',)#, alpha=.2)



# %%
ds['N50-500'].mean('time')

# %%
ds

# %%
ds['N50-500'].plot()

# %%
ds['N100-500'].plot()

# %%
ds['N100-500'].mean('time')

# %%
ds['N200-500'].mean('time')

# %% [markdown]
# ### *Add ACSM data: 

# %%
ds = ds.assign({'OA':df_ATTO['org (ug m-3)'].resample('h').mean()})

# %%
fig, ax = plt.subplots()
ds['OA'].plot(c = 'b', alpha=0.5, marker='.', linewidth=0, markersize=.1)

ds['N100-500'].plot(ax = ax.twinx(),c = 'r', alpha=0.5, marker='.', linewidth=0, markersize=.1)
ax.set_ylim([-10,40])

# %% [markdown]
# ### *Add BC data: 

# %%
ds = ds.assign({'BC_conc':df_ATTO_bc['BC_conc'].resample('1h').mean()})


# %% [markdown]
# ## Save dataset

# %%
ds.to_netcdf(path_comb_data_full_time)

# %%

# %%
ds = xr.open_dataset(path_comb_data_full_time)

# %%
ds

# %% [markdown]
# ## Extra

# %% [markdown]
# Filter data for bc below 0.01 micro g/m3, or .05.

# %% [markdown] tags=[]
# ## JFM 

# %%
ds_jfm = ds.where(ds['time.month'].isin([1,2,3]))

# %%
_all = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<10000).resample(time='1D').median().count().values

_1 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<1).resample(time='1D').median().count().values

_p5 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.5).resample(time='1D').median().count().values

_p1 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.1).resample(time='1D').median().count().values

_p05 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.05).resample(time='1D').median().count().values

_p01 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.01).resample(time='1D').median().count().values

# %%
print(f'All data: \t \t {_all} days \n'
      f'BC less than 1: \t {_1} days \n'
      f'BC less than 0.5: \t {_p5} days \n'
      f'BC less than 0.1: \t {_p1} days \n'
      f'BC less than 0.05: \t {_p05} days \n'
      f'BC less than 0.01: \t {_p01} days \n'
     )

# %%
_ds_jfm_daily_med = ds_jfm.resample(time='1D').median()

# %%
_all = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<10000).count().values
_1 = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<1).count().values
_p5 = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<.5).count().values
_p1 = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<.1).count().values
_p05 = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<.05).count().values
_p01 = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<.01).count().values


# %%
print(f'All data: \t \t {_all} days \n'
      f'BC less than 1: \t {_1} days \n'
      f'BC less than 0.5: \t {_p5} days \n'
      f'BC less than 0.1: \t {_p1} days \n'
      f'BC less than 0.05: \t {_p05} days \n'
      f'BC less than 0.01: \t {_p01} days \n'
     )

# %%
_ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<.05).plot.hist(bins=np.linspace(0,10), alpha=.5)

_ds_jfm_daily_med['OA'].plot.hist(bins=np.linspace(0,10), alpha = .5)

# %% [markdown]
# ## FMAM

# %%
ds_jfm = ds.where(ds['time.month'].isin([2,3,4,5]))

# %%
_all = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<10000).resample(time='1D').median().count().values

_1 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<1).resample(time='1D').median().count().values

_p5 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.5).resample(time='1D').median().count().values

_p1 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.1).resample(time='1D').median().count().values

_p05 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.05).resample(time='1D').median().count().values

_p01 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.01).resample(time='1D').median().count().values

# %%
print(f'All data: \t \t {_all} days \n'
      f'BC less than 1: \t {_1} days \n'
      f'BC less than 0.5: \t {_p5} days \n'
      f'BC less than 0.1: \t {_p1} days \n'
      f'BC less than 0.05: \t {_p05} days \n'
      f'BC less than 0.01: \t {_p01} days \n'
     )

# %%
_ds_jfm_daily_med = ds_jfm.resample(time='1D').median()

# %%
_all = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<10000).count().values
_1 = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<1).count().values
_p5 = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<.5).count().values
_p1 = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<.1).count().values
_p05 = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<.05).count().values
_p01 = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<.01).count().values


# %%
print(f'All data: \t \t {_all} days \n'
      f'BC less than 1: \t {_1} days \n'
      f'BC less than 0.5: \t {_p5} days \n'
      f'BC less than 0.1: \t {_p1} days \n'
      f'BC less than 0.05: \t {_p05} days \n'
      f'BC less than 0.01: \t {_p01} days \n'
     )

# %%
_ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<.05).plot.hist(bins=np.linspace(0,10), alpha=.5)

_ds_jfm_daily_med['OA'].plot.hist(bins=np.linspace(0,10), alpha = .5)

# %%
ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.05).resample(time='1D').median().plot.hist(bins=np.linspace(0,10), alpha=.5)

ds_jfm['OA'].resample(time='1D').median().plot.hist(bins=np.linspace(0,10), alpha = .5)

# %% [markdown]
# ## JFMAM

# %%
ds_jfm = ds.where(ds['time.month'].isin([1,2,3,4,5]))

# %%
_all = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<10000).resample(time='1D').median().count().values

_1 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<1).resample(time='1D').median().count().values

_p5 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.5).resample(time='1D').median().count().values

_p1 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.1).resample(time='1D').median().count().values

_p05 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.05).resample(time='1D').median().count().values

_p01 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.01).resample(time='1D').median().count().values

# %%
print(f'All data: \t \t {_all} days \n'
      f'BC less than 1: \t {_1} days \n'
      f'BC less than 0.5: \t {_p5} days \n'
      f'BC less than 0.1: \t {_p1} days \n'
      f'BC less than 0.05: \t {_p05} days \n'
      f'BC less than 0.01: \t {_p01} days \n'
     )

# %%
_ds_jfm_daily_med = ds_jfm.resample(time='1D').median()

# %%
_all = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<10000).count().values
_1 = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<1).count().values
_p5 = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<.5).count().values
_p1 = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<.1).count().values
_p05 = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<.05).count().values
_p01 = _ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<.01).count().values


# %%
print(f'All data: \t \t {_all} days \n'
      f'BC less than 1: \t {_1} days \n'
      f'BC less than 0.5: \t {_p5} days \n'
      f'BC less than 0.1: \t {_p1} days \n'
      f'BC less than 0.05: \t {_p05} days \n'
      f'BC less than 0.01: \t {_p01} days \n'
     )

# %%
_ds_jfm_daily_med['OA'].where(_ds_jfm_daily_med['BC_conc'].fillna(0)<.05).plot.hist(bins=np.linspace(0,10), alpha=.5)

_ds_jfm_daily_med['OA'].plot.hist(bins=np.linspace(0,10), alpha = .5)

# %%
ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.05).resample(time='1D').median().plot.hist(bins=np.linspace(0,10), alpha=.5)

ds_jfm['OA'].resample(time='1D').median().plot.hist(bins=np.linspace(0,10), alpha = .5)

# %% [markdown] tags=[]
# ## MAM 

# %%
ds_jfm = ds.where(ds['time.month'].isin([3,4,5]))

# %%
_all = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<10000).resample(time='1D').median().count().values

_1 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<1).resample(time='1D').median().count().values

_p5 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.5).resample(time='1D').median().count().values

_p1 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.1).resample(time='1D').median().count().values

_p05 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.05).resample(time='1D').median().count().values

_p01 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.01).resample(time='1D').median().count().values

# %%
print(f'All data: \t \t {_all} days \n'
      f'BC less than 1: \t {_1} days \n'
      f'BC less than 0.5: \t {_p5} days \n'
      f'BC less than 0.1: \t {_p1} days \n'
      f'BC less than 0.05: \t {_p05} days \n'
      f'BC less than 0.01: \t {_p01} days \n'
     )

# %%
ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.05).resample(time='1D').median().plot.hist(bins=np.linspace(0,10), alpha=.5)

ds_jfm['OA'].resample(time='1D').median().plot.hist(bins=np.linspace(0,10), alpha = .5)

# %% [markdown] tags=[]
# ## FMA 

# %%
ds_jfm = ds.where(ds['time.month'].isin([2,3,4]))

# %%
_all = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<10000).resample(time='1D').median().count().values

_1 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<1).resample(time='1D').median().count().values

_p5 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.5).resample(time='1D').median().count().values

_p1 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.1).resample(time='1D').median().count().values

_p05 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.05).resample(time='1D').median().count().values

_p01 = ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.01).resample(time='1D').median().count().values

# %%
print(f'All data: \t \t {_all} days \n'
      f'BC less than 1: \t {_1} days \n'
      f'BC less than 0.5: \t {_p5} days \n'
      f'BC less than 0.1: \t {_p1} days \n'
      f'BC less than 0.05: \t {_p05} days \n'
      f'BC less than 0.01: \t {_p01} days \n'
     )

# %%
ds_jfm['OA'].where(ds_jfm['BC_conc'].fillna(0)<.05).resample(time='1D').median().plot.hist(bins=np.linspace(0,10), alpha=.5)

ds_jfm['OA'].resample(time='1D').median().plot.hist(bins=np.linspace(0,10), alpha = .5)

# %%
ds_jfm.plot.scatter(x='OA',y='BC_conc')

# %%
ds_jfm['OA'].plot()

# %%
ds_jfm['OA'].where(ds_jfm['BC_conc']<2).plot.hist(bins=np.linspace(0,10), alpha=.5)

ds_jfm['OA'].plot.hist(bins=np.linspace(0,10), alpha = .5)

# %% [markdown]
# ## Testing bc filtering 

# %%

# %%

# %%

# %%
