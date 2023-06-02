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
# %load_ext autoreload
# %autoreload 2


# %%
import useful_scit.util.log as log
log.ger.setLevel(log.log.INFO)
import xarray as xr

import matplotlib.pyplot as plt
from bs_fdbck.constants import path_extract_latlon_outdata
from dask.diagnostics import ProgressBar

from bs_fdbck.util.imports import import_fields_xr_echam

from bs_fdbck.util.BSOA_datamanip import compute_total_tau, broadcase_station_data, change_units_and_compute_vars, \
    get_dic_df_mod, change_units_and_compute_vars_echam, extract_2D_cloud_time_echam, rn_dic_echam_cloud,rn_dic_noresm_cloud

from bs_fdbck.util.BSOA_datamanip import fix_echam_time

import pandas as pd


# %%
from timeit import default_timer as timer



from dask.diagnostics import ProgressBar

from bs_fdbck.util.BSOA_datamanip.atto import season2month

# %%

select_station = 'ATTO'

# %%
xr.set_options(keep_attrs=True) 

# %%
calc_seasons = ['ALL_year']


# %%

# %%
tau_lims = [5,50]
r_eff_lim = 1
cloud_top_temp_above = -15 
cld_water_path_above = 50
#include_months = [7,8]


# %%
from bs_fdbck.constants import path_measurement_data
postproc_data = path_measurement_data /'model_station'/select_station
postproc_data_obs = path_measurement_data /select_station/'processed'



# %% [markdown] tags=[]
# ## Daytime values
#
#
# Set the daytime to be from 10 to 17 each day

# %%
daytime_from = 9
daytime_to = daytime_from + 7


# %% [markdown] tags=[]
# ## Read in model station data:

# %%
models = ['ECHAM-SALSA','NorESM']
mod2cases = {'ECHAM-SALSA':['SALSA_BSOA_feedback'],
             'NorESM':['OsloAero_intBVOC_f09_f09_mg17_fssp']
            }
di_mod2cases = mod2cases.copy()


# %%

# %%
dic_df_station=dict()
for mod in models:
    print(mod)
    dic_df_station[mod] = dict()
    for ca in mod2cases[mod]:
        print(mod, ca)
        fn_out = postproc_data/f'{select_station}_station_{mod}_{ca}.csv'
        print(fn_out)
        dic_df_station[mod][ca] = pd.read_csv(fn_out, index_col=0)
        dic_df_station[mod][ca].index = pd.to_datetime(dic_df_station[mod][ca].index)
        #dic_df_mod_case[mod][ca].to_csv(fn_out)

# %% [markdown] tags=[]
# ## NorESM settings

# %%
lon_lims = [293.,308.]
lat_lims = [-8.,-1.]

lat_smr = -2.150
lon_smr = 360-59.009
model_lev_i=-2


temperature = 273.15  # K


from_time1 = '2012-01-01'
to_time1 = '2015-01-01'
from_time2 ='2015-01-01'
to_time2 ='2019-01-01'
sel_years_from_files = ['2012','2014','2015','2018']

# %%
case_name = 'OsloAero_intBVOC_f09_f09_mg17_fssp245'
case_name_noresm = 'OsloAero_intBVOC_f09_f09_mg17_fssp245'

case_name1 = 'OsloAero_intBVOC_f09_f09_mg17_full'
case_name2 = 'OsloAero_intBVOC_f09_f09_mg17_ssp245'

# %%
cases = [case_name]

# %% [markdown]
# ## ECHAM-SALSA

# %% [markdown]
# ### Names etc

# %%

case_name = 'SALSA_BSOA_feedback'
case_name_echam = 'SALSA_BSOA_feedback'
time_res = 'hour'
space_res='locations'
model_name='ECHAM-SALSA'
model_name_echam  ='ECHAM-SALSA'

# %% [markdown]
# ### Input path:

# %%
input_path_echam = path_extract_latlon_outdata / model_name_echam / case_name_echam 

# %%

cases_echam = [case_name_echam]

# %% [markdown]
# ### Station variables  and others

# %%
varl_st_echam = [
    'mmrtrN500',
    'mmrtrN250',
    'mmrtrN200',
    'mmrtrN100',
    'mmrtrN50',
    'mmrtrN3',
    'SO2_gas',
    'APIN_gas',
    'TBETAOCI_gas',
    'BPIN_gas',
    'LIMON_gas',
    'SABIN_gas',
    'MYRC_gas',
    'CARENE3_gas',
    'ISOP_gas',
    'VBS0_gas',
    'V*BS1_gas',
    'VBS10_gas',
    'ORG_mass',
    'oh_con',
    'tempair',
    'ccn02',
    'ccn10',
]


varl_cl_echam = [
    'airdens',
    'uw',
    'vw',
    'cod',
    'cwp',
    'ceff',
    'ceff_ct',
    #'ceff_ct_incl',
    'lcdnc',
    'lcdnc_ct',
    'clfr',
    'cl_time',
    'aot550nm',
    'up_sw',
    'up_sw_cs',
    'up_sw_noa',
    'up_sw_cs_noa',
    'up_lw',
    'up_lw_cs',
    'up_lw_noa',
    'up_lw_cs_noa',
    'emi_monot_bio',
    'emi_isop_bio',
    'tempair',
    'tempair_ct',
    'T_ct',

]


# %% [markdown] tags=[]
# ### Define some strings for files

# %%

str_from_t = pd.to_datetime(from_time1).strftime('%Y%m')
str_to = pd.to_datetime(to_time2).strftime('%Y%m')
str_lonlim = '%.1f-%.1f'%(*lon_lims,)
str_latlim = '%.1f-%.1f'%(*lat_lims,)
str_coordlims = f'{str_lonlim}_{str_latlim}'
str_coordlims

# %% [markdown]
# ### Filenames: 

# %%
fn_final_echam = input_path_echam / f'{case_name}_{from_time1}-{to_time2}_ALL-VARS_concat_subs_{str_coordlims}.nc'
fn_final_echam_csv = input_path_echam / f'{case_name}_{from_time1}-{to_time2}_ALL-VARS_concat_subs_{str_coordlims}_wet_season.csv'
fn_final_echam_csv_stem = input_path_echam / f'{case_name}_{from_time1}-{to_time2}_ALL-VARS_concat_subs_{str_coordlims}.csv'

# %% [markdown]
# ### Open data

# %%

# %%
fl_open = []

for v in varl_cl_echam+ varl_st_echam:
    fn = input_path_echam / f'{case_name}_{from_time1}-{to_time2}_{v}_concat_subs_{str_coordlims}.nc'
    #print(fn)
    if fn.exists():
        fl_open.append(fn)
    else:
        print(f'{v} not found')

# %% [markdown] tags=[]
# ### Open files, decode time, drop excess coords, select bottom layer, broadcast station vars to whole grid and compute units etc

# %%
dic_df_station['ECHAM-SALSA']['SALSA_BSOA_feedback'].head()

# %%
fl_open = list(set(fl_open))

# %%
ds_all = xr.open_mfdataset(fl_open, decode_cf = False)


# %% tags=[]
ds_all

# %% tags=[]
ds_all['time'].attrs['units'] = 'days since 2010-01-01 00:00:00'

ds_all['time'].attrs['calendar'] = 'gregorian'

ds_all['time'] = xr.decode_cf(ds_all[['tempair']])['time']

# %%
_dm = ds_all.where(ds_all['time.month'].isin([1,2,3]), drop=True).isel(lat=1,lon=1)

# %%
_dm['diff'] = _dm['cl_time']-_dm['clfr']

# %%
num_days = 5
_dm['diff'].isel(time=slice(0,24*num_days)).plot(x='time', ylim=[47,0])

# %%
lev = _dm['hyam'].isel(time=0) + _dm['hybm'].isel(time=0)*100000
lev.plot()

# %%
lev

# %%
_dm['pres'] = xr.DataArray(lev.values, dims='lev')

# %%
_dm = _dm.swap_dims({'lev':'pres'}).rename({'lev':'lev_orig2','pres':'lev'})
_dm

# %%
num_days = 5
_dm['clfr'].isel(time=slice(0,24*num_days)).plot(x='time',yscale='log',ylim=[1e5,1000] )#ylim=[47,0])

# %%
num_days = 50
_dm['clfr'].isel(time=slice(0,24*num_days)).plot(x='time',yscale='log',ylim=[1e5,1000] )#ylim=[47,0])

# %%
num_days = 50
_dm['cl_time'].isel(time=slice(0,24*num_days)).plot(x='time',yscale='log',ylim=[1e5,1000] )#ylim=[47,0])

# %%
_dm['clfr'].isel(time=slice(0,24*num_days)).sum('lev').plot()

# %%
_dm['test_measure_2'] = _dm['clfr'].sum('lev')

# %%
num_days = 5
_dm['cl_time'].isel(time=slice(0,24*num_days)).plot(x='time', ylim=[47,0])

# %%
_dm['test_measure']=_dm['diff'].where(_dm['diff']<0).sum('lev')
_dm['test_measure_p']=-_dm['diff'].where(_dm['diff']<0).sum('lev')

# %%
fig, ax = plt.subplots()
_dm['diff'].isel(time=slice(0,24*num_days)).plot(x='time',yscale='log', ylim=[1000e2, 10e2])
ax2 = ax.twinx()
_dm['test_measure_p'].isel(time=slice(0,24*num_days)).plot(ax=ax2)
_dm['test_measure_2'].isel(time=slice(0,24*num_days)).plot(ax=ax2)

# %%
fig, ax = plt.subplots()
_dm['diff'].isel(time=slice(0,24*50)).plot(x='time', ylim=[47,0])
_dm['test_measure_p'].isel(time=slice(0,24*50)).plot(ax=ax.twinx())

# %%
_dm['cl_time'].max('lev')

# %%
_dm['cwp_incld'] = _dm['cwp']/_dm['cl_time'].max('lev')
_dm['cwp_incld']  = _dm['cwp_incld'].where(_dm['cl_time'].max('lev')>.1)

# %%
_dm['cwp_incld'].plot.hist()

# %%
import numpy as np

# %%
_dm['temp_ct'] = _dm['tempair'].where(_dm['cl_time']>.1).min('lev')

# %%
_dm['temp_ct_cf'] = _dm['tempair'].where(_dm['clfr']>.1).min('lev')

# %%
_dmf = _dm[['cwp_incld','cwp','cod','test_measure_p','test_measure_2','temp_ct', 'temp_ct_cf']].to_dataframe()

# %%
_dmf['cwp_bin'] = pd.cut(_dmf['cwp_incld'], bins = np.linspace(.01,.350, 10))
_dmf['cwp_bn'] = _dmf['cwp_bin'].apply(lambda x: x.mid)

# %%
_dmf['test_bin'] = pd.cut(_dmf['test_measure_p'], bins = np.arange(15))
_dmf['test_bn'] = _dmf['test_bin'].apply(lambda x: x.mid)

# %%
_dmf['test_measure_p'].sample(5000).plot.hist()

# %%
_dmf['test_measure_2'].sample(5000).plot.hist()

# %%
(_dmf['temp_ct']-273.15).sample(2000).plot.hist()

# %%
(_dmf['temp_ct_cf']-273.15).sample(2000).plot.hist()

# %%
(_dmf['temp_ct_cf']-_dmf['temp_ct']).sample(2000).plot.hist()

# %%
import seaborn as sns

# %%
fig, ax = plt.subplots(figsize=[20,10])
_dmfl = _dmf[_dmf['test_measure_2'] > 15]#.sample(2000)

sns.swarmplot(x='cwp_bn', y ='cod', data = _dmfl, ax = ax, size=2)

# %%
fig, ax = plt.subplots(figsize=[20,10])
_dmfl = _dmf[_dmf['cod']>1]
_dmfl = _dmfl[_dmfl['cwp_incld']>.05]
_dmfl = _dmfl[_dmfl['test_measure_2']<10]
_dmfl = _dmfl[_dmfl['test_measure_p']<10]
_dmfl = _dmfl[_dmfl['temp_ct']>(273.15-15)]
_dmfl = _dmfl.sample(2000)

sns.swarmplot(x='cwp_bn', y ='cod', data = _dmfl, ax = ax, size=2)

# %%
fig, ax = plt.subplots(figsize=[20,10])
_dmfl = _dmf[_dmf['cod']>1]
_dmfl = _dmfl[_dmfl['cwp_incld']>.05]
_dmfl = _dmfl[_dmfl['test_measure_2']>10]
_dmfl = _dmfl[_dmfl['test_measure_p']>10]
_dmfl = _dmfl.sample(2000)

sns.swarmplot(x='cwp_bn', y ='cod', data = _dmfl, ax = ax, size=2)

# %%
fig, ax = plt.subplots(figsize=[20,10])
_dmfl = _dmf[_dmf['cod']>1]
_dmfl = _dmfl[_dmfl['cwp_incld']>.05]
_dmfl = _dmfl[_dmfl['test_measure_2']>10]
_dmfl = _dmfl[_dmfl['test_measure_p']>10]
_dmfl = _dmfl.sample(2000)

sns.swarmplot(x='cwp_bn', y ='cod', data = _dmfl, ax = ax, size=2)

# %%
fig, ax = plt.subplots(figsize=[20,10])
_dmfl = _dmf[_dmf['cod']>1]
_dmfl = _dmfl[_dmfl['cwp_incld']>.05]
_dmfl = _dmfl[_dmfl['test_measure_2'] > 10].sample(2000)

sns.swarmplot(x='cwp_bn', y ='cod', data = _dmfl, ax = ax, size=2)

# %%
fig, ax = plt.subplots(figsize=[20,10])
_dmfl = _dmf[_dmf['test_measure_p'] < 20].sample(2000)
_dmfl = _dmfl[_dmfl['cod']>5]
_dmfl = _dmfl[_dmfl['cwp']>.05]
sns.swarmplot(x='cwp_bn', y ='cod', data = _dmfl, ax = ax, size=4, hue='test_bn', palette='viridis')

# %%
fig, ax = plt.subplots(figsize=[20,10])
_dmfl = _dmf[_dmf['test_measure_p'] < 20].sample(5000)
_dmfl = _dmfl[_dmfl['cod']>5]
_dmfl = _dmfl[_dmfl['cwp']>.05]
sns.swarmplot(x='cwp_bn', y ='cod', data = _dmfl, ax = ax, size=4, hue='test_bn', palette='viridis')

# %%
fig, ax = plt.subplots(figsize=[20,10])
_dmfl = _dmf.sample(2000)
_dmfl = _dmfl[_dmfl['cod']>5]
_dmfl = _dmfl[_dmfl['cwp']>0]

sns.swarmplot(x='test_bn', y ='cod', data = _dmfl, ax = ax, size=2)

# %%
fig, ax = plt.subplots(figsize=[20,10])
_dmfl = _dmf.sample(2000)
_dmfl = _dmfl[_dmfl['cod']>5]
_dmfl = _dmfl[_dmfl['cwp_incld']>.050]
sns.swarmplot(x='test_bn', y ='cwp_incld', data = _dmfl, ax = ax, size=2)

# %%
fig, ax = plt.subplots(figsize=[20,20])
_dmfl = _dmf[_dmf['test_measure_p'] < 10].iloc

sns.swarmplot(x='cwp_bn', y ='cod', data = _dmfl, ax = ax)

# %%
fig, ax = plt.subplots(figsize=[10,10])
_dmfl = _dmf[_dmf['test_measure_p'] > 10]

sns.swarmplot(x='cwp_bn', y ='cod', data = _dmfl, ax = ax)

# %%
_dmfl = _dmf[_dmf['test_measure_p']<10]
sns.swarmplot(x='cwp_bn', y ='cod', data = _dmfl)

# %%
_dmfl = _dmf[_dmf['test_measure_p']<10]
sns.swarmplot(x='cwp_bn', y ='cod', data = _dmfl)

# %%
sns.swarmplot(x='cwp_bn', y ='cod', data = _dmf)

# %%
fi, ax = plt.subplots()
_dm['clfr'].sel(time=slice('2014-01-01','2014-01-14')).plot(x='time', ylim=[47,0], ax=ax)
_dm['diff'].sum('lev').sel(time=slice('2014-01-01','2014-01-14')).plot(x='time',  c='r', ax = ax.twinx())
_dm['test_measure'].sel(time=slice('2014-01-01','2014-01-14')).plot(x='time',  c='m', ax = ax.twinx())
_dm['cwp'].sel(time=slice('2014-01-01','2014-01-14')).plot(x='time',  c='b', ax = ax.twinx())

# %%
fi, ax = plt.subplots()
_dm['clfr'].sel(time=slice('2014-01-01','2014-01-14')).plot(x='time', ylim=[47,0], ax=ax)
#_dm['diff'].sum('lev').sel(time=slice('2014-01-01','2014-01-14')).plot(x='time',  c='r', ax = ax.twinx())
(-_dm['test_measure']).sel(time=slice('2014-01-01','2014-01-14')).plot(x='time',  c='m', ax = ax.twinx())
#_dm['cwp'].sel(time=slice('2014-01-01','2014-01-14')).plot(x='time',  c='b', ax = ax.twinx())

# %%
fi, ax = plt.subplots()
_dm['diff'].sel(time=slice('2014-01-01','2014-01-14')).plot(x='time', ylim=[47,0], ax=ax)
#_dm['diff'].sum('lev').sel(time=slice('2014-01-01','2014-01-14')).plot(x='time',  c='r', ax = ax.twinx())
(-_dm['test_measure']).sel(time=slice('2014-01-01','2014-01-14')).plot(x='time',  c='m', ax = ax.twinx())
#_dm['cwp'].sel(time=slice('2014-01-01','2014-01-14')).plot(x='time',  c='b', ax = ax.twinx())

# %%
fi, ax = plt.subplots()
#_dm['clfr'].sel(time=slice('2014-01-01','2014-01-14')).plot(x='time', ylim=[47,0], ax=ax)
#_dm['diff'].sum('lev').sel(time=slice('2014-01-01','2014-01-14')).plot(x='time',  c='r', ax = ax.twinx())
(-_dm['test_measure']).sel(time=slice('2014-01-01','2014-01-14')).plot(x='time',  c='m', ax = ax)
_dm['cwp'].sel(time=slice('2014-01-01','2014-01-14')).plot(x='time',  c='b', ax = ax.twinx())

# %%
_dm['temp_ct'] = _dm['tempair'].where(_dm['cl_time']>0.011).min('lev')

# %%
(_dm['temp_ct']-273.15).plot.hist()

# %%
_dm['clfr'].isel(time=200).plot()
_dm['clfr'].isel(time=210).plot()
_dm['clfr'].isel(time=220).plot()


# %%
_dmslev=_dm.sel(lev=slice(23,38))

# %%
fi, ax = plt.subplots()

_dmslev['clfr'].sel(time=slice('2014-01-01','2014-01-14')).plot(x='time', ylim=[47,0], ax=ax)
_dmslev['clfr'].sum('lev').plot(x='time', ax = ax.twinx())


# %%
fi, ax = plt.subplots()

_dmslev['clfr'].where(_dmslev['clfr'].sum('lev')<12).sel(time=slice('2014-01-01','2014-01-14')).plot(x='time', ylim=[47,0], ax=ax)

# %%
_dm['criteria'] = _dmslev['clfr'].sum('lev')<10

# %%
fi, ax = plt.subplots()

# %%
(_dmslev['tempair']-273.15).sel(time=slice('2014-01-01','2014-01-14')).plot(x='time', ylim=[47,0], vmax=0, vmin=-20)

# %%
_dm.plot.scatter(x='cod',y='cwp', alpha=0.2)

# %%
_dm.plot.scatter(x='cod',y='cwp', alpha=0.2, hue='clfr')

# %%
_dm.plot.scatter(x='cod',y='test_measure', alpha=0.2, hue='cwp')

# %%
_dm.plot.scatter(x='cwp',y='test_measure', alpha=0.2, hue='cod')

# %%
fi, ax = plt.subplots()

_dm['clfr'].where(_dm['criteria']).sel(time=slice('2014-01-01','2014-01-14')).plot(x='time', ylim=[47,0], ax=ax)

_dm['cwp'].plot(ax=ax.twinx())

# %%
fi, ax = plt.subplots()

_dm['diff'].sel(time=slice('2014-01-01','2014-01-14')).plot(x='time', ylim=[47,0], ax=ax)

_dm['clfr'].sum('lev').plot(ax=ax.twinx())

# %%

# %%
_df = _dm[['diff','cwp','cod','test_measure', 'criteria','temp_ct']].sum('lev').to_dataframe()

# %%
import seaborn as sns



# %%
_df

# %%
_df['crit2'] = _df['criteria']&(_df['temp_ct']>-15+273.15)

# %%
sns.jointplot(x='cod',y='cwp', data = _df[(((_df['cwp']>.01)&(_df['cwp']<.3))&_df['crit2'])& (_df['cod']<30)], kind='hex', xlim=[0,40])

# %%
sns.jointplot(x='cod',y='cwp', data = _df[(((_df['cwp']>.01)&(_df['cwp']<.2)))& ((_df['cod']<10)&(_df['cod']>0.1))], kind='hex', )

# %%
sns.jointplot(x='cod',y='cwp', data = _df[(((_df['cwp']>.01)&(_df['cwp']<.2))&_df['crit2'])& ((_df['cod']<10)&(_df['cod']>0.1))], kind='hex', )

# %%
sns.jointplot(x='cod',y='cwp', data = _df[((_df['cwp']>.01)&(_df['cwp']<.3))&_df['criteria']], kind='hex',xlim=[0,40])

# %%
_df['test_measure_pos'] = -_df['test_measure']

# %%
sns.jointplot(x='test_measure_pos',y='cwp', data = _df[((_df['cwp']>.01)&(_df['cwp']<.3))], kind='hex',xlim=[0,15])

# %%
sns.jointplot(x='test_measure_pos',y='cod', data = _df[((_df['cwp']>.01)&(_df['cwp']<.3))], kind='hex',xlim=[0,15])

# %%
import numpy as np

# %%
sns.jointplot(x='test_measure_pos',y='cod', data = _df[((_df['cod']>5)&(_df['cod']<30))], kind='hex',xlim=[0,15], )

# %%
sns.jointplot(x='test_measure_pos',y='cod', data = _df[((_df['cod']>5)&(_df['cod']<30))], kind='hex',xlim=[0,15], )

# %%
_df_test = (_df[
    ((_df['cod']>1)&(_df['cod']<30)
     &(_df['test_measure_pos']>10)
    )
]
)
sns.jointplot(x='cod',y='cwp', data = _df_test, kind='hex',xlim=[0,40])

# %%
_df_test = (_df[
    ((_df['cod']>1)&(_df['cod']<30)
     &(_df['test_measure_pos']<10)
    )
]
)
sns.jointplot(x='cod',y='cwp', data = _df_test, kind='hex',xlim=[0,40])

# %%
_df_test = (_df[
    ((_df['cod']>5)&(_df['cod']<30)
     &(_df['test_measure_pos']<10)
    &(_df['cwp']>.1)&(_df['cwp']<.5)
    )
]
)
sns.jointplot(x='cod',y='cwp', data = _df_test, kind='hex')

# %%
_df_test = (_df[
    ((_df['cod']>5)&(_df['cod']<30)
     &(_df['test_measure_pos']>10)
    &(_df['cwp']>.1)&(_df['cwp']<.5)
    )
]
)
sns.jointplot(x='cod',y='cwp', data = _df_test, kind='hex')

# %%
sns.jointplot(x='cod',y='cwp', data = _df[((_df['cwp']>.01)&(_df['cwp']<.3))], kind='hex',xlim=[0,40])

# %%
sns.jointplot(x='test_measure',y='cwp', data = _df[(_df['cwp']>.01)&(_df['cwp']<.3)], kind='hex')

# %%
sns.jointplot(x='diff',y='cwp', data = _df[(_df['cwp']>.01)&(_df['cwp']<.3)], kind='hex')

# %%
sns.jointplot(x='diff',y='cwp', data = _df[(_df['cwp']>.01)&(_df['cwp']<.3)], kind='hex')

# %%
sns.jointplot(x='test_measure',y='cod', data = _df[(_df['cwp']>.01)&(_df['cod']<30)], kind='hex')

# %%
sns.jointplot(x='diff',y='cod', data = _df[(_df['cwp']>.01)&(_df['cod']<30)], kind='hex')

# %%
fi, ax = plt.subplots()
_dm['cl_time'].sel(time=slice('2014-01-01','2014-01-14')).plot(x='time', ylim=[47,0], ax=ax)
_dm['diff'].sum('lev').sel(time=slice('2014-01-01','2014-01-14')).plot(x='time',  c='r', ax = ax.twinx())

# %%
_dm['diff'].sum('lev').sel(time=slice('2014-01-01','2014-01-14')).plot(x='time',  c='r')

# %%
fi, ax = plt.subplots()
_dm['clfr'].sel(time=slice('2014-01-01','2014-01-14')).plot(x='time', ylim=[47,0], ax=ax)

_dm['cwp'].sel(time=slice('2014-01-01','2014-01-14')).plot(x='time',  ax = ax.twinx(), c='r')

# %%
fi, ax = plt.subplots()
_dm['clfr'].sel(time=slice('2014-01-01','2014-05-14')).plot(x='time', ylim=[47,0], ax=ax)

_dm['cwp'].sel(time=slice('2014-01-01','2014-05-14')).plot(x='time',  ax = ax.twinx(), c='r')

# %%
_dm['cl_top_tmp'] = _dm['tempair'].where(_dm['clfr']>.5).min('lev')

# %%

# %%
(_dm['cl_top_tmp']-273.15).plot.hist()

# %%
_dm['cl_time'].sel(time=slice('2014-01-01','2014-01-14')).plot(x='time', ylim=[47,0])

# %%
(ds_all['tempair'].isel(lat=0,lon=0)-273.15).plot(x='time', ylim=[47,0])

# %%
(ds_all['cl_time'].isel(lat=0,lon=0)).plot(x='time', ylim=[47,0])

# %%

# %%
(ds_all['clfr'].isel(lat=0,lon=0)).plot(x='time', ylim=[47,0])

# %%
(ds_all['ceff'].isel(lat=0,lon=0)).plot(x='time', ylim=[47,0], robust=True)

# %%
(ds_all['clfr'].isel(lat=0,lon=0)).isel(time=slice(0,400)).plot(x='time', ylim=[47,0])

# %%
fi, ax = plt.subplots()
(ds_all['cl_time'].isel(lat=0,lon=0)).isel(time=slice(0,400)).plot(x='time', ylim=[47,0])

ds_all['cwp'].isel(lat=0,lon=0).isel(time=slice(0,400)).plot(ax=ax.twinx(), c='r')

# %%
fi, ax = plt.subplots()
(ds_all['clfr'].isel(lat=0,lon=0)).isel(time=slice(0,400)).plot(x='time', ylim=[47,0])

ds_all['cwp'].isel(lat=0,lon=0).isel(time=slice(0,400)).plot(ax=ax.twinx(), c='r')

# %%
fi, ax = plt.subplots()
(ds_all['clfr'].isel(lat=0,lon=0)).isel(time=slice(0,400)).plot(x='time', ylim=[47,0])

ds_all['cod'].isel(lat=0,lon=0).isel(time=slice(0,400)).plot(ax=ax.twinx(), c='r')

# %%
fi, ax = plt.subplots()
from_days = 365
num_days = 375
(ds_all['clfr'].isel(lat=0,lon=0)-273.15).isel(time=slice(24*from_days,24*num_days)).plot(x='time', ylim=[47,0])

ds_all['cwp'].isel(lat=0,lon=0).isel(time=slice(24*from_days,24*num_days)).plot(ax=ax.twinx(), c='r')

# %%
ds_all['min_cl_temp'] = ds_all['tempair'].where(ds_all['clfr']>0.9).min('lev')
ds_all['min_clt_temp'] = ds_all['tempair'].where(ds_all['cl_time']>0.9).min('lev')

# %%
fi, ax = plt.subplots()
from_days = 365*5
num_days = 365*5+10
(ds_all['clfr'].isel(lat=0,lon=0)).isel(time=slice(24*from_days,24*num_days)).plot(x='time', ylim=[47,0])

#ds_all['cwp'].isel(lat=0,lon=0).isel(time=slice(24*from_days,24*num_days)).plot(ax=ax.twinx(), c='r')
(ds_all['min_cl_temp']-273.15).isel(lat=0,lon=0).isel(time=slice(24*from_days,24*num_days)).plot(ax=ax.twinx(), c='r')


# %%
fi, ax = plt.subplots()
from_days = 0#365*5
num_days = 365*7+30
(ds_all['clfr'].isel(lat=0,lon=0)).isel(time=slice(24*from_days,24*num_days)).plot(x='time', ylim=[47,0])

#ds_all['cwp'].isel(lat=0,lon=0).isel(time=slice(24*from_days,24*num_days)).plot(ax=ax.twinx(), c='r')
(ds_all['min_cl_temp']-273.15).where((ds_all['min_cl_temp']-273.15)>-15).isel(lat=0,lon=0).isel(time=slice(24*from_days,24*num_days)).plot(ax=ax.twinx(),marker='.', c='r')


# %%
(ds_all['min_cl_temp']-273.15).where((ds_all['min_cl_temp']-273.15)>-80).isel(lat=0,lon=0).plot.hist()

# %%
(ds_all['min_clt_temp']-ds_all['min_cl_temp']).plot.hist()

# %%
ds_all['diff'] = ds_all['min_clt_temp']-ds_all['min_cl_temp']
ds_all.plot.scatter(x='diff',y='cod', alpha=.1)


# %%
ds_all['tempair_incld'] = ds_all['tempair'].where(ds_all['clfr']>0.99)

# %%
ds_all['tempair_minlev_incld'] = ds_all['tempair_incld'].min('lev')
ds_all['tempair_medianlev_incld'] = ds_all['tempair_incld'].median('lev')
ds_all['tempair_meanlev_incld'] = ds_all['tempair_incld'].mean('lev')

# %%
ds_all['tempair_minlev_incld2'] = ds_all['tempair'].where(ds_all['clfr']>0.9).min('lev')
ds_all['tempair_medianlev_incld2'] = ds_all['tempair'].where(ds_all['clfr']>0.9).median('lev')
ds_all['tempair_meanlev_incld2'] = ds_all['tempair'].where(ds_all['clfr']>0.9).mean('lev')

# %%
(ds_all['tempair_minlev_incld']-273.15).isel(lat=0,lon=0).plot()

# %%
(ds_all['tempair_minlev_incld'].isel(lat=0,lon=0)-273.15).plot.hist(alpha=.5)
(ds_all['tempair_minlev_incld2'].isel(lat=0,lon=0)-273.15).plot.hist(alpha=.5)

# %%
ds_all['tempair_meanlev_incld'].isel(lat=0,lon=0).count().compute()

# %%
ds_all['tempair_meanlev_incld'].where(ds_all['tempair_meanlev_incld']>-15+273.15).isel(lat=0,lon=0).count().compute()

# %%
ds_all['tempair_meanlev_incld'].where(ds_all['tempair_minlev_incld']>-15+273.15).isel(lat=0,lon=0).count().compute()

# %%
(ds_all['tempair_meanlev_incld'].isel(lat=0,lon=0)-273.15).plot.hist(alpha=.5)
(ds_all['tempair_meanlev_incld2'].isel(lat=0,lon=0)-273.15).plot.hist(alpha=.5)

# %%
(ds_all['tempair_medianlev_incld'].isel(lat=0,lon=0)-273.15).plot.hist(alpha=.5)
(ds_all['tempair_medianlev_incld2'].isel(lat=0,lon=0)-273.15).plot.hist(alpha=.5)

# %%
(ds_all['tempair_minlev_incld2'].isel(lat=0,lon=0)-273.15).plot.hist()

# %% tags=[]
ds_all.isel(lat=0,lon=0)

# %%
_df = ds_all.isel(lat=0,lon=0).to_dataframe()

# %%
import seaborn as sns
sns.histplot(x = 'diff',y='cod', data = ds_all.isel(lat=0,lon=0).to_dataframe())

# %%
(ds_all['min_clt_temp']-273.15).where((ds_all['min_cl_temp']-273.15)>-80).isel(lat=0,lon=0).plot.hist()

# %%
(ds_all['min_cl_temp']-273.15).where((ds_all['min_cl_temp']-273.15)>-80).isel(lat=0,lon=0).plot.hist()

# %%
fi, ax = plt.subplots()
num_days = 5
(ds_all['tempair'].isel(lat=0,lon=0)-273.15).isel(time=slice(0,24*num_days)).plot(x='time', ylim=[47,0])

ds_all['cod'].isel(lat=0,lon=0).isel(time=slice(0,24*num_days)).plot(ax=ax.twinx(), c='r')

# %% tags=[]
if True:#not fn_final_echam.exists():
    ds_all = xr.open_mfdataset(fl_open, decode_cf = False)
    #ds_iso = xr.open_dataset(fl_open[21])
    #ds = xr.merge([ds_iso,ds])
    ds_all = import_fields_xr_echam.decode_cf_echam(ds_all)




    #ds_all = import_fields_xr_echam.decode_cf_echam(ds_all)
    ds_all = extract_2D_cloud_time_echam(ds_all)

    #ds_sel = ds_all.sel(lat = lat_smr, lon= lon_smr, method='nearest').isel( lev=model_lev_i)#.load()
    ds_all = ds_all.squeeze()
    ds_all=ds_all.drop(['hyai','hybi','hyam','hybm']).squeeze()
    ds_all = ds_all.isel( lev=model_lev_i)
    


    #ds_all = broadcase_station_data(ds_all, varl_st=varl_st_echam, lon = lon_smr, lat = lat_smr)
    

    ds_all = change_units_and_compute_vars_echam(ds_all)

    delayed_obj = ds_all.to_netcdf(fn_final_echam, compute=False)
    print('hey')
    with ProgressBar():
        results = delayed_obj.compute()


# %% [markdown]
# ### Use station data computed before:

# %%
df_comb_station = dic_df_station['ECHAM-SALSA']['SALSA_BSOA_feedback']


# %% [markdown]
# #### Compute Nx-500

# %%

for v in ['N50','N100','N200','N150','N25','N70','N250']:
    if v in df_comb_station.columns:
        df_comb_station[v+'-500'] = df_comb_station[v] - df_comb_station['N500']
        varl_st_computed.append(v+'-500')
        print(v)

# %%
ds_comb_station = df_comb_station.to_xarray()
ds_comb_station=ds_comb_station.assign_coords(station=['ATTO'])

# %%
ds_all = xr.open_dataset(fn_final_echam,engine='netcdf4')
ds_all['time'].attrs['timezone'] = 'utc'
ds_all['N50'].mean('time').plot()#.isel(lat=0, time=0).plot()#.shape#.plot()


# %%
ds_all['time'].attrs['timezone'] = 'utc'


# %%
ds_all['cwp_incld'].isel(lat=1, lon=1).plot()

# %%
ds_all['tempair_ct']

# %% [markdown]
# ### Fix time for echam

# %%
with xr.set_options(keep_attrs=True):
    attrs = ds_all['time'].attrs.copy()
    ds_all['time'] = ds_all['time'].to_dataframe()['time'].apply(fix_echam_time).values
    ds_all['time'].attrs = attrs

# %%
ds_all['time']

# %% [markdown]
# ## TESTING STUFF: 

# %% [markdown]
# ### Check how many days we loose for masking for cloud top temperature

# %% jupyter={"outputs_hidden": true} tags=[]
ds_all

# %% tags=[] jupyter={"outputs_hidden": true}
_d = ds_all.resample(time='d').mean()

# %%
_d['month'] = _d['time.month']
_dm = _d.where(_d['month'].isin([1,2,3]), drop=True)

# %%
_dm['cod'].isel(lat=0,lon=0).count()

# %% [markdown]
# So starting point is 632 points in each grid cell

# %% jupyter={"outputs_hidden": true} tags=[]
_dm

# %%
_dm.where(_dm['min_cl_tempair']>-15+273.15)['cod'].isel(lat=0,lon=0).count()

# %%
_

# %% [markdown]
# So using cloud top temperature reduces the number of points by a factor of 10

# %% [markdown]
# ## Any other way of checking? 

# %% [markdown]
# ### Finally produce daily median dataset:

# %%
dic_ds = dict()
dic_ds[case_name] =ds_all

# %%
import numpy as np

# %%

# %%
ds_all['cwp'].plot(bins=np.linspace(0,1000,20),alpha=.5, )

ds_all['cwp_incld'].plot(bins=np.linspace(0,1000,20), alpha=.5, color='r')

# %%
ds_all['cwp_incld2'] = ds_all['cwp']/ds_all['cl_clfr_max']


# %%
ds_all['cwp'].plot(bins=np.linspace(0,1000,20),alpha=.5, )

ds_all['cwp_incld'].plot(bins=np.linspace(0,1000,20), alpha=.5, color='r')

ds_all['cwp_incld2'].plot(bins=np.linspace(0,1000,20), alpha=.5, color='r')

# %%
ds_all['cwp'].plot(bins=np.linspace(0,1000,20),alpha=.5, )

ds_all['cwp_incld2'].plot(bins=np.linspace(0,1000,20), alpha=.5, color='r')
ds_all['cwp_incld2'].where(ds_all['cl_clfr_max']>0.8).plot(bins=np.linspace(0,1000,20), alpha=.5, color='m')

# %%
fig, ax = plt.subplots()
ds_all['hour'] = ds_all['time.hour']
ds_all['cod'].groupby(ds_all['hour']).mean().sel(lat=lat_smr,lon=lon_smr, method='nearest').plot()
ds_all['cwp'].groupby(ds_all['hour']).mean().sel(lat=lat_smr,lon=lon_smr, method='nearest').plot(ax = ax.twinx(), c='r')
#ds_all['cod_incld'].groupby(ds_all['hour']).mean().sel(lat=lat_smr,lon=lon_smr, method='nearest').plot(marker='*')


# %%
fig, ax = plt.subplots()
ds_all['hour'] = ds_all['time.hour']
#ds_all['T_C'].mean('time').plot()
ds_all['T_C'].groupby(ds_all['hour']).mean().sel(lat=lat_smr,lon=lon_smr, method='nearest').plot(ax = ax.twinx(), c='r')
#ds_all['cod_incld'].groupby(ds_all['hour']).mean().sel(lat=lat_smr,lon=lon_smr, method='nearest').plot(marker='*')


# %%
ds_['co

# %%
fig, ax = plt.subplots()
ds_all['hour'] = ds_all['time.hour']
ds_all['cod_incld'].mean('time').plot()
#ds_all['cwp'].groupby(ds_all['hour']).mean().sel(lat=lat_smr,lon=lon_smr, method='nearest').plot(ax = ax.twinx(), c='r')
#ds_all['cod_incld'].groupby(ds_all['hour']).mean().sel(lat=lat_smr,lon=lon_smr, method='nearest').plot(marker='*')


# %%
ds_all.where(ds_all['cwp_incld']>50).plot.scatter(x='cwp', y='cod', alpha=0.01)
plt.ylim([0,400])

# %%
ds_all.where(ds_all['cwp_incld']>50).plot.scatter(x='cwp_incld', y='cod_incld', alpha=0.01)
plt.ylim([0,400])

# %%
ds_all.where(ds_all['cl_clfr_max']>.5).plot.scatter(x='ceff_ct', y='cod', alpha=0.01)
plt.ylim([0,400])

# %%
ds_all.where(ds_all['cl_clfr_max']>.9)['cod'].plot.hist(bins=np.arange(-1,50)  )


# %%
ds_all.where((ds_all['cl_time_max']>.2)&(ds_all['cwp']>50))['cod'].plot.hist(bins=np.arange(-1,50)  )


# %%
ds_all.where((ds_all['cl_time_max']>.2)&(ds_all['cwp']>50))['cod'].plot.hist(bins=np.arange(-1,50)  )


# %%
ds_all.where((ds_all['cwp_incld']>50)&(ds_all['cl_time_ct']>.5))['cod'].plot.hist(bins=np.arange(-1,50),  alpha=.5)

ds_all.where((ds_all['cwp_incld']>50)&(ds_all['cl_time_max']>.5))['cod'].plot.hist(bins=np.arange(-1,50)  , alpha=.5)


# %%
ds_all.where((ds_all['cwp_incld']>50)&(ds_all['cl_time_ct']>.1))['cod'].plot.hist(bins=np.arange(-1,50),  alpha=.5)


ds_all.where((ds_all['cwp_incld']>50)&(ds_all['cl_time_max']>.1))['cod'].plot.hist(bins=np.arange(-1,50)  , alpha=.5)


# %%
ds_all.where((ds_all['cwp_incld']>50)&(ds_all['cl_time_ct']>.1))['cod_incld'].plot.hist(bins=np.arange(-1,50),  alpha=.5)
ds_all.where((ds_all['cwp_incld']>50)&(ds_all['cl_time_ct']>.1))['cod'].plot.hist(bins=np.arange(-1,50),  alpha=.5)

# %%
ds_all.where((ds_all['cwp_incld']>50)&(ds_all['cl_time_max']>.1))['cod_incld'].plot.hist(bins=np.arange(-1,50)  , alpha=.5)
ds_all.where((ds_all['cwp_incld']>50)&(ds_all['cl_time_max']>.1))['cod'].plot.hist(bins=np.arange(-1,50)  , alpha=.5)

# %%
ds_all.where(ds_all['cwp_incld']>50).plot.scatter(x='cwp_incld', y='cod_incld', hue = 'ceff_ct', alpha=0.01)
plt.ylim([0,400])

# %%
ds_all.plot.scatter(x='cwp_incld', y='cod_incld', alpha=0.5)

# %%
ds_all.plot.scatter(x='cwp_incld', y='cod_incld', alpha=0.5)

# %%
f, ax = plt.subplots(1)
ds_all['cwp'].plot.hist(bins=np.logspace(0,3.1), alpha=.5,ax = ax)


ds_noresm['TGCLDCWP'].plot.hist(bins=np.logspace(0,3.1),alpha=0.5,ax = ax.twinx(), color='r')
plt.xscale('log')

# %%
(ds_all['tempair_ct']-273.15).plot.hist(bins=100)

# %%
(ds_all['T'].max('time')-273.15).plot()

# %%
lat_smr


# %%
(ds_all['T'].sel(lat = lat_smr, lon = lon_smr, method='nearest')-273.15).plot()

# %%
ts = (ds_all['T'].sel(lat = lat_smr, lon = lon_smr, method='nearest')-273.15)

# %%
tsn = (ds_noresm['T'].sel(lat = lat_smr, lon = lon_smr, method='nearest')-273.15)

# %%
ts.where(ts['time.month'].isin([1,2,3])).plot()
tsn.where(ts['time.month'].isin([1,2,3])).plot()


# %%
tsds = ts.to_dataset()
tsds['hour'] = tsds['time.hour']
tsdsnor =tsn.to_dataset()
tsdsnor['hour'] = tsdsnor['time.hour']

# %%

tsdsnor.groupby(tsdsnor['hour']).median()['T'].plot(label='NorESM')
tsds.groupby(tsds['hour']).median()['T'].plot(label='ECHAM')

plt.legend()

# %%
ts['time'] = pd.to_datetime(ts['time'])

# %%
tsn['time'] = pd.to_datetime(tsn['time'])

# %%
#ts.plot(alpha=.5)
#tsn.resample(time='1d').max().plot(alpha=0.5, label='NorESM', c='b')
ma = ts.resample(time='1d').max()#.plot(alpha=0.5, label='ECHAM-SALSA', c='r')
mi = ts.resample(time='1d').min()#.plot(alpha=0.5, label='ECHAM-SALSA', c='r')
plt.fill_between(ma['time'].values, mi, ma,alpha=0.5, label='ECHAM-SALSA')
ma = tsn.resample(time='1d').max()#.plot(alpha=0.5, label='ECHAM-SALSA', c='r')
mi = tsn.resample(time='1d').min()#.plot(alpha=0.5, label='ECHAM-SALSA', c='r')
plt.fill_between(ma['time'].values, mi, ma, alpha=0.5, label='NorESM')

#ts.plot(alpha=.5)
#tsn.resample(time='1d').min().plot(alpha=0.5, label='NorESM', c='b')
#ts.resample(time='1d').min().plot(alpha=0.5, label='ECHAM-SALSA', c='r')
plt.legend()

# %%
#ts.plot(alpha=.5)
tsn.resample(time='1d').mean().plot(alpha=0.5, label='NorESM')
ts.resample(time='1d').mean().plot(alpha=0.5, label='ECHAM-SALSA')
plt.legend()

# %%
#ts.plot(alpha=.5)
tsn.plot(alpha=0.5, label='NorESM')
ts.plot(alpha=0.5, label='ECHAM-SALSA')
plt.legend()

# %%
#ts.plot(alpha=.5)
tsn.plot(alpha=0.5, label='NorESM')
ts.plot(alpha=0.5, label='ECHAM-SALSA')
plt.legend()

# %%
f, ax = plt.subplots(1)
ds_all['cwp'].sel(lat=-2,lon=290, method='nearest').plot.hist(bins=np.linspace(-1,60), alpha=.5,ax = ax,density=True)
plt.yscale('log')

ds_noresm['TGCLDCWP'].sel(lat=-2,lon=290, method='nearest').plot.hist(bins=np.linspace(-1,60),alpha=0.5,ax = ax, color='r',density=True)
ds_noresm['TGCLDCWP_incld'].sel(lat=-2,lon=290, method='nearest').plot.hist(bins=np.linspace(-1,60),alpha=0.5,ax = ax, color='m',density=True)
plt.yscale('log')

# %%
f, ax = plt.subplots(1)
ds_all['cwp'].plot.hist(bins=np.logspace(0,3.1), alpha=.5,ax = ax)
ds_all['cwp_incld'].plot.hist(bins=np.logspace(0,3.1), alpha=.5,ax = ax)


# %%
f, ax = plt.subplots(1)
ds_all['cwp_incld'].plot.hist(bins=np.logspace(0,3.1), alpha=.5,ax = ax)


ds_noresm['TGCLDCWP_incld'].plot.hist(bins=np.logspace(0,3.1),alpha=0.5,ax = ax.twinx(), color='r')
plt.xscale('log')

# %% [markdown]
# ## Mask values where cloud time max is less than 10 percent

# %%
ds_all = ds_all.where(ds_all['cl_time_max']>.1)

# %%
from datetime import timedelta

# %% [markdown] tags=[]
# ## Shift timezone

# %%
from datetime import timedelta
with ProgressBar():
    ds_all.load()
    
    
if ds_all['time'].attrs['timezone']=='utc':
    ds_all['time'] = ds_all['time'].to_pandas().index - timedelta(hours=4)
    ds_all['time'].attrs['timezone'] = 'utc-4'
    print('shifted time by -4')
    #dic_ds[k] = _ds

# %% [markdown] tags=[]
# ## Broadcast computed variables so that only station value is in the gridcells. 

# %%
ds_smll = ds_all[['mmrtrN100']]

# %%
ds_comb_station = df_comb_station.to_xarray()
ds_comb_station = ds_comb_station.assign_coords(station=['SMR'])

# %%
ds_all['hour'] = ds_all['time.hour']
ds_all['T_C'].groupby(ds_all['hour']).mean().sel(lat=lat_smr,lon=lon_smr, method='nearest').plot()
ds_comb_station['T_C'].groupby(ds_comb_station['time.hour']).mean().plot()

# %%
#ds_comb_station['T_C'].groupby(ds_comb_station['time.hour']).mean().plot()

# %%
lat_smr

# %%
ds_all['hour'] = ds_all['time.hour']
ds_all['T_C'].groupby(ds_all['hour']).mean().sel(lat=lat_smr,lon=lon_smr, method='nearest').plot()
ds_comb_station['T_C'].groupby(ds_comb_station['time.hour']).mean().plot()

# %%

# %%
ds_comb_station = ds_comb_station.drop(['lon'])

# %%
varl_tmp = varl_st_echam + varl_st_computed 

varl_tmp = list(set(df_comb_station.columns).intersection(set(varl_tmp)))


# %% tags=[]
ds_smll = broadcast_vars_in_ds_sel(ds_smll, ds_comb_station, varl_tmp, only_already_in_ds= False)

# %% [markdown]
# ## Replace all values by station values

# %%
for v in varl_tmp:
    ds_all[v] = ds_smll[v]

# %%
ds_all.where((ds_all['cwp_incld']>50)&(ds_all['cl_time_max']>.1))['cod_incld'].plot.hist(bins=np.arange(-1,50)  , alpha=.5)


# %% [markdown]
# ## Save for different seasons: 
#

# %%
dic_ds = dict()
dic_ds[case_name_echam] =ds_all

# %%

# %%
#calc_seasons = ['WET','DRY', 'WET_mid','WET_early','WET_late', 'DRY_early','DRY_late']

for key in dic_ds:
    dic_ds[key] = dic_ds[key].rename(rn_dic_echam_cloud)

# %%
season2month[seas]

# %%
for seas in calc_seasons:
    _fn_csv = fn_final_echam_csv_stem.parent / (fn_final_echam_csv_stem.stem + seas+'.csv')
    print(_fn_csv)
    if True:#not _fn_csv.exists():
        #for key in dic_ds.keys():
    
        dic_df = get_dic_df_mod(dic_ds, select_hours_clouds=True, summer_months=season2month[seas],mask_cloud_values =True,
                                from_hour=daytime_from,
                                to_hour=daytime_to,
                                #kwrgs_mask_clouds = dict(min_reff=1,min_cwp =50, tau_bounds = [5,50])
                                kwrgs_mask_clouds = dict(min_reff=r_eff_lim,min_cwp =cld_water_path_above, tau_bounds = tau_lims),
                               
                               )

        df_mod = dic_df[case_name_echam]
        #with ProgressBar():
        df_mod = df_mod.dropna()    
        #df_mod.to_csv(_fn_csv)

# %%
df_mod.plot.scatter(x='CWP', y='COT')

# %%
_fn_csv

# %% [markdown]
# ## EXTRA

# %%

ds_all = xr.open_dataset(fn_comb_lev1_final)

ds_all['NCONC01'].isel(lat=1, lon=1).plot()


# %%
ds_all['OA'].attrs['units'] = 'ug'

# %%
ds_all.plot.scatter(x='OA',y='CLDFREE', alpha=0.1)

# %%
ds_all['TOT_CLD_VISTAU_s_incld']

# %%
ds_all['month'] = ds_all['time.month']

# %%
ma = ((ds_all['TGCLDCWP_incld']<200 ) & (ds_all['TGCLDCWP_incld']>150 ) )& (ds_all['time.hour']<16 ) & (ds_all['time.hour']>10 ) 
ma = ma & ( (ds_all['month']>=6 ) &(ds_all['month']<=8 ))

# %%
ds_all.where(ma).plot.scatter(x='CLDFREE',y='TOT_CLD_VISTAU_s_incld', alpha=0.1,)

# %%
ds_m = ds_all#.where(ma)

# %%
ds_m.isel(lat=0, lon = 0).groupby(ds_m['time.hour']).mean()['FLNTCDRF'].plot()

# %%
ds_all.where(ma).plot.scatter(x='OA',y='TOT_CLD_VISTAU_s_incld', alpha=0.1,)

# %%
ds_all['hour'] = ds_all['time.hour']
ds_all['month'] = ds_all['time.month']

# %%
ds_all.where(ma).plot.scatter(x='TGCLDCWP_incld',y='TOT_CLD_VISTAU_s_incld', alpha=0.1)
plt.ylim([0,100])

# %%
ds_all.where((ma&(ds_all['month']==8))).plot.scatter(x='hour',y='TOT_CLD_VISTAU_s_incld', alpha=0.1)
plt.ylim([0,100])

# %%
ds_all.where(ma).plot.scatter(x='TGCLDCWP_incld',y='TOT_CLD_VISTAU_s_incld', alpha=0.1)
plt.ylim([0,100])

# %%
ds_all.where(ma).plot.scatter(x='TGCLDCWP',y='TOT_CLD_VISTAU_s_incld', alpha=0.1)
plt.ylim([0,100])

# %%
ds_all.where(ma).plot.scatter(x='TGCLDCWP',y='TOT_CLD_VISTAU_s', alpha=0.1)
plt.ylim([0,100])

# %%
ds_all.where(ma).plot.scatter(x='CLDTOT',y='TOT_CLD_VISTAU_s_incld', alpha=0.1,)

# %%
ds_all.where((ds_all['time.hour']<15) & (ds_all['time.hour']>10) ).plot.scatter(x='CLDFREE',y='TOT_CLD_VISTAU_s_incld', alpha=0.1,)

# %%
ds_all.where((ds_all['time.hour']<15) & (ds_all['time.hour']>10) ).plot.scatter(x='CLDFREE',y='TOT_CLD_VISTAU_s_incld', alpha=0.1,)

# %%
ds_all.where(ma).plot.scatter(x='CLDFREE',y='TOT_CLD_VISTAU_s_incld', alpha=0.1,)

# %%
ds_all.where(ma).plot.scatter(x='CLDTOT',y='CLDFREE', alpha=0.1,)

# %%
ds_all['CLDFREE'].sel(lat=lat_smr, lon  = lon_smr, method = 'nearest').plot.hist()

# %%
ds_all['CLDTOT'].sel(lat=lat_smr, lon  = lon_smr, method = 'nearest').plot.hist()

# %%
ds_all['TGCLDCWP_incld'].where(ds_all['CLDFREE']<.99).mean('time').plot()

# %%
ds_all['ACTNL_incld'].where(ds_all['CLDFREE']<.99).mean('time').plot()

# %%
ds_all['TGCLDCWP_incld'].mean('time').plot()

# %%
ds_all['TGCLDCWP'].mean('time').plot()

# %%
ds_all['TOT_CLD_VISTAU_s'].mean('time').plot()

# %%
ds_all['TOT_CLD_VISTAU_s_incld'].where(ma).mean('time').plot()

# %%
ds_all['TOT_CLD_VISTAU_s_incld'].where(ma).median('time').plot()

# %%

# %%

# %%

# %%

# %%

# %%
