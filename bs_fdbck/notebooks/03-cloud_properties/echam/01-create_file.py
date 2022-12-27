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

select_station = 'SMR'

# %%
xr.set_options(keep_attrs=True) 

# %%
calc_seasons = ['ALL_year']#'ALL_year','DRY','WET_mid','WET','WET_34','WET_old','WET_early','WET_late', 'DRY_early','DRY_late', 'ALL_year']


# %%
from bs_fdbck.constants import path_measurement_data
postproc_data = path_measurement_data /'model_station'/select_station
postproc_data_obs = path_measurement_data /select_station/'processed'

fn_obs_comb_data_full_time =postproc_data_obs /'ATTO_data_comb_hourly.nc'


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
lon_lims = [22.,30.]
lat_lims = [60.,66.]

lat_smr = 61.85
lon_smr = 24.28
model_lev_i=-1

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
# ### Path input data

# %%

path_input_data_noresm = path_extract_latlon_outdata / case_name

# %% [markdown]
# ### Filenames

# %% [markdown]
# #### Define some strings

# %%
str_from_t = pd.to_datetime(from_time1).strftime('%Y%m')
str_to = pd.to_datetime(to_time2).strftime('%Y%m')
str_lonlim = '%.1f-%.1f'%(*lon_lims,)
str_latlim = '%.1f-%.1f'%(*lat_lims,)
str_coordlims = f'{str_lonlim}_{str_latlim}'

# %% [markdown]
# ## Filenames to store products in 3d/2d

# %%
# Filename for case1 concatinated over time
fn1   = path_extract_latlon_outdata/case_name1/f'{case_name1}.h1._{from_time1}-{to_time1}_concat_subs_{str_coordlims}.nc'

# Select variables and time:
fn1_2 = fn1.parent / f'{fn1.stem}_sort.nc'
# Sortby time:
fn1_3 = fn1.parent / f'{fn1.stem}_sort3.nc'

# Filename for case1 concatinated over time

fn2   = path_extract_latlon_outdata/case_name2 /f'{case_name2}.h1._{from_time2}-{to_time2}_concat_subs_{str_coordlims}.nc'

# Select variables and time:
fn2_2 = fn2.parent / f'{fn2.stem}_sort.nc'
# Sortby time:
fn2_3 = fn2.parent / f'{fn2.stem}_sort3.nc'

# Concatinated case1 and case2
fn_comb                = path_input_data_noresm /f'{case_name}.h1._{from_time1}-{to_time2}_concat_subs_{str_coordlims}.nc'
# Concatinated only

#fn_comb_lev1           = path_input_data_noresm /f'{case_name}.h1._{from_time1}-{to_time2}_concat_subs_{str_coordlims}_lev1.nc'
fn_comb_lev1_final     = path_input_data_noresm /f'{case_name}.h1._{from_time1}-{to_time2}_concat_subs_{str_coordlims}_lev1_final.nc'
fn_comb_lev1_finaler    = path_input_data_noresm /f'{case_name}.h1._{from_time1}-{to_time2}_concat_subs_{str_coordlims}_lev1_finaler.nc'
fn_comb_lev1_final_csv = path_input_data_noresm /f'{case_name}.h1._{from_time1}-{to_time2}_concat_subs_{str_coordlims}_lev1_final_wet_season.csv'
fn_final_csv_stem      = path_input_data_noresm /f'{case_name}.h1._{from_time1}-{to_time2}_concat_subs_{str_coordlims}_lev1_final.csv'

# %%
fn_comb_lev1_final_csv

# %% [markdown]
# ### Station variables and cloud variables

# %%
varl_st = [      'SOA_NA','SOA_A1','OM_NI','OM_AI','OM_AC','SO4_NA','SO4_A1','SO4_A2','SO4_AC','SO4_PR',
      'BC_N','BC_AX','BC_NI','BC_A','BC_AI','BC_AC','SS_A1','SS_A2','SS_A3','DST_A2','DST_A3',
           'N50','N100', 'N150', 'N200', 'N500',
#           'N50-500','N100-500', 'N150-500', 'N200-500',
           #'OA',
                 ]
varl_st_computed = ['OA','T_C',]

varl_cl = ['TOT_CLD_VISTAU','TOT_ICLD_VISTAU','TGCLDCWP','TGCLDLWP','TGCLDIWP',
           'TOT_CLD_VISTAU_s','TOT_ICLD_VISTAU_s','optical_depth',
           'CLDFREE',
           'FCTL',
           'ACTREL','ACTNL','TGCLDLWP',
           'FSDSC','FSDSCDRF',
           'FCTI',
           'FCTL',
           'FLNS',
           'FLNSC',
           'FLNT',
           'FLNTCDRF',
           'FLNT_DRF',
           'FLUS',
           'FLUTC','FORMRATE',
           'FREQI',
           'FREQL',
           'FSDSCDRF',
           'FSDS_DRF',
           'FSNS',
           'FSNSC',
           'FSNT',
           'FSNTCDRF',
           'FSNT_DRF',
           'FSUS_DRF',
           'FSUTADRF',
           ]


# %% [markdown] tags=[]
# ### Read files and compute vertical variables, extract surface layer,  and merge timeseries

# %%
fn_comb#.exists()

# %% [markdown]
# ### Produce files if not already computed

# %% tags=[]
if not fn_comb.exists():
    if (not fn1_2.exists()) or (not fn2_2.exists()):
        ds_mod1 = xr.open_dataset(fn1, chunks = {'time':96},engine='netcdf4')#[fn1,fn2])#.sortby('time')
        ds_mod2 = xr.open_dataset(fn2, chunks = {'time':96},engine='netcdf4')

        varl1 = set(ds_mod1.data_vars)

        varl2 = set(ds_mod2.data_vars)


        varl =list(varl1.intersection(varl2))

        ds_mod1 = ds_mod1[varl].sel(time=slice(sel_years_from_files[0],sel_years_from_files[1]))#.sortby('time')

        ds_mod2 = ds_mod2[varl].sel(time=slice(sel_years_from_files[2],sel_years_from_files[3]))#.sortby('time')
        if not fn1_2.exists():
            delayed_obj = ds_mod1.to_netcdf(fn1_2, compute=False)
            with ProgressBar():
                results = delayed_obj.compute()
        if not fn2_2.exists():
            delayed_obj = ds_mod2.to_netcdf(fn2_2, compute=False)
            with ProgressBar():
                results = delayed_obj.compute()
    
    if not fn1_3.exists():
            ds_mod1 = xr.open_dataset(fn1_2, chunks = {'time':48},engine='netcdf4')#[fn1,fn2])#.sortby('time')
            ds_mod1 = compute_total_tau(ds_mod1)
            ds_mod1 = ds_mod1.isel(lev = model_lev_i)
            ds_mod1 = ds_mod1.sortby('time')#.sel(time=slice('2012','2014'))
            delayed_obj = ds_mod1.to_netcdf(fn1_3, compute=False)
            print('hey 1')
            with ProgressBar():
                results = delayed_obj.compute()
    if not fn2_3.exists():
            ds_mod2 = xr.open_dataset(fn2_2, chunks = {'time':48},engine='netcdf4')#[fn1,fn2])#.sortby('time')
            ds_mod2 = compute_total_tau(ds_mod2)
            ds_mod2 = ds_mod2.isel(lev = model_lev_i)
            ds_mod2 = ds_mod2.sortby('time')#.sel(time=slice('2012','2014'))
            delayed_obj = ds_mod2.to_netcdf(fn2_3, compute=False)
            print('hey')
            with ProgressBar():
                results = delayed_obj.compute()
    
    
    ds_mod = xr.open_mfdataset([fn1_3,fn2_3], combine='by_coords', concat_dim='time')

    fn_comb.parent.mkdir(exist_ok=True,)

    delayed_obj = ds_mod.to_netcdf(fn_comb, compute = False)
    with ProgressBar():
        results = delayed_obj.compute()

    #ds_mod = xr.concat([ds_mod1[varl].sel(time=slice('2012','2014')), ds_mod2[varl].sel(time=slice('2015','2018'))], dim='time')


# %% [markdown]
# ### Check: 

# %%
ds_mod = xr.open_dataset(fn_comb,engine='netcdf4', chunks = {'time':48})
(1e-6*ds_mod['NCONC01'].isel(lat=0, lon=0)).plot()

# %% [markdown] tags=[]
# ### Broadcast station variables to every gridcell and manipulate units etc

# %% [markdown]
# We use only hyytiala for org etc, but all grid cells over finland for cloud properties

# %% tags=[]
if not fn_comb_lev1_final.exists():
    ds_all = xr.open_dataset(fn_comb,engine='netcdf4').isel(ilev=model_lev_i)
    ds_sel = ds_all.sel(lat = lat_smr, lon= lon_smr, method='nearest')#.isel( ilev=model_lev_i)#.load()
    ds_all = ds_all.isel(
             nbnd=0
    ).squeeze()
    ds_all = broadcase_station_data(ds_all, lon = lon_smr, lat = lat_smr)
    ds_all = change_units_and_compute_vars(ds_all, temperature=temperature)


    delayed_obj = ds_all.to_netcdf(fn_comb_lev1_final, compute=False)
    print('hey')
    with ProgressBar():
        results = delayed_obj.compute()


# %% [markdown]
#
# ### Add station variables:

# %%
df_comb_station = dic_df_station['NorESM']['OsloAero_intBVOC_f09_f09_mg17_fssp']
df_comb_station.head()


# %% [markdown]
# ### Compute Nx-500

# %%

for v in ['N50','N100','N200','N150','N25','N70','N250']:
    if v in df_comb_station.columns:
        df_comb_station[v+'-500'] = df_comb_station[v] - df_comb_station['N500']
        varl_st_computed.append(v+'-500')
        print(v)

# %%
from bs_fdbck.util.BSOA_datamanip import broadcast_vars_in_ds_sel

# %%

ds_all = xr.open_dataset(fn_comb_lev1_final, chunks = {'lon':1},engine='netcdf4')
ds_all['time'].attrs['timezone'] = 'utc'
ds_all['NCONC01'].isel(lat=1, lon=1).plot()




# %% [markdown]
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
ds_smll = ds_all[['NCONC01']]

# %%
ds_comb_station = df_comb_station.to_xarray()
ds_comb_station=ds_comb_station.assign_coords(station=['ATTO'])

# %%
ds_all['hour'] = ds_all['time.hour']

# %%
ds_all['T_C'].groupby(ds_all['hour']).mean().isel(lat=0,lon=0).plot()
ds_comb_station['T_C'].groupby(ds_comb_station['time.hour']).mean().plot()

# %%
varl_tmp = varl_st + varl_st_computed

varl_tmp = list(set(df_comb_station.columns).intersection(set(varl_tmp)))


# %% tags=[]
ds_smll = broadcast_vars_in_ds_sel(ds_smll, ds_comb_station, varl_tmp, only_already_in_ds= False)

# %% [markdown]
# ## Replace all values by station values

# %%
for v in varl_tmp:
    ds_all[v] = ds_smll[v]

# %% [markdown]
# ## Finally produce daily median dataframe:

# %%
ds_all['TGCLDCWP_incld'].sel(time = '2012-05-30 02:00:00').plot()

# %%
ds_all['TOT_CLD_VISTAU_s_incld'] = ds_all['TOT_CLD_VISTAU_s']/ds_all['CLDTOT']

# %%
ds_all.where(ds_all['TOT_ICLD_VISTAU_s']<500).plot.scatter(x = 'TOT_CLD_VISTAU_s',alpha=.01,
                                                                                      y = 'TOT_ICLD_VISTAU_s')

# %% tags=[]
ds_all.where(ds_all['CLDTOT']>.1).mean('time')['TOT_CLD_VISTAU_s'].plot()

# %% tags=[]
ds_all.mean('time')['TOT_CLD_VISTAU_s'].plot()

# %% tags=[]
ds_all.where(ds_all['CLDTOT']>.1).std('time')['TOT_CLD_VISTAU_s_incld'].plot()

# %% tags=[]
ds_all.where(ds_all['CLDTOT']>.01).std('time')['TOT_CLD_VISTAU_s_incld'].plot()

# %% tags=[]
ds_all.mean('time')['TOT_CLD_VISTAU_s_incld'].plot()

# %% tags=[]
ds_all.where(ds_all['CLDTOT']>.1).mean('time')['TOT_CLD_VISTAU_s_incld'].plot()

# %%

# %%
ds_all.where(ds_all['CLDTOT']>.1).where(ds_all['TOT_ICLD_VISTAU_s']<500).plot.scatter(x = 'TOT_CLD_VISTAU_s',alpha=.01,
                                                                                      y = 'TOT_ICLD_VISTAU_s')

# %%
dic_ds = dict()
dic_ds[case_name_noresm] =ds_all

# %%
from timeit import default_timer as timer



from dask.diagnostics import ProgressBar

from bs_fdbck.util.BSOA_datamanip.atto import season2month

# %%
for key in dic_ds:
    dic_ds[key] = dic_ds[key].rename(rn_dic_noresm_cloud)



# %%
if not fn_comb_lev1_finaler.exists():
    with ProgressBar():
        dic_ds[case_name_noresm].to_netcdf(fn_comb_lev1_finaler)

# %%
dic_ds[key]['OA'].mean('lon').plot()

# %%
for seas in calc_seasons:
    _fn_csv = fn_final_csv_stem.parent / (fn_final_csv_stem.stem + seas+'.csv')
    print(_fn_csv)
    if True:#not _fn_csv.exists():
        start = timer()


        dic_df = get_dic_df_mod(dic_ds, select_hours_clouds=True, summer_months=season2month[seas],
                                from_hour=daytime_from,
                                kwrgs_mask_clouds = dict(min_reff=1),
                   to_hour=daytime_to,)

        df_mod = dic_df[case_name_noresm]

        df_mod= df_mod.dropna()
        print(_fn_csv)
        df_mod.to_csv(_fn_csv)
        end = timer()
        print(end - start) # Time in seconds, e.g. 5.38091952400282
        print(f'DONE! That took {(end-start)} seconds')
        print(f'That is  {((end-start)/60)} minuts')



# %%

# %%
_ds = dic_ds['OsloAero_intBVOC_f09_f09_mg17_fssp245']
_ds['N100'].sel(time = '2012-05-30 02:00:00').plot()

# %%
_ds = dic_ds['OsloAero_intBVOC_f09_f09_mg17_fssp245']
_ds['COT'].sel(time = '2012-05-30 02:00:00').plot()

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

# %%
len(fl_open)

# %% [markdown] tags=[]
# ### Open files, decode time, drop excess coords, select bottom layer, broadcast station vars to whole grid and compute units etc

# %%
dic_df_station['ECHAM-SALSA']['SALSA_BSOA_feedback']

# %%
model_lev_i

# %%
if not fn_final_echam.exists():
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
    


    ds_all = broadcase_station_data(ds_all, varl_st=varl_st_echam, lon = lon_smr, lat = lat_smr)
    

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
# ### Finally produce daily median dataset:

# %%
dic_ds = dict()
dic_ds[case_name] =ds_all

# %%
import numpy as np

# %%
ds_all['cwp'].plot(bins=np.linspace(0,1000,20),alpha=.5, )

ds_all['cwp_incld'].plot(bins=np.linspace(0,1000,20), alpha=.5, color='r')

# %%

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
ds_comb_station=ds_comb_station.assign_coords(station=['ATTO'])

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

# %% [markdown]
# ## Save for different seasons:
#

# %%
#calc_seasons = ['WET','DRY', 'WET_mid','WET_early','WET_late', 'DRY_early','DRY_late']

for key in dic_ds:
    dic_ds[key] = dic_ds[key].rename(rn_dic_echam_cloud)

# %%
daytime_from

# %%
for seas in calc_seasons:
    _fn_csv = fn_final_echam_csv_stem.parent / (fn_final_echam_csv_stem.stem + seas+'.csv')
    print(_fn_csv)
    if True:#not _fn_csv.exists():
        #for key in dic_ds.keys():

        dic_df = get_dic_df_mod(dic_ds, select_hours_clouds=True, summer_months=season2month[seas],mask_cloud_values =True,
                                from_hour=daytime_from,
                                to_hour=daytime_to,
                                kwrgs_mask_clouds = dict(min_reff=1))

        df_mod = dic_df[case_name_echam]
        #with ProgressBar():
        df_mod = df_mod.dropna()
        df_mod.to_csv(_fn_csv)

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
fn_final_echam_csv

# %%
