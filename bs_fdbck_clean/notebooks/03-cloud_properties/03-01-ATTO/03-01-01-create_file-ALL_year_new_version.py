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
# # Compute files for cloud plots for models

# %%
# %load_ext autoreload
# %autoreload 2


from datetime import timedelta
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import useful_scit.util.log as log
import xarray as xr
from dask.diagnostics import ProgressBar

from bs_fdbck_clean.constants import path_extract_latlon_outdata
# %%
from bs_fdbck_clean.constants import path_measurement_data
from bs_fdbck_clean.util.BSOA_datamanip import broadcast_vars_in_ds_sel, rn_dic_ec_earth_cloud
from bs_fdbck_clean.util.BSOA_datamanip import compute_total_tau, change_units_and_compute_vars, \
    get_dic_df_mod, change_units_and_compute_vars_echam, extract_2D_cloud_time_echam, rn_dic_echam_cloud, \
    rn_dic_noresm_cloud
from bs_fdbck_clean.util.BSOA_datamanip import fix_echam_time
from bs_fdbck_clean.util.BSOA_datamanip.atto import season2month
from bs_fdbck_clean.util.imports import import_fields_xr_echam

# %%

# %%
xr.set_options(keep_attrs=True)
log.ger.setLevel(log.log.INFO)

# %%

# %% [markdown]
# ## General settings

# %%

select_station = 'ATTO'

calc_seasons = ['ALL_year']

tau_lims = [5, 50]
r_eff_lim = 1
cloud_top_temp_above = -15
cld_water_path_above = 50

postproc_data = path_measurement_data / 'model_station' / select_station
postproc_data_obs = path_measurement_data / select_station / 'processed'

# %%
lon_lims = [293., 308.]
lat_lims = [-8., -1.]

lat_station = -2.150
lon_station = 360 - 59.009
model_lev_i = -2

temperature = 273.15  # K

from_time1 = '2012-01-01'
to_time1 = '2015-01-01'
from_time2 = '2015-01-01'
to_time2 = '2019-01-01'
sel_years_from_files = ['2012', '2014', '2015', '2018']

# %% [markdown]
# #### Model level different in UKESM because model has much finer resolution close to the ground 

# %%
model_lev_i_ukesm = -2

# %%
str_from_t = pd.to_datetime(from_time1).strftime('%Y%m')
str_to = pd.to_datetime(to_time2).strftime('%Y%m')
str_lonlim = '%.1f-%.1f' % (*lon_lims,)
str_latlim = '%.1f-%.1f' % (*lat_lims,)
str_coordlims = f'{str_lonlim}_{str_latlim}'

# %% [markdown] tags=[]
# #### Daytime values
#
#
# Set the daytime to be from 10 to 17 each day

# %%
daytime_from = 9
daytime_to = daytime_from + 7

# %% [markdown] tags=[]
# ## Read in model station data:

# %%
models = ['UKESM','ECHAM-SALSA', 'NorESM', 'EC-Earth']

mod2cases = {'ECHAM-SALSA': ['SALSA_BSOA_feedback'],
             'NorESM': ['OsloAero_intBVOC_f09_f09_mg17_fssp'],
             'EC-Earth': ['ECE3_output_Sara'],
             'UKESM':['AEROCOMTRAJ'],
             }
di_mod2cases = mod2cases.copy()

# %%
dic_df_pre=dict()
for mod in models:
    print(mod)
    dic_df_pre[mod] = dict()
    for ca in mod2cases[mod]:
        print(mod, ca)
        #if model_lev_i !=-2:
        fn_out = postproc_data/f'{select_station}_station_{mod}_{ca}_ilev{model_lev_i}.csv'
        #else:
        #    fn_out = postproc_data/f'{select_station}_station_{mod}_{ca}.csv'

        #fn_out = postproc_data/f'{select_station}_station_{mod}_{ca}.csv'
        print(fn_out)
        dic_df_pre[mod][ca] = pd.read_csv(fn_out, index_col=0)
        dic_df_pre[mod][ca].index = pd.to_datetime(dic_df_pre[mod][ca].index)
        #dic_df_mod_case[mod][ca].to_csv(fn_out)

# %%
dic_df_station = dict()
for mod in models:
    print(mod)
    dic_df_station[mod] = dict()
    for ca in mod2cases[mod]:
        print(mod, ca)
        #if model_lev_i !=-2:
        fn_out = postproc_data/f'{select_station}_station_{mod}_{ca}_ilev{model_lev_i}.csv'
        #else:
        #    fn_out = postproc_data/f'{select_station}_station_{mod}_{ca}.csv'
        
        print(fn_out)
        dic_df_station[mod][ca] = pd.read_csv(fn_out, index_col=0)
        dic_df_station[mod][ca].index = pd.to_datetime(dic_df_station[mod][ca].index)
        # dic_df_mod_case[mod][ca].to_csv(fn_out)

# %%
pd.read_csv('/proj/bolinc/users/x_sarbl/analysis/BS-FDBCK/Data/model_station/ATTO/ATTO_station_UKESM_AEROCOMTRAJ_ilev-1.csv', index_col=0)

# %% [markdown]
# ## Calculate datasets for each model

# %% [markdown] tags=[]
# # NorESM

# %%
case_name = 'OsloAero_intBVOC_f09_f09_mg17_fssp245'
case_name_noresm = 'OsloAero_intBVOC_f09_f09_mg17_fssp245'

case_name1 = 'OsloAero_intBVOC_f09_f09_mg17_full'
case_name2 = 'OsloAero_intBVOC_f09_f09_mg17_ssp245'

# %%
cases = [case_name]

# %% [markdown]
# #### Path input data

# %%
path_input_data_noresm = path_extract_latlon_outdata / case_name

# %% [markdown]
# #### Filenames to store products in 3d/2d

# %%
# Filename for case1 concatinated over time 
fn1 = path_extract_latlon_outdata / case_name1 / f'{case_name1}.h1._{from_time1}-{to_time1}_concat_subs_{str_coordlims}.nc'

# Select variables and time: 
fn1_2 = fn1.parent / f'{fn1.stem}_sort.nc'
# Sortby time: 
fn1_3 = fn1.parent / f'{fn1.stem}_sort3.nc'

# Filename for case1 concatinated over time 

fn2 = path_extract_latlon_outdata / case_name2 / f'{case_name2}.h1._{from_time2}-{to_time2}_concat_subs_{str_coordlims}.nc'

# Select variables and time: 
fn2_2 = fn2.parent / f'{fn2.stem}_sort.nc'
# Sortby time: 
fn2_3 = fn2.parent / f'{fn2.stem}_sort3.nc'

# Concatinated case1 and case2 
fn_comb = path_input_data_noresm / f'{case_name}.h1._{from_time1}-{to_time2}_concat_subs_{str_coordlims}.nc'
# Concatinated only 

# fn_comb_lev1           = path_input_data_noresm /f'{case_name}.h1._{from_time1}-{to_time2}_concat_subs_{str_coordlims}_lev1.nc'
fn_comb_lev1_final = path_input_data_noresm / f'{case_name}.h1._{from_time1}-{to_time2}_concat_subs_{str_coordlims}_lev1_final.nc'
fn_comb_lev1_finaler = path_input_data_noresm / f'{case_name}.h1._{from_time1}-{to_time2}_concat_subs_{str_coordlims}_lev1_finaler.nc'
fn_comb_lev1_final_csv = path_input_data_noresm / f'{case_name}.h1._{from_time1}-{to_time2}_concat_subs_{str_coordlims}_lev1_final_wet_season.csv'
fn_final_csv_stem = path_input_data_noresm / f'{case_name}.h1._{from_time1}-{to_time2}_concat_subs_{str_coordlims}_lev1_final.csv'

# %%
print(fn_comb_lev1_final_csv)

# %% [markdown]
# #### Define NorESM station variables and cloud variables

# %%
varl_st = ['SOA_NA', 'SOA_A1', 'OM_NI', 'OM_AI', 'OM_AC', 'SO4_NA', 'SO4_A1', 'SO4_A2', 'SO4_AC', 'SO4_PR',
           'BC_N', 'BC_AX', 'BC_NI', 'BC_A', 'BC_AI', 'BC_AC', 'SS_A1', 'SS_A2', 'SS_A3', 'DST_A2', 'DST_A3',
           'N50', 'N100', 'N150', 'N200', 'N500',
           #           'N50-500','N100-500', 'N150-500', 'N200-500',
           # 'OA',
           ]
varl_st_computed = ['OA', 'OA_STP', 'OA_amb', 'T_C',  
                    'N50_STP', 'N100_STP', 'N150_STP', 'N200_STP', 'N500_STP',
                    'N50-500_STP', 'N100-500_STP', 'N150-500_STP', 
                    'N200-500_STP', 'N500_STP',
                   ]

varl_cl = ['TOT_CLD_VISTAU', 'TOT_ICLD_VISTAU', 'TGCLDCWP', 'TGCLDLWP', 'TGCLDIWP',
           'TOT_CLD_VISTAU_s', 'TOT_ICLD_VISTAU_s', 'optical_depth',
           'CLDFREE',
           'FCTL',
           'ACTREL', 'ACTNL', 'TGCLDLWP',
           'FSDSC', 'FSDSCDRF',
           'FCTI',
           'FCTL',
           'FLNS',
           'FLNSC',
           'FLNT',
           'FLNTCDRF',
           'FLNT_DRF',
           'FLUS',
           'FLUTC', 'FORMRATE',
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

# %% [markdown]
# #### Concatinate files, compute 2D fields from 3D fields (compute tau) and sortby time.

# %%
fn_comb.exists()

# %% tags=[]
if not fn_comb.exists():
    if (not fn1_2.exists()) or (not fn2_2.exists()):
        ds_mod1 = xr.open_dataset(fn1, chunks={'time': 96}, engine='netcdf4')  # [fn1,fn2])#.sortby('time')
        ds_mod2 = xr.open_dataset(fn2, chunks={'time': 96}, engine='netcdf4')

        varl1 = set(ds_mod1.data_vars)

        varl2 = set(ds_mod2.data_vars)

        varl = list(varl1.intersection(varl2))

        ds_mod1 = ds_mod1[varl].sel(time=slice(sel_years_from_files[0], sel_years_from_files[1]))  # .sortby('time')

        ds_mod2 = ds_mod2[varl].sel(time=slice(sel_years_from_files[2], sel_years_from_files[3]))  # .sortby('time')
        if not fn1_2.exists():
            delayed_obj = ds_mod1.to_netcdf(fn1_2, compute=False)
            with ProgressBar():
                results = delayed_obj.compute()
        if not fn2_2.exists():
            delayed_obj = ds_mod2.to_netcdf(fn2_2, compute=False)
            with ProgressBar():
                results = delayed_obj.compute()

    if not fn1_3.exists():
        ds_mod1 = xr.open_dataset(fn1_2, chunks={'time': 48}, engine='netcdf4')  # [fn1,fn2])#.sortby('time')
        ds_mod1 = compute_total_tau(ds_mod1)
        ds_mod1 = ds_mod1.isel(lev=model_lev_i)
        ds_mod1 = ds_mod1.sortby('time')  # .sel(time=slice('2012','2014'))
        delayed_obj = ds_mod1.to_netcdf(fn1_3, compute=False)
        print('hey 1')
        with ProgressBar():
            results = delayed_obj.compute()
    if not fn2_3.exists():
        ds_mod2 = xr.open_dataset(fn2_2, chunks={'time': 48}, engine='netcdf4')  # [fn1,fn2])#.sortby('time')
        ds_mod2 = compute_total_tau(ds_mod2)
        ds_mod2 = ds_mod2.isel(lev=model_lev_i)
        ds_mod2 = ds_mod2.sortby('time')  # .sel(time=slice('2012','2014'))
        delayed_obj = ds_mod2.to_netcdf(fn2_3, compute=False)
        print('hey')
        with ProgressBar():
            results = delayed_obj.compute()

    ds_mod = xr.open_mfdataset([fn1_3, fn2_3], combine='by_coords', concat_dim='time')

    fn_comb.parent.mkdir(exist_ok=True, )

    delayed_obj = ds_mod.to_netcdf(fn_comb, compute=False)
    with ProgressBar():
        results = delayed_obj.compute()

    # ds_mod = xr.concat([ds_mod1[varl].sel(time=slice('2012','2014')), ds_mod2[varl].sel(time=slice('2015','2018'))], dim='time')

# %%
fn1_3.exists()

# %%
fn2_3.exists()

# %%
fn_comb

# %% [markdown]
# ##### Check: 

# %%
ds_mod = xr.open_dataset(fn_comb, engine='netcdf4', chunks={'time': 48})
(1e-6 * ds_mod['NCONC01'].isel(lat=0, lon=0)).plot()

# %% [markdown] tags=[]
# #### Change units and compute variables

# %% [markdown]
# We use only hyytiala for org etc, but all grid cells over finland for cloud properties

# %% tags=[]
if not fn_comb_lev1_final.exists():
    ds_all = xr.open_dataset(fn_comb, engine='netcdf4').isel(ilev=model_lev_i)
    # ds_sel = ds_all.sel(lat = lat_station, lon= lon_station, method='nearest')#.isel( ilev=model_lev_i)#.load()
    ds_all = ds_all.isel(
        nbnd=0
    ).squeeze()
    # ds_all = broadcase_station_data(ds_all, lon = lon_station, lat = lat_station)
    ds_all = change_units_and_compute_vars(ds_all, temperature=temperature)

    delayed_obj = ds_all.to_netcdf(fn_comb_lev1_final, compute=False)
    print('hey')
    with ProgressBar():
        results = delayed_obj.compute()

# %% [markdown]
#
# #### Add variables from station data to imitate using station measurements

# %%
df_comb_station = dic_df_station['NorESM']['OsloAero_intBVOC_f09_f09_mg17_fssp']
df_comb_station.head()

# %% [markdown]
# #### Open dataset computed above

# %%

ds_all = xr.open_dataset(fn_comb_lev1_final, chunks={'lon': 1}, engine='netcdf4')
ds_all['time'].attrs['timezone'] = 'utc'
ds_all['NCONC01'].isel(lat=1, lon=1).plot()

# %% [markdown] tags=[]
# #### *Mask if ice water path more than 5% of total water path

# %%
mask_liq_cloudtop = (ds_all['FCTL']>0.05) & (ds_all['FCTL']/(ds_all['FCTL']+ds_all['FCTI'])>.8)

mask_liq_cloudtop
ds_all['mask_liq_cloudtop'] = mask_liq_cloudtop
#ds_all = ds_all.where(mask_liq_cloudtop)


# %%
ds_all['frac_lwp2cwp'] = ds_all['TGCLDLWP']/(ds_all['TGCLDIWP']+ds_all['TGCLDLWP'])
ds_all['mask_by_lwp2cwp'] = ds_all['frac_lwp2cwp']>0.95

# %%
ds_all['frac_lwp2cwp'].plot.hist(alpha=0.5,bins=np.linspace(0,1,10))

ds_all['frac_lwp2cwp'].where(ds_all['frac_lwp2cwp']>0.9).plot.hist(alpha=0.5, bins=np.linspace(0,1,10),label='lwp>.9')
ds_all['frac_lwp2cwp'].where(ds_all['mask_liq_cloudtop']).plot.hist(alpha=0.5, bins=np.linspace(0,1,10), label='liq cloudtop')

plt.legend()

# %%
ds_all= ds_all.where(ds_all['mask_by_lwp2cwp'])

# %% [markdown]
# #### * Mask if cloud top fraction of liquid is below 10 %

# %%
ds_all['FCTL'].where(ds_all['TGCLDCWP_incld']>50).plot(alpha=.5)
ds_all.where(ds_all['FCTL']>.1)['FCTL'].where(ds_all['TGCLDCWP_incld']>50).plot(alpha=.5)

# %%
ds_all = ds_all.where(ds_all['FCTL']>.1)

# %% [markdown] tags=[]
# #### NorESM Shift time step to start of period to be consistent with measurements and other models

# %%
#ind = df.index
#df.index = ind - datetime.timedelta(hours=1)
if ds_all['time'].attrs['timezone'] == 'utc':
    ds_all['time'] = ds_all['time'].to_pandas().index - timedelta(hours=1)
    #ds_all['time'].attrs['timezone'] = 'utc+2'
    ds_all['time'].attrs['timezone'] = 'utc'
    
    print('shifted time by -1 for NorESM')


# %% [markdown] tags=[]
# #### Shift timezone

# %%

with ProgressBar():
    ds_all.load()

if ds_all['time'].attrs['timezone'] == 'utc':
    ds_all['time'] = ds_all['time'].to_pandas().index - timedelta(hours=4)
    ds_all['time'].attrs['timezone'] = 'utc-4'
    print('shifted time by -4')
    # dic_ds[k] = _ds

# %% [markdown] tags=[]
# #### Broadcast computed variables so that only station value is in the gridcells. 

# %%
ds_smll = ds_all[['NCONC01']]

# %%
ds_comb_station = df_comb_station.to_xarray()
ds_comb_station = ds_comb_station.assign_coords(station=[select_station])

# %%
varl_tmp = varl_st + varl_st_computed

varl_tmp = list(set(df_comb_station.columns).intersection(set(varl_tmp)))

# %%
varl_tmp = varl_st + varl_st_computed
set(df_comb_station.columns).intersection(set(varl_tmp))

# %% tags=[]
ds_smll = broadcast_vars_in_ds_sel(ds_smll, ds_comb_station, varl_tmp, only_already_in_ds=False)

# %% [markdown]
# #### Replace all values by station values

# %%
for v in varl_tmp:
    if v not in ds_smll:
        print(f'skipping {v} because not in dataset')
        continue
    ds_all[v] = ds_smll[v]

# %% [markdown]
# ##### Controle plots

# %%
ds_all['TGCLDCWP_incld'].sel(time='2012-05-30 02:00:00').plot()

# %%
ds_all['TGCLDLWP_incld'].sel(time='2012-05-30 02:00:00').plot()

# %%
ds_all['TGCLDLWP_incld'].sel(time='2012-05-30 02:00:00').plot()

# %% [markdown]
# #### Finally steps

# %%
dic_ds = dict()
dic_ds[case_name_noresm] = ds_all

# %% [markdown]
# ##### Rename vars

# %%
for key in dic_ds:
    dic_ds[key] = dic_ds[key].rename(rn_dic_noresm_cloud)

# %% [markdown]
# #### Save netcdf file

# %%
if True:#not :#fn_comb_lev1_finaler.exists():
    with ProgressBar():
        dic_ds[case_name_noresm].to_netcdf(fn_comb_lev1_finaler)

# %% [markdown]
# #### Controle plots

# %%
dic_ds[case_name_noresm]['OA'].mean('lon').plot()

# %%
dic_ds[case_name_noresm]['COT'].sel(time='2018-01-07 00:00:00').plot()

# %%
_ds = dic_ds['OsloAero_intBVOC_f09_f09_mg17_fssp245']

_ds.where(_ds['COT'] > 0).where(_ds['CWP'] > 50).plot.scatter(x='CWP', y='COT', alpha=0.01)
plt.ylim([0, 400])

# %%
_ds = dic_ds['OsloAero_intBVOC_f09_f09_mg17_fssp245']

_ds.where(_ds['COT'] > 0).where(_ds['CWP'] > 50).plot.scatter(x='TGCLDCWP', y='TOT_CLD_VISTAU_s', alpha=0.01)
plt.ylim([0, 400])

# %% [markdown]
# #### Save final csv

# %%
ds_noresm = ds_all.copy()

# %%
for seas in calc_seasons:
    _fn_csv = fn_final_csv_stem.parent / (fn_final_csv_stem.stem + seas + '.csv')
    print(_fn_csv)
    if True:# not _fn_csv.exists():
        start = timer()

        dic_df = get_dic_df_mod(
            dic_ds,
            select_hours_clouds=True,
            summer_months=season2month[seas],
            from_hour=daytime_from,
            # kwrgs_mask_clouds = dict(min_reff=1,min_cwp =50, tau_bounds = [5,50]),
            kwrgs_mask_clouds=dict(min_reff = r_eff_lim, min_cwp=cld_water_path_above, tau_bounds=tau_lims),

            # kwrgs_mask_clouds = dict(min_reff = 1),
            to_hour=daytime_to,
        )

        df_mod = dic_df[case_name_noresm]

        # df_mod= df_mod.dropna()
        print(_fn_csv)
        df_mod.to_csv(_fn_csv)
        end = timer()
        print(end - start)  # Time in seconds, e.g. 5.38091952400282
        print(f'DONE! That took {(end - start)} seconds')
        print(f'That is  {((end - start) / 60)} minuts')

# %%

# %%
_ds = dic_ds['OsloAero_intBVOC_f09_f09_mg17_fssp245']
_ds['COT'].sel(time='2012-05-30 02:00:00').plot()

# %%
_ds = dic_ds['OsloAero_intBVOC_f09_f09_mg17_fssp245']
_ds['COT'].sel(time='2012-05-30 23:00:00').plot()

# %%
_ds = dic_ds['OsloAero_intBVOC_f09_f09_mg17_fssp245']
_ds['OA_STP'].sel(time='2012-05-30 02:00:00').plot()

# %%
_ds = dic_ds['OsloAero_intBVOC_f09_f09_mg17_fssp245']
_ds['OA_STP'].sel(time='2012-05-30 02:00:00').plot()

# %% [markdown]
# # ECHAM-SALSA

# %% [markdown]
# #### Names etc

# %%

case_name = 'SALSA_BSOA_feedback'
case_name_echam = 'SALSA_BSOA_feedback'
time_res = 'hour'
space_res = 'locations'
model_name = 'ECHAM-SALSA'
model_name_echam = 'ECHAM-SALSA'

# %% [markdown]
# #### Input path 

# %%
input_path_echam = path_extract_latlon_outdata / model_name_echam / case_name_echam

# %%

cases_echam = [case_name_echam]

# %% [markdown]
# #### Station variables  and others

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
    'N50',
    'N100',
    'N200',
    'N500',
    'N50-500',
    'N100-500',
    'N200-500',
    'N50_STP',
    'N100_STP',
    'N200_STP',
    'N500_STP',
    'N50-500_STP',
    'N100-500_STP',
    'N200-500_STP',

]

varl_cl_echam = [
    'airdens',
    'uw',
    'vw',
    'cod',
    'cwp',
    'ceff',
    'ceff_ct',
    # 'ceff_ct_incl',
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



# %% [markdown]
# #### Filenames: 

# %%
fn_final_echam = input_path_echam / f'{case_name}_{from_time1}-{to_time2}_ALL-VARS_concat_subs_{str_coordlims}.nc'
fn_final_echam_csv = input_path_echam / f'{case_name}_{from_time1}-{to_time2}_ALL-VARS_concat_subs_{str_coordlims}_wet_season.csv'
fn_final_echam_csv_stem = input_path_echam / f'{case_name}_{from_time1}-{to_time2}_ALL-VARS_concat_subs_{str_coordlims}.csv'

# %% [markdown]
# #### Open data area around station

# %%
fl_open = []

for v in varl_cl_echam + varl_st_echam:
    fn = input_path_echam / f'{case_name}_{from_time1}-{to_time2}_{v}_concat_subs_{str_coordlims}.nc'
    # print(fn)
    if fn.exists():
        fl_open.append(fn)
    else:
        print(f'{v} not found')

# %% [markdown] tags=[]
# #### Open files, decode time, drop excess coords, select bottom layer, broadcast station vars to whole grid and compute units etc

# %%
fl_open = list(set(fl_open))

# %%
ds_all = xr.open_mfdataset(fl_open, decode_cf=False)

# %% tags=[]
if not fn_final_echam.exists():
    ds_all = xr.open_mfdataset(fl_open, decode_cf=False)
    # ds_iso = xr.open_dataset(fl_open[21])
    # ds = xr.merge([ds_iso,ds])
    ds_all = import_fields_xr_echam.decode_cf_echam(ds_all)

    # ds_all = import_fields_xr_echam.decode_cf_echam(ds_all)
    ds_all = extract_2D_cloud_time_echam(ds_all)

    # ds_sel = ds_all.sel(lat = lat_station, lon= lon_station, method='nearest').isel( lev=model_lev_i)#.load()
    ds_all = ds_all.squeeze()
    ds_all = ds_all.drop(['hyai', 'hybi', 'hyam', 'hybm']).squeeze()
    ds_all = ds_all.isel(lev=model_lev_i)

    # ds_all = broadcase_station_data(ds_all, varl_st=varl_st_echam, lon = lon_station, lat = lat_station)

    ds_all = change_units_and_compute_vars_echam(ds_all)

    delayed_obj = ds_all.to_netcdf(fn_final_echam, compute=False)
    print('hey')
    with ProgressBar():
        results = delayed_obj.compute()

# %% [markdown]
#
# #### Add variables from station data to imitate using station measurements

# %%
df_comb_station = dic_df_station['ECHAM-SALSA']['SALSA_BSOA_feedback']

# %% [markdown]
# #### Moved to preprocess: Compute Nx-500

# %% [markdown]
# for v in ['N50', 'N100', 'N200', 'N150', 'N25', 'N70', 'N250']:
#     if v in df_comb_station.columns:
#         df_comb_station[v + '-500'] = df_comb_station[v] - df_comb_station['N500']
#         varl_st_computed.append(v + '-500')
#         print(v)

# %%
ds_comb_station = df_comb_station.to_xarray()
ds_comb_station = ds_comb_station.assign_coords(station=[select_station])

# %%
ds_all = xr.open_dataset(fn_final_echam, engine='netcdf4')
ds_all['time'].attrs['timezone'] = 'utc'
ds_all['N50'].mean('time').plot()  # .isel(lat=0, time=0).plot()#.shape#.plot()

# %%
ds_all['time'].attrs['timezone'] = 'utc'

# %%
ds_all['cwp_incld'].isel(lat=1, lon=1).plot()

# %% [markdown]
# #### Fix time for echam

# %%
with xr.set_options(keep_attrs=True):
    attrs = ds_all['time'].attrs.copy()
    ds_all['time'] = ds_all['time'].to_dataframe()['time'].apply(fix_echam_time).values
    ds_all['time'].attrs = attrs

# %% [markdown]
# #### Finally produce daily median dataset:

# %%
dic_ds = dict()
dic_ds[case_name] = ds_all

# %%

# %% [markdown]
# ##### Controle plots

# %%
ds_all['cwp'].plot(bins=np.linspace(0, 1000, 20), alpha=.5, )

ds_all['cwp_incld'].plot(bins=np.linspace(0, 1000, 20), alpha=.5, color='r')

# %%
ds_all['cwp_incld2'] = ds_all['cwp'] / ds_all['cl_clfr_max']

# %%
f, ax = plt.subplots(1)
ds_all['cwp'].plot.hist(bins=np.logspace(0, 3.1), alpha=.5, ax=ax)
ds_all['cwp_incld'].plot.hist(bins=np.logspace(0, 3.1), alpha=.5, ax=ax)

# %%
f, ax = plt.subplots(1)
ds_all['cwp_incld'].plot.hist(bins=np.logspace(0, 3.1), alpha=.5, ax=ax)

ds_noresm['TGCLDCWP_incld'].plot.hist(bins=np.logspace(0, 3.1), alpha=0.5, ax=ax.twinx(), color='r')
plt.xscale('log')

# %% [markdown] tags=[]
# #### *Mask values where cloud time max and cloud top cloud time is less than 10 percent 

# %%
ds_all['cl_time_max'].plot.hist()

# %%
ds_all['cl_time_ct'].where(ds_all['cl_time_max']>.1).plot.hist()

# %%
number_before_mask = ds_all['ceff_ct_incld'].count()

# %%
ds_all = ds_all.where(ds_all['cl_time_max'] > .1)
ds_all = ds_all.where(ds_all['cl_time_ct'] > .1)

# %%
(ds_all['ceff_ct_incld'].count()-number_before_mask)/number_before_mask

# %% [markdown] tags=[]
# #### Shift timezone

# %%

with ProgressBar():
    ds_all.load()

if ds_all['time'].attrs['timezone'] == 'utc':
    ds_all['time'] = ds_all['time'].to_pandas().index - timedelta(hours=4)
    ds_all['time'].attrs['timezone'] = 'utc-4'
    print('shifted time by -4')
    # dic_ds[k] = _ds

# %% [markdown] tags=[]
# #### Broadcast computed variables so that only station value is in the gridcells. 

# %%
ds_smll = ds_all[['mmrtrN100']]

# %%
ds_comb_station = df_comb_station.to_xarray()
ds_comb_station = ds_comb_station.assign_coords(station=[select_station])

# %% [markdown]
# ##### Check time by comparing to station dataset

# %%
ds_all['hour'] = ds_all['time.hour']
ds_all['T_C'].groupby(ds_all['hour']).mean().sel(lat=lat_station, lon=lon_station, method='nearest').plot()
ds_comb_station['T_C'].groupby(ds_comb_station['time.hour']).mean().plot()

# %%

ds_all['T_C'].sel(time=slice('2013-06','2013-08')).sel(lat=lat_station, lon=lon_station, method='nearest').plot(label='full dataset',linewidth=0, marker='.', markersize=5)
ds_comb_station['T_C'].sel(time=slice('2013-06','2013-08')).plot(label='station dataset',linewidth=0,  marker='.', markersize=5, alpha=.2)

# %%
ds_all['hour'] = ds_all['time.hour']
ds_all['T_C'].groupby(ds_all['hour']).mean().sel(lat=lat_station, lon=lon_station, method='nearest').plot()
ds_comb_station['T_C'].groupby(ds_comb_station['time.hour']).mean().plot()

# %%
ds_comb_station = ds_comb_station.drop(['lon'])

# %%
varl_tmp = varl_st_echam + varl_st_computed

varl_tmp = list(set(df_comb_station.columns).intersection(set(varl_tmp)))

# %% tags=[]
ds_smll = broadcast_vars_in_ds_sel(ds_smll, ds_comb_station, varl_tmp, only_already_in_ds=False)

# %% [markdown]
# #### Replace all values by station values

# %%
for v in varl_tmp:
    ds_all[v] = ds_smll[v]

# %%
ds_all.where((ds_all['cwp_incld'] > 50) & (ds_all['cl_time_max'] > .1))['cod_incld'].plot.hist(bins=np.arange(-1, 50),
                                                                                               alpha=.5)

# %% [markdown]
# #### Final steps

# %%
dic_ds = dict()
dic_ds[case_name_echam] = ds_all

# %% [markdown]
# ##### Rename vars

# %%
for key in dic_ds:
    dic_ds[key] = dic_ds[key].rename(rn_dic_echam_cloud)

# %% [markdown]
# #### Save final csv

# %%
for seas in calc_seasons:
    _fn_csv = fn_final_echam_csv_stem.parent / (fn_final_echam_csv_stem.stem + seas + '.csv')
    print(_fn_csv)
    if True:  # not _fn_csv.exists():
        # for key in dic_ds.keys():

        dic_df = get_dic_df_mod(dic_ds, select_hours_clouds=True, summer_months=season2month[seas],
                                mask_cloud_values=True,
                                from_hour=daytime_from,
                                to_hour=daytime_to,
                                # kwrgs_mask_clouds = dict(min_reff=1,min_cwp =50, tau_bounds = [5,50])
                                kwrgs_mask_clouds=dict(min_reff=r_eff_lim, min_cwp=cld_water_path_above,
                                                       tau_bounds=tau_lims),

                                )

        df_mod = dic_df[case_name_echam]
        # with ProgressBar():
        # df_mod = df_mod.dropna()
        df_mod.to_csv(_fn_csv)

# %%
df_mod.plot.scatter(x='CWP', y='COT')

# %%
_fn_csv

# %% [markdown] tags=[]
# # EC-Earth

# %% [markdown]
# #### Names etc

# %%

case_name = 'ECE3_output_Sara'
case_name_ec_earth = 'ECE3_output_Sara'
time_res = 'hour'
space_res = 'locations'
model_name = 'EC-Earth'
model_name_ec_earth = 'EC-Earth'

# %% [markdown]
# #### Input path:

# %%
input_path_ec_earth = path_extract_latlon_outdata / model_name_ec_earth / case_name_ec_earth

# %%

cases_ec_earth = [case_name_ec_earth]

# %% [markdown]
# #### Filenames: 

# %%
fn_intermediate_ec_earth = input_path_ec_earth / f'{case_name}_{from_time1}-{to_time2}_ALL-VARS_concat_subs_{str_coordlims}_intermediate.nc'
fn_intermediate_ec_earth_lev = input_path_ec_earth / f'{case_name}_{from_time1}-{to_time2}_ALL-VARS_concat_subs_{str_coordlims}_intermediate_lev.nc'

fn_final_ec_earth = input_path_ec_earth / f'{case_name}_{from_time1}-{to_time2}_ALL-VARS_concat_subs_{str_coordlims}.nc'
fn_final_ec_earth_csv = input_path_ec_earth / f'{case_name}_{from_time1}-{to_time2}_ALL-VARS_concat_subs_{str_coordlims}.csv'
fn_final_ec_earth_csv_stem = input_path_ec_earth / f'{case_name}_{from_time1}-{to_time2}_ALL-VARS_concat_subs_{str_coordlims}'

# %%
fn_final_ec_earth_csv_stem

# %% [markdown]
# #### Open pre calculated extracted fields

# %%
which = 'IFS'

# %%
fn_t = input_path_ec_earth / f'{case_name}_{which}_{from_time1}-{to_time2}_concat_subs_{str_coordlims}.nc'

# %%
fl_open = []
# ds_list =[]
dic_ds = dict()

for which in ['IFS', 'IFS_T']:
    fn = input_path_ec_earth / f'{case_name}_{which}_{from_time1}-{to_time2}_concat_subs_{str_coordlims}.nc'
    print(fn)
    if fn.exists():
        fl_open.append(fn)
        _ds = xr.open_dataset(fn)
        dic_ds[which] = _ds
    else:
        print(f'{v} not found')

# %% [markdown] tags=[]
# #### Open files, decode time, drop excess coords, select bottom layer, broadcast station vars to whole grid and compute units etc

# %%

# %%
from bs_fdbck_clean.util.BSOA_datamanip.ec_earth import (
    rename_ifs_vars,
    fix_units_ec_earth,
    extract_cloud_top,
    calculate_incld_values_warmclouds,
)

# %% [markdown]
# #### Fix units, calc cloud properties etc.

# %%
fn_intermediate_ec_earth

# %%
if not fn_intermediate_ec_earth.exists():

    for key in dic_ds:
        _ds = dic_ds[key]
        _ds = rename_ifs_vars(_ds)

        _ds = fix_units_ec_earth(_ds)
        # _ds = calculate_incld_values_warmclouds(_ds)
        # _ds = extract_cloud_top(_ds)
        # _ds['lat'] = np.round(_ds['lat'], decimals=2)
        # _ds['lon'] = np.round(_ds['lon'], decimals=2)
        _ds = _ds.sortby('lon')
        _ds = _ds.sortby('lat')
        _ds = (
            _ds
            .assign(
                lat=lambda d: d['lat'].astype('float').round(2))
            .assign(
                lon=lambda d: d['lon'].astype('float').round(2))
        )

        dic_ds[key] = _ds

    ds = dic_ds['IFS']

    ds = calculate_incld_values_warmclouds(ds)

    ds = extract_cloud_top(ds)

    dic_ds['IFS'] = ds

    for key in dic_ds:
        _ds = dic_ds[key]
        ds_l = _ds.isel(lev=model_lev_i)
        dic_ds[key] = ds_l

    ds_t = dic_ds['IFS_T']

    ds = dic_ds['IFS']
    ds = ds.sortby('lat')
    ds = ds.sortby('lon')

    ds_t['lev'] = ds['lev']
    ds_t = ds_t.sortby('lat')
    ds_t = ds_t.sortby('lon')
    ds_t['temp'].plot()
    plt.show()

    drop_list = ['U', 'V', 'temp']
    ds = xr.merge([ds.drop_vars(drop_list).drop_dims(['plev']), ds_t[['temp']]])
    ds['temp'].plot()
    plt.show()

    # ds =fix_units_ec_earth(ds)
    # ds = calculate_incld_values_warmclouds(ds)
    # ds = extract_cloud_top(ds)

    delayed_obj = ds.to_netcdf(fn_intermediate_ec_earth, compute=False)
    with ProgressBar():
        delayed_obj.compute()

# %% [markdown]
# #### Open file with fixed units and extracted cloud params:

# %%
ds_ifs = xr.open_dataset(fn_intermediate_ec_earth, decode_times=False)

# %% [markdown]
# #### Fix units and decode time

# %%
ds_ifs['ttc'].attrs['units'] = 1

for v in ds_ifs.data_vars:
    if 'units' in ds_ifs[v].attrs:
        print(v, ds_ifs[v].attrs['units'])
        if ds_ifs[v].attrs['units'] is np.nan:
            print('******')
        if ds_ifs[v].attrs['units'] == 1:
            ds_ifs[v].attrs['units'] = '1'
            print(f'{v} unit is 1')

ds_ifs = xr.decode_cf(ds_ifs)
ds_ifs['time'].attrs['timezone'] = 'utc'

# %% [markdown]
# #### Overview plots

# %%
ds_ifs['re_liq_cltop'].count('time').plot()

# %%
ds_ifs['cl_frac_where_cltime_pos'].mean('time').plot()

# %%
ds_ifs['cc_all'].mean('time').plot()

# %%
ds_ifs['cc'].mean('time').plot()

# %%
ds_ifs['ttc'].mean('time').plot()

# %%
ds_ifs['re_liq'].isel(lat=2, lon=0).plot(x='time', linewidth=0, marker='.')

# %%
ds_ifs['argmax'].mean('time').plot()

# %%
ds_ifs['re_liq_cltop'].mean('time').plot()

# %%
ds_ifs['cdnc_incld_cltop'].mean('time').plot()

# %%
ds_ifs['re_liq'].mean('time').plot()

# %%
ds_ifs['tclw'].mean('time').plot()

# %%
ds_ifs['cloud_time_norm'].mean('time').plot()

# %%

# %%

# %% [markdown]
# #### Masking and computing vars

# %% [markdown] tags=[]
# ##### *Mask values where cloud fraction is less than 10 percent

# %%
xr.set_options(keep_attrs=True)

# %%
ds_ifs['cc_cltop'].plot.hist()

# %%
ds_ifs = ds_ifs.where(ds_ifs['cc_cltop'] > .1)

# %%

# %%
ds_ifs['liq_frac_cwp'].plot(bins=np.linspace(0,1), alpha=.5)
ds_ifs['liq_frac_cwp'].where(ds_ifs['liq_frac_cwp'] > .90).plot(bins=np.linspace(0,1), alpha=.5)
ds_ifs['liq_frac_cwp'].where(ds_ifs['liq_frac_cwp'] > .95).plot(bins=np.linspace(0,1), alpha=.5)

# %% [markdown] tags=[]
# ##### *Mask if ice water path more than 5% of total water path

# %%
ds_ifs = ds_ifs.where(ds_ifs['liq_frac_cwp'] > .95)

# %% [markdown] tags=[]
# #### Shift timezone

# %%

with ProgressBar():
    ds_ifs.load()

if ds_ifs['time'].attrs['timezone'] == 'utc':
    ds_ifs['time'] = ds_ifs['time'].to_pandas().index - timedelta(hours=4)
    ds_ifs['time'].attrs['timezone'] = 'utc-4'
    print('shifted time by -4')
    # dic_ds[k] = _ds

# %% [markdown] tags=[]
# #### Use station data computed before:

# %%
df_comb_station = dic_df_station[model_name_ec_earth][case_name_ec_earth]

# %%
ds_comb_station = df_comb_station.to_xarray()
ds_comb_station = ds_comb_station.assign_coords(station=[select_station])

# %%
ds_ifs['temp'].plot()

# %% [markdown]
# ##### Check time against station data

# %%
ds_ifs['hour'] = ds_ifs['time.hour']
_ds1 = ds_ifs.sel(time=slice('2012-07', '2012-08'))
_ds2 = ds_comb_station.sel(time=slice('2012-07', '2012-08'))
(_ds1['temp'] - 273.15).groupby(_ds1['hour']).mean().sel(lat=lat_station, lon=lon_station, method='nearest').plot()
_ds2['T_C'].groupby(_ds2['time.hour']).mean().plot(marker='*')

# %%
varl_station_ec_earth = [
    'CCN0.20',
    'CCN1.00',
    'M_BCACS',
    'M_BCAII',
    'M_BCAIS',
    'M_BCCOS',
    'M_DUACI',
    'M_DUACS',
    'M_DUCOI',
    'M_DUCOS',
    'M_POMACS',
    'M_POMAII',
    'M_POMAIS',
    'M_POMCOS',
    'M_SO4ACS',
    'M_SO4COS',
    'M_SO4NUS',
    'M_SOAACS',
    'M_SOAAII',
    'M_SOAAIS',
    'M_SOACOS',
    'M_SOANUS',
    'M_SSACS',
    'M_SSCOS',
    'OA',
    'SOA',
    'N_ACI',
    'N_ACS',
    'N_AII',
    'N_AIS',
    'N_COI',
    'N_COS',
    'N_NUS',
    'RDRY_ACS',
    'RDRY_AIS',
    'RDRY_COS',
    'RDRY_NUS',
    'RWET_ACI',
    'RWET_ACS',
    'RWET_AII',
    'RWET_AIS',
    'RWET_COI',
    'RWET_COS',
    'RWET_NUS',
    'emiisop',
    'emiterp',
    'T',
    'DDRY_NUS',
    'DDRY_AIS',
    'DDRY_ACS',
    'DDRY_COS',
    'DWET_AII',
    'DWET_ACI',
    'DWET_COI',
    'N50',
    'N70',
    'N100',
    'N150',
    'N200',
    'N500',
    'N50-500',
    'N70-500',
    'N100-500',
    'N150-500',
    'N200-500',
    'N50-500_STP',
    'N100-500_STP',
    'N200-500_STP',
    'N50_STP',
    'N100_STP',
    'N200_STP',
    'N500_STP',
    'OA_STP',
    'POM',
    'SOA',
    'SOA2',
    'T_C',

]

# %%

varl_tmp = list(set(df_comb_station.columns).intersection(set(varl_station_ec_earth)))

# %%
ds_smll = ds_ifs[['temp']]

# %%

# %% tags=[]
ds_smll = broadcast_vars_in_ds_sel(ds_smll, ds_comb_station, varl_tmp, only_already_in_ds=False)

# %% [markdown]
# ##### Replace all values by station values

# %%
for v in varl_tmp:
    ds_ifs[v] = ds_smll[v]

# %% [markdown]
# #### Final adjustments
#

# %%
dic_ds = dict()
dic_ds[case_name_ec_earth] = ds_ifs

# %% [markdown]
# ##### Rename variables

# %%
# calc_seasons = ['WET','DRY', 'WET_mid','WET_early','WET_late', 'DRY_early','DRY_late']

for key in dic_ds:
    dic_ds[key] = dic_ds[key].rename(rn_dic_ec_earth_cloud)

# %%
ds = dic_ds[key]

# %%
ds['r_eff'].plot(bins=np.linspace(1, 40));

# %%
ds['CWP_unweigth'] = ds['tclw']

# %%
ds['CWP_unweigth'].plot(bins=np.linspace(0, 500), alpha=.5);
ds['CWP'].plot(bins=np.linspace(0, 500), alpha=.5);

# %% [markdown]
# ##### Controle plots normalizing by cloud fraction

# %%
ds['r_eff'].where(ds['ttc'] > .9).plot(bins=np.linspace(0, 30), alpha=.5, density=True, label='cloud frac above 0.9');
ds['r_eff'].where(ds['ttc'] < .4).plot(bins=np.linspace(0, 30), alpha=.5, density=True, label='cloud frac below 0.1');
plt.legend()
plt.title('CWP divided by cloud fraction')

# %%
ds['CWP'].where(ds['ttc'] > .9).plot(bins=np.linspace(0, 500), alpha=.5, density=True, label='cloud frac above 0.9');
ds['CWP'].where(ds['ttc'] < .3).plot(bins=np.linspace(0, 500), alpha=.5, density=True, label='cloud frac below 0.3');
plt.legend()
plt.title('CWP divided by cloud fraction')

# %%
ds['CWP_unweigth'].where(ds['ttc'] > .9).plot(bins=np.linspace(0, 500), alpha=.5, density=True,
                                              label='cloud frac above 0.9');
ds['CWP_unweigth'].where(ds['ttc'] < .3).plot(bins=np.linspace(0, 500), alpha=.5, density=True,
                                              label='cloud frac below 0.3');
plt.legend()
plt.title('CWP not divided ')

# %% [markdown]
# #### Final save csv

# %% tags=[]
for seas in calc_seasons:
    _fn_csv = fn_final_ec_earth_csv_stem.parent / (fn_final_ec_earth_csv_stem.name + seas + '.csv')
    print(_fn_csv)

    if True:  # not _fn_csv.exists():
        # for key in dic_ds.keys():

        dic_df = get_dic_df_mod(dic_ds,
                                select_hours_clouds=True,
                                summer_months=season2month[seas],
                                mask_cloud_values=True,
                                from_hour=daytime_from,
                                to_hour=daytime_to,
                                # kwrgs_mask_clouds = dict(min_reff=1,min_cwp =50, tau_bounds = [5,50])
                                kwrgs_mask_clouds=dict(min_reff=r_eff_lim,
                                                       min_cwp=cld_water_path_above,
                                                       tau_bounds=tau_lims
                                                       ),

                                )

        df_mod = dic_df[case_name_ec_earth]
        # with ProgressBar():
        # df_mod = df_mod.dropna()
        df_mod.to_csv(_fn_csv)

# %%
df_mod['r_eff'].plot.hist()

# %%
print('Done')

# %% [markdown] tags=[]
# # UKESM

# %%

case_name_ukesm = 'AEROCOMTRAJ'
case_name = case_name_ukesm
time_res = 'hour'
space_res = 'locations'
model_name_ukesm = 'UKESM'
model_name = model_name_ukesm


# %% [markdown]
# #### Input path

# %%
input_path_ukesm = path_extract_latlon_outdata / model_name_ukesm / case_name_ukesm

# %%

cases_ukesm = [case_name_ukesm]

# %%

# %% [markdown]
# #### Station variables  and others

# %%
varl_st_ukesm = [
'Mass_Conc_OM_NS',
'Mass_Conc_OM_KS',
'Mass_Conc_OM_KI',
'Mass_Conc_OM_AS',
'Mass_Conc_OM_CS',
'mmrtr_OM_NS',
'mmrtr_OM_KS',
'mmrtr_OM_KI',
'mmrtr_OM_AS',
'mmrtr_OM_CS',
'nconcNS',
'nconcKS',
'nconcKI',
'nconcAS',
'nconcCS',
'ddryNS',
'ddryKS',
'ddryKI',
'ddryAS',
'ddryCS',
'Temp',
    'N100',
    'N50',
    'N200',
    'N500',
    'OA',
    'N100_STP',
    'N50_STP',
    'N200_STP',
    'N500_STP',
    'N50-500_STP',
    'N100-500_STP',
    'N200-500_STP',
    'OA_STP',
    'T_C',
]

varl_cl_ukesm = [
    'Reff_2d_distrib_x_weight',
    'Reff_2d_x_weight_warm_cloud',
    'area_cloud_fraction_in_each_layer',
    'bulk_cloud_fraction_in_each_layer',
    'cloud_ice_content_after_ls_precip',
    'dry_rho',
    'frozen_cloud_fraction_in_each_layer',
    'liq_cloud_fraction_in_each_layer',
    'qcf',
    'qcl',
    'supercooled_liq_water_content',
    'weight_Reff_2d_distrib',
    'weight_Reff_2d',
    'cdnc_top_cloud_x_weight',
    'weight_of_cdnc_top_cloud',
    'ls_lwp',
    'ls_iwp',
    'conv_iwp',
    'conv_lwp',
    'rho',
    'layer_thickness',
    
]



# %% [markdown]
# #### Filenames:

# %%
fn_final_ukesm = input_path_ukesm / f'{case_name}_{from_time1}-{to_time2}_ALL-VARS_concat_subs_{str_coordlims}.nc'
fn_final_ukesm_csv = input_path_ukesm / f'{case_name}_{from_time1}-{to_time2}_ALL-VARS_concat_subs_{str_coordlims}_wet_season.csv'
fn_final_ukesm_csv_stem = input_path_ukesm / f'{case_name}_{from_time1}-{to_time2}_ALL-VARS_concat_subs_{str_coordlims}.csv'

# %% [markdown]
# #### Open data area around station

# %% tags=[]
fl_open = []
fl_rho = []

for v in varl_cl_ukesm + varl_st_ukesm:
    fn = input_path_ukesm / f'{case_name}_{from_time1}-{to_time2}_{v}_concat_subs_{str_coordlims}.nc'
    #print(fn)
    if fn.exists():
        if (v=='dry_rho') or (v=='rho'):
            fl_rho.append(fn)
            print(f'Adding {v} to rho filelist: {fn}')
            continue
            
        fl_open.append(fn)
        _ds = xr.open_dataset(fn)
        try:
            _ds[v].isel(lat=0,lon=0).plot()
            plt.show()
        except:
            print('ups, coult not plot')
        print(f'Opening {fn}')
    else:
        print(f'{v} not found')
        print(fn)


# %% [markdown] tags=[]
# #### Open files, decode time etc

# %%
fl_open = list(set(fl_open))

# %% [markdown] tags=[]
# #### Some timestamps have small errors in them (20 min off the hour), so we round some files:
#

# %%


ls_ds = []
for f in fl_open:
    _ds = xr.open_dataset(f, decode_times=False)
    if 'time' in _ds.coords:
        if 'hours since' in _ds['time'].units:
            _ds['time'] = np.floor(_ds['time'])
            _ds = xr.decode_cf(_ds)
    ls_ds.append(_ds)

ds_all = xr.merge(ls_ds)

# %%
ls_ds_rho = []
for f in fl_rho:
    _ds = xr.open_dataset(f, decode_times=False)
    if 'time' in _ds.coords:
        if 'hours since' in _ds['time'].units:
            _ds['time'] = np.floor(_ds['time'])
            _ds = xr.decode_cf(_ds)
    ls_ds_rho.append(_ds)

ds_rho = xr.merge(ls_ds_rho)

# %% [markdown]
# ### Somehow the rho has different level than the other, but this is due to it being at the grid box interface, not mid point, so we can ignore it

# %%
for v in ['rho','dry_rho']:
    ds_all[v] = ds_rho[v]


# %%
fn_final_ukesm

# %% [markdown]
# ### layer thickness has the wrong vertical coordinate name:

# %%
ds_all['layer'] = ds_all['layer'].swap_dims({'lev':'model_level'})

# %%
ds_all['ls_lwp'].quantile(.95)*1000

# %%
from bs_fdbck_clean.util.BSOA_datamanip.ukesm import extract_2D_cloud_time_ukesm, change_units_and_compute_vars_ukesm

# %% tags=[]
if True:#not fn_final_ukesm.exists():
    #ds_all = xr.open_mfdataset(fl_open, decode_cf=False)
    # ds_iso = xr.open_dataset(fl_open[21])
    # ds = xr.merge([ds_iso,ds])
    #ds_all = import_fields_xr_echam.decode_cf_echam(ds_all)

    # ds_all = import_fields_xr_echam.decode_cf_echam(ds_all)
    ds_all = extract_2D_cloud_time_ukesm(ds_all)

    # ds_sel = ds_all.sel(lat = lat_smr, lon= lon_smr, method='nearest').isel( lev=model_lev_i)#.load()
    ds_all = ds_all.squeeze()
    #ds_all = ds_all.drop(['hyai', 'hybi', 'hyam', 'hybm']).squeeze()
    if 'model_level' in ds_all.coords:
        ds_all = ds_all.isel(model_level=(-1-model_lev_i_ukesm))

    # ds_all = broadcase_station_data(ds_all, varl_st=varl_st_echam, lon = lon_smr, lat = lat_smr)

    ds_all = change_units_and_compute_vars_ukesm(ds_all)

    
    
    
    delayed_obj = ds_all.to_netcdf(fn_final_ukesm, compute=False)
    print('hey')
    with ProgressBar():
        results = delayed_obj.compute()

# %%
ds_all['lwp'].quantile(.95)

# %%

# %% tags=[]
ds_all = xr.open_dataset(fn_final_ukesm)
ds_all['time'].attrs['timezone'] = 'utc'


# %%

# %% [markdown]
#
# #### Add variables from station data to imitate using station measurements

# %%
df_comb_station = dic_df_station[model_name_ukesm][case_name_ukesm]

# %%
ds_comb_station = df_comb_station.to_xarray()
ds_comb_station = ds_comb_station.assign_coords(station=[select_station])

# %%
ds_comb_station

# %%
ds_all['lwp'].quantile(.95)

# %%
ds_all['lwp_incld'].quantile(.95)

# %%
ds_all['conv_lwp'].isel(lat=1, lon=1).plot()

# %%
ds_all['max_cloud_cover'].isel(lat=1, lon=1).plot()

# %% [markdown]
# #### Finally produce daily median dataset:

# %%
dic_ds = dict()
dic_ds[case_name_ukesm] = ds_all

# %%

# %% [markdown]
# ##### Controle plots

# %%
ds_all['lwp_incld'].where(ds_all['max_cloud_cover']>.1).quantile(0.90)

# %%
ds_all['lwp'].where(ds_all['max_cloud_cover']>.10).quantile(0.95)

# %%
ds_all['lwp_incld'].where(ds_all['max_cloud_cover']>.10).quantile(0.95)

# %%
ds_all['lwp'].plot(bins=np.linspace(0, 400, 20), alpha=.5, )

ds_all['lwp_incld'].plot(bins=np.linspace(0, 400, 20), alpha=.5, color='r')

# %%
f, ax = plt.subplots(1)
ds_all['lwp'].plot.hist(bins=np.logspace(0, 3.1), alpha=.5, ax=ax)
ds_all['lwp_incld'].plot.hist(bins=np.logspace(0, 3.1), alpha=.5, ax=ax)
plt.xscale('log')

# %% [markdown]
# #### Masking and computing vars

# %% [markdown] tags=[]
# #### *Mask where cloud fraction is above 10 percent 

# %%
ds_all['max_cloud_cover'].plot.hist(alpha=.4)
ds_all['max_cloud_cover'].where(ds_all['max_cloud_cover']>0.1).plot.hist(alpha=.4)

# %% [markdown] tags=[]
# ##### *Mask if ice water path more than 5% of total water path

# %%
ds_all = ds_all.where(ds_all['liq_frac_cwp'] > .95)

# %%
ds_all['max_cloud_cover'].plot.hist(alpha=.4)
ds_all['max_cloud_cover'].where(ds_all['max_cloud_cover']>0.1).plot.hist(alpha=.4)

# %%
ds_all = ds_all.where(ds_all['max_cloud_cover']>0.1)

# %% [markdown] tags=[]
# ### *Mask where cloud top weight less than 10 percent: 

# %%
ds_all['lwp_incld'].plot.hist(bins=np.linspace(0,800), alpha =.5)

ds_all['lwp_incld'].where(ds_all['weight_of_cdnc_top_cloud']>0.1).plot.hist(bins=np.linspace(0,800), alpha =.5)

# %%
ds_all = ds_all.where(ds_all['weight_Reff_2d_distrib']>0.1)

# %% [markdown]
# #### UKESM Shift time step to start of period to be consistent with measurements and other models
# Note that station data is already corrected in the same way. 

# %%
if ds_all['time'].attrs['timezone'] == 'utc':
    ds_all['time'] = ds_all['time'].to_pandas().index - timedelta(hours=1)
    #ds_all['time'].attrs['timezone'] = 'utc+2'
    ds_all['time'].attrs['timezone'] = 'utc'
    
    print('shifted time by -1 for UKESM')


# %% [markdown] tags=[]
# #### Shift timezone

# %%

with ProgressBar():
    ds_all.load()

if ds_all['time'].attrs['timezone'] == 'utc':
    ds_all['time'] = ds_all['time'].to_pandas().index - timedelta(hours=4)
    ds_all['time'].attrs['timezone'] = 'utc-4'
    print('shifted time by -4')
    # dic_ds[k] = _ds

# %% [markdown] tags=[]
# #### Broadcast computed variables so that only station value is in the gridcells. 

# %% tags=[]
ds_all

# %%
ds_all['lwp'].where(ds_all['max_cloud_cover']>.10).quantile(0.95)

# %%
ds_all['lwp_incld'].where(ds_all['max_cloud_cover']>.10).quantile(0.95)

# %%
ds_smll = ds_all[['qcf']]

# %%
ds_comb_station = df_comb_station.to_xarray()
ds_comb_station = ds_comb_station.assign_coords(station=[select_station])

# %% [markdown]
# ##### Check time by comparing to station dataset

# %%
lon_station


# %%
ds_comb_station

# %%
ds_all

# %%
_ds = xr.merge([ds_all[['Temp']].sel(lat=lat_station, lon=lon_station, method='nearest'), ds_comb_station[['T_C']]])
_ds['notnull'] = (_ds['Temp'].notnull()) & (_ds['T_C'].notnull())

# %%
(_ds['Temp']-273.15).where(_ds['notnull']).plot(marker='*')
(_ds['T_C']).where(_ds['notnull']).plot(marker='*')


# %%
_da = ds_comb_station['T_C'].sel(time=slice('2013','2018'))

# %%
_da2 = (ds_all['Temp']-273.15).sel(time=slice('2013','2018')).sel(lat=lat_station, lon=lon_station, method='nearest')

# %%
_da.where(_da2.notnull()).plot()
_da2.plot()

# %%
df_comb_station['OA_STP'].plot()

# %%
varl_st_ukesm


# %%
varl_tmp = varl_st_ukesm

varl_tmp = list(set(df_comb_station.columns).intersection(set(varl_tmp)))

# %%
varl_tmp

# %% tags=[]
ds_smll = broadcast_vars_in_ds_sel(ds_smll, ds_comb_station, varl_tmp, only_already_in_ds=False)

# %% [markdown]
# #### Replace all values by station values

# %%
for v in varl_tmp:
    ds_all[v] = ds_smll[v]

# %% [markdown]
# #### Final steps

# %%
dic_ds = dict()
dic_ds[case_name_ukesm] = ds_all

# %% [markdown]
# ##### Rename vars

# %%
from bs_fdbck_clean.util.BSOA_datamanip import rn_dic_ukesm_cloud

# %%
rn_dic_ukesm_cloud

# %%
for key in dic_ds:
    dic_ds[key] = dic_ds[key].rename(rn_dic_ukesm_cloud)

# %% [markdown]
# #### Save final csv

# %%
dic_ds[key]['CWP'].where(ds_all['max_cloud_cover']>.10).quantile(0.95)

# %%
ds_all['lwp_incld'].where(ds_all['max_cloud_cover']>.10).quantile(0.95)

# %%
for seas in calc_seasons:
    _fn_csv = fn_final_ukesm_csv_stem.parent / (fn_final_ukesm_csv_stem.stem + seas + '.csv')
    print(_fn_csv)
    if True:  # not _fn_csv.exists():
        # for key in dic_ds.keys():

        dic_df = get_dic_df_mod(dic_ds, select_hours_clouds=True, summer_months=season2month[seas],
                                mask_cloud_values=True,
                                from_hour=daytime_from,
                                to_hour=daytime_to,
                                # kwrgs_mask_clouds = dict(min_reff=1,min_cwp =50, tau_bounds = [5,50])
                                kwrgs_mask_clouds=dict(min_reff=r_eff_lim, min_cwp=cld_water_path_above,
                                                       tau_bounds=tau_lims),

                                )

        df_mod = dic_df[case_name_ukesm]
        # with ProgressBar():
        # df_mod = df_mod.dropna()
        df_mod.to_csv(_fn_csv)

# %%
pd.read_csv('/proj/bolinc/users/x_sarbl/analysis/Output_data_BS-FDBCK/extracted_latlon_subset/UKESM/AEROCOMTRAJ/AEROCOMTRAJ_2012-01-01-2019-01-01_ALL-VARS_concat_subs_293.0-308.0_-8.0--1.0ALL_year.csv',index_col=0)

# %%
_ds = dic_ds['AEROCOMTRAJ']
_ds = _ds.where(_ds['max_cloud_cover']>0.1)

# %%
#_ds['computed_lwp_sum'].plot(bins=np.linspace(50,1000), alpha=.5);

#(_ds['computed_lwp_sum']/_ds['max_cloud_cover']).plot(bins=np.linspace(50,1000), alpha=.5);

#_ds['lwp'].plot(bins=np.linspace(50,1000), alpha=.5);


(_ds['lwp']/_ds['max_cloud_cover']).plot(bins=np.linspace(50,1000), alpha=.5);



_ds['CWP'].plot(bins=np.linspace(50,1000), alpha=.5);

# %%
df_mod['CWP'].quantile(.95)

# %%
_ds = dic_ds['AEROCOMTRAJ']
(_ds['CWP']
 .where(_ds['r_eff']>1)
 .where(_ds['CWP']>50)
 .where(_ds['liq_frac_cwp']>0.95)
 .plot(bins= np.linspace(50,1000))
)

# %%
_ds = dic_ds['AEROCOMTRAJ']
(_ds['CWP']
 .where(_ds['r_eff']>1)
 .where(_ds['CWP']>50)
 .where(_ds['liq_frac_cwp']>0.95)
 .plot(bins= np.linspace(50,1000))
)

# %%
_ds = dic_ds['AEROCOMTRAJ']
(_ds['lwp']
 .where(_ds['r_eff2']>1)
 .where(_ds['CWP']>50)
 .where(_ds['liq_frac_cwp']>0.95)
 .quantile(.90)
)

# %%

_ds = df_mod.to_xarray()

# %%
_ds['CWP'].plot()

# %%
_ds['CWP'].isel(lat=0,lon=0).plot(marker='.')

_ds['CWP'].isel(lat=1,lon=1).plot(marker='.')

# %%
_ds['OA_STP'].isel(lat=0,lon=0).plot()

_ds['OA_STP'].isel(lat=1,lon=1).plot()

# %%
_ds['OA_STP'].isel(time=270).plot()

# %%
for i in range(0,50):
    _ds['OA_STP'].isel(time=(400+i)).plot()
    plt.show()

# %% [markdown]
#
# #### Add variables from station data to imitate using station measurements

# %%
df_comb_station = dic_df_station['ECHAM-SALSA']['SALSA_BSOA_feedback']

# %%
print()

# %% [markdown]
# #### Moved to preprocess: Compute Nx-500

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
