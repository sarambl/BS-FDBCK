# -*- coding: utf-8 -*-
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
import xarray as xr

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from bs_fdbck.constants import path_extract_latlon_outdata
from dask.diagnostics import ProgressBar
import seaborn as sns


# %%
# %load_ext autoreload

# %autoreload 2


# %%
from bs_fdbck.util.BSOA_datamanip import compute_total_tau, broadcase_station_data, change_units_and_compute_vars, \
    get_dic_df_mod

# %%
def make_fn(case, v_x, v_y, comment=''):
    _x = v_x.split('(')[0]
    _y = v_y.split('(')[0]
    f = f'dist_plots_f09_f09_allyears_Nx{comment}_{case}_{_x}_{_y}.png'
    return plot_path /f



# %%
plot_path = Path('Plots')

# %%
xr.set_options(keep_attrs=True) 

# %% [markdown]
# ## Get observational data

# %%
import pandas as pd

# %%

# %%
import pandas as pd


# %%
from bs_fdbck.constants import path_measurement_data

# %%
from bs_fdbck.constants import path_measurement_data

# %%
fn = path_measurement_data / 'SourceData_Yli_Juuti2021.xls'

df_hyy_1 = pd.read_excel(fn, sheet_name=4, header=1,)# usecols=range(7,12),nrows=7)

df_hyy_1.head()
#df_hyy_1y= df_hyy_1y.rename({'year.1':'year',
#                            'T (degree C).1':'T (degree C)',
#                             'OA (microgram m^-3).1':'OA (microgram m^-3)',
#                             'N100 (cm^-3).1':'N100 (cm^-3)'
#                            }, axis=1)
#df_hyy_1y['year'] = pd.to_datetime(df_hyy_1y['year'].apply(x:str(x)))

df_hyy_1

# %%
import pandas as pd

# %%
df_hyy_1['date'] = df_hyy_1.apply(lambda x: f'{x.year:.0f}-{x.month:02.0f}-{x.day:02.0f}', axis=1)

df_hyy_1['date'] = pd.to_datetime(df_hyy_1['date'] )


# %%
df_hyy_1 = df_hyy_1.set_index(['date','LAT','LON'])

# %% [markdown]
# ## Pick up sizedist info as well

# %%
df_hyy_2 = pd.read_excel(fn, sheet_name=0, header=2, usecols=range(6))
df_hyy_2['date'] = df_hyy_2.apply(lambda x: f'{x.year:.0f}-{x.month:02.0f}-{x.day:02.0f}', axis=1)

df_hyy_2['date'] = pd.to_datetime(df_hyy_2['date'] )



# %%
from bs_fdbck.util.EBAS_data import get_ebas_dataset_Nx_daily_JA_median_df



df_ebas_Nx, ds_ebas_Nx = get_ebas_dataset_Nx_daily_JA_median_df()#x_list = [90,100,110,120])


# %%

df_hyy_2['date'] = df_hyy_2.apply(lambda x: f'{x.year:.0f}-{x.month:02.0f}-{x.day:02.0f}', axis=1)

df_hyy_2['date'] = pd.to_datetime(df_hyy_2['date'] )


df_hyy_2 = df_hyy_2.set_index('date')

# %%
df_hyy_2.index = df_hyy_2.index.rename('time')

# %%
df_hyy_2['N100 (cm^-3)'].plot.hist(bins=50, alpha=0.4, label='obs')

plt.show()



# %% [markdown] tags=[]
# ## Why is my method 20% off their method? Is it integration?

# %%

df_joint_hyy = pd.merge(df_ebas_Nx, df_hyy_2, left_index=True, right_index=True)
(df_joint_hyy['N100']).loc['2014-07':'2014-09'].plot(label='mine')
(df_joint_hyy['N100 (cm^-3)']).loc['2014-07':'2014-09'].plot(label='orig')
plt.legend()
plt.show()



print(df_joint_hyy['N100'][df_joint_hyy['N100 (cm^-3)'].notnull()].mean()/df_joint_hyy['N100 (cm^-3)'].mean())
# %%
df_hyy_1

# %%
take_vars = ['N50','N100','N150','N200','N100 (cm^-3)']

# %%
import numpy as np

# %%
for v in take_vars:
    df_hyy_1[v] = np.nan

for d in df_hyy_1.index.get_level_values(0).unique():
    #print(d)
    for v in take_vars:
        df_hyy_1.loc[d,v] = df_joint_hyy.loc[d,v]

# %%
import pandas as pd

# %%
df_hyy_1

# %%
for v in take_vars:
    df_hyy_1[f'{v}_low'] = df_hyy_1[v]<df_hyy_1[v].quantile(.34)
    df_hyy_1[f'{v}_high']= df_hyy_1[v]>df_hyy_1[v].quantile(.66)
    df_hyy_1[f'{v}_category'] = pd.NA#df_hyy_1.assign(OA_category= pd.NA)

    df_hyy_1.loc[df_hyy_1[f'{v}_high'], f'{v}_category'] = f'{v} high'
    df_hyy_1.loc[df_hyy_1[f'{v}_low'], f'{v}_category'] = f'{v} low'


# %% tags=[]
#df_hyy_1['OA_category']

df_hyy_1['OA_low'] = df_hyy_1['OA (microgram m^-3)']<2
df_hyy_1['OA_high']= df_hyy_1['OA (microgram m^-3)']>2

ddf_hyy_1=df_hyy_1.assign(OA_category= pd.NA)
df_hyy_1.loc[df_hyy_1['OA_high'], 'OA_category'] = 'OA high'
df_hyy_1.loc[df_hyy_1['OA_low'], 'OA_category'] = 'OA low'



# %%
bins = pd.IntervalIndex.from_tuples([(60, 100), (100, 140), (140, 180), (180, 220), (220, 260), (260, 300), (300, 340)])

# %%
labels=[ 80, 120, 160, 200, 240, 280, 320]

# %%
df_hyy_1['CWP_cut']=pd.cut(df_hyy_1['CWP (g m^-2)'], bins=bins, labels=labels)
df_hyy_1['CWP_qcut']=pd.qcut(df_hyy_1['CWP (g m^-2)'], 6)#bins=bins, labels=labels)

# %%
df_hyy_1['CWP_qcutl'] = df_hyy_1['CWP_qcut'].apply(lambda x:x.mid)

df_hyy_1['CWP_cutl'] = df_hyy_1['CWP_cut'].apply(lambda x:x.mid)

# %%
df_hyy_1['OA (microgram m^-3)'][df_hyy_1['OA_low']].plot.hist(bins=50, alpha=0.4, label='obs')
df_hyy_1['OA (microgram m^-3)'][df_hyy_1['OA_high']].plot.hist(bins=50, alpha=0.4, label='obs')



# %%
df_hyy_1['N50'][df_hyy_1['N50_low']].plot.hist(bins=50, alpha=0.4, label='obs')
df_hyy_1['N50'][df_hyy_1['N50_high']].plot.hist(bins=50, alpha=0.4, label='obs')



# %% [markdown] tags=[]
# ## Get model data:

# %% [markdown]
# ## Get station vars:

# %%
from pathlib import Path

from bs_fdbck.util.BSOA_datamanip import ds2df_inc_preprocessing
from bs_fdbck.util.collocate.collocateLONLAToutput import CollocateLONLATout
import useful_scit.util.log as log

from bs_fdbck.util.plot.BSOA_plots import make_cool_grid, plot_scatter

log.ger.setLevel(log.log.INFO)
import time
import xarray as xr
import matplotlib.pyplot as plt

# %%
nr_of_bins = 5
maxDiameter = 39.6  #    23.6 #e-9
minDiameter = 5.0  # e-9
history_field='.h1.'

# %%
from_t = '2012-01-01'
to_t = '2015-01-01'

# %%
from_t2 = '2015-01-01'
to_t2 = '2019-01-01'

# %% [markdown] tags=[]
# ### Cases:

# %%
cases_orig1 = ['OsloAero_intBVOC_f19_f19_mg17_full']
cases_orig2 = ['OsloAero_intBVOC_f19_f19_mg17_ssp245']
# %%
case_mod = 'OsloAero_intBVOC_f19_f19_mg17_fssp'

# %% [markdown] tags=[]
# ### Variables

# %%

varl =['NCONC01','N50','N100','N150','N200',
      'SOA_NA','SOA_A1','OM_NI','OM_AI','OM_AC','SO4_NA','SO4_A1','SO4_A2','SO4_AC','SO4_PR',
      'BC_N','BC_AX','BC_NI','BC_A','BC_AI','BC_AC','SS_A1','SS_A2','SS_A3','DST_A2','DST_A3',
      ]


# %%
for case_name in cases_orig1:
    varlist = varl
    c = CollocateLONLATout(case_name, from_t, to_t,
                           True,
                           'hour',
                           history_field=history_field)
    if c.check_if_load_raw_necessary(varlist ):
        time1 = time.time()
        a = c.make_station_data_merge_monthly(varlist)
        print(a)

        time2 = time.time()
        print('DONE : took {:.3f} s'.format( (time2-time1)))
    else:
        print('UUUPS')
# %%
for case_name in cases_orig2:
    varlist = varl# list_sized_vars_noresm
    c = CollocateLONLATout(case_name, from_t2, to_t2,
                           False,
                           'hour',
                           history_field=history_field)
    if c.check_if_load_raw_necessary(varlist ):
        time1 = time.time()
        a = c.make_station_data_merge_monthly(varlist)
        print(a)

        time2 = time.time()
        print('DONE : took {:.3f} s'.format( (time2-time1)))
    else:
        print('UUUPS')

# %% tags=[]
dic_ds = dict()
for ca in cases_orig1:
    c = CollocateLONLATout(ca, from_t, to_t,
                           False,
                           'hour',
                           history_field=history_field)
    ds = c.get_collocated_dataset(varl)
    if 'location' in ds.coords:
        ds = ds.rename({'location':'station'})
    dic_ds[ca]=ds

# %% tags=[]
#dic_ds = dict()
for ca in cases_orig2:
    c = CollocateLONLATout(ca, from_t2, to_t2,
                           False,
                           'hour',
                           history_field=history_field)
    ds = c.get_collocated_dataset(varl)
    if 'location' in ds.coords:
        ds = ds.rename({'location':'station'})
    dic_ds[ca]=ds

# %%
case1 = cases_orig1[0]
case2 = cases_orig2[0]

ds1 = dic_ds[case1]
ds2 = dic_ds[case2]

from_t

st_y = from_t.split('-')[0]
mid_y_t = str(int(to_t.split('-')[0])-1)
mid_y_f = to_t.split('-')[0]
end_y = to_t2.split('-')[0]

print(st_y, mid_y_t, mid_y_f, end_y)

_ds1 = ds1.sel(time=slice(st_y, mid_y_t))
_ds2 = ds2.sel(time=slice(mid_y_f, end_y))
ds_comb_station = xr.concat([_ds1, _ds2], dim='time')#.sortby('time')

# %%
mid_y_f


# %% [markdown]
# ## Get full grid vars:
#

# %%

# %% [markdown] tags=[]
# ### Settings

# %%
lat_smr = 61.85
lon_smr = 24.28
model_lev_i=-1

# %%
temperature = 273.15  # K

# %%
case_name = 'OsloAero_intBVOC_f09_f09_mg17_fssp245'

from_time1 = '2012-01-01'
to_time1 = '2015-01-01'
from_time2 ='2015-01-01'
to_time2 ='2019-01-01'
sel_years_from_files = [
    '2012-01-01',
    '2014-12-31',
    '2015-01-01',
    '2018-12-31',

]
# %%
case_name = 'OsloAero_intBVOC_f09_f09_mg17_fssp245'

case_name1 = 'OsloAero_intBVOC_f09_f09_mg17_full'
case_name2 = 'OsloAero_intBVOC_f09_f09_mg17_ssp245'

# %%
fn1 = path_extract_latlon_outdata/ case_name1/f'{case_name1}.h1._{from_time1}-{to_time1}_concat_subs_22.0-30.0_60.0-66.0.nc'
fn1_2 = fn1.parent / f'{fn1.stem}_sort.nc'
fn1_3 = fn1.parent / f'{fn1.stem}_sort3.nc'

fn2 = path_extract_latlon_outdata/ case_name2/f'{case_name2}.h1._{from_time2}-{to_time2}_concat_subs_22.0-30.0_60.0-66.0.nc'

fn2_2 = fn2.parent / f'{fn2.stem}_sort.nc'
fn2_3 = fn2.parent / f'{fn2.stem}_sort3.nc'

fn_comb =path_extract_latlon_outdata / case_name /f'{case_name}.h1._{from_time1}-{to_time2}_concat_subs_22.0-30.0_60.0-66.0.nc'
fn_comb_lev1 = path_extract_latlon_outdata/ case_name/f'{case_name}.h1._{from_time1}-{to_time2}_concat_subs_22.0-30.0_60.0-66.0_lev1.nc'
fn_comb_lev1_final = path_extract_latlon_outdata/ case_name/f'{case_name}.h1._{from_time1}-{to_time2}_concat_subs_22.0-30.0_60.0-66.0_lev1_final.nc'
fn_comb_lev1_finaler = path_extract_latlon_outdata/ case_name/f'{case_name}.h1._2012-01-01-2015-01-01_concat_subs_22.0-30.0_60.0-66.0_lev1_finaler.nc'
fn_comb_lev1_final_csv = path_extract_latlon_outdata/ case_name/f'{case_name}.h1._{from_time1}-{to_time2}_concat_subs_22.0-30.0_60.0-66.0_lev1_final.csv'

# %%

cases = [case_name]

# %%
varl =['DOD500','DOD440','ACTREL','ACTNL','TGCLDLWP', #,'SOA_A1',
       'H2SO4','SOA_LV','COAGNUCL','FORMRATE','T'
       ,'FCTL',
       'TOT_CLD_VISTAU','TOT_ICLD_VISTAU','TGCLDCWP',
       'CLDFREE',
      'SOA_NA','SOA_A1','OM_NI','OM_AI','OM_AC','SO4_NA','SO4_A1','SO4_A2','SO4_AC','SO4_PR',
      'BC_N','BC_AX','BC_NI','BC_A','BC_AI','BC_AC','SS_A1','SS_A2','SS_A3','DST_A2','DST_A3', 
       'FSDSC','FSDSCDRF',
       'N50','N100','N150','N200'
      ]


# %% [markdown]
# ## Station variables

# %%
varl_st = [      'SOA_NA','SOA_A1','OM_NI','OM_AI','OM_AC','SO4_NA','SO4_A1','SO4_A2','SO4_AC','SO4_PR',
      'BC_N','BC_AX','BC_NI','BC_A','BC_AI','BC_AC','SS_A1','SS_A2','SS_A3','DST_A2','DST_A3',
                 ]


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
# ## If file not already createad already, skip this part

# %%
fn_comb_lev1.exists()

# %% tags=[]
if not fn_comb.exists():
    if (not fn1_2.exists()) or (not fn2_2.exists()):
        ds_mod1 = xr.open_dataset(fn1, chunks = {'time':48})#[fn1,fn2])#.sortby('time')
        ds_mod2 = xr.open_dataset(fn2, chunks = {'time':48})

        varl1 = set(ds_mod1.data_vars)

        varl2 = set(ds_mod2.data_vars)


        varl =list(varl1.intersection(varl2))

        ds_mod1 = ds_mod1[varl].sel(time=slice(sel_years_from_files[0],sel_years_from_files[1]))#.sortby('time')

        ds_mod2 = ds_mod2[varl].sel(time=slice(sel_years_from_files[2],sel_years_from_files[3]))#.sortby('time')
        print('HEEEEY')
        if not fn1_2.exists():
            delayed_obj = ds_mod1.to_netcdf(fn1_2, compute=False)
            with ProgressBar():
                results = delayed_obj.compute()
        if not fn2_2.exists():
            delayed_obj = ds_mod2.to_netcdf(fn2_2, compute=False)
            with ProgressBar():
                results = delayed_obj.compute()
    
    if not fn1_3.exists():
            ds_mod1 = xr.open_dataset(fn1_2, chunks = {'time':48})#[fn1,fn2])#.sortby('time')
            ds_mod1 = compute_total_tau(ds_mod1)
            ds_mod1 = ds_mod1.isel(lev = model_lev_i)
            ds_mod1 = ds_mod1.sortby('time')#.sel(time=slice('2012','2014'))
            delayed_obj = ds_mod1.to_netcdf(fn1_3, compute=False)
            print('hey 1')
            with ProgressBar():
                results = delayed_obj.compute()
    if not fn2_3.exists():
            ds_mod2 = xr.open_dataset(fn2_2, chunks = {'time':48})#[fn1,fn2])#.sortby('time')
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


# %% [markdown] tags=[]
# ### Select hyytiala grid cell:
#     


# %%
ds_mod = xr.open_dataset(fn_comb, chunks = {'time':48})
ds_mod['NCONC01'].isel(lat=0, lon=0).plot()

# %% [markdown] tags=[]
#     #if not fn_comb_lev1.exists():
#     ds_mod = xr.open_dataset(fn_comb, chunks = {'time':48})#[fn1,fn2])#.sortby('time')
#     #ds_mod2 = xr.open_dataset(fn2, chunks = {'time':48})
#
#     #ds_mod = compute_total_tau(ds_mod)
#
#     #ds_mod = ds_mod.sortby('time')#.sel(time=slice('2012','2014'))
#
#     #ds_mod = ds_mod.isel(lev = model_lev_i)
#
#
#     delayed_obj = ds_mod.to_netcdf(fn_test)#, compute=False)
#     #    print('hey')
#     #with ProgressBar():
#     #    results = delayed_obj.compute()


# %% [markdown]
# ## If file createad already, skip to here

# %% [markdown] tags=[]
# ### Select hyytiala grid cell:

# %% [markdown]
# We use only hyytiala for org etc, but all grid cells over finland for cloud properties

# %%
fn_comb_lev1_final.exists()

# %%
if not fn_comb_lev1_final.exists():
    ds_all = xr.open_dataset(fn_comb).isel(ilev=model_lev_i)
    ds_sel = ds_all.sel(lat = lat_smr, lon= lon_smr, method='nearest')#.isel( ilev=model_lev_i)#.load()
    ds_all = ds_all.isel(
        #ilev=-1,
        # cosp_tau_modis=0,
        #                                                    cosp_tau=0,
        #                                                   cosp_dbze=0,
        #                                                    cosp_ht=0,
        #                                                    cosp_prs = 0,
        #                                                   cosp_reffice=0,
        #                                                    cosp_htmisr=0,
        #                                                    cosp_reffliq=0,
        #                                                    cosp_scol=0,
        #                                                    cosp_sr=0,
        #                                                    cosp_sza=0,
        nbnd=0
    ).squeeze()
    ds_all = broadcase_station_data(ds_all)
    ds_all = change_units_and_compute_vars(ds_all, temperature=temperature)


    delayed_obj = ds_all.to_netcdf(fn_comb_lev1_final, compute=False)
    print('hey')
    with ProgressBar():
        results = delayed_obj.compute()


# %%
fn_comb_lev1.exists()

# %% [markdown] tags=[]
# if not fn_comb_lev1.exists():
#     ds_mod = xr.open_dataset(fn_comb, chunks = {'time':48})#[fn1,fn2])#.sortby('time')
#     #ds_mod2 = xr.open_dataset(fn2, chunks = {'time':48})
#
#     #ds_mod = compute_total_tau(ds_mod)
#
#     ds_mod = ds_mod.sortby('time')#.sel(time=slice('2012','2014'))
#
#     #ds_mod = ds_mod.isel(lev = model_lev_i)
#  
#
#     delayed_obj = ds_mod.to_netcdf(fn_comb_lev1, compute=False)
#     print('hey')
#     with ProgressBar():
#         results = delayed_obj.compute()


# %% [markdown] tags=[]
# ## If file createad already, skip to here

# %% [markdown] tags=[]
# ### Select hyytiala grid cell:

# %% [markdown]
# We use only hyytiala for org etc, but all grid cells over finland for cloud properties

# %%
if not fn_comb_lev1_final.exists():
    ds_all = xr.open_dataset(fn_comb).isel(ilev=model_lev_i)
    ds_sel = ds_all.sel(lat = lat_smr, lon= lon_smr, method='nearest')#.isel( ilev=model_lev_i)#.load()
    ds_all = ds_all.isel(
        #ilev=-1,
        # cosp_tau_modis=0,
        #                                                    cosp_tau=0,
        #                                                   cosp_dbze=0,
        #                                                    cosp_ht=0,
        #                                                    cosp_prs = 0,
        #                                                   cosp_reffice=0,
        #                                                    cosp_htmisr=0,
        #                                                    cosp_reffliq=0,
        #                                                    cosp_scol=0,
        #                                                    cosp_sr=0,
        #                                                    cosp_sza=0,
        nbnd=0
    ).squeeze()
    ds_all = broadcase_station_data(ds_all)
    ds_all = change_units_and_compute_vars(ds_all, temperature=temperature)


    delayed_obj = ds_all.to_netcdf(fn_comb_lev1_final, compute=False)
    print('hey')
    with ProgressBar():
        results = delayed_obj.compute()


# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ### Broadcast ds_sel to same grid

# %%

ds_all = xr.open_dataset(fn_comb_lev1_final)


# %% [markdown]
# ## Combine station and full grid data for model:

# %%
ds_comb_smr

# %%
if not fn_comb_lev1_finaler.exists():
    ds_all = xr.open_dataset(fn_comb_lev1_final)
    _, index = np.unique(ds_all['time'], return_index=True)
    index


    ds_all = ds_all.isel(time=index)
    ds_all =  ds_all.sel(time=slice('2012-07',None))

    ds_all

    ds_comb_smr = ds_comb_station.sel(station='SMR').drop('NCONC01')

    ds_comb_smr = ds_comb_smr.sel(time=slice(from_time1,None)).isel(lev=-1)
    ds_comb_smr

    ds_comb_smr.load()
    ds_all.load()
    drop_vars = set(ds_comb_smr.data_vars).intersection(set(ds_all.data_vars))
    print('Dropping following data from full grid:')
    print(drop_vars)
    (ds1, _)= xr.broadcast(ds_comb_smr, ds_all.drop(drop_vars))

    ds1['N100'].isel(time=slice(1,400)).mean('lon').plot()

    ds_all_merged = xr.merge([ds_all.drop(drop_vars),ds1])

    ds_all_merged.to_netcdf(fn_comb_lev1_finaler)


#dic_df = get_dic_df_mod()

# %%
ds_all = xr.open_dataset(fn_comb_lev1_finaler)

# %% [markdown]
# ds_all = ds_all.drop(['ilev','lev','station'])

# %%
ds_all

# %%
ds_all.load()

# %%
dic_ds = dict()
dic_ds[case_name] =ds_all



# %%
dic_ds['OsloAero_intBVOC_f09_f09_mg17_fssp245']

# %%
type(df_hyy_1.index) is pd.MultiIndex

# %%
if not fn_comb_lev1_final_csv.exists():
    dic_df = get_dic_df_mod(dic_ds, select_hours_clouds=True)

    df_mod = dic_df[case_name]
    df_mod.to_csv(fn_comb_lev1_final_csv)

# %%
df_mod = pd.read_csv(fn_comb_lev1_final_csv, index_col=[0,1,2] )

# %%
df_mod['N50']

# %% [markdown]
# df_modean data:

# %% [markdown]
# ### Remove gridcells that don't have a lot of cloud?

# %%
df_mod = df_mod#[df_mod['CLDFREE']<.5]#.index.get_level_values(1)

# %% [markdown]
# ### Remove grid vells with no cloud top liquid

# %%
mask_liq_cloudtop = (df_mod['FCTL']>0.1) & (df_mod['FCTL']/(df_mod['FCTL']+df_mod['FCTI'])>.8)

df_mod.loc[:,'mask_liq_cloudtop'] = mask_liq_cloudtop

# %%
one_gc = (df_mod.index.get_level_values(1)==61.57894736842104) & (df_mod.index.get_level_values(2) ==30.0)

# %%
df_mod.index.get_level_values(1)

# %%
_ma = mask_liq_cloudtop[one_gc]

# %%
df_mod[one_gc][_ma].reset_index().set_index('time')['FCTL'].plot()#ylim=[-.0,.01])


# %%
_df = df_mod.reset_index()

# %%
df_mod = df_mod[df_mod['mask_liq_cloudtop']]

# %% [markdown] tags=[]
# ## Cloud water path above 50

# %%
mask_cl_waterpath = df_mod['TGCLDCWP_incld']>50

# %%
df_mod = df_mod[df_mod['mask_liq_cloudtop']& mask_cl_waterpath]

# %%
len(df_mod)

# %% [markdown] tags=[]
# ## Group by cloud water path

# %%
#_df = ((df_mod['TOT_ICLD_VISTAU_s']>1))
_df = df_mod[df_mod['TGCLDCWP']>10]
sns.displot(#x='TGCLDLWP',
            x='TGCLDCWP',
            data=_df,
            #hue='OA_category',
           #kind='swarm'
           )
#plt.ylim([0,250])


# %%
df_mod['CWP_qcut']=pd.qcut(df_mod['TGCLDCWP'],6)# bins=bins, labels=labels)§

df_mod['CWP_qcutl'] = df_mod['CWP_qcut'].apply(lambda x:x.mid)



# %%
bins = pd.IntervalIndex.from_breaks([   50,  80,  110, 140, 170, 200,230, 500])


df_mod['CWP_cut']=pd.cut(df_mod['TGCLDCWP_incld'], bins=bins)#, labels=labels)

df_mod['CWP_cutl'] = df_mod['CWP_cut'].apply(lambda x:x.mid)

# %% [markdown]
# ## Category of OA concentration

# %%
df_mod

# %%
for v in take_vars:
    if v not in df_mod.columns:
        continue 
    df_mod[f'{v}_low'] = df_mod[v]<df_mod[v].quantile(.34)
    df_mod[f'{v}_high']= df_mod[v]>df_mod[v].quantile(.66)
    df_mod[f'{v}_category'] = pd.NA#df_hyy_1.assign(OA_category= pd.NA)
    
    df_mod.loc[df_mod[f'{v}_high'], f'{v}_category'] = f'{v} high'
    df_mod.loc[df_mod[f'{v}_low'], f'{v}_category'] = f'{v} low'


# %%
df_mod['N100'][df_mod['N100_high']].plot.hist()
df_mod['N100'][df_mod['N100_low']].plot.hist()

# %%
df_mod['OA_low'] = df_mod['OA']<df_mod['OA'].quantile(.34)
df_mod['OA_high']= df_mod['OA']>df_mod['OA'].quantile(.66)

# %%
#df_mod['OA_low'].loc[:,:] = df_mod['OA']<df_mod['OA'].quantile(.34)
mid_range = ( df_mod['OA'].quantile(.34)<df_mod['OA']) & (df_mod['OA']<df_mod['OA'].quantile(.66))
df_mod['OA_mid_range'] = mid_range


# %%

df_mod=df_mod.assign(OA_category= pd.NA)
df_mod.loc[df_mod['OA_high'], 'OA_category'] = 'OA high'
df_mod.loc[df_mod['OA_low'], 'OA_category'] = 'OA low'



# %% [markdown]
# ## Distribution plots:

# %%
palette = 'Set2'

# %%
import numpy as np

# %%
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# %%
#cmap = cm.get_cmap(name=palette, )
cmap_list = ['#441FE0','#BBE01F'][::-1]#cmap(a) for a in np.linspace(0,1,8)]

# %%
palette_OA = cmap_list[0:2]


# %%
fig, axs = plt.subplots(2,1, sharex=True, figsize =[6,6])
v_x = 'N100'
x_cut = 10000
v_hue = 'N100_category'
category = 'N100_category'
hue_order=['N100 low', 'N100 high'][::-1]

_palette = palette_OA#cmap_list[0:2]


_df = df_hyy_1
_df = _df[_df[v_x]<x_cut]

ax = axs[0]
sns.histplot(#x='TGCLDLWP',
            x=v_x,
            data=_df,
            hue=v_hue,
    hue_order=hue_order,
    palette=_palette,
    legend=False,
    edgecolor='w',

    ax = ax
           )
#plt.ylim([0,250])
print(len(_df))
ax.set_title('Observations')


ax = axs[1]

_df = (df_mod[(df_mod[category].notna())])
_df = _df[_df[v_x]<x_cut]
sns.histplot(#x='TGCLDLWP',
            x=v_x,
            data=_df,
            hue=v_hue,
    hue_order=hue_order,
    ax = ax,
    legend=False,

    palette = _palette,
    edgecolor='w',

           #kind='swarm'
           )
ax.set_title('OsloAero')

custom_lines = [Line2D([0], [0], color=cmap_list[0], lw=4),
                Line2D([0], [0], color=cmap_list[1], lw=4),
               # Line2D([0], [0], color=cmap(1.), lw=4)

               ]

leg_els = [

    Patch(edgecolor='w',alpha = .5, facecolor=_palette[1], label=hue_order[0]),
    Patch(edgecolor='w', alpha = .5,facecolor=_palette[0], label=hue_order[1]),

]

ax.legend(handles = leg_els, frameon=False)
ax.set_xlabel('N100')
#plt.ylim([0,250])
print(len(_df))
sns.despine(fig)

fn = make_fn(case_name, v_x,'obs',comment='distribution')

fig.savefig(fn, dpi=150)



# %%
_df = df_hyy_1
_df = _df[_df['CWP (g m^-2)']<600]
sns.displot(#x='TGCLDLWP',
            x='CWP (g m^-2)',
            data=_df,
            hue='OA_category',
           #kind='swarm'

           )
#plt.ylim([0,250])
print(len(df_hyy_1[df_hyy_1['OA_category'].notna()]))

# %%
_df = (df_mod[(df_mod['OA_category'].notna()) & (df_mod['TOT_ICLD_VISTAU_s']>1)])
_df = _df[_df['TGCLDCWP_incld']<600]
sns.displot(#x='TGCLDLWP',
            x='TGCLDCWP_incld',
            data=_df,
            hue='OA_category',
           #kind='swarm'
           )
#plt.ylim([0,250])
print(len(df_mod[df_mod['OA_category'].notna()]))

# %%
_df = df_mod#[(df_mod['OA_category'].notna()) & (df_mod['TOT_ICLD_VISTAU_s']>1)])
_df = _df#[_df['TGCLDCWP']<700]
sns.displot(#x='TGCLDLWP',
            x='FCTL',
            data=_df,
            #hue='OA_category',
    palette=palette_OA,
           #kind='swarm'
           )
#plt.ylim([0,250])
print(len(df_mod[df_mod['OA_category'].notna()]))

# %% [markdown]
# ### Take only points where 30% of cloud top is liquid

# %%
_df = df_mod
_df = _df[_df['FCTL']>.3]
sns.displot(
    x='TGCLDCWP_incld',
    data=_df,
    hue='OA_category',
    palette=palette_OA,
)
print(len(df_mod[df_mod['OA_category'].notna()]))

# %%
sns.displot(
    x='ACTNL_incld',
    data=df_mod[~df_mod['OA_mid_range']].reset_index(),
    hue='OA_category',
    palette=palette_OA,
)
print(len(df_mod[~df_mod['OA_mid_range']]))

# %%
f, axs = plt.subplots(1,2, figsize=[10,5])
sns.scatterplot(
    x='N50',
    y='N200',
    data=df_mod,#~df_mod['OA_mid_range']].reset_index(),
    hue='OA',
    ax = axs[0]
    #palette=palette_OA,
)
print(len(df_mod[~df_mod['OA_mid_range']]))

sns.scatterplot(
    x='N50',
    y='N200',
    data=df_hyy_1,
    hue='OA (microgram m^-3)',
    ax = axs[1],
    #palette=palette_OA,
)
print(len(df_mod[~df_mod['OA_mid_range']]))

# %%
f, axs = plt.subplots(1,2, figsize=[10,5])
sns.scatterplot(
    x='N100',
    y='N200',
    data=df_mod,#~df_mod['OA_mid_range']].reset_index(),
    hue='OA',
    ax = axs[0]
    #palette=palette_OA,
)
axs[0].set_title('MODEL')
print(len(df_mod[~df_mod['OA_mid_range']]))

sns.scatterplot(
    x='N100',
    y='N200',
    data=df_hyy_1,
    hue='OA (microgram m^-3)',
    ax = axs[1],
    #palette=palette_OA,
)
axs[1].set_title('OBSERVATIONS')
print(len(df_mod[~df_mod['OA_mid_range']]))

# %%
f, axs = plt.subplots(1,2, figsize=[10,5])
sns.scatterplot(
    x='N100',
    y='N200',
    data=df_mod,#~df_mod['OA_mid_range']].reset_index(),
    hue='OA',
    ax = axs[0]
    #palette=palette_OA,
)
axs[0].set_title('MODEL')
print(len(df_mod[~df_mod['OA_mid_range']]))

sns.scatterplot(
    x='N100',
    y='N200',
    data=df_hyy_1,
    hue='OA (microgram m^-3)',
    ax = axs[1],
    #palette=palette_OA,
)
axs[1].set_title('OBSERVATIONS')
print(len(df_mod[~df_mod['OA_mid_range']]))

# %% [markdown]
# ## Cloud optical thickness

# %% [markdown]
# ### Incloud

# %%
hue_order = ['OA low','OA high']
palette_OA_2 = palette_OA[::-1]

# %% [markdown]
# # N100

# %%
x_mod = 'CWP_cutl'
x_obs = 'CWP_cutl'
y_mod = 'TOT_ICLD_VISTAU_s'

y_obs = 'COT'
v_hue = 'N100_category'
category = 'N100_category'
hue_order=['N100 low', 'N100 high']#[::-1]

ylim = [0,52]
figsize = [12,10]
figsize = [11,7]
_palette = palette_OA_2

#fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex=True)
fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1[df_hyy_1[category].notna()]#df_mod[df_mod['OA_category'].notna()].reset_index()
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]


ax = axs[0,1]
sns.swarmplot(
    x=x_obs,
    y=y_obs,
    data=_df_obs_lim,
    hue_order=hue_order,
    hue=category,
    palette=_palette,
    size = markersize,
    ax = ax,
)


ax = axs[1,1]
sns.boxenplot(
    x=x_obs,
    y=y_obs,
    data= _df_obs,
    hue_order=hue_order,#['OA low','OA high'],
    hue=category,

    #kind='boxen',
    ax = ax,
    palette=_palette,
           )


## PLOT MODEL



_df_mod = df_mod[df_mod[category].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]

markersize=3
ax = axs[0,0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,

    hue=category,

    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1,0]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=hue_order,

    ax = ax,
    hue=category,

    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label=hue_order[0]),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label=hue_order[1]),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label=hue_order[0]
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label=hue_order[1]
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)


for ax in axs[:,1]:
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)
for ax in axs[:,0]:
    ax.set_ylabel('Cloud optical depth []')
for ax in axs[1,:]:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)


axs[0,0].set_title('OsloAero')
axs[0,1].set_title('Observations')

sns.despine(fig)

fn = make_fn(case_name+f'_{category}_', x_mod,y_mod, comment='binned_by_x')

fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%
x_mod = 'CWP_cutl'
x_obs = 'CWP_cutl'
y_mod = 'TOT_CLD_VISTAU_s'
y_obs = 'COT'
v_hue = 'N100_category'
category = 'N100_category'
hue_order=['N100 low', 'N100 high']#[::-1]

ylim = [0,52]
figsize = [12,10]
figsize = [11,7]
_palette = palette_OA_2

#fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex=True)
fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1[df_hyy_1[category].notna()]#df_mod[df_mod['OA_category'].notna()].reset_index()
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]


ax = axs[0,1]
sns.swarmplot(
    x=x_obs,
    y=y_obs,
    data=_df_obs_lim,
    hue_order=hue_order,
    hue=category,
    palette=_palette,
    size = markersize,
    ax = ax,
)


ax = axs[1,1]
sns.boxenplot(
    x=x_obs,
    y=y_obs,
    data= _df_obs,
    hue_order=hue_order,#['OA low','OA high'],
    hue=category,

    #kind='boxen',
    ax = ax,
    palette=_palette,
           )


## PLOT MODEL



_df_mod = df_mod[df_mod[category].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]

markersize=3
ax = axs[0,0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,

    hue=category,

    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1,0]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=hue_order,

    ax = ax,
    hue=category,

    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label=hue_order[0]),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label=hue_order[1]),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label=hue_order[0]
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label=hue_order[1]
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)


for ax in axs[:,1]:
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)
for ax in axs[:,0]:
    ax.set_ylabel('Cloud optical depth []')
for ax in axs[1,:]:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)


axs[0,0].set_title('OsloAero')
axs[0,1].set_title('Observations')

sns.despine(fig)
fn = make_fn(case_name+f'_{category}_', x_mod,y_mod, comment='binned_by_x')
fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%
x_mod = 'CWP_cutl'
x_obs = 'CWP_cutl'
y_mod = 'ACTREL_incld'
y_obs = 'CER (micrometer)'

ylabel = 'Cloud effective radius [$\mu m$]'
ylim = [3,25]
figsize = [11,7]

v_hue = 'N100_category'





category = 'N100_category'
hue_order=['N100 low', 'N100 high']#[::-1]

_palette = palette_OA_2

#fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex=True)
fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1[df_hyy_1[category].notna()]#df_mod[df_mod['OA_category'].notna()].reset_index()
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]


ax = axs[0,1]
sns.swarmplot(
    x=x_obs,
    y=y_obs,
    data=_df_obs_lim,
    hue_order=hue_order,
    hue=category,
    palette=_palette,
    size = markersize,
    ax = ax,
)


ax = axs[1,1]
sns.boxenplot(
    x=x_obs,
    y=y_obs,
    data= _df_obs,
    hue_order=hue_order,#['OA low','OA high'],
    hue=category,

    #kind='boxen',
    ax = ax,
    palette=_palette,
           )


## PLOT MODEL



_df_mod = df_mod[df_mod[category].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]

markersize=3
ax = axs[0,0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,

    hue=category,

    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1,0]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=hue_order,

    ax = ax,
    hue=category,

    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label=hue_order[0]),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label=hue_order[1]),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label=hue_order[0]
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label=hue_order[1]
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)


for ax in axs[:,1]:
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)
for ax in axs[:,0]:
    ax.set_ylabel(ylabel)
for ax in axs[1,:]:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)


axs[0,0].set_title('OsloAero')
axs[0,1].set_title('Observations')

sns.despine(fig)
fn = make_fn(case_name+f'_{category}_', x_mod,y_mod, comment='binned_by_x')
fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%
x_mod = 'CWP_cutl'
y_mod = 'ACTNL_incld'
ylim = [0,150]
figsize = [9,7]
_palette = palette_OA_2


v_hue = 'N100_category'





category = 'N100_category'
hue_order=['N100 low', 'N100 high']#[::-1]

_palette = palette_OA_2



fig, axs = plt.subplots(2,1,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]

_df_mod = df_mod[df_mod[category].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]



## PLOT MODEL

markersize=3
ax = axs[0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,
    hue=category,
    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=hue_order,
    hue=category,
    ax = ax,
    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label=hue_order[0]),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label=hue_order[1]),
          ]
axs[1].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label=hue_order[0]
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label=hue_order[1]
           ),
          ]
axs[0].legend(handles = leg_els, frameon=False)


#for ax in axs[:,1]:
#    ax.legend([],[], frameon=False)
#    ax.set_ylabel(None)
for ax in axs:
    ax.set_ylabel(r'Cloud droplet number conc. [cm$^{-3}$]')
for ax in axs:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[:-1]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)

axs[0].set_title('OsloAero')
axs[1].set_title('Observations')


sns.despine(fig)

fn = make_fn(case_name+f'_{category}_', x_mod,y_mod, comment='binned_by_x')

fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %% [markdown]
# # N50

# %%

# %%
x_mod = 'CWP_cutl'
x_obs = 'CWP_cutl'
y_mod = 'TOT_ICLD_VISTAU_s'

y_obs = 'COT'
v_hue = 'N50_category'
category = 'N50_category'
hue_order=['N50 low', 'N50 high']#[::-1]

ylim = [0,52]
figsize = [12,10]
figsize = [11,7]
_palette = palette_OA_2

#fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex=True)
fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1[df_hyy_1[category].notna()]#df_mod[df_mod['OA_category'].notna()].reset_index()
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]


ax = axs[0,1]
sns.swarmplot(
    x=x_obs,
    y=y_obs,
    data=_df_obs_lim,
    hue_order=hue_order,
    hue=category,
    palette=_palette,
    size = markersize,
    ax = ax,
)


ax = axs[1,1]
sns.boxenplot(
    x=x_obs,
    y=y_obs,
    data= _df_obs,
    hue_order=hue_order,#['OA low','OA high'],
    hue=category,

    #kind='boxen',
    ax = ax,
    palette=_palette,
           )


## PLOT MODEL



_df_mod = df_mod[df_mod[category].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]

markersize=3
ax = axs[0,0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,

    hue=category,

    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1,0]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=hue_order,

    ax = ax,
    hue=category,

    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label=hue_order[0]),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label=hue_order[1]),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label=hue_order[0]
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label=hue_order[1]
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)


for ax in axs[:,1]:
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)
for ax in axs[:,0]:
    ax.set_ylabel('Cloud optical depth []')
for ax in axs[1,:]:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)


axs[0,0].set_title('OsloAero')
axs[0,1].set_title('Observations')

sns.despine(fig)

fn = make_fn(case_name+f'_{category}_', x_mod,y_mod, comment='binned_by_x')

fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%
x_mod = 'CWP_cutl'
x_obs = 'CWP_cutl'
y_mod = 'TOT_CLD_VISTAU_s'
y_obs = 'COT'
v_hue = 'N50_category'
category = 'N50_category'
hue_order=['N50 low', 'N50 high']#[::-1]

ylim = [0,52]
figsize = [12,10]
figsize = [11,7]
_palette = palette_OA_2

#fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex=True)
fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1[df_hyy_1[category].notna()]#df_mod[df_mod['OA_category'].notna()].reset_index()
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]


ax = axs[0,1]
sns.swarmplot(
    x=x_obs,
    y=y_obs,
    data=_df_obs_lim,
    hue_order=hue_order,
    hue=category,
    palette=_palette,
    size = markersize,
    ax = ax,
)


ax = axs[1,1]
sns.boxenplot(
    x=x_obs,
    y=y_obs,
    data= _df_obs,
    hue_order=hue_order,#['OA low','OA high'],
    hue=category,

    #kind='boxen',
    ax = ax,
    palette=_palette,
           )


## PLOT MODEL



_df_mod = df_mod[df_mod[category].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]

markersize=3
ax = axs[0,0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,

    hue=category,

    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1,0]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=hue_order,

    ax = ax,
    hue=category,

    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label=hue_order[0]),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label=hue_order[1]),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label=hue_order[0]
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label=hue_order[1]
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)


for ax in axs[:,1]:
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)
for ax in axs[:,0]:
    ax.set_ylabel('Cloud optical depth []')
for ax in axs[1,:]:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)


axs[0,0].set_title('OsloAero')
axs[0,1].set_title('Observations')

sns.despine(fig)

fn = make_fn(case_name+f'_{category}_', x_mod,y_mod, comment='binned_by_x')

fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%
x_mod = 'CWP_cutl'
x_obs = 'CWP_cutl'
y_mod = 'ACTREL_incld'
y_obs = 'CER (micrometer)'

ylabel = 'Cloud effective radius [$\mu m$]'
ylim = [3,25]
figsize = [11,7]

v_hue = 'N50_category'





category = 'N50_category'
hue_order=['N50 low', 'N50 high']#[::-1]

_palette = palette_OA_2

#fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex=True)
fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1[df_hyy_1[category].notna()]#df_mod[df_mod['OA_category'].notna()].reset_index()
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]


ax = axs[0,1]
sns.swarmplot(
    x=x_obs,
    y=y_obs,
    data=_df_obs_lim,
    hue_order=hue_order,
    hue=category,
    palette=_palette,
    size = markersize,
    ax = ax,
)


ax = axs[1,1]
sns.boxenplot(
    x=x_obs,
    y=y_obs,
    data= _df_obs,
    hue_order=hue_order,#['OA low','OA high'],
    hue=category,

    #kind='boxen',
    ax = ax,
    palette=_palette,
           )


## PLOT MODEL



_df_mod = df_mod[df_mod[category].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]

markersize=3
ax = axs[0,0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,

    hue=category,

    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1,0]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=hue_order,

    ax = ax,
    hue=category,

    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label=hue_order[0]),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label=hue_order[1]),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label=hue_order[0]
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label=hue_order[1]
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)


for ax in axs[:,1]:
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)
for ax in axs[:,0]:
    ax.set_ylabel(ylabel)
for ax in axs[1,:]:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)


axs[0,0].set_title('OsloAero')
axs[0,1].set_title('Observations')

sns.despine(fig)

fn = make_fn(case_name+f'_{category}_', x_mod,y_mod, comment='binned_by_x')

fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%
x_mod = 'CWP_cutl'
y_mod = 'ACTNL_incld'
ylim = [0,150]
figsize = [9,7]
_palette = palette_OA_2


v_hue = 'N50_category'





category = 'N50_category'
hue_order=['N50 low', 'N50 high']#[::-1]

_palette = palette_OA_2



fig, axs = plt.subplots(2,1,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]

_df_mod = df_mod[df_mod[category].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]



## PLOT MODEL

markersize=3
ax = axs[0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,
    hue=category,
    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=hue_order,
    hue=category,
    ax = ax,
    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label='OA low'),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label='OA high'),
          ]
axs[1].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label='OA low'
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label='OA high'
           ),
          ]
axs[0].legend(handles = leg_els, frameon=False)


#for ax in axs[:,1]:
#    ax.legend([],[], frameon=False)
#    ax.set_ylabel(None)
for ax in axs:
    ax.set_ylabel(r'Cloud droplet number conc. [cm$^{-3}$]')
for ax in axs:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[:-1]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)

axs[0].set_title('OsloAero')
axs[1].set_title('Observations')


sns.despine(fig)

fn = make_fn(case_name+f'_{category}_', x_mod,y_mod, comment='binned_by_x')

fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %% [markdown]
# ## N150

# %%
x_mod = 'CWP_cutl'
x_obs = 'CWP_cutl'
y_mod = 'TOT_ICLD_VISTAU_s'

y_obs = 'COT'
v_hue = 'N150_category'
category = 'N150_category'
hue_order=['N150 low', 'N150 high']#[::-1]

ylim = [0,52]
figsize = [12,10]
figsize = [11,7]
_palette = palette_OA_2

#fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex=True)
fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1[df_hyy_1[category].notna()]#df_mod[df_mod['OA_category'].notna()].reset_index()
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]


ax = axs[0,1]
sns.swarmplot(
    x=x_obs,
    y=y_obs,
    data=_df_obs_lim,
    hue_order=hue_order,
    hue=category,
    palette=_palette,
    size = markersize,
    ax = ax,
)


ax = axs[1,1]
sns.boxenplot(
    x=x_obs,
    y=y_obs,
    data= _df_obs,
    hue_order=hue_order,#['OA low','OA high'],
    hue=category,

    #kind='boxen',
    ax = ax,
    palette=_palette,
           )


## PLOT MODEL



_df_mod = df_mod[df_mod[category].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]

markersize=3
ax = axs[0,0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,

    hue=category,

    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1,0]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=hue_order,

    ax = ax,
    hue=category,

    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label=hue_order[0]),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label=hue_order[1]),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label=hue_order[0]
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label=hue_order[1]
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)


for ax in axs[:,1]:
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)
for ax in axs[:,0]:
    ax.set_ylabel('Cloud optical depth []')
for ax in axs[1,:]:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)


axs[0,0].set_title('OsloAero')
axs[0,1].set_title('Observations')

sns.despine(fig)

fn = make_fn(case_name, x_mod,y_mod, comment='binned_by_x')
fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%
x_mod = 'CWP_cutl'
x_obs = 'CWP_cutl'
y_mod = 'TOT_CLD_VISTAU_s'
y_obs = 'COT'
v_hue = 'N150_category'
category = 'N150_category'
hue_order=['N150 low', 'N150 high']#[::-1]

ylim = [0,52]
figsize = [12,10]
figsize = [11,7]
_palette = palette_OA_2

#fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex=True)
fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1[df_hyy_1[category].notna()]#df_mod[df_mod['OA_category'].notna()].reset_index()
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]


ax = axs[0,1]
sns.swarmplot(
    x=x_obs,
    y=y_obs,
    data=_df_obs_lim,
    hue_order=hue_order,
    hue=category,
    palette=_palette,
    size = markersize,
    ax = ax,
)


ax = axs[1,1]
sns.boxenplot(
    x=x_obs,
    y=y_obs,
    data= _df_obs,
    hue_order=hue_order,#['OA low','OA high'],
    hue=category,

    #kind='boxen',
    ax = ax,
    palette=_palette,
           )


## PLOT MODEL



_df_mod = df_mod[df_mod[category].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]

markersize=3
ax = axs[0,0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,

    hue=category,

    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1,0]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=hue_order,

    ax = ax,
    hue=category,

    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label=hue_order[0]),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label=hue_order[1]),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label=hue_order[0]
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label=hue_order[1]
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)


for ax in axs[:,1]:
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)
for ax in axs[:,0]:
    ax.set_ylabel('Cloud optical depth []')
for ax in axs[1,:]:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)


axs[0,0].set_title('OsloAero')
axs[0,1].set_title('Observations')

sns.despine(fig)

fn = make_fn(case_name, x_mod,y_mod, comment='binned_by_x')
fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%
x_mod = 'CWP_cutl'
x_obs = 'CWP_cutl'
y_mod = 'ACTREL_incld'
y_obs = 'CER (micrometer)'

ylabel = 'Cloud effective radius [$\mu m$]'
ylim = [3,25]
figsize = [11,7]

v_hue = 'N150_category'





category = 'N150_category'
hue_order=['N150 low', 'N150 high']#[::-1]

_palette = palette_OA_2

#fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex=True)
fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1[df_hyy_1[category].notna()]#df_mod[df_mod['OA_category'].notna()].reset_index()
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]


ax = axs[0,1]
sns.swarmplot(
    x=x_obs,
    y=y_obs,
    data=_df_obs_lim,
    hue_order=hue_order,
    hue=category,
    palette=_palette,
    size = markersize,
    ax = ax,
)


ax = axs[1,1]
sns.boxenplot(
    x=x_obs,
    y=y_obs,
    data= _df_obs,
    hue_order=hue_order,#['OA low','OA high'],
    hue=category,

    #kind='boxen',
    ax = ax,
    palette=_palette,
           )


## PLOT MODEL



_df_mod = df_mod[df_mod[category].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]

markersize=3
ax = axs[0,0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,

    hue=category,

    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1,0]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=hue_order,

    ax = ax,
    hue=category,

    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label=hue_order[0]),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label=hue_order[1]),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label=hue_order[0]
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label=hue_order[1]
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)


for ax in axs[:,1]:
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)
for ax in axs[:,0]:
    ax.set_ylabel(ylabel)
for ax in axs[1,:]:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)


axs[0,0].set_title('OsloAero')
axs[0,1].set_title('Observations')

sns.despine(fig)

fn = make_fn(case_name+f'_{category}_', x_mod,y_mod, comment='binned_by_x')
fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%
x_mod = 'CWP_cutl'
y_mod = 'ACTNL_incld'
ylim = [0,150]
figsize = [9,7]
_palette = palette_OA_2


v_hue = 'N150_category'





category = 'N150_category'
hue_order=['N150 low', 'N150 high']#[::-1]

_palette = palette_OA_2



fig, axs = plt.subplots(2,1,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]

_df_mod = df_mod[df_mod[category].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]



## PLOT MODEL

markersize=3
ax = axs[0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,
    hue=category,
    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=hue_order,
    hue=category,
    ax = ax,
    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label=hue_order[0]),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label=hue_order[1]),
          ]
axs[1].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label=hue_order[0]
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label=hue_order[1]
           ),
          ]
axs[0].legend(handles = leg_els, frameon=False)


#for ax in axs[:,1]:
#    ax.legend([],[], frameon=False)
#    ax.set_ylabel(None)
for ax in axs:
    ax.set_ylabel(r'Cloud droplet number conc. [cm$^{-3}$]')
for ax in axs:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[:-1]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)

axs[0].set_title('OsloAero')
axs[1].set_title('Observations')


sns.despine(fig)
fn = make_fn(case_name+f'_{category}_', x_mod,y_mod, comment='binned_by_x')

fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %% [markdown] toc-hr-collapsed=true
# ## N200

# %%
x_mod = 'CWP_cutl'
x_obs = 'CWP_cutl'
y_mod = 'TOT_ICLD_VISTAU_s'

y_obs = 'COT'
v_hue = 'N200_category'
category = 'N200_category'
hue_order=['N200 low', 'N200 high']#[::-1]

ylim = [0,52]
figsize = [12,10]
figsize = [11,7]
_palette = palette_OA_2

#fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex=True)
fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1[df_hyy_1[category].notna()]#df_mod[df_mod['OA_category'].notna()].reset_index()
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]


ax = axs[0,1]
sns.swarmplot(
    x=x_obs,
    y=y_obs,
    data=_df_obs_lim,
    hue_order=hue_order,
    hue=category,
    palette=_palette,
    size = markersize,
    ax = ax,
)


ax = axs[1,1]
sns.boxenplot(
    x=x_obs,
    y=y_obs,
    data= _df_obs,
    hue_order=hue_order,#['OA low','OA high'],
    hue=category,

    #kind='boxen',
    ax = ax,
    palette=_palette,
           )


## PLOT MODEL



_df_mod = df_mod[df_mod[category].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]

markersize=3
ax = axs[0,0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,

    hue=category,

    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1,0]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=hue_order,

    ax = ax,
    hue=category,

    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label=hue_order[0]),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label=hue_order[1]),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label=hue_order[0]
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label=hue_order[1]
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)


for ax in axs[:,1]:
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)
for ax in axs[:,0]:
    ax.set_ylabel('Cloud optical depth []')
for ax in axs[1,:]:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)


axs[0,0].set_title('OsloAero')
axs[0,1].set_title('Observations')

sns.despine(fig)
fn = make_fn(case_name+f'_{category}_', x_mod,y_mod, comment='binned_by_x')

fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%
x_mod = 'CWP_cutl'
x_obs = 'CWP_cutl'
y_mod = 'TOT_CLD_VISTAU_s'
y_obs = 'COT'
v_hue = 'N200_category'
category = 'N200_category'
hue_order=['N200 low', 'N200 high']#[::-1]

ylim = [0,52]
figsize = [12,10]
figsize = [11,7]
_palette = palette_OA_2

#fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex=True)
fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1[df_hyy_1[category].notna()]#df_mod[df_mod['OA_category'].notna()].reset_index()
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]


ax = axs[0,1]
sns.swarmplot(
    x=x_obs,
    y=y_obs,
    data=_df_obs_lim,
    hue_order=hue_order,
    hue=category,
    palette=_palette,
    size = markersize,
    ax = ax,
)


ax = axs[1,1]
sns.boxenplot(
    x=x_obs,
    y=y_obs,
    data= _df_obs,
    hue_order=hue_order,#['OA low','OA high'],
    hue=category,

    #kind='boxen',
    ax = ax,
    palette=_palette,
           )


## PLOT MODEL



_df_mod = df_mod[df_mod[category].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]

markersize=3
ax = axs[0,0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,

    hue=category,

    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1,0]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=hue_order,

    ax = ax,
    hue=category,

    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label=hue_order[0]),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label=hue_order[1]),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label=hue_order[0]
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label=hue_order[1]
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)


for ax in axs[:,1]:
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)
for ax in axs[:,0]:
    ax.set_ylabel('Cloud optical depth []')
for ax in axs[1,:]:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)


axs[0,0].set_title('OsloAero')
axs[0,1].set_title('Observations')

sns.despine(fig)
fn = make_fn(case_name+f'_{category}_', x_mod,y_mod, comment='binned_by_x')
fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%
x_mod = 'CWP_cutl'
x_obs = 'CWP_cutl'
y_mod = 'ACTREL_incld'
y_obs = 'CER (micrometer)'

ylabel = 'Cloud effective radius [$\mu m$]'
ylim = [3,25]
figsize = [11,7]

v_hue = 'N200_category'





category = 'N200_category'
hue_order=['N200 low', 'N200 high']#[::-1]

_palette = palette_OA_2

#fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex=True)
fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1[df_hyy_1[category].notna()]#df_mod[df_mod['OA_category'].notna()].reset_index()
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]


ax = axs[0,1]
sns.swarmplot(
    x=x_obs,
    y=y_obs,
    data=_df_obs_lim,
    hue_order=hue_order,
    hue=category,
    palette=_palette,
    size = markersize,
    ax = ax,
)


ax = axs[1,1]
sns.boxenplot(
    x=x_obs,
    y=y_obs,
    data= _df_obs,
    hue_order=hue_order,#['OA low','OA high'],
    hue=category,

    #kind='boxen',
    ax = ax,
    palette=_palette,
           )


## PLOT MODEL



_df_mod = df_mod[df_mod[category].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]

markersize=3
ax = axs[0,0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,

    hue=category,

    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1,0]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=hue_order,

    ax = ax,
    hue=category,

    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label=hue_order[0]),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label=hue_order[1]),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label=hue_order[0]
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label=hue_order[1]
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)


for ax in axs[:,1]:
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)
for ax in axs[:,0]:
    ax.set_ylabel(ylabel)
for ax in axs[1,:]:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)


axs[0,0].set_title('OsloAero')
axs[0,1].set_title('Observations')

sns.despine(fig)
fn = make_fn(case_name+f'_{category}_', x_mod,y_mod, comment='binned_by_x')
fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%
x_mod = 'CWP_cutl'
y_mod = 'ACTNL_incld'
ylim = [0,150]
figsize = [9,7]
_palette = palette_OA_2


v_hue = 'N200_category'
category = 'N200_category'
hue_order=['N200 low', 'N200 high']#[::-1]

_palette = palette_OA_2



fig, axs = plt.subplots(2,1,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]

_df_mod = df_mod[df_mod[category].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]



## PLOT MODEL

markersize=3
ax = axs[0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,
    hue=category,
    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=['OA low','OA high'],
    ax = ax,
    hue='OA_category',
    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label='OA low'),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label='OA high'),
          ]
axs[1].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label='OA low'
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label='OA high'
           ),
          ]
axs[0].legend(handles = leg_els, frameon=False)


#for ax in axs[:,1]:
#    ax.legend([],[], frameon=False)
#    ax.set_ylabel(None)
for ax in axs:
    ax.set_ylabel(r'Cloud droplet number conc. [cm$^{-3}$]')
for ax in axs:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[:-1]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)

axs[0].set_title('OsloAero')
axs[1].set_title('Observations')


sns.despine(fig)
fn = make_fn(case_name+f'_{category}_', x_mod,y_mod, comment='binned_by_x')
fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%

# %%

# %%

# %%
