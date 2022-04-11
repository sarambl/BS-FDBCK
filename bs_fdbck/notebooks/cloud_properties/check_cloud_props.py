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
import numpy as np

from pathlib import Path

from bs_fdbck.constants import measurements_path, path_outdata, path_extract_latlon_outdata

# %%
xr.set_options(keep_attrs=True) 

# %%
path_extract_latlon_outdata

# %%
lat_smr = 61.85
lon_smr = 24.28

# %%
fn1 = Path('/proj/bolinc/users/x_sarbl/noresm_archive/OsloAero_intBVOC_f19_f19_mg17_full/atm/OsloAero_intBVOC_f19_f19_mg17_full.h1._2012-01-01-2015-01-01_concat_subs_22.0-30.0_60.0-66.0.nc')

# %%
fn1.stem

# %%
fn1_2 = fn1.parent / f'{fn1.stem}_sort.nc'

# %%
fn2 = path_extract_latlon_outdata/ 'OsloAero_intBVOC_f19_f19_mg17_ssp245/OsloAero_intBVOC_f19_f19_mg17_ssp245.h1._2015-01-01-2019-01-01_concat_subs_22.0-30.0_60.0-66.0.nc'

# %%

# %%
fn2_2 = fn2.parent / f'{fn2.stem}_sort.nc'

# %%

# %%
case_name = 'OsloAero_intBVOC_f19_f19_mg17_full'


cases = [case_name]

# %%
from pathlib import Path

# %%
plot_path = Path('Plots')


# %%
def make_fn(case, v_x, v_y):
    _x = v_x.split('(')[0]
    _y = v_y.split('(')[0]
    f = f'scat_{case}_{_x}_{_y}.png'
    return plot_path /f


# %%
plot_path.mkdir(exist_ok=True, parents=True)

# %%
varl =['DOD500','DOD440','ACTREL','ACTNL','TGCLDLWP', #,'SOA_A1',
       'H2SO4','SOA_LV','COAGNUCL','FORMRATE','T','FCTL',
       'TOT_CLD_VISTAU','TOT_ICLD_VISTAU','TGCLDCWP',
       'TAUTLOGMODIS',
       'MEANTAU_ISCCP',
       'LWPMODIS','CLWMODIS','REFFCLWMODIS','TAUTMODIS','TAUWMODIS',
      
      'SOA_NA','SOA_A1','OM_NI','OM_AI','OM_AC','SO4_NA','SO4_A1','SO4_A2','SO4_AC','SO4_PR',
      'BC_N','BC_AX','BC_NI','BC_A','BC_AI','BC_AC','SS_A1','SS_A2','SS_A3','DST_A2','DST_A3', 
      ] 


# %%
varl_st = [      'SOA_NA','SOA_A1','OM_NI','OM_AI','OM_AC','SO4_NA','SO4_A1','SO4_A2','SO4_AC','SO4_PR',
      'BC_N','BC_AX','BC_NI','BC_A','BC_AI','BC_AC','SS_A1','SS_A2','SS_A3','DST_A2','DST_A3']

# %% [markdown]
# ## Load observations: 

# %% [markdown]
# ## Open model dataset: 
#

# %% [markdown]
# ds_mod1 = xr.open_dataset(fn1, chunks = {'time':48})#[fn1,fn2])#.sortby('time')
# ds_mod2 = xr.open_dataset(fn2, chunks = {'time':48})

# %% [markdown]
# varl1 = set(ds_mod1.data_vars)
#
# varl2 = set(ds_mod2.data_vars)
#
#
# varl =list(varl1.intersection(varl2))

# %% [markdown]
# ds_mod1 = ds_mod1[varl].sortby('time').sel(time=slice('2012','2014'))
#
# ds_mod2 = ds_mod2[varl].sortby('time').sel(time=slice('2015','2018'))

# %% [markdown]
# ds_mod1

# %%
import dask.array as da
from dask.diagnostics import ProgressBar


# %% [markdown]
# delayed_obj = ds_mod1.to_netcdf(fn1_2, compute=False)
# with ProgressBar():
#     results = delayed_obj.compute()
#     
#     

# %% [markdown]
# delayed_obj = ds_mod2.to_netcdf(fn2_2, compute=False)
# with ProgressBar():
#     results = delayed_obj.compute()

# %% [markdown]
# ds_mod = xr.open_mfdataset([fn1_2,fn2_2], combine='by_coords', concat_dim='time')

# %%
fn_comb = path_extract_latlon_outdata/ 'OsloAero_intBVOC_f19_f19_mg17_fssp245'/'OsloAero_intBVOC_f19_f19_mg17_fssp245.h1._2012-01-01-2019-01-01_concat_subs_22.0-30.0_60.0-66.0.nc'


# %%
fn_comb_lev1 = path_extract_latlon_outdata/ 'OsloAero_intBVOC_f19_f19_mg17_fssp245'/'OsloAero_intBVOC_f19_f19_mg17_fssp245.h1._2012-01-01-2019-01-01_concat_subs_22.0-30.0_60.0-66.0_lev-1.nc'
fn_comb_lev1_dailymedian = path_extract_latlon_outdata/ 'OsloAero_intBVOC_f19_f19_mg17_fssp245'/'OsloAero_intBVOC_f19_f19_mg17_fssp245.h1._2012-01-01-2019-01-01_concat_subs_22.0-30.0_60.0-66.0_lev-1_dailymedian.nc'




# %%
fn_comb.parent.mkdir(exist_ok=True,)

# %% [markdown]
#
# delayed_obj = ds_mod.to_netcdf(fn_comb, compute = False)
# with ProgressBar():
#     results = delayed_obj.compute()

# %% [markdown] tags=[]
# ds_mod = xr.concat([ds_mod1[varl].sel(time=slice('2012','2014')), ds_mod2[varl].sel(time=slice('2015','2018'))], dim='time')

# %% [markdown]
# Load data: 
#

# %% [markdown]
# ds_mod = xr.open_dataset(fn_comb) 


# %% [markdown]
# ds_mod.compute()
#
# ds_mod.load()

# %% [markdown]
# Somehow unsorted

# %% [markdown]
# ds_mod = ds_mod.sortby('time')

# %% [markdown]
# ### Select hyytiala grid cell: 

# %% [markdown]
# We use only hyytiala for org etc, but all grid cells over finland for cloud properties

# %% [markdown]
# ds_mod['TOT_ICLD_VISTAU_s']= ds_mod['TOT_ICLD_VISTAU'].sum('lev')
# ds_mod['TOT_CLD_VISTAU_s']= ds_mod['TOT_CLD_VISTAU'].sum('lev')
#

# %% [markdown]
# model_lev_i=-1
# ds_sel = ds_mod.sel(lat = lat_smr, lon= lon_smr, method='nearest').isel( lev=model_lev_i)#.load()
# ds_all = ds_mod.isel(lev=model_lev_i)#.load()
#
# #ds_sel.load()
# #ds_all.load()

# %% [markdown]
# ds_all.to_netcdf(fn_comb_lev1)

# %%
model_lev_i=-1

# %%
ds_all = xr.open_dataset(fn_comb_lev1).isel(ilev=model_lev_i, nbnd=0)
ds_sel = ds_all.sel(lat = lat_smr, lon= lon_smr, method='nearest')#.isel( ilev=model_lev_i)#.load()


# %%
ds_mod = ds_all

# %%
ds_all['ACTNL'].isel(lat=0).plot()

# %%
dic_ds=dict()
dic_ds[case_name]= ds_mod

# %% [markdown]
# ### Broadcast ds_sel to same grid 

# %% [markdown]
# Copying the same for as hyytiala for all grid cells for the station variables (st measurements) 

# %%
ds_all['ACTNL'].load().isel(time=20000).plot()

# %%
ds_all['SOA_NA'].load().isel(time=20000).plot()

# %%
ds_1, ds_2 =xr.broadcast(ds_sel, ds_all)
for v in varl_st:
    ds_all[v] = ds_1[v]


# %%
ds_sel

# %% [markdown]
# ### Set dic_ds : 

# %%
dic_ds[case_name] =ds_all

# %% [markdown]
# Constants:

# %%
R = 287.058
pressure = 1000. #hPa
kg2ug = 1e9

# %%
ds_all.load()


# %%
def get_dic_df_mod(model_lev_i=-1):
    


    # %%
    dic_df = dict()
    dic_df_sm = dict()

    for ca in dic_ds.keys():
        ds = dic_ds[ca]
        #ds['TOT_ICLD_VISTAU_s']= ds['TOT_ICLD_VISTAU'].sum('lev')
        #ds['TOT_CLD_VISTAU_s']= ds['TOT_CLD_VISTAU'].sum('lev')
        for v in ['TGCLDLWP','TGCLDIWP','TGCLDCWP']:
            if v in ds.data_vars:
                if ds[v].attrs['units'] =='kg/m2':
                    ds[v] = ds[v]*1000
                    ds[v].attrs['units'] = 'g/m2'
                
        
        ds_sel = ds.sel(lat = lat_smr, lon= lon_smr, method='nearest')#.isel( lev=model_lev_i)


# %%

        # %%
        ds_all = ds#.isel(lev=model_lev_i)
        ds_sel =ds_sel[varl_st]
       
        s_1, ds_2 =xr.broadcast(ds_sel, ds_all)
        for v in varl_st:
            ds_all[v] = ds_1[v]
        ds_sel = ds_all
        rho = pressure*100/(R*ds_sel['T'])


        # %%
        ds_sel['rho'] = rho
        ds_sel['ACTNL_incld'] = ds_sel['ACTNL']/ds_sel['FCTL']
        ds_sel['ACTREL_incld'] = ds_sel['ACTREL']/ds_sel['FCTL']


        # %%
        ds_sel_median = ds_sel.resample({'time':'D'}).median()

        # %%
        ds_sel_median['ACTNL_incld'].plot()

        # %%
        #df = ds_sel_median.to_dataframe()


        # %%
        ls_so4 = [c for c in ds_sel_median.data_vars if 'SO4_' in c]#['SO4_NA']

        # %%
        ls_so4

        # %%
        for s in ['SOA_NA','SOA_A1','OM_AC','OM_AI','OM_NI']+ls_so4:
            un = '$\micro$g/m3'
            if ds_sel_median[s].attrs['units']!=un:
                ds_sel_median[s] = ds_sel_median[s]*ds_sel_median['rho']*kg2ug
                ds_sel_median[s].attrs['units']=un
        #ds_sel_med= ds_sel_median.resample(time='D').median()


        # %%
        df = ds_sel_median.to_dataframe()
        df = df.drop([co for co in df.columns if (('lat_' in co)|('lon_' in co))], 
                     axis=1)

        df['SOA'] = df['SOA_NA'] + df['SOA_A1']

        df['OA']  = df['SOA_NA'] + df['SOA_A1'] + df['OM_AC'] + df['OM_AI'] + df['OM_NI']
        df['POA'] = df['OM_AC']  + df['OM_AI' ] + df['OM_NI']
    
        df['SO4']=0
        for s in ls_so4:
            print(s)
            
            print(df[s].mean())
            df['SO4'] = df['SO4'] + df[s]
        
        #df['ACTNL_incld'] = df['ACTNL']/df['FCTL']
        #df['ACTREL_incld'] = df['ACTREL']/df['FCTL']
        
    
        df_daily = df#.resample('D').median()

        months = (df.index.get_level_values(0).month==7 )|(df.index.get_level_values(0).month==8  )

        df_s = df_daily[months]
        df_s.loc[:,'year'] = df_s.index.get_level_values(0).year.values

        df_s.loc[:,'T_C'] = df_s['T'].values-273.15
        #df_s.index = df_s.index.rename('date')
        df_merge = df_s#pd.merge(df_s, df_hyy_1, right_on='date', left_on='date')
        
        df_merge['year'] = df_merge.index.get_level_values(0).year

        
        dic_df[ca] = df_merge
        print(ca)
    
        months = (df.index.get_level_values(0).month==7 )|(df.index.get_level_values(0).month==8  )

        df_s = df[months]
        ds_month_mask = ds_sel.where((ds_sel['time.month']==7) | (ds_sel['time.month']==8))
        ds_sel_med_y= ds_month_mask.resample(time='Y').median()
        df_ym =ds_sel_med_y.to_dataframe()
        #df_ym = df_s.resample('Y').median()
        #df_ym.loc[:,'year'] = df_ym.index.year.values

        df_ym.loc[:,'T_C'] = df_ym['T'].values-273.15
        
        dic_df_sm[ca] = df_merge
        print(ca)


    # %%
    return dic_df_sm, dic_df


dic_df_sm, dic_df = get_dic_df_mod(model_lev_i=-1)

# %% tags=[]
dic_df_sm[case_name].columns

# %% tags=[]
df_mod = dic_df_sm[case_name]


# %%
df_mod.index.get_level_values(1)

# %%
df_mod#.index.get_level_values(1)

# %%
mask_liq_cloudtop = df_mod['FCTL']>0.0001

# %%
df_mod['mask_liq_cloudtop'] = mask_liq_cloudtop

# %% tags=[]
sel_latlon = (df_mod.index.get_level_values(2)==27.5)&(df_mod.index.get_level_values(1)==61.57894736842104)

df_mod[sel_latlon].reset_index().set_index('time')['TGCLDLWP'].plot()


# %%
df_mod[mask_liq_cloudtop].reset_index().set_index('time')['FCTL'].plot()#ylim=[-.0,.01])


# %%
df_mod[mask_liq_cloudtop].reset_index().set_index('time')['ACTNL_incld'].plot()#ylim=[-.0,.01])


# %% [markdown]
# #### Mask values that don't have cloud top liquid

# %%
df_mod = df_mod[df_mod['mask_liq_cloudtop']]

# %%
df_mod['CWP_qcut']=pd.qcut(df_mod['TGCLDLWP'],6)# bins=bins, labels=labels)§

df_mod['CWP_qcutl'] = df_mod['CWP_qcut'].apply(lambda x:x.mid)



# %%
bins = pd.IntervalIndex.from_breaks([ 10,  30,  50,  70, 90, 110, 130,500])


df_mod['CWP_cut']=pd.cut(df_mod['TGCLDLWP'], bins=bins)#, labels=labels)

df_mod['CWP_cutl'] = df_mod['CWP_cut'].apply(lambda x:x.mid)

# %% [markdown]
# ## Category of OA concentration

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



# %%
df_mod[(df_mod.index.get_level_values('lat') >65)& (df_mod.index.get_level_values('lon') == 25.0)]

# %%
import seaborn as sns

# %%
df_mod['ACTREL_incld'].plot.hist()

# %%
df_mod = df_mod[((df_mod['OA_category'].notna()) & (df_mod['TOT_ICLD_VISTAU_s']>2))& (df_mod['ACTREL_incld']>2)]

# %%
_df = (df_mod[((df_mod['OA_category'].notna()) & (df_mod['TOT_ICLD_VISTAU_s']>2))& (df_mod['ACTREL_incld']>2)])
_df = _df[_df['TOT_ICLD_VISTAU_s']<50]
sns.displot(#x='TGCLDLWP', 
            x='TOT_ICLD_VISTAU_s',
            data=_df,
            hue='OA_category',
           #kind='swarm'
           )
#plt.ylim([0,250])
print(len(_df['OA_category']))

# %%
sns.displot(#x='TGCLDLWP', 
            x='ACTNL_incld',
            data=df_mod[~df_mod['OA_mid_range']].reset_index(),
            hue='OA_category',
           #kind='swarm'
           )
#plt.ylim([0,250])

# %%
import seaborn as sns

# %%
sns.catplot(x='CWP_cutl', 
            y='ACTNL_incld',
            #data=df_mod.reset_index(),
            #data=df_mod[~df_mod['OA_mid_range']].reset_index(),
            data=df_mod[df_mod['OA_category'].notna()].reset_index(),
            hue_order=['OA low','OA high'],

            hue='OA_category',
           kind='swarm'
           )
#plt.ylim([0,250])

# %%
sns.catplot(x='CWP_cutl', 
            y='ACTNL_incld',
            #data=df_mod.reset_index(),
            #data=df_mod[~df_mod['OA_mid_range']].reset_index(),
            data=df_mod[df_mod['OA_category'].notna()].reset_index(),
            hue_order=['OA low','OA high'],

            hue='OA_category',
           kind='boxen'
           )
#plt.ylim([0,250])

# %%
sns.catplot(x='CWP_cutl', 
            y='TOT_ICLD_VISTAU_s',
            data=df_mod[df_mod['OA_category'].notna()].reset_index(),
            hue_order=['OA low','OA high'],

            hue='OA_category',
            kind='boxen',
           )
#plt.ylim([0,250])

# %%
sns.catplot(x='CWP_cutl', 
            y='TOT_ICLD_VISTAU_s',
            #data=df_mod.reset_index(),
            data=df_mod[df_mod['OA_category'].notna()].reset_index(),
            hue_order=['OA low','OA high'],

            hue='OA_category',
           kind='swarm'
           )
#plt.ylim([0,250])

# %%
sns.catplot(x='CWP_cutl', 
            y='TOT_CLD_VISTAU_s',
            data=df_mod[df_mod['OA_category'].notna()].reset_index(),
            hue ='OA_category',
            kind='boxen',
            hue_order=['OA low','OA high'],
           )
#plt.ylim([0,250])

# %%
sns.catplot(x='CWP_cutl', 
            y='TOT_CLD_VISTAU_s',
            data=df_mod[df_mod['OA_category'].notna()].reset_index(),
            hue ='OA_category',
            kind='swarm',
            hue_order=['OA low','OA high'],
           )
#plt.ylim([0,250])

# %%
sns.catplot(x='CWP_cutl', 
            y='ACTNL_incld',
            data=df_mod[df_mod['OA_category'].notna()].reset_index(),

            hue='OA_category',
           kind='swarm'
           )

# %%
sns.catplot(x='CWP_cutl', 
            y='ACTNL_incld',
            data=df_mod[df_mod['OA_category'].notna()].reset_index(),

            hue='OA_category',
            kind='violin'
           )

# %%
sns.catplot(x='CWP_cutl', 
            y='ACTREL_incld',
            data=df_mod[df_mod['OA_category'].notna()].reset_index(),
            hue_order=['OA low','OA high'],

            hue='OA_category',
            kind='boxen',
           )
plt.ylim([0,25])

# %%
sns.catplot(x='CWP_cutl', 
            y='ACTREL_incld',
            data=df_mod[df_mod['OA_category'].notna()].reset_index(),
            hue_order=['OA low','OA high'],

            hue='OA_category',
            kind='swarm',
           )
plt.ylim([0,25])

# %%
sns.catplot(x='CWP_cutl', 
            y='ACTREL_incld',
            data=df_mod[df_mod['OA_category'].notna()].reset_index(),
            hue ='OA_category',
            kind='swarm',
            hue_order=['OA low','OA high'],
           )
#plt.ylim([0,250])
plt.ylim([0,25])

# %%

# %%

# %%

# %%

# %%
