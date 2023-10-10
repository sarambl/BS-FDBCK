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
from pathlib import Path
import matplotlib as mpl
import xarray as xr


# %%
from pathlib import Path

from bs_fdbck_clean.util.BSOA_datamanip import ds2df_inc_preprocessing
from bs_fdbck_clean.util.collocate.collocateLONLAToutput import CollocateLONLATout
from bs_fdbck_clean.util.collocate.collocate_echam_salsa import CollocateModelEcham
import useful_scit.util.log as log

from bs_fdbck_clean.util.plot.BSOA_plots import make_cool_grid, plot_scatter

log.ger.setLevel(log.log.INFO)
import time
import xarray as xr
import matplotlib.pyplot as plt

# %%
from bs_fdbck_clean.util.plot.BSOA_plots import cdic_model

# %%
from bs_fdbck_clean.constants import path_measurement_data
select_station = 'SMR'
postproc_data = path_measurement_data /'model_station'/select_station
postproc_data_obs = path_measurement_data /'SMEARII'/'processed'


# %%
path_comb_data_full_time =postproc_data_obs /'SMEAR_data_comb_hourly.csv'

# %%
plot_path = Path(f'Plots/{select_station}')


# %%
def make_fn_eval(case,_type):
    #_x = v_x.split('(')[0]
    #_y = v_y.split('(')[0]
    f = f'evalOA_echam_{case}_{_type}_{select_station}.png'
    return plot_path /f


# %%
plot_path.mkdir(exist_ok=True, parents=True)

# %% [markdown] tags=[]
# ## Read in model data. 

# %% tags=[]
models = ['ECHAM-SALSA','NorESM', 'EC-Earth', 'UKESM']
mod2cases = {'ECHAM-SALSA' : ['SALSA_BSOA_feedback'],
             'NorESM' : ['OsloAero_intBVOC_f09_f09_mg17_fssp'],
             'EC-Earth' : ['ECE3_output_Sara'],
             'UKESM' : ['AEROCOMTRAJ'],
             'Observations':['Obs'],
            }
di_mod2cases = mod2cases.copy()


# %%
from bs_fdbck_clean.preprocess.launch_monthly_station_collocation import launch_monthly_station_output
from bs_fdbck_clean.util.Nd.sizedist_class_v2.SizedistributionBins import SizedistributionStationBins
from bs_fdbck_clean.util.collocate.collocateLONLAToutput import CollocateLONLATout
from bs_fdbck_clean.data_info.variable_info import list_sized_vars_nonsec, list_sized_vars_noresm
import useful_scit.util.log as log
log.ger.setLevel(log.log.INFO)
import time

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
import numpy as np

# %%
import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge

# %%

# %% [markdown] tags=[]
# ## Constants:

# %%
R = 287.058
pressure = 1000. #hPa
kg2ug = 1e9
temperature = 273.15


# %%
import pandas as pd

# %% [markdown]
# ## Read in model data

# %%
dic_df_pre=dict()
for mod in models:
    print(mod)
    dic_df_pre[mod] = dict()
    for ca in mod2cases[mod]:
        print(mod, ca)
        fn_out = postproc_data/f'{select_station}_station_{mod}_{ca}.csv'
        print(fn_out)
        dic_df_pre[mod][ca] = pd.read_csv(fn_out, index_col=0)
        dic_df_pre[mod][ca].index = pd.to_datetime(dic_df_pre[mod][ca].index)
        #dic_df_mod_case[mod][ca].to_csv(fn_out)

# %%
for mod in models:
    for ca in mod2cases[mod]:
        _df = dic_df_pre[mod][ca]
        for v in ['OA','N50','N100','N200']:
            if f'{v}_STP' in _df.columns:
                if v in _df.columns:
                    _df = _df.rename({v:f'{v}_orig'}, axis=1)
                _df = _df.rename({f'{v}_STP':v}, axis=1)
        dic_df_pre[mod][ca] = _df

# %%
for mo in ['EC-Earth','UKESM']:
    if mo in models:
        for ca in mod2cases[mo]:
            dic_df_pre[mo][ca]['OAG'] = dic_df_pre[mo][ca]['OA']


# %% [markdown]
# ### Double check EC-Earth

# %%
mo = 'EC-Earth'
_df = dic_df_pre[mo][mod2cases[mo][0]]

# %%
_df['hour'] = _df.index.hour

# %% tags=[]
_df['hour'].unique()


# %%
_df.groupby('hour').mean()['OA'].plot(marker='.')

# %%
_df.groupby('hour').mean()['emiisop'].plot(marker='.')

# %%
_df.groupby('hour').mean()['POM'].plot()

# %%
_df.groupby('hour').mean()['POM'].plot()

# %%
_df.groupby('hour').mean()['N100'].plot()

# %% [markdown]
# ## Read in observations

# %%
df_obs = pd.read_csv(path_comb_data_full_time,index_col=0)

# %%
df_obs = df_obs.rename({'Org_STP':'OA'}, axis=1)

# %%
dic_df_pre['Observations']=dict()
dic_df_pre['Observations']['Observations'] = df_obs
dic_df_pre['Observations']['Observations'].index = pd.to_datetime(dic_df_pre['Observations']['Observations'].index)

# %%
mod2cases['Observations']= ['Observations']


# %% [markdown]
# ## Set uo dic with all OA values from models

# %%
dic_df_mod_case = dic_df_pre

# %% tags=[]
dic_mod_oa = dict()
#dic_mod_soa = dict()
dic_mod_all = dict()


for mod in models:
    dic_mod_oa[mod] = dict()
    #dic_mod_soa[mod] = dict()
    dic_mod_all[mod] = dict()
    for ca in di_mod2cases[mod]:
        dic_mod_oa[mod][ca] = dict()
        #dic_mod_soa[mod][ca] = dict()
        dic_mod_all[mod][ca] = dict()
    
    
for mo in models:
    print(mo)
    for ca in di_mod2cases[mo]:
        print(ca)
        _df = dic_df_mod_case[mo][ca]
        dic_mod_oa[mo][ca] =_df['OA']
        #dic_mod_soa[mo][ca][i] =_df['SOA']
        dic_mod_all[mo][ca] =_df
    



# %%
dic_df_pre['ECHAM-SALSA']['SALSA_BSOA_feedback']['OA'].plot()


# %%
dic_df_pre['ECHAM-SALSA']['SALSA_BSOA_feedback'].resample('h').ffill()['OA'].plot()         
dic_df_pre['ECHAM-SALSA']['SALSA_BSOA_feedback']['OA'].plot()
plt.xlim(['2014-07','2014-08'])

# %% [markdown] tags=[]
# ## Merge with observations:

# %%
dic_df_mod_case = dic_df_pre.copy()

# %%
mask_obs_OA =  dic_df_pre['Observations']['Observations']['OA'].notnull()

# %%
_df = dic_df_pre['Observations']['Observations']['OA'].rename('Obs')

df_OA_all = pd.DataFrame(_df)
df_OAG_all = pd.DataFrame(_df)


# %%
df_OA_all

# %%
for mod in dic_df_pre.keys():
    if mod=='Observations':
        continue
    print(mod)
    for ca in dic_df_pre[mod].keys():
        if len(dic_df_pre[mod].keys())==1:
            use_name = mod
        else: 
            use_name = f'{mod}: {ca}'
        df_OA_all[use_name] = dic_df_pre[mod][ca]['OA']
        df_OAG_all[use_name] = dic_df_pre[mod][ca]['OAG']


#df_OA_all = df_OA_all[df_OA_all[mod].notna()]
df_OA_all = df_OA_all[df_OA_all['Obs'].notna()]
#df_OAG_all = df_OAG_all[df_OAG_all[mod].notna()]
df_OAG_all = df_OAG_all[df_OAG_all['Obs'].notna()]


# %%
df_OA_all

# %%
orgname={'NorESM' : 'OA',
         'ECHAM-SALSA': 'OA'}

# %%
seasons2months = {'DJF':[12,1,2],
        'MAM': [3,4,5],
        'JJA':[6,7,8],
        'SON':[9,10,11],
       }

# %%
seasons2months2 = {'JFM': [ 1, 2,3], 'AMJ': [ 4, 5,6], 'JAS': [ 7, 8,9], 'OND': [ 10, 11,12]}

# %% [markdown]
# ### Calculate anomaly from daily average

# %%
df_anom_OA = df_OA_all-df_OA_all.resample('D').mean().resample('h').ffill(limit=24)
df_anom_OAG = df_OAG_all-df_OAG_all.resample('D').mean().resample('h').ffill(limit=24)

# %%
df_anom_OAG = df_anom_OAG[df_anom_OAG['Obs'].notna()]
df_anom_OA = df_anom_OA[df_anom_OA['Obs'].notna()]

# %%
df_anom_OA

# %%
linestyle_dic = {
    'Obs': 'solid',
    'NorESM':'dashdot',
    'UKESM':'dashdot',
    'ECHAM-SALSA':'-.',
    'EC-Earth':'-.',
}


# %% [markdown]
# ## Choose months: 

# %%
months = [1,2,3,4,5,6,7,8,9,10,11,12]
df_OA_all = df_OA_all[df_OA_all.index.month.isin(months)]
df_OAG_all = df_OAG_all[df_OAG_all.index.month.isin(months)]                      

df_anom_OA = df_anom_OA[df_anom_OA.index.month.isin(months)]
df_anom_OAG = df_anom_OAG[df_anom_OAG.index.month.isin(months)]

# %%
df_OA_all

# %%
import datetime

# %%
for m in models + ['Observations']:
    c = list(dic_df_pre[m].keys())[0]
    _df = dic_df_pre[m][c].copy(deep=True)
    #if m in ['NorESM','UKESM']:
    #    _df.index = _df.index -  datetime.timedelta(hours=1)
    #if m in ['ECHAM-SALSA','EC-Earth']:
    #    _df.index = _df.index -  datetime.timedelta(hours=0)

    print(m)
    if m=='Observations':
        a =_df.groupby(_df.index.hour).mean()['HYY_META.T168']
    else:
        a = _df.groupby(_df.index.hour).mean()['T_C'].dropna(axis=0)#.plot(marker='*',label=m)         
    ((a-a.mean())/(a.max()-a.min())).plot(label=m)         
    
plt.legend()
#plt.xlim(['2014-07','2014-08'])

# %% [markdown]
# ### Controle that EC-Earth has correct time zone also for TM5 by checking that diurnal cycle of isoprene emissions and temperature align

# %%
m = 'EC-Earth'

# %%
c = list(dic_df_pre[m].keys())[0]
_df = dic_df_pre[m][c]
print(m)
f, ax = plt.subplots()
_df.groupby(_df.index.hour).mean()['T_C'].dropna(axis=0).plot(marker='*',label='temperature', c='r')      
ax.legend(bbox_to_anchor=(1.1,.5,))
ax2 = ax.twinx()
_df.groupby(_df.index.hour).mean()['emiisop'].dropna(axis=0).plot(marker='*',label='emissions isoprene', ax=ax2)         
    
ax2.legend(bbox_to_anchor=(1,1.1,))
ax2.legend(bbox_to_anchor=(1.5,.8,))

ax.set_xticks(np.arange(24))
ax.grid()

# %% [markdown]
# ## Timeseries

# %%
fig, axs = plt.subplots(5,1, sharex=True, sharey=True, figsize=[5,8], dpi=100)
ax = axs[1]
#pl_obs = df_anom_OA['Obs'].groupby(df_anom_OA['obs'].index.hour).mean()
#pl_obs.plot(ax=ax,label='Observations', c='k')

for ax, mo in zip( axs, df_anom_OA.columns):
 
    _df_anom_OA = df_OA_all.copy()
    _df_anom_OA = _df_anom_OA[_df_anom_OA['Obs'].notna()] # .dropna(axis=0)
    print(mo)
    if mo=='Obs':
        c = 'k'
        ls = 'solid'
        c2 = 'w'
        
    else:
        c=cdic_model[mo]
        ls = linestyle_dic[mo]
        c2='k'
    _df = _df_anom_OA[mo]
        
    _df.plot(ax=ax,c=c,
             #linestyle=ls,
             marker='.', 
             linewidth=0,
             markersize=.7,
             alpha=.4,
             rasterized=True,
             
             label=mo)
    
    ax.set_title(mo)
    #'OsloAeroSec',)# c='k')
        #ax.fill_between(_df_gbm.index, _df_gb33,_df_gb66, color = c,alpha=.2)
    # ax.set_title("$OA'$: Average diurnal anomaly") 
#ax.set_title(f"{seas} $OA'$: Median diurnal anomaly") 

#    ax.set_ylim([-2,2])

ax.legend(frameon=False)


plt.legend(frameon=False)
ax.set_xlabel('Time of day [h]')
for ax in axs: 
    ax.set_ylabel('OA  [$\mu$gm$^{-3}$]')
plt.tight_layout()

#ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,])

#ax.set_xlim([0,23])
fn = make_fn_eval('_'.join(models), 'timeseries')

fig.savefig(fn.with_suffix('.png'))
fig.savefig(fn.with_suffix('.pdf'))
print(fn)

# %%
from bs_fdbck_clean.util.BSOA_datamanip.atto import season2month

# %%
fig, ax = plt.subplots(1,1, sharex=True, figsize=[7,3], dpi=100)
seas = 'JA'
    
_df_anom_OA = df_anom_OA[df_anom_OA.index.month.isin(season2month[seas])]
_df_anom_OA = _df_anom_OA[_df_anom_OA['Obs'].notna()]#.dropna(axis=0)

for mod in df_anom_OA.columns:
    if mod=='Obs':
        c = 'k'
    else:
        c=cdic_model[mod]
    if mod in ['UKESM']:
        # 3h resolution so need to drop nans
        _df_anom_OA = _df_anom_OA[_df_anom_OA[mod].notna()]
    ls = linestyle_dic[mod]
    _df_gb = _df_anom_OA[mod].groupby(_df_anom_OA.index.hour)
    _df_gbm = _df_gb.median()
    _df_gbstd = _df_gb.std()
    _df_gb33 = _df_gb.quantile(.84)
    _df_gb66 = _df_gb.quantile(.16)
    
    
    _df_gbm.plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
    ax.fill_between(_df_gbm.index, _df_gb33,_df_gb66, color = c,alpha=.2)
ax.set_title(f"Jul & Aug $OA'$: Median diurnal anomaly") 

ax.set_ylim([-1.5,1.5])


plt.legend(frameon=False, bbox_to_anchor=(1,1,))
ax.set_xlabel('Time of day [h]')

ax.set_ylabel('OA  [$\mu$gm$^{-3}$]')
plt.tight_layout()

ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,])

ax.set_xlim([0,23])
fn = make_fn_eval('_'.join(models), 'diurnal_mean_dev_JA')

fig.savefig(fn.with_suffix('.png'))
fig.savefig(fn.with_suffix('.pdf'))
print(fn)

# %%
fig, axs = plt.subplots(4,1, sharex=True, figsize=[7,8], dpi=100)
ax = axs[1]
#pl_obs = df_anom_OA['Obs'].groupby(df_anom_OA['obs'].index.hour).mean()
#pl_obs.plot(ax=ax,label='Observations', c='k')

for seas, ax in zip(seasons2months, axs):
    
    _df_anom_OA = df_anom_OA[df_anom_OA.index.month.isin(seasons2months[seas])]
    #_df_anom_OA = _df_anom_OA.dropna(axis=0)
    for mod in df_anom_OA.columns:
        if mod=='Obs':
            c = 'k'
        else:
            c=cdic_model[mod]
            #continue
        if mod in ['UKESM']:
            # 3h resolution so need to drop nans
            _df_anom_OA = _df_anom_OA[_df_anom_OA[mod].notna()]

        ls = linestyle_dic[mod]
        _df_gb = _df_anom_OA[mod].groupby(_df_anom_OA.index.hour)
        _df_gbm = _df_gb.median()
        _df_gbstd = _df_gb.std()
        _df_gb33 = _df_gb.quantile(.84)
        _df_gb66 = _df_gb.quantile(.16)
        
        
        _df_gbm.plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
        ax.fill_between(_df_gbm.index, _df_gb33,_df_gb66, color = c,alpha=.2)
    # ax.set_title("$OA'$: Average diurnal anomaly") 
    ax.set_title(f"{seas} $OA'$: Median diurnal anomaly") 

    ax.set_ylim([-.7,.7])

#ax.legend(frameon=False)


plt.legend(frameon=False, bbox_to_anchor=(1,1,))
ax.set_xlabel('Time of day [h]')
for ax in axs: 
    ax.set_ylabel('OA  [$\mu$gm$^{-3}$]')
plt.tight_layout()

ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,])

ax.set_xlim([0,23])
fn = make_fn_eval('_'.join(models), 'diurnal_mean_dev')

fig.savefig(fn.with_suffix('.png'))
fig.savefig(fn.with_suffix('.pdf'))

# %%
_df_OA_all = df_OA_all[df_OA_all.index.month.isin(seasons2months[seas])]

_df_OA_all

# %%
fig, axs = plt.subplots(4,1, sharex=True, figsize=[7,8], dpi=100)
ax = axs[1]
#pl_obs = df_anom_OA['Obs'].groupby(df_anom_OA['obs'].index.hour).mean()
#pl_obs.plot(ax=ax,label='Observations', c='k')

for seas, ax in zip(seasons2months, axs):
    
    _df_OA_all = df_OA_all[df_OA_all.index.month.isin(seasons2months[seas])]
    #_df_OA_all = _df_OA_all.dropna(axis=0)
    for mod in df_OA_all.columns:
        if mod=='Obs':
            c = 'k'
        else:
            c=cdic_model[mod]
            #continue
        if mod in ['UKESM']:
            # 3h resolution so need to drop nans
            _df_OA_all = _df_OA_all[_df_OA_all[mod].notna()]

        ls = linestyle_dic[mod]
        _df_gb = _df_OA_all[mod].groupby(_df_OA_all.index.hour)
        _df_gbm = _df_gb.median()
        _df_gbstd = _df_gb.std()
        _df_gb33 = _df_gb.quantile(.84)
        _df_gb66 = _df_gb.quantile(.16)
        
        
        _df_gbm.plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
        ax.fill_between(_df_gbm.index, _df_gb33,_df_gb66, color = c,alpha=.2)
    # ax.set_title("$OA'$: Average diurnal anomaly") 
    ax.set_title(f"{seas} $OA$: Median diurnal variation") 

    #ax.set_ylim([-3,3])

plt.legend(frameon=False, bbox_to_anchor=(1,1,))


ax.set_xlabel('Time of day [h]')
for ax in axs: 
    ax.set_ylabel('OA  [$\mu$gm$^{-3}$]')
plt.tight_layout()

ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,])

ax.set_xlim([0,23])
fn = make_fn_eval('_'.join(models), 'diurnal_mean')

fig.savefig(fn.with_suffix('.png'))
fig.savefig(fn.with_suffix('.pdf'))

# %%
fig, axs = plt.subplots(1,1, sharex=True, figsize=[6,4], dpi=100)
ax = axs
#pl_obs = df_anom_OA['Obs'].groupby(df_anom_OA['obs'].index.hour).mean()
#pl_obs.plot(ax=ax,label='Observations', c='k')

    
_df_OA_all = df_OA_all.copy()#[df_OA_all.index.month.isin(seasons2months[seas])]
#_df_OA_all = _df_OA_all.dropna(axis=0)
for mod in df_OA_all.columns:
    if mod=='Obs':
        c = 'k'
    else:
        c=cdic_model[mod]
    ls = linestyle_dic[mod]
    _df_gb = _df_OA_all[mod].groupby(_df_OA_all.index.month)
    _df_gbm = _df_gb.median()
    _df_gbstd = _df_gb.std()
    _df_gb33 = _df_gb.quantile(.84)
    _df_gb66 = _df_gb.quantile(.16)
    
    _df_gbm.plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
    ax.fill_between(_df_gbm.index, _df_gb33,_df_gb66, color = c,alpha=.2)
    # ax.set_title("$OA'$: Average diurnal anomaly") 
ax.set_title(f"$OA$: Annual cycle") 

ax.set_ylim([0,6])

ax.legend(frameon=False)


plt.legend(frameon=False)
ax.set_xlabel('Month')
ax.set_ylabel('OA  [$\mu$gm$^{-3}$]')
plt.tight_layout()

ax.set_xticks(np.arange(1,13))

ax.set_xlim([1,12])
fn = make_fn_eval('_'.join(models), 'diurnal_mean')

fig.savefig(fn.with_suffix('.png'))
fig.savefig(fn.with_suffix('.pdf'))
print(fn)

# %%
import matplotlib as mpl

# %%
mi = np.min(df_OA_all[df_OA_all['Obs']>0]['Obs'])*60
ma = np.max(df_OA_all[df_OA_all['Obs']>0]['Obs'])*.9
bins_ = 10 ** np.linspace(np.log10(mi), np.log10(ma), 50)

# %%
fig, axs = plt.subplots(2,2,figsize = [9,5,],sharex=True, sharey=True)

for seas,ax in zip(seasons2months, axs.flatten()):
    df_OA_all_sub =  df_OA_all.copy()[df_OA_all['Obs'].notna()]
    for mo in ['UKESM']:
        if mo in df_OA_all.columns:
            df_OA_all_sub[mo] =df_OA_all_sub.loc[:,mo].ffill(limit=3).copy()
            df_OA_all_sub[mo] =df_OA_all_sub[mo][df_OA_all_sub['Obs'].notna()]
        
    df_OA_all_sub = df_OA_all_sub[df_OA_all_sub.index.month.isin(seasons2months[seas])].copy()
    df_OA_all_sub['Obs'].plot.hist(bins=bins_, alpha=1, 
                                     color='k',
                                     label='Observations',
                                   ax = ax,
                                   histtype='step',
                                  )
    df_OA_all_sub['Obs'].plot.hist(bins=bins_, alpha=0.2, 
                                     color='k',
                                     label='__nolabel__',
                                   ax = ax,
                                  )
    for mo in models:

        df_OA_all_sub[mo].plot.hist(bins=bins_, alpha=1, 
                                     color=cdic_model[mo],
                                     label=mo,
                                    ax = ax,
                                    
                                    histtype='step'
                                    )
        df_OA_all_sub[mo].plot.hist(bins=bins_, alpha=0.2, 
                                     color=cdic_model[mo],
                                    
                                     label='__nolabel__',
                                    ax = ax
                                    )

        
    ax.set_xscale('log')
    #_mod_an.plot.hist(bins=bins_, alpha=0.5,label='OsloAero, SOA')

    ax.set_xlabel('OA [$\mu$gm$^{-3}$]')

    ax.set_title(f'{seas}', y=.95)

ax.legend(frameon=False, bbox_to_anchor=(1,1,))

plt.suptitle(f'Distribution at {select_station}')
fn = make_fn_eval('noresm_echam_seasons','hist')
fig.tight_layout()
sns.despine(fig)
plt.savefig(fn, dpi=300)

print(fn)
plt.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
from bs_fdbck_clean.util.BSOA_datamanip.atto import season2month

# %%
season2month

# %%
_df = df_OA_all[df_OA_all['Obs']>0]
_df [_df.index.month.isin(season2month[seas])]
mi = np.min(_df['Obs'])*300
ma = np.max(_df['Obs'])*.9
bins_ = 10 ** np.linspace(np.log10(mi), np.log10(ma), 50)

# %%
fig, ax = plt.subplots(1,figsize = [6,4,],)
seas ='JA'

df_OA_all_sub = df_OA_all.copy()
for mo in ['UKESM']:
    if mo in df_OA_all.columns:
        #mo = 'UKESM'
        df_OA_all_sub[mo] =df_OA_all_sub.loc[:,mo].ffill(limit=3).copy()
        df_OA_all_sub[mo] =df_OA_all_sub[mo][df_OA_all_sub['Obs'].notna()]

df_OA_all_sub = df_OA_all_sub[df_OA_all_sub.index.month.isin(season2month[seas])]
df_OA_all_sub['Obs'].plot.hist(bins=bins_, alpha=1, 
                                     color='k',
                                     label='Observations',
                                   ax = ax,
                                   histtype='step',
                                  )
df_OA_all_sub['Obs'].plot.hist(bins=bins_, alpha=0.2, 
                                     color='k',
                                     label='__nolabel__',
                                   ax = ax,
                                  )
for mo in models:
        df_OA_all_sub[mo].plot.hist(bins=bins_, alpha=1, 
                                     color=cdic_model[mo],
                                     label=mo,
                                    ax = ax,
                                    histtype='step'
                                    )
        df_OA_all_sub[mo].plot.hist(bins=bins_, alpha=0.2, 
                                     color=cdic_model[mo],
                                     label='__nolabel__',
                                    ax = ax
                                    )

    #df_OA_all_sub[mo].plot.hist(bins=bins_, alpha=0.5, 
    #                                 color=cdic_model[mo],
    #                                 label=mo,
    #                                ax = ax,
                                #histtype="step",
                                #element='step',
    #                                )
ax.set_xscale('log')
    #_mod_an.plot.hist(bins=bins_, alpha=0.5,label='OsloAero, SOA')

ax.set_xlabel('OA [$\mu$gm$^{-3}$]')


ax.legend(frameon=False)
ax.set_title(f'Distribution at {select_station} in July & August')
fn = make_fn_eval('noresm_echam_JA','hist')
fig.tight_layout()
sns.despine(fig)
plt.savefig(fn, dpi=300)


plt.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
fig, axs = plt.subplots(1,2,figsize = [8,4],sharex=True, sharey=False)
seas = 'JA'
_df = df_OA_all[df_OA_all.index.month.isin(season2month[seas])]
_df['hour'] = _df.index.hour
for mo, ax in zip(models,axs):
    sns.scatterplot(y=mo, x='Obs',#orbins=bins_, alpha=0.5, 
                                     hue='hour', 
                #col = 'dir',
                ax=ax,
                alpha=0.4,
                palette='viridis',
                
                edgecolors=None,
                data = _df)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.1,50])
    ax.set_ylim([0.1,50])

    ax.set_ylabel(f'{mo} OA '+'[$\mu$gm$^{-3}$]')
    ax.set_xlabel('Observed OA [$\mu$gm$^{-3}$]')
    ax.set_title(seas)


    lims = ax.get_xlim()
    ax.plot(lims,lims,'k', linewidth=.5)

fig.tight_layout()

fn = make_fn_eval('_'.join(models),'scatt_JA')
fig.savefig(fn, dpi=300)
fig.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
fig, axs = plt.subplots(4,figsize = [4,8,],sharex=True, sharey=True)

for seas,ax in zip(seasons2months, axs.flatten()):
    df_OA_all_sub = df_OA_all.copy()
    for mo in ['UKESM']:
        if mo in df_OA_all.columns:
        
            df_OA_all_sub[mo] =df_OA_all_sub.loc[:,mo].ffill(limit=3).copy()
            df_OA_all_sub[mo] =df_OA_all_sub[mo][df_OA_all_sub['Obs'].notna()]

    df_OA_all_sub = df_OA_all_sub[df_OA_all_sub.index.month.isin(season2month[seas])]
    

    df_OA_all_sub['Obs'].plot.hist(bins=bins_, alpha=1, 
                                     color='k',
                                     label='Observations',
                                   ax = ax,
                                   histtype='step',
                                  )
    df_OA_all_sub['Obs'].plot.hist(bins=bins_, alpha=0.2, 
                                     color='k',
                                     label='__nolabel__',
                                   ax = ax,
                                  )

    for mo in models:
        df_OA_all_sub[mo].plot.hist(bins=bins_, alpha=1, 
                                     color=cdic_model[mo],
                                     label=mo,
                                    ax = ax,
                                    histtype='step'
                                    )
        df_OA_all_sub[mo].plot.hist(bins=bins_, alpha=0.2, 
                                     color=cdic_model[mo],
                                     label='__nolabel__',
                                    ax = ax
                                    )
    ax.set_xscale('log')
    #_mod_an.plot.hist(bins=bins_, alpha=0.5,label='OsloAero, SOA')

    ax.set_xlabel('OA [$\mu$gm$^{-3}$]')

    ax.set_title(f'{seas}', y=.95)

ax.legend(frameon=False, bbox_to_anchor=(.6,.2,))
plt.suptitle('Distribution at SMEAR-II')
fn = make_fn_eval('noresm_echam_seasons','hist')
fig.tight_layout()
sns.despine(fig)
plt.savefig(fn, dpi=300)
print(fn)

plt.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
df_OA_all_sub = df_OA_all.copy()
for mo in ['UKESM']:
    if mo in df_OA_all.columns:
        df_OA_all_sub[mo] =df_OA_all_sub.loc[:,mo].ffill(limit=3).copy()
        df_OA_all_sub[mo] =df_OA_all_sub[mo][df_OA_all_sub['Obs'].notna()]

for mo in models:
    
    (df_OA_all_sub[mo]-df_OA_all_sub['Obs']).plot.hist(#bins=bins_, 
        alpha=1, 
        bins=200,
        color = cdic_model[mo],
                                     #color=None,
                                     label=mo,
        histtype='step',
                                    )
    (df_OA_all_sub[mo]-df_OA_all_sub['Obs']).plot.hist(#bins=bins_, 
        alpha=0.2, 
        bins=200,
        color = cdic_model[mo],
                                     #color=None,
                                     label='__nolabel__',
                                    )

    #plt.xscale('log')
#_mod_an.plot.hist(bins=bins_, alpha=0.5,label='OsloAero, SOA')
plt.xlim([-8,8])
plt.xlabel('OA [$\mu$gm$^{-3}$]')

plt.title('Model anomaly, July and August 2012-2018, Hyytiälä')

plt.legend()
fn = make_fn_eval('diff_hist','hist')
plt.savefig(fn, dpi=300)
plt.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
import numpy as np

# %%
_df = df_OA_all[df_OA_all['Obs']>0]
_df [_df.index.month.isin(season2month[seas])]
mi = np.min(_df['Obs'])*300
ma = np.max(_df['Obs'])*.9
bins_ = 10 ** np.linspace(np.log10(mi), np.log10(ma), 50)

# %%
fig, axs = plt.subplots(4,4,figsize = [14,13],sharex=True, sharey=False)

for seas,i in zip(seasons2months, range(4)):
    axs_sub = axs[i]
    _df = df_OA_all[df_OA_all.index.month.isin(seasons2months[seas])]

    _df['hour'] = _df.index.hour
    for mo, ax in zip(models,axs_sub):
        sns.scatterplot(y=mo, x='Obs',#orbins=bins_, alpha=0.5, 
                                     hue='hour', 
                #col = 'dir',
                ax=ax,
                alpha=0.4,
                palette='viridis',
                
                edgecolors=None,
                data = _df)

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([0.1,50])
        ax.set_ylim([0.1,50])

        ax.set_ylabel(f'{mo} OA '+'[$\mu$gm$^{-3}$]')
        ax.set_xlabel('Observed OA [$\mu$gm$^{-3}$]')
        ax.set_title(seas)


        lims = ax.get_xlim()
        ax.plot(lims,lims,'k', linewidth=.5)

fig.tight_layout()

fn = make_fn_eval('_'.join(models),'scatt')
fig.savefig(fn, dpi=300)
fig.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
fig, axs = plt.subplots(4,4,figsize = [15,13],sharex=True, sharey=False)

for seas,i in zip(['DJF','MAM','JJA','SON'], range(4)):
    axs_sub = axs[i]
    _df = df_OA_all[df_OA_all.index.month.isin(season2month[seas])]
    bins = np.linspace(-1,2,40)
    #_df['hour'] = _df.index.hour
    for mo, ax in zip(models,axs_sub):
        sns.histplot(y=mo, x='Obs',
                ax=ax,
                cmap = sns.color_palette("mako_r", as_cmap=True),
                log_scale=(True, True),
                     cbar=True, cbar_kws=dict(shrink=.75),
                
                edgecolors=None,
             rasterized=True,
                     
                     bins=(bins,bins,),
                data = _df)


        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([0.1,30])
        ax.set_ylim([0.1,30])

        ax.set_ylabel(f'{mo} OA '+'[$\mu$gm$^{-3}$]')
        ax.set_xlabel('Observed OA [$\mu$gm$^{-3}$]')

        ax.set_title(seas)


        lims = ax.get_xlim()
        ax.plot(lims,lims,'k', linewidth=.5)

fig.tight_layout()

fn = make_fn_eval('_'.join(models),'2Dhist_seasons_OA')
fig.savefig(fn, dpi=300)
fig.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
fig, axs = plt.subplots(1,4,figsize = [15,4],sharex=True, sharey=False)
seas = 'JA'
_df = df_OA_all[df_OA_all.index.month.isin(season2month[seas])]
_df['hour'] = _df.index.hour
for mo, ax in zip(models,axs):
    sns.histplot(y=mo, x='Obs',#orbins=bins_, alpha=0.5, 
                                    # hue='hour', 
                #col = 'dir',
                ax=ax,
                cmap = sns.color_palette("mako_r", as_cmap=True),
                log_scale=(True, True),
                     cbar=True, cbar_kws=dict(shrink=.75),
                
                edgecolors=None,
             rasterized=True,
                 
                data = _df)


    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.1,30])
    ax.set_ylim([0.1,30])

    ax.set_ylabel(f'{mo} OA '+'[$\mu$gm$^{-3}$]')
    ax.set_xlabel('Observed OA [$\mu$gm$^{-3}$]')



    lims = ax.get_xlim()
    ax.plot(lims,lims,'k', linewidth=.5)

fig.suptitle(f'OA concentrations in Jul&Aug')

fn = make_fn_eval('_'.join(models),f'season_{seas}_2dist_OA_conc')
print(fn)
sns.despine(fig)
fig.tight_layout()
fig.savefig(fn, dpi=300)
fig.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
fig, axs = plt.subplots(4,sharey=True, figsize=[10,8])
ax = axs[0]
_df = df_anom_OA
_df['hour'] = _df.index.hour
for mo, ax in zip(models,axs.flatten()):
    sns.histplot(y=f'{mo}', x=f'hour',#orbins=bins_, alpha=0.5, 
                                     #hue=f'Obs', 
                #col = 'dir',
                ax=ax,
                #alpha=0.4,
                cmap='YlGnBu',
                 
                
                #edgecolors=None,
                data = _df)

    #ax.set_yscale('log')
    # ax.set_xscale('symlog')
    ax.set_ylim([-3.5,3.5])
    #ax.set_ylim([0.1,30])

    #ax.set_xlabel(f'{mo} OA '+'[$\mu$gm$^{-3}$]')
    #ax.set_ylabel('Observed OA [$\mu$gm$^{-3}$]')



    lims = ax.get_xlim()
    #ax.plot(lims,lims,'k', linewidth=.5)

    #_df.groupby('hour').count().plot(ax = ax.twinx())

ax.set_ylim([-1.5,1.5])
fn = make_fn_eval('_'.join(models),'2dhist_dist_time')
fig.savefig(fn, dpi=300)
fig.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
df_OA_diffObs = (df_OA_all.drop('Obs', axis=1).T-df_OA_all['Obs']).T

# %%
df_OA_diffObs

# %%
df_OA_diffObs.tail(50)

# %%
fig, axs = plt.subplots(4,sharex=False, figsize=[10,8])
ax = axs[0]
_df = df_OA_diffObs
_df['hour'] = _df.index.hour
for mo, ax in zip(models,axs.flatten()):
    
    sns.histplot(y=f'{mo}', x='hour',#orbins=bins_, alpha=0.5, 
                                     #hue=f'Obs', 
                #col = 'dir',
                ax=ax,
                #alpha=0.4,
                cmap='YlGnBu',
                 
                
                #edgecolors=None,
                data = _df[[mo,'hour']].dropna(axis=0))

    #ax.set_yscale('log')
    # ax.set_xscale('symlog')
    #ax.set_ylim([-3.5,3.5])
    #ax.set_ylim([0.1,30])

    #ax.set_xlabel(f'{mo} OA '+'[$\mu$gm$^{-3}$]')
    #ax.set_ylabel('Observed OA [$\mu$gm$^{-3}$]')



    #lims = ax.get_xlim()
    #ax.plot(lims,lims,'k', linewidth=.5)

    #_df.groupby('hour').count().plot(ax = ax.twinx())

    ax.set_ylim([-1.5,1.5])
fn = make_fn_eval('_'.join(models),'2dhist_dist_time')
fig.savefig(fn, dpi=300)
fig.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%

# %%

# %%

# %%
