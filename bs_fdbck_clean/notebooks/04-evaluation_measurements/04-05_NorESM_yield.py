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

# %% [markdown]
# # LOAD ATTO station

# %%
# %load_ext autoreload

# %autoreload 2

from pathlib import Path
from bs_fdbck.util.BSOA_datamanip import ds2df_inc_preprocessing, ds2df_echam
from bs_fdbck.util.collocate.collocateLONLAToutput import CollocateLONLATout
from bs_fdbck.util.collocate.collocate_echam_salsa import CollocateModelEcham
import useful_scit.util.log as log
from bs_fdbck.util.plot.BSOA_plots import make_cool_grid, plot_scatter
log.ger.setLevel(log.log.INFO)
import time
import xarray as xr
import matplotlib.pyplot as plt

# %%
import pandas as pd


# %%
from bs_fdbck.constants import path_measurement_data

# %%

# %%

import numpy as np



# %%

select_station = 'ATTO'
model_lev_i = -1

# %%
plot_path = Path(f'Plots/{select_station}')


# %% pycharm={"name": "#%% \n"}
def make_fn_scat(case, v_x, v_y):
    _x = v_x.split('(')[0]
    _y = v_y.split('(')[0]
    f = f'scat_all_years_echam_noresm_{case}_{_x}_{_y}-ATTO_ukesm.png'
    return plot_path /f


# %%
plot_path.mkdir(exist_ok=True, parents=True)

# %%
plot_path

# %%
from bs_fdbck.constants import path_measurement_data
postproc_data = path_measurement_data /'model_station'/select_station
postproc_data_obs = path_measurement_data /select_station/'processed'


# %%
fn_obs_comb_data_full_time =postproc_data_obs /'ATTO_data_comb_hourly.nc'

# %% [markdown]
# # Load observational data: 

# %%
postproc_data_obs = path_measurement_data /'ATTO'/'processed'

# %%


ds_ATTO = xr.open_dataset(fn_obs_comb_data_full_time)

# %%
fn_obs_comb_data_full_time

# %% [markdown] tags=[]
# # Read in model data:

# %%
models = ['ECHAM-SALSA','NorESM', 'EC-Earth', 'UKESM']
mod2cases = {'ECHAM-SALSA' : ['SALSA_BSOA_feedback'],
             'NorESM' : ['OsloAero_intBVOC_f09_f09_mg17_fssp'],
             'EC-Earth' : ['ECE3_output_Sara'],
             'UKESM' : ['AEROCOMTRAJ'],
             'Observations':['Obs'],
            }
di_mod2cases = mod2cases.copy()


# %%
select_station='ATTO'

# %%
dic_df_pre=dict()
for mod in models:
    print(mod)
    dic_df_pre[mod] = dict()
    for ca in mod2cases[mod]:
        print(mod, ca)
        if model_lev_i !=-2:
            fn_out = postproc_data/f'{select_station}_station_{mod}_{ca}_ilev{model_lev_i}.csv'
        else:
            fn_out = postproc_data/f'{select_station}_station_{mod}_{ca}.csv'
        #fn_out = postproc_data/f'{select_station}_station_{mod}_{ca}.csv'
        print(fn_out)
        dic_df_pre[mod][ca] = pd.read_csv(fn_out, index_col=0)
        dic_df_pre[mod][ca].index = pd.to_datetime(dic_df_pre[mod][ca].index)
        #dic_df_mod_case[mod][ca].to_csv(fn_out)

# %% [markdown]
# ## Read in observations

# %%
ds_ATTO = xr.open_dataset(fn_obs_comb_data_full_time, engine='netcdf4')

# %%
ds_ATTO = ds_ATTO.sel(time_traj=0)

# %%
varl = ['Pressure_reanalysis', 'Potential_Temperature_reanalysis', 'Temperature_reanalysis', 'Rainfall', 'Mixing_Depth', 'Relative_Humidity', 'Specific_Humidity_reanalysis',
 'Mixing_Ratio','Solar_Radiation', 'condensation_sink', 'N50-500', 'N100-500', 'N200-500', 'timeUTC-3', 'pressure', 'temperature', 'humidity','precip',#'Solar_inc_Wm2',
        #'Solar_out_Wm2',
        #'wind_speed',
        #'wind_speed_v',
        'OA'
]
ds_ATTO[varl].squeeze().to_dataframe()

# %%
dic_df_pre['Observations'] = dict()
dic_df_pre['Observations']['Observations'] = ds_ATTO[varl].squeeze().to_dataframe()
dic_df_pre['Observations']['Observations'].index = pd.to_datetime(dic_df_pre['Observations']['Observations'].index)

# %%
mod2cases['Observations'] = ['Observations']


# %%
dic_mod_ca = dic_df_pre.copy()

# %%
_ds =dic_mod_ca['ECHAM-SALSA']['SALSA_BSOA_feedback']

_ds =dic_mod_ca['NorESM'][mod2cases['NorESM'][0]]

_ds['SFisoprene'].plot()

# %% [markdown]
# ### Save result in dictionary

# %%
dic_df_mod_case = di_mod2cases
for mo in models:
    cs = mod2cases[mo]
    for c in cs: 
        if len(cs)>1:
            use_name = f'{mo}_{c}'
        else:
            use_name =mo


# %%
dic_df_mod_case = dic_mod_ca.copy()

# %%
from bs_fdbck.util.BSOA_datamanip import calculate_daily_median_summer,calculate_summer_median

# %%
for mod in models:
    for ca in mod2cases[mod]:
        _df = dic_df_mod_case[mod][ca]
        for v in ['OA','N50','N100','N200','N500']:
            if f'{v}_STP' in _df.columns:
                if v in _df.columns:
                    _df = _df.rename({v:f'{v}_orig'}, axis=1)
                _df = _df.rename({f'{v}_STP':v}, axis=1)
        dic_df_mod_case[mod][ca] = _df

# %% [markdown] tags=[]
# ### Calculate Nx-500:
#

# %% tags=[]
for mod in models:
    print(mod)
    for ca in dic_df_mod_case[mod].keys():
        print(ca)
        _df = dic_df_mod_case[mod][ca]
        for v in ['N50','N100','N200']:
            _df[f'{v}-500'] = _df[v] -_df['N500'] 
        dic_df_mod_case[mod][ca] = _df

# %%
dic_df_atto = dic_df_pre.copy()

# %%
dic_df_pre_ATTO =dict()
for key in dic_df_pre.keys():
    dic_df_pre_ATTO[key] = dict()
    for key2 in dic_df_pre[key].keys():
        dic_df_pre_ATTO[key][key2] = dic_df_pre[key][key2].copy()



# %% [markdown] tags=[]
# # Merge with observations:

# %%
dic_df_pre = dict()#dic_df_mod_case.copy()#deep=True)
for mod in dic_df_mod_case.keys():
    dic_df_pre[mod] = dic_df_mod_case[mod].copy()

# %%
vars_obs = ['OA', 'N100-500','N50-500','N200-500','temperature']

# %%
ds_ATTO

# %%
df_ATTO = ds_ATTO[vars_obs].drop('time_traj').to_dataframe()

df_ATTO['some_obs_missing'] = df_ATTO.isnull().any(axis=1)

# %%
df_for_merge = df_ATTO[['OA','N100-500', 'some_obs_missing']].rename({'OA':'obs_OA','N100-500':'obs_N100-500',},axis=1)

# %%
for mod in dic_df_mod_case.keys():
    print(mod)
    for ca in dic_df_mod_case[mod].keys():
        dic_df_mod_case[mod][ca] = pd.merge(dic_df_pre[mod][ca], df_for_merge ,right_on='time', left_on='time', how='outer')
        dic_df_mod_case[mod][ca]['year'] = dic_df_mod_case[mod][ca].index.year

# %%
df_ATTO_obs_rename = df_ATTO.rename({'Org':'OA','temperature':'T_C'}, axis=1)

# %%
df_ATTO_obs_rename

# %% [markdown]
# ## Add observations to dictionary

# %%
dic_df_mod_case['Observations'] = dict()
dic_df_mod_case['Observations']['Observations'] = df_ATTO_obs_rename

# %%
dic_df_mod_case['Observations'].keys()

# %%
dic_mod2case = mod2cases


# %%
def add_log(df, varl=None):
    if varl is None:
        varl = ['OA','N100', 'Org','N100 (cm^-3)','N50','N150','N200']
    var_exist = df.columns
    
    varl_f = set(varl).intersection(var_exist)
    print(varl_f)
    for v in varl_f:
        df[f'log10({v})'] = np.log10(df[v])
    return df


for mod in dic_df_mod_case.keys():
    for c in dic_df_mod_case[mod].keys():
    
        dic_df_mod_case[mod][c] = add_log(dic_df_mod_case[mod][c])
        #dic_dfsm_mod_case[mod][c] = add_log(dic_dfsm_mod_case[mod][c])
        
df_ATTO = add_log(df_ATTO)


# %%
mod='NorESM'

# %%
ca = mod2cases[mod][0]

# %%
mask_obs_N = dic_df_mod_case[mod][ca]['obs_N100-500'].notnull()
mask_obs_OA = dic_df_mod_case[mod][ca]['obs_OA'].notnull()

# %%
from bs_fdbck.util.plot.BSOA_plots import cdic_model
import seaborn as sns
from matplotlib import pyplot as plt, gridspec as gridspec
from bs_fdbck.util.plot.BSOA_plots import make_cool_grid2, make_cool_grid3
import scipy

# %% [markdown]
# # Load data SMEAR: 
#

# %%
from pathlib import Path
import matplotlib as mpl
import xarray as xr


# %%
from pathlib import Path

from bs_fdbck.util.BSOA_datamanip import ds2df_inc_preprocessing
from bs_fdbck.util.collocate.collocateLONLAToutput import CollocateLONLATout
from bs_fdbck.util.collocate.collocate_echam_salsa import CollocateModelEcham
import useful_scit.util.log as log

from bs_fdbck.util.plot.BSOA_plots import make_cool_grid, plot_scatter

log.ger.setLevel(log.log.INFO)
import time
import xarray as xr
import matplotlib.pyplot as plt

# %%
from bs_fdbck.util.plot.BSOA_plots import cdic_model

# %%
from bs_fdbck.constants import path_measurement_data
select_station = 'SMR'
postproc_data = path_measurement_data /'model_station'/select_station
postproc_data_obs = path_measurement_data /select_station/'processed'


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
from bs_fdbck.preprocess.launch_monthly_station_collocation import launch_monthly_station_output
from bs_fdbck.util.Nd.sizedist_class_v2.SizedistributionBins import SizedistributionStationBins
from bs_fdbck.util.collocate.collocateLONLAToutput import CollocateLONLATout
from bs_fdbck.data_info.variable_info import list_sized_vars_nonsec, list_sized_vars_noresm
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
dic_df_pre['Observations']['Observations'].columns

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
_df.columns

# %%
_df.groupby('hour').mean()['T_C'].plot(marker='.')

# %%
_df.groupby('hour').mean()['emiisop'].plot()

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

# %%
dic_df_pre_SMR =dict()
for key in dic_df_pre.keys():
    dic_df_pre_SMR[key] = dict()
    for key2 in dic_df_pre[key].keys():
        dic_df_pre_SMR[key][key2] = dic_df_pre[key][key2].copy()



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
    'FMA': [  2,3,4], 
                  
       }

# %%
seasons2months2 = {
    'JFM': [ 1, 2,3], 
    'FMA': [  2,3,4], 
    'AMJ': [ 4, 5,6], 
    'JAS': [ 7, 8,9], 
    'OND': [ 10, 11,12],
}

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


# %% [markdown]
# # ANALYSIS
#

# %%
dic_df_pre_SMR['NorESM']['OsloAero_intBVOC_f09_f09_mg17_fssp']['T_C'].plot()

dic_df_pre_ATTO['NorESM']['OsloAero_intBVOC_f09_f09_mg17_fssp']['T_C'].plot()

# %%
df_noresm_smr = dic_df_pre_SMR['NorESM']['OsloAero_intBVOC_f09_f09_mg17_fssp'].resample('d').mean()
df_noresm_atto = dic_df_pre_ATTO['NorESM']['OsloAero_intBVOC_f09_f09_mg17_fssp'].resample('d').mean()


# %%
df_obs_smr = dic_df_pre_SMR['Observations']['Observations'].resample('d').mean()
df_obs_atto = dic_df_pre_ATTO['Observations']['Observations'].resample('d').mean()


# %%
dic_df = dict(
    ATTO = dict(NorESM=df_noresm_atto, Observations=df_obs_atto),
    SMR = dict(NorESM=df_noresm_smr, Observations=df_obs_smr),
)
              

# %%
dic_season= dict(
    ATTO = 'FMA',
    SMR = 'JJA',
)

# %%
station = 'SMR'



# %%
df_noresm_smr['SFisoprene'].plot()

# %%
seasons2months['JJA']


# %%
station = 'SMR'
season = dic_season[station]
_df = dic_df[station]['NorESM']

months = seasons2months[season]
_df = _df[_df.index.month.isin(months)]
#_df['SFisoprene'].plot()
sns.scatterplot(x='SFisoprene', y='OA', data = _df)

# %%
station = 'SMR'
season = dic_season[station]
_df = dic_df[station]['NorESM'].resample('d').mean()

months = seasons2months[season]
_df = _df[_df.index.month.isin(months)]
#_df['SFisoprene'].plot()
sns.scatterplot(y='SFisoprene', x='T_C', data = _df)

# %%
station = 'SMR'
season = dic_season[station]
_df = dic_df[station]['NorESM'].resample('d').mean()

months = seasons2months[season]
_df = _df[_df.index.month.isin(months)]
#_df['SFisoprene'].plot()
sns.scatterplot(y='SFisoprene', x='OA', data = _df)

# %%
station = 'SMR'
season = dic_season[station]
_df = dic_df[station]['NorESM'].resample('d').mean()

months = seasons2months[season]
_df = _df[_df.index.month.isin(months)]
#_df['SFisoprene'].plot()
sns.scatterplot(y='SFmonoterp', x='OA', data = _df)

# %%

# %%
station = 'ATTO'
season = dic_season[station]
_df = dic_df[station]['NorESM'].resample('d').mean()

months = seasons2months[season]
_df = _df[_df.index.month.isin(months)]
#_df['SFisoprene'].plot()
sns.scatterplot(y='SFmonoterp', x='T_C', data = _df)

# %%
station = 'ATTO'
season = dic_season[station]
_df = dic_df[station]['NorESM'].resample('d').mean()

months = seasons2months[season]
_df = _df[_df.index.month.isin(months)]
#_df['SFisoprene'].plot()
sns.scatterplot(y='SFisoprene', x='T_C', data = _df)

# %%
_df['T_C'].plot()


# %%
station = 'SMR'
season = dic_season[station]
_df = dic_df[station]['NorESM'].resample('d').mean()

_df['emissions_monoterp2isoprene'] = _df['SFmonoterp']/_df['SFisoprene']
_df = _df[ _df['SFmonoterp']>0]
_df = _df[ _df['SFisoprene']>0]
_df = _df[_df['emissions_monoterp2isoprene'].notna()]

m = _df['emissions_monoterp2isoprene'].median()
label = f'{station}, median={m}'
_df['emissions_monoterp2isoprene'].plot.hist( bins = np.logspace(-1,3), alpha=0.5, label=label)
plt.xscale('log')

station = 'ATTO'
season = dic_season[station]
_df = dic_df[station]['NorESM'].resample('d').mean()

_df['emissions_monoterp2isoprene'] = _df['SFmonoterp']/_df['SFisoprene']
_df = _df[ _df['SFmonoterp']>0]
_df = _df[ _df['SFisoprene']>0]
_df = _df[_df['emissions_monoterp2isoprene'].notna()]
m = _df['emissions_monoterp2isoprene'].median()
print(m)
label = f'{station}, median={m}'
_df['emissions_monoterp2isoprene'].plot.hist( bins = np.logspace(-.5,3), alpha=0.5, label=label)
plt.xscale('log')
plt.title('Mass emission ratio of monterpene/isoprene ')
plt.legend()

# %%
isoprene_mass = 68
monoterp_mass = isoprene_mass*2

# %%
final_mass = 168

# %%
monoterp_yield = 0.15
isoprene_yield = 0.05
monoterp_yield_mass = monoterp_yield*final_mass/monoterp_mass
isoprene_yield_mass = isoprene_yield*final_mass/isoprene_mass
print('monoterp mass yields now:',monoterp_yield_mass)
print('isoprene mass yields now:',isoprene_yield_mass)

# %%

# %%
dic_colors = {
    'NorESM' : {'ATTO':'#e41a1c', 'SMR':'#e41a1c'},
    'Observations' : {'ATTO':'k', 'SMR':'k'},
}

# %%
fig, axs = plt.subplots(2,sharex=True)
station = 'SMR'
season = dic_season[station]
ax = axs[0]
for k in dic_df[station].keys():
    _df = dic_df[station][k].resample('d').mean()


    m = _df['OA'].median()
    label = f'{station},{k} median={m:.2f}'
    _df['OA'].plot.hist( bins = np.logspace(-2,2), alpha=0.5, label=label, color = dic_colors[k][station],
                        linewidth=3,
                        ax = ax,
                       histtype='stepfilled', linestyle=':')
    plt.xscale('log')
ax.set_xscale('log')
ax.legend(frameon=False)
ax = axs[1]
station = 'ATTO'
season = dic_season[station]
for k in dic_df[station].keys():
    _df = dic_df[station][k].resample('d').mean()


    m = _df['OA'].median()
    label = f'{station},{k} median={m:.2f}'
    _df['OA'].plot.hist( bins = np.logspace(-2,2), alpha=.5, label=label,color = dic_colors[k][station], linewidth=3,
                        ax = ax,
                        
                       histtype='stepfilled', )
    plt.xscale('log')
ax.set_xscale('log')
ax.legend(frameon=False)

# %%
_df = dic_df['ATTO']['NorESM']

# %%
_df = dic_df['ATTO']['NorESM'].resample('d').mean()
_df['prodOA_mono'] = _df['SFmonoterp']*monoterp_yield_mass
_df['prodOA_iso'] = _df['SFisoprene']*isoprene_yield_mass
_df['prodOA_tot'] = _df['prodOA_mono']+_df['prodOA_iso']
_df['monoterp_fraction_total'] =_df['prodOA_mono']/_df['prodOA_tot'] 
_df['isoprene_fraction_total'] =_df['prodOA_iso']/_df['prodOA_tot'] 



_df['isoprene_fraction_total'].resample('Y').mean().plot()
_df['monoterp_fraction_total'].resample('Y').mean().plot()
#_df['scale_OA'].resample('Y').mean().plot()
plt.legend()
plt.show()
_df[['prodOA_mono','prodOA_iso']].resample('Y').mean().plot()


plt.legend()
plt.show()
_df_rs = _df.resample('Y').mean()
(_df_rs['prodOA_mono']/_df_rs['prodOA_tot']).plot(label='mono')
(_df_rs['prodOA_iso']/_df_rs['prodOA_tot']).plot(label='iso')
plt.legend()
plt.show()

# %%
_df = dic_df['ATTO']['NorESM'].resample('d').mean()
_df['prodOA_mono'] = _df['SFmonoterp']*monoterp_yield_mass
_df['prodOA_iso'] = _df['SFisoprene']*isoprene_yield_mass
_df['prodOA_tot'] = _df['prodOA_mono']+_df['prodOA_iso']
_df['monoterp_fraction_total'] =_df['prodOA_mono']/_df['prodOA_tot'] 
_df['isoprene_fraction_total'] =_df['prodOA_iso']/_df['prodOA_tot'] 



_df['isoprene_fraction_total'].resample('Y').mean().plot()
_df['monoterp_fraction_total'].resample('Y').mean().plot()
#_df['scale_OA'].resample('Y').mean().plot()
plt.legend()
plt.show()
_df[['prodOA_mono','prodOA_iso']].resample('Y').mean().plot()


plt.legend()
plt.show()
_df_rs = _df.resample('Y').mean()
(_df_rs['prodOA_mono']/_df_rs['prodOA_tot']).plot(label='mono')
(_df_rs['prodOA_iso']/_df_rs['prodOA_tot']).plot(label='iso')
plt.legend()
plt.show()

# %%
_df = dic_df['ATTO']['NorESM'].resample('d').mean()
_df['SFisoprene'].mean()/_df['SFmonoterp'].mean()

# %%
_df = dic_df['ATTO']['NorESM'].resample('d').mean()

_df['SFisoprene'].mean()/_df['SFmonoterp'].mean()

# %%
_df = dic_df['ATTO']['NorESM'].resample('d').mean()

_df['SFmonoterp'].mean()

# %%
_df = dic_df['ATTO']['NorESM'].resample('d').mean()

_df['SFmonoterp'].mean()

# %%
_df = dic_df['ATTO']['NorESM'].resample('d').mean()

_df['SFisoprene'].mean()

# %%
_df = dic_df['ATTO']['NorESM'].resample('d').mean()

_df['SFisoprene'].mean()

# %%
monoterp_yield_mass


# %%
def merge_two_sources(df1,df2,name1,name2, v1,v2):
    _df1 = df1[v1].rename(name1)
    _df2 = df2[v2].rename(name2)
    return pd.concat([_df1,_df2], axis=1)
    
    

# %%

# %%
s_mono = 1.5
s_isop = .1
for station in dic_df.keys():
    print(station)
    _df = dic_df[station]['NorESM'].resample('d').mean()
    _df['prodOA_mono'] = _df['SFmonoterp']*monoterp_yield_mass
    _df['prodOA_iso'] = _df['SFisoprene']*isoprene_yield_mass

    _df['monoterp_fraction_total'] =_df['prodOA_mono']/(_df['prodOA_mono']+ _df['prodOA_iso'])
    _df['isoprene_fraction_total'] =_df['prodOA_iso']/(_df['prodOA_mono']+ _df['prodOA_iso'])    
    _df['scale_OA'] =_df['monoterp_fraction_total']*s_mono + _df['isoprene_fraction_total']*s_isop
    _df['OA_new'] = _df['OA']*(_df['monoterp_fraction_total']*s_mono + _df['isoprene_fraction_total']*s_isop)
    dic_df[station]['NorESM'] = _df
    
    

    
    
fig, axs = plt.subplots(2, sharex=True, figsize=(10,10,))
ax = axs[0]
station = 'SMR'
season = dic_season[station]
#for k in dic_df[station].keys():
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA')
_df = _df[_df.index.month.isin(months)]
_df = _df.dropna()

for k in dic_df[station].keys():
    m = _df[k].median()
    label = f'{station},{k} median={m}'
    _df[k].plot.hist( bins = np.logspace(-2,2), alpha=0.3, label=label, color = dic_colors[k][station],
                        linewidth=3,
                     ax = ax,
                       histtype='stepfilled', linestyle=':')

k= 'NorESM'
#_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA_new')
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA_new')
_df = _df[_df.index.month.isin(months)]
m = _df[k].median()
_df = _df.dropna()

label = f'{station},{k}, new median={m}'
_df[k].plot.hist( bins = np.logspace(-2,2), alpha=1, label=label, color =  dic_colors[k][station],
                        linewidth=4,
                     ax = ax,
                 
                       histtype='step', )
ax.set_xscale('log')
ax.set_title(f'Season: {season}')

ax.legend(frameon=False, loc=2)    

ax = axs[1]

station = 'ATTO'
season = dic_season[station]
months = seasons2months[season]
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA')
_df = _df[_df.index.month.isin(months)]
_df = _df.dropna()

for k in dic_df[station].keys():
    m = _df[k].median()
    label = f'{station},{k} median={m}'
    _df[k].plot.hist( bins = np.logspace(-2,2), alpha=.4, label=label, color = dic_colors[k][station],
                        linewidth=3,
                     ax = ax,
                     
                       histtype='stepfilled', linestyle='--')

k= 'NorESM'
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA_new')
_df = _df[_df.index.month.isin(months)]
_df = _df.dropna()
m = _df[k].median()
label = f'{station},{k}, new median={m}'
_df[k].plot.hist( bins = np.logspace(-2,2), alpha=1, label=label, color = dic_colors[k][station],
                        linewidth=4,
                     ax = ax,
                 
                       histtype='step', )

ax.set_xscale('log')
ax.legend(frameon=False, loc=2)    
ax.set_title(f'Season: {season}')
plt.suptitle(f'OA distribution and estimated distribution with MT and IP \n yields scaled by {s_mono} and {s_isop} respectively')
sns.despine(fig)
ax.set_xlabel('OA concentration [$\mu$gm$^{-3}$]')
fn = f'Plots/noresm_yield_adjust_ym{s_mono}_yiso{s_isop}.pdf'
fig.tight_layout()
fig.savefig(fn)

# %%
s_mono = 1.5
s_isop = 0.1
for station in dic_df.keys():
    print(station)
    _df = dic_df[station]['NorESM'].resample('d').mean()
    _df['prodOA_mono'] = _df['SFmonoterp']*monoterp_yield_mass
    _df['prodOA_iso'] = _df['SFisoprene']*isoprene_yield_mass

    _df['monoterp_fraction_total'] =_df['prodOA_mono']/(_df['prodOA_mono']+ _df['prodOA_iso'])
    _df['isoprene_fraction_total'] =_df['prodOA_iso']/(_df['prodOA_mono']+ _df['prodOA_iso'])    
    _df['scale_OA'] =_df['monoterp_fraction_total']*s_mono + _df['isoprene_fraction_total']*s_isop
    _df['OA_new'] = _df['OA']*(_df['monoterp_fraction_total']*s_mono + _df['isoprene_fraction_total']*s_isop)
    dic_df[station]['NorESM'] = _df
    
    

    
    
fig, axs = plt.subplots(2, sharex=True, figsize=(10,10,))
ax = axs[0]
station = 'SMR'
season = dic_season[station]
#for k in dic_df[station].keys():
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA')
_df = _df[_df.index.month.isin(months)]
_df = _df.dropna()

for k in dic_df[station].keys():
    m = _df[k].median()
    label = f'{station},{k} median={m}'
    _df[k].plot.hist( bins = np.logspace(-2,2), alpha=0.3, label=label, color = dic_colors[k][station],
                        linewidth=3,
                     ax = ax,
                       histtype='stepfilled', linestyle=':')

k= 'NorESM'
#_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA_new')
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA_new')
_df = _df[_df.index.month.isin(months)]
m = _df[k].median()
_df = _df.dropna()

label = f'{station},{k}, new median={m}'
_df[k].plot.hist( bins = np.logspace(-2,2), alpha=1, label=label, color =  dic_colors[k][station],
                        linewidth=4,
                     ax = ax,
                 
                       histtype='step', )
ax.set_xscale('log')
ax.set_title(f'Season: {season}')

ax.legend(frameon=False, loc=2)    

ax = axs[1]

station = 'ATTO'
season = dic_season[station]
months = seasons2months[season]
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA')
_df = _df[_df.index.month.isin(months)]
_df = _df.dropna()

for k in dic_df[station].keys():
    m = _df[k].median()
    label = f'{station},{k} median={m}'
    _df[k].plot.hist( bins = np.logspace(-2,2), alpha=.4, label=label, color = dic_colors[k][station],
                        linewidth=3,
                     ax = ax,
                     
                       histtype='stepfilled', linestyle='--')

k= 'NorESM'
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA_new')
_df = _df[_df.index.month.isin(months)]
_df = _df.dropna()
m = _df[k].median()
label = f'{station},{k}, new median={m}'
_df[k].plot.hist( bins = np.logspace(-2,2), alpha=1, label=label, color = dic_colors[k][station],
                        linewidth=4,
                     ax = ax,
                 
                       histtype='step', )

ax.set_xscale('log')
ax.legend(frameon=False, loc=2)    
ax.set_title(f'Season: {season}')
yields_string =  f'$Y(MT)={(monoterp_yield_mass*s_mono):.3f}$, $Y(IP)={(isoprene_yield_mass*s_isop):.3f}$'
plt.suptitle(f'OA distribution and estimated distribution with MT and IP  yields scaled by {s_mono} and {s_isop} respectively \n {yields_string}')
sns.despine(fig)
ax.set_xlabel('OA concentration [$\mu$gm$^{-3}$]')
fn = f'Plots/noresm_yield_adjust_ym{s_mono}_yiso{s_isop}.pdf'
fig.tight_layout()
fig.savefig(fn)

# %%
s_mono = 1
s_isop = 0.1
for station in dic_df.keys():
    print(station)
    _df = dic_df[station]['NorESM'].resample('d').mean()
    _df['prodOA_mono'] = _df['SFmonoterp']*monoterp_yield_mass
    _df['prodOA_iso'] = _df['SFisoprene']*isoprene_yield_mass

    _df['monoterp_fraction_total'] =_df['prodOA_mono']/(_df['prodOA_mono']+ _df['prodOA_iso'])
    _df['isoprene_fraction_total'] =_df['prodOA_iso']/(_df['prodOA_mono']+ _df['prodOA_iso'])    
    _df['scale_OA'] =_df['monoterp_fraction_total']*s_mono + _df['isoprene_fraction_total']*s_isop
    _df['OA_new'] = _df['OA']*(_df['monoterp_fraction_total']*s_mono + _df['isoprene_fraction_total']*s_isop)
    dic_df[station]['NorESM'] = _df
    
    

    
    
fig, axs = plt.subplots(2, sharex=True, figsize=(10,10,))
ax = axs[0]
station = 'SMR'
season = dic_season[station]
#for k in dic_df[station].keys():
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA')
_df = _df[_df.index.month.isin(months)]
_df = _df.dropna()

for k in dic_df[station].keys():
    m = _df[k].median()
    label = f'{station},{k} median={m}'
    _df[k].plot.hist( bins = np.logspace(-2,2), alpha=0.3, label=label, color = dic_colors[k][station],
                        linewidth=3,
                     ax = ax,
                       histtype='stepfilled', linestyle=':')

k= 'NorESM'
#_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA_new')
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA_new')
_df = _df[_df.index.month.isin(months)]
m = _df[k].median()
_df = _df.dropna()

label = f'{station},{k}, new median={m}'
_df[k].plot.hist( bins = np.logspace(-2,2), alpha=1, label=label, color =  dic_colors[k][station],
                        linewidth=4,
                     ax = ax,
                 
                       histtype='step', )
ax.set_xscale('log')
ax.set_title(f'Season: {season}')

ax.legend(frameon=False, loc=2)    

ax = axs[1]

station = 'ATTO'
season = dic_season[station]
months = seasons2months[season]
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA')
_df = _df[_df.index.month.isin(months)]
_df = _df.dropna()

for k in dic_df[station].keys():
    m = _df[k].median()
    label = f'{station},{k} median={m}'
    _df[k].plot.hist( bins = np.logspace(-2,2), alpha=.4, label=label, color = dic_colors[k][station],
                        linewidth=3,
                     ax = ax,
                     
                       histtype='stepfilled', linestyle='--')

k= 'NorESM'
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA_new')
_df = _df[_df.index.month.isin(months)]
_df = _df.dropna()
m = _df[k].median()
label = f'{station},{k}, new median={m}'
_df[k].plot.hist( bins = np.logspace(-2,2), alpha=1, label=label, color = dic_colors[k][station],
                        linewidth=4,
                     ax = ax,
                 
                       histtype='step', )

ax.set_xscale('log')
ax.legend(frameon=False, loc=2)    
ax.set_title(f'Season: {season}')
yields_string =  f'$Y(MT)={(monoterp_yield_mass*s_mono):.3f}$, $Y(IP)={(isoprene_yield_mass*s_isop):.3f}$'
plt.suptitle(f'OA distribution and estimated distribution with MT and IP  yields scaled by {s_mono} and {s_isop} respectively \n {yields_string}')
sns.despine(fig)
ax.set_xlabel('OA concentration [$\mu$gm$^{-3}$]')
fn = f'Plots/noresm_yield_adjust_ym{s_mono}_yiso{s_isop}.pdf'
fig.tight_layout()
fig.savefig(fn)

# %%
s_mono = 1
s_isop = 0
for station in dic_df.keys():
    print(station)
    _df = dic_df[station]['NorESM'].resample('d').mean()
    _df['prodOA_mono'] = _df['SFmonoterp']*monoterp_yield_mass
    _df['prodOA_iso'] = _df['SFisoprene']*isoprene_yield_mass

    _df['monoterp_fraction_total'] =_df['prodOA_mono']/(_df['prodOA_mono']+ _df['prodOA_iso'])
    _df['isoprene_fraction_total'] =_df['prodOA_iso']/(_df['prodOA_mono']+ _df['prodOA_iso'])    
    _df['scale_OA'] =_df['monoterp_fraction_total']*s_mono + _df['isoprene_fraction_total']*s_isop
    _df['OA_new'] = _df['OA']*(_df['monoterp_fraction_total']*s_mono + _df['isoprene_fraction_total']*s_isop)
    dic_df[station]['NorESM'] = _df
    
    

    
    
fig, axs = plt.subplots(2, sharex=True, figsize=(10,10,))
ax = axs[0]
station = 'SMR'
season = dic_season[station]
#for k in dic_df[station].keys():
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA')
_df = _df[_df.index.month.isin(months)]
_df = _df.dropna()

for k in dic_df[station].keys():
    m = _df[k].median()
    label = f'{station},{k} median={m}'
    _df[k].plot.hist( bins = np.logspace(-2,2), alpha=0.3, label=label, color = dic_colors[k][station],
                        linewidth=3,
                     ax = ax,
                       histtype='stepfilled', linestyle=':')

k= 'NorESM'
#_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA_new')
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA_new')
_df = _df[_df.index.month.isin(months)]
m = _df[k].median()
_df = _df.dropna()

label = f'{station},{k}, new median={m}'
_df[k].plot.hist( bins = np.logspace(-2,2), alpha=1, label=label, color =  dic_colors[k][station],
                        linewidth=4,
                     ax = ax,
                 
                       histtype='step', )
ax.set_xscale('log')
ax.set_title(f'Season: {season}')

ax.legend(frameon=False, loc=2)    

ax = axs[1]

station = 'ATTO'
season = dic_season[station]
months = seasons2months[season]
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA')
_df = _df[_df.index.month.isin(months)]
_df = _df.dropna()

for k in dic_df[station].keys():
    m = _df[k].median()
    label = f'{station},{k} median={m}'
    _df[k].plot.hist( bins = np.logspace(-2,2), alpha=.4, label=label, color = dic_colors[k][station],
                        linewidth=3,
                     ax = ax,
                     
                       histtype='stepfilled', linestyle='--')

k= 'NorESM'
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA_new')
_df = _df[_df.index.month.isin(months)]
_df = _df.dropna()
m = _df[k].median()
label = f'{station},{k}, new median={m}'
_df[k].plot.hist( bins = np.logspace(-2,2), alpha=1, label=label, color = dic_colors[k][station],
                        linewidth=4,
                     ax = ax,
                 
                       histtype='step', )

ax.set_xscale('log')
ax.legend(frameon=False, loc=2)    
ax.set_title(f'Season: {season}')
yields_string =  f'$Y(MT)={monoterp_yield_mass*s_mono}$, $Y(IP)={isoprene_yield_mass*s_isop}$'
plt.suptitle(f'OA distribution and estimated distribution with MT and IP  yields scaled by {s_mono} and {s_isop} respectively \n {yields_string}')
sns.despine(fig)
ax.set_xlabel('OA concentration [$\mu$gm$^{-3}$]')
fn = f'Plots/noresm_yield_adjust_ym{s_mono}_yiso{s_isop}.pdf'
fig.tight_layout()
fig.savefig(fn)

# %%
s_mono = .4
s_isop = 0
for station in dic_df.keys():
    print(station)
    _df = dic_df[station]['NorESM'].resample('d').mean()
    _df['prodOA_mono'] = _df['SFmonoterp']*monoterp_yield_mass
    _df['prodOA_iso'] = _df['SFisoprene']*isoprene_yield_mass

    _df['monoterp_fraction_total'] =_df['prodOA_mono']/(_df['prodOA_mono']+ _df['prodOA_iso'])
    _df['isoprene_fraction_total'] =_df['prodOA_iso']/(_df['prodOA_mono']+ _df['prodOA_iso'])    
    _df['scale_OA'] =_df['monoterp_fraction_total']*s_mono + _df['isoprene_fraction_total']*s_isop
    _df['OA_new'] = _df['OA']*(_df['monoterp_fraction_total']*s_mono + _df['isoprene_fraction_total']*s_isop)
    dic_df[station]['NorESM'] = _df
    
    

    
    
fig, axs = plt.subplots(2, sharex=True, figsize=(10,10,))
ax = axs[0]
station = 'SMR'
season = dic_season[station]
#for k in dic_df[station].keys():
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA')
_df = _df[_df.index.month.isin(months)]
_df = _df.dropna()

for k in dic_df[station].keys():
    m = _df[k].median()
    label = f'{station},{k} median={m}'
    _df[k].plot.hist( bins = np.logspace(-2,2), alpha=0.3, label=label, color = dic_colors[k][station],
                        linewidth=3,
                     ax = ax,
                       histtype='stepfilled', linestyle=':')

k= 'NorESM'
#_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA_new')
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA_new')
_df = _df[_df.index.month.isin(months)]
m = _df[k].median()
_df = _df.dropna()

label = f'{station},{k}, new median={m}'
_df[k].plot.hist( bins = np.logspace(-2,2), alpha=1, label=label, color =  dic_colors[k][station],
                        linewidth=4,
                     ax = ax,
                 
                       histtype='step', )
ax.set_xscale('log')
ax.set_title(f'Season: {season}')

ax.legend(frameon=False, loc=2)    

ax = axs[1]

station = 'ATTO'
season = dic_season[station]
months = seasons2months[season]
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA')
_df = _df[_df.index.month.isin(months)]
_df = _df.dropna()

for k in dic_df[station].keys():
    m = _df[k].median()
    label = f'{station},{k} median={m}'
    _df[k].plot.hist( bins = np.logspace(-2,2), alpha=.4, label=label, color = dic_colors[k][station],
                        linewidth=3,
                     ax = ax,
                     
                       histtype='stepfilled', linestyle='--')

k= 'NorESM'
_df= merge_two_sources(dic_df[station]['Observations'],dic_df[station]['NorESM'], 'Observations','NorESM','OA', 'OA_new')
_df = _df[_df.index.month.isin(months)]
_df = _df.dropna()
m = _df[k].median()
label = f'{station},{k}, new median={m}'
_df[k].plot.hist( bins = np.logspace(-2,2), alpha=1, label=label, color = dic_colors[k][station],
                        linewidth=4,
                     ax = ax,
                 
                       histtype='step', )

ax.set_xscale('log')
ax.legend(frameon=False, loc=2)    
ax.set_title(f'Season: {season}')
yields_string =  f'$Y(MT)={monoterp_yield_mass*s_mono}$, $Y(IP)={isoprene_yield_mass*s_isop}$'
plt.suptitle(f'OA distribution and estimated distribution with MT and IP  yields scaled by {s_mono} and {s_isop} respectively \n {yields_string}')
sns.despine(fig)
ax.set_xlabel('OA concentration [$\mu$gm$^{-3}$]')
fn = f'Plots/noresm_yield_adjust_ym{s_mono}_yiso{s_isop}.pdf'
fig.tight_layout()
fig.savefig(fn)

# %%
