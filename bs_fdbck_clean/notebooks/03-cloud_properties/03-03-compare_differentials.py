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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

# %%
import scienceplots
import scienceplots
plt.style.use([
    'default',
    # 'science',
    'acp',
    # 'sp-grid',
    'no-black',
    'no-latex',
    'illustrator-safe'
])


# %%
def make_fn(variable, season1, season2, comment='', relplot = False, distplot=False):
    _x = variable.split('(')[0]
    #_y = v_y.split('(')[0]
    f = f'delta_percentile_{season1}_{season2}_{_x}{comment}.png'
    if relplot:
        return plot_path_relplot/f
    if distplot:
        return plot_path_distplot/f

    return plot_path /f


plot_path = Path('Plots')
plot_path.mkdir(parents=True, exist_ok=True) 


# %% tags=[]
models = ['ECHAM-SALSA','NorESM', 'EC-Earth', 'UKESM']
mod2cases = {'ECHAM-SALSA' : ['SALSA_BSOA_feedback'],
             'NorESM' : ['OsloAero_intBVOC_f09_f09_mg17_fssp'],
             'EC-Earth' : ['ECE3_output_Sara'],
             'UKESM' : ['AEROCOMTRAJ'],
            }
di_mod2cases = mod2cases.copy()


# %%
models_and_obs =  models + ['Observations'] 
models_and_obs

# %%
season_atto = 'FMA'
season_smr = 'JA'


# %%
dic_station2season = dict(
    SMR=season_smr,
    ATTO=season_atto,
)
dic_station2nicename= dict(
    SMR='SMEAR-II',
    ATTO='ATTO'
)

# %%
f_smr = f'03-02-SMR/Plots/cloud_props__percentile_OA_OA_perc_{season_smr}.csv'
print(f_smr)


# %%
f_atto =f'03-01-ATTO/Plots/cloud_props__percentile_OA_OA_perc_{season_atto}.csv'
print(f_atto)


# %%
fit_overview = '../02-T2OA_OA2Nx/Plots/Both_stations/all_fits.csv'

# %%
df_smr = pd.read_csv(f_smr, index_col=0)

# %%
df_smr['station']= 'SMEAR-II'
df_smr['diff_med'] = df_smr['med_high'] - df_smr['med_low'] 

# %%
df_smr.loc['Observations','med_high']

# %%
df_atto =  pd.read_csv(f_atto, index_col=0)

# %%
df_atto['station'] = 'ATTO'
df_atto['diff_med'] = df_atto['med_high'] - df_atto['med_low'] 

# %%
df_atto.reset_index()

# %%
df_tot = pd.concat([df_smr.reset_index(), df_atto.reset_index()])

df_tot = df_tot.rename({'index':'Source'}, axis=1)

# %%
from bs_fdbck.util.plot.BSOA_plots import cdic_model


# %%
cols = [cdic_model[mo] for mo in df_tot['Source'].unique()]
order_sources = df_tot['Source'].unique()

# %%
fig, ax = plt.subplots()
sns.barplot(y='diff',data=df_tot, hue='Source',x='station', palette=cols)

ax.set_ylabel('$\Delta$OA 66$^{th}$ to 33$^{rd}$ perc. [$\mu$gm$^{-3}$]')
ax.grid()
ax.set_xlabel('')

ax.set_title('Difference in OA between 66th and 33rd percentile')

fn = make_fn('OA', season_smr,season_atto)
fig.savefig(fn.with_suffix('.pdf'))


# %%
fig, ax = plt.subplots()
sns.barplot(y='diff',data=df_tot, hue='Source',x='station', palette=cols)

ax.set_ylabel('$\Delta$OA 66$^{th}$ to 33$^{rd}$ perc. [$\mu$gm$^{-3}$]')
ax.grid()
ax.set_xlabel('')

ax.set_title('Difference in OA between 66th and 33rd percentile')

fn = make_fn('OA', season_smr,season_atto)
fig.savefig(fn.with_suffix('.pdf'))


# %%
fig, ax = plt.subplots()
sns.barplot(y='diff_med',data=df_tot, hue='Source',x='station', palette=cols)

ax.set_ylabel('$\Delta$OA median above 66$^{th}$ to below 33$^{rd}$ perc. [$\mu$gm$^{-3}$]')
ax.grid()
ax.set_xlabel('')

ax.set_title('Difference in median high OA and median low OA')

fn = make_fn('OA', season_smr,season_atto)
fig.savefig(fn.with_suffix('.pdf'))


# %% [markdown]
# ### Do weighted average 

# %%
station_dic_dir = {'ATTO':'03-01-ATTO', 'SMR':'03-02-SMR'}  
def make_fn2(case, v_x, v_y, season, station, comment='',  ):
    plot_path = Path(f'{station_dic_dir[station]}/Plots/')
    print(v_x)
    print(v_y)
    
    _x = v_x.split('(')[0]
    _y = v_y.split('(')[0]
    f = f'cloud_props_{comment}_{case}_{_x}_{_y}_{season}.png'

    return plot_path /f



# %%
v = 'COT'
station = 'SMR'
dic_v_st_src = dict()

for station in ['SMR','ATTO']:
    st_nicename = dic_station2nicename[station]
    dic_v_med = dict()

    for s in ['Observations', 'NorESM','ECHAM-SALSA']:
        fn = make_fn2('sample_stats', 'OA', v, dic_station2season[station],station,comment=s).with_suffix('.csv')
        #print(fn.exists())
        print(fn)
        dic_v_med[s] = pd.read_csv(fn, index_col=0)

    #_dic_smr = dict()
    print(station)
    dic_v_st_src[st_nicename] = dict()
    for source in dic_v_med.keys():
        #print(source)
        _df = dic_v_med[source]
        delta_v_weigthed = (_df[v]*_df['n_tot']).sum()/ _df['n_tot'].sum()

        dic_v_st_src[st_nicename][source] = delta_v_weigthed
        print(delta_v_weigthed)
    print(dic_v_st_src)
    

for source in ['UKESM','EC-Earth']:
    for st in ['SMR','ATTO']:
        st_nn = dic_station2nicename[st]
        dic_v_st_src[st_nn][source] =0

_dic_delta_v_weigthed =dic_v_st_src

_dic_delta_v_weigthed

df_integrated_COT = (pd.DataFrame(_dic_delta_v_weigthed)
                     .stack()
                     .reset_index()
                     .rename({'level_1':'station','level_0':'Source',0:f'd{v}'},axis=1)
                    )



# %%
v = 'r_eff'
station = 'SMR'
dic_v_st_src = dict()

for station in ['SMR','ATTO']:
    st_nicename = dic_station2nicename[station]
    dic_v_med = dict()

    for s in models_and_obs:
        fn = make_fn2('sample_stats', 'OA', v, dic_station2season[station],station,comment=s).with_suffix('.csv')
        #print(fn.exists())
        print(fn)
        dic_v_med[s] = pd.read_csv(fn, index_col=0)

    #_dic_smr = dict()
    print(station)
    dic_v_st_src[st_nicename] = dict()
    for source in dic_v_med.keys():
        #print(source)
        _df = dic_v_med[source]
        delta_v_weigthed = (_df[v]*_df['n_tot']).sum()/ _df['n_tot'].sum()

        dic_v_st_src[st_nicename][source] = delta_v_weigthed
        print(delta_v_weigthed)
    print(dic_v_st_src)


_dic_delta_v_weigthed =dic_v_st_src

_dic_delta_v_weigthed

df_integrated_r_eff = (pd.DataFrame(_dic_delta_v_weigthed)
                     .stack()
                     .reset_index()
                     .rename({'level_1':'station','level_0':'Source',0:f'd{v}'},axis=1)
                    )



# %%

# %%
df_integrated_cloud_prop = pd.merge(df_integrated_COT, df_integrated_r_eff, on=['Source','station'])

# %%
df_together = pd.merge(df_tot.rename({'diff_med':'dOA'},axis=1), df_integrated_cloud_prop, on = ['station', 'Source'],)

df_together['dCOT/dOA'] = df_together['dCOT']/df_together['dOA']
df_together['dr_eff/dOA'] = df_together['dr_eff']/df_together['dOA']

# %%
df_together

# %%
df_together['dCOT/dOA'].plot()

# %%
df_together['dCOT/dOA'].plot()

# %% [markdown] tags=[]
# ## Lets assume $\Delta T$ around 3 degrees

# %% [markdown] tags=[]
# ### Read in data

# %%
dic_df_med_station=dict()


# %%
from bs_fdbck.constants import path_measurement_data

select_station = 'SMR'
postproc_data = path_measurement_data /'model_station'/select_station
postproc_data_obs = path_measurement_data /select_station/'processed'

fn_obs_comb_data_full_time =postproc_data_obs /'SMEAR_data_comb_hourly.csv'

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
        
        # dic_df_mod_case[mod][ca].to_csv(fn_out)

df_obs = pd.read_csv(fn_obs_comb_data_full_time,index_col=0)

df_obs_rename = df_obs.rename({'Org_STP':'OA', 'HYY_META.T168':'T_C'}, axis=1)
vars_obs = ['OA', 'N100','N50','N200','T_C']

df_obs_rename = df_obs_rename[vars_obs]

df_obs_rename['some_obs_missing'] = df_obs_rename[vars_obs].isnull().any(axis=1)
df_obs_rename.index = pd.to_datetime(df_obs_rename.index)

dic_df_pre['Observations']=dict()
dic_df_pre['Observations']['Observations'] = df_obs_rename#_rename


### Save result in dictionary

dic_df_mod_case = dic_df_pre.copy()

#### Merge with observations:


df_for_merge = df_obs_rename[['OA','N100', 'some_obs_missing']].rename({'OA':'obs_OA','N100':'obs_N100',},axis=1)

for mod in dic_df_mod_case.keys():
    if mod=='Observations':
        dic_df_mod_case[mod][mod] = dic_df_mod_case[mod][mod].rename({'some_obs_missing':'some_obs_missing_x'}, axis=1)
    print(mod)
    for ca in dic_df_mod_case[mod].keys():
        dic_df_mod_case[mod][ca] = pd.merge(dic_df_pre[mod][ca], df_for_merge ,right_on='time', left_on='time', how='outer')
        dic_df_mod_case[mod][ca]['year'] = dic_df_mod_case[mod][ca].index.year

dic_mod2case = mod2cases

## Compute daily medians:

dic_df_med = dict()
for mo in dic_df_mod_case.keys():
    for ca in dic_df_mod_case[mo].keys():
        if len(dic_df_mod_case[mo].keys())>1:
            use_name = f'{mo}_{ca}'
        else:
            use_name = mo
            
        _df = dic_df_mod_case[mo][ca]
        
        _df = _df[_df['some_obs_missing']==False]
        dic_df_med[use_name] = _df.resample('D').median()
        
        
dic_df_med_station[select_station] = dic_df_med.copy()


# %%

# %%
from bs_fdbck.constants import path_measurement_data



# %%
fn_obs_comb_data_full_time =postproc_data_obs /'ATTO_data_comb_hourly.nc'

# %%
import xarray as xr

# %%
# !ls /proj/bolinc/users/x_sarbl/analysis/BS-FDBCK/Data/model_station/ATTO/

# %% [markdown]
# ### Set level to read

# %%
dic_model_level = dict(
    SMR= -1,
    ATTO=-1,
)

# %%
from bs_fdbck.constants import path_measurement_data

select_station = 'ATTO'
postproc_data = path_measurement_data /'model_station'/select_station
postproc_data_obs = path_measurement_data /select_station/'processed'

fn_obs_comb_data_full_time =postproc_data_obs /'ATTO_data_comb_hourly.nc'

dic_df_pre=dict()
for mod in models:
    print(mod)
    dic_df_pre[mod] = dict()
    for ca in mod2cases[mod]:
        print(mod, ca)
        fn_out = postproc_data/f'{select_station}_station_{mod}_{ca}_ilev{dic_model_level[select_station]}.csv'
        print(fn_out)
        dic_df_pre[mod][ca] = pd.read_csv(fn_out, index_col=0)
        dic_df_pre[mod][ca].index = pd.to_datetime(dic_df_pre[mod][ca].index)
        #dic_df_mod_case[mod][ca].to_csv(fn_out)


        
ds_ATTO = xr.open_dataset(fn_obs_comb_data_full_time, engine='netcdf4')

ds_ATTO = ds_ATTO.sel(time_traj=0)
vars_obs = ['OA', 'N100-500','N50-500','N200-500','temperature']

varl = ['Pressure_reanalysis', 'Potential_Temperature_reanalysis', 'Temperature_reanalysis', 'Rainfall', 'Mixing_Depth', 'Relative_Humidity', 'Specific_Humidity_reanalysis',
 'Mixing_Ratio','Solar_Radiation', 'condensation_sink', 'N50-500', 'N100-500', 'N200-500', 'timeUTC-3', 'pressure', 'temperature', 'humidity',#'wind_dir',
        'precip',#'Solar_inc_Wm2',
        #'Solar_out_Wm2',
        #'wind_speed','wind_speed_v',
        'OA'
]

df_ATTO = ds_ATTO[varl].drop('time_traj').to_dataframe()
df_ATTO['some_obs_missing'] = df_ATTO[vars_obs].isnull().any(axis=1)
df_ATTO_obs_rename = df_ATTO.rename({'Org':'OA','temperature':'T_C'}, axis=1)
dic_df_pre['Observations'] = dict()
dic_df_pre['Observations']['Observations'] = df_ATTO_obs_rename
dic_df_pre['Observations']['Observations'].index = pd.to_datetime(dic_df_pre['Observations']['Observations'].index)


dic_df_ATTO = dic_df_pre.copy()

dic_df_mod_case = dic_df_pre.copy()

df_for_merge = df_ATTO[['OA','N100-500', 'some_obs_missing']].rename({'OA':'obs_OA','N100-500':'obs_N100-500',},axis=1)

for mod in dic_df_mod_case.keys():
    if mod=='Observations':
        dic_df_mod_case[mod][mod] = dic_df_mod_case[mod][mod].rename({'some_obs_missing':'some_obs_missing_x'}, axis=1)
    print(mod)
    for ca in dic_df_mod_case[mod].keys():
        dic_df_mod_case[mod][ca] = pd.merge(dic_df_pre[mod][ca], df_for_merge ,right_on='time', left_on='time', how='outer')
        dic_df_mod_case[mod][ca]['year'] = dic_df_mod_case[mod][ca].index.year

dic_df_med = dict()
for mo in dic_df_mod_case.keys():
    print(mo) 
    for ca in dic_df_mod_case[mo].keys():
        if len(dic_df_mod_case[mo].keys())>1:
            use_name = f'{mo}_{ca}'
        else:
            use_name = mo
            
        _df = dic_df_mod_case[mo][ca]
        
        _df = _df[_df['some_obs_missing']==False]
        dic_df_med[use_name] = _df.resample('D').median()
        


dic_df_med_station[select_station] = dic_df_med.copy()

# %% [markdown] tags=[]
# ## Rename STP 

# %%

# %%
for st in dic_df_med_station.keys():
    for mod in dic_df_med_station[st].keys():
        print(mod)
        for v in ['OA','N100','N50','N200','N500','N100-500','N50-500','N200-500']:
            if (f'{v}_STP' in dic_df_med_station[st][mod].columns):
                if v in dic_df_med_station[st][mod].columns:
                    dic_df_med_station[st][mod] = dic_df_med_station[st][mod].drop([v], axis=1)
                    print('dropping OA in favor of OA_STP')
                dic_df_med_station[st][mod] = dic_df_med_station[st][mod].rename({f'{v}_STP':v}, axis=1)
                print(f'remaning {v}_STP to {v}')


# %%
from bs_fdbck.util.BSOA_datamanip.fits import *
from bs_fdbck.util.BSOA_datamanip.atto import season2month


# %% [markdown] tags=[]
# ### season to monthseason2month

# %%
def select_months(df, season = None, month_list=None):
    if season is not None: 
        month_list = season2month[season]
    

    df['month'] = df.index.month
    return df['month'].isin(month_list)

# %%
mod2cases['Observations']= ['Observations']


# %%
dic_med_temperature = dict()

# %%
station = 'SMR'
season=season_smr
dic_med_temperature[station] = dict()
dic_df_med = dic_df_med_station[station]
for mo in models_and_obs:
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    print(df_s['T_C'].median())
    dic_med_temperature[station][mo] = df_s['T_C'].median()

# %%
station = 'ATTO'
season=dic_station2season[station]
dic_med_temperature[station] = dict()
dic_df_med = dic_df_med_station[station]
for mo in models_and_obs:
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    print(df_s['T_C'].median())
    dic_med_temperature[station][mo] = df_s['T_C'].median()

# %% [markdown]
# ## Median OA: 

# %%
station = 'SMR'
season=dic_station2season[station]

dic_med_temperature[station] = dict()
dic_df_med = dic_df_med_station[station]
for mo in models_and_obs:
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    print(df_s['OA'].median())
    dic_med_temperature[station][mo] = df_s['T_C'].median()

# %%
station = 'ATTO'
season=dic_station2season[station]

dic_med_temperature[station] = dict()
dic_df_med = dic_df_med_station[station]
for mo in models_and_obs:
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    print(df_s['OA'].median())
    dic_med_temperature[station][mo] = df_s['T_C'].median()

# %%
station = 'ATTO'
season=dic_station2season[station]

dic_med_temperature[station] = dict()
dic_df_med = dic_df_med_station[station]
for mo in models_and_obs:
    df_s =  dic_df_med[mo]
    print(mo)
    if mo!='Observations':
        df_s['N100-500'] = df_s['N100']- df_s['N500']
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    print(df_s['N100-500'].median())
    dic_med_temperature[station][mo] = df_s['T_C'].median()

# %%
station = 'SMR'
season=dic_station2season[station]

dic_med_temperature[station] = dict()
dic_df_med = dic_df_med_station[station]
for mo in models_and_obs:
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    print(df_s['N100'].median())
    dic_med_temperature[station][mo] = df_s['T_C'].median()

# %%
176.24964/560.5083086698271

# %%
151.0/177.8

# %%
smr_start = 15 
atto_start = 25 

# %%

# %%
df_fits = pd.read_csv(fit_overview, index_col=[0,1,2,3])

# %%
df_fits.head()


# %%
def exp_mod(T, mod, station):
    print(df_fits.loc[(station, 'OA',mod,'$a\cdot \exp{(bx)}$'),'a'])
    print(df_fits.loc[(station, 'OA',mod,'$a\cdot \exp{(bx)}$'),'a'])
    print(df_fits.loc[(station, 'OA',mod,'$a\cdot \exp{(bx)}$'),'b'])
    print('Correlation:')
    corr =df_fits.loc[(station, 'OA',mod,'$a\cdot \exp{(bx)}$'),'r$^2$'] 
    print(df_fits.loc[(station, 'OA',mod,'$a\cdot \exp{(bx)}$'),'r$^2$'])
    if np.abs(corr)<0.1:
        print(f'correlation for {mod} at {station} is below .1, r2={corr}')
        return 0
    
    
    a = float(df_fits.loc[(station, 'OA',mod,'$a\cdot \exp{(bx)}$'),'a'].split('$\pm$')[0])
    b = float(df_fits.loc[(station, 'OA',mod,'$a\cdot \exp{(bx)}$'),'b'].split('$\pm$')[0])
    print(a,b)
    print(f'${a}exp({b}T)$')
    return a*np.exp(b*T)
exp_mod(15,'UKESM','ATTO-no2015/2016')


# %%
def exp_noresm_smr(T):
    return exp_mod(T, 'NorESM', 'SMR')
def exp_echam_salsa_smr(T):
    return exp_mod(T, 'ECHAM-SALSA', 'SMR')
def exp_ec_earth_smr(T):
    return 0.19*np.exp(0.14*T)
def exp_obs_smr(T):
    return 0.35*np.exp(0.13*T)


# %%
def exp_noresm_atto(T):
    return 2.3e-4*np.exp(0.4*T)
def exp_echam_salsa_atto(T):
    return 0.46*np.exp(0.05*T)
def exp_ec_earth_atto(T):
    return 0.52*np.exp(0.05*T)
def exp_obs_atto(T):
    return 8.5e-6*np.exp(0.45*T)


# %% [markdown]
# dic_exp_funcs={
#     'SMEAR-II': {'Observations':exp_obs_smr,
#               'NorESM':exp_noresm_smr,
#               'ECHAM-SALSA':exp_echam_salsa_smr,
#               'EC-Earth':exp_ec_earth_smr,
#              },
#     'ATTO':{'Observations':exp_obs_atto,
#               'NorESM':exp_noresm_atto,
#               'ECHAM-SALSA':exp_echam_salsa_atto,
#               'EC-Earth':exp_ec_earth_atto,
#              },   
# }

# %%
def exp_station_source(station, source, T=None ):
    #func = dic_exp_funcs[station][source]
    if T is None:
        T = dic_med_temperature[station][source]
    print(f'Temperature:{T}')
    
    
    return exp_mod(T, source,station)


# %% [markdown]
# ## Making exponential function 
#
# - For ECHAM-SALSA, use no 2015/2016 
# - For UKESM, check if corr>0.1 and if not set to zero. 

# %%
dic_dOA_dT =dict()
for st in ['SMR','ATTO']:

        
    dic_dOA_dT[st] = dict()
    for so in order_sources:
        print(so)
        print(st)
        T = dic_med_temperature[st]['Observations']        
        if (so=='ECHAM-SALSA' and st=='ATTO'):
            print('Using no 2015/2016 for ECHAM-SALSA at ATTO')
            st_alt = 'ATTO-no2015/2016'
            dic_dOA_dT[st][so] = (exp_station_source(st_alt, so, T+3)- exp_station_source(st_alt, so, T))/3
            continue
        dic_dOA_dT[st][so] = (exp_station_source(st, so, T+3)- exp_station_source(st, so, T))/3


# %%
df_dOA_dT = pd.DataFrame(dic_dOA_dT).stack().reset_index().rename({'level_0':'Source','level_1':'station',0:'dOA/dT'},axis=1)

# %%
df_dOA_dT = df_dOA_dT.replace('SMR','SMEAR-II')

# %%
df_dOA_dT

# %%
cols

# %%

sns.barplot(y='dOA/dT', x='station', hue='Source', data= df_dOA_dT, palette=cols)

# %%
df_together

# %%

sns.barplot(y='dCOT/dOA', x='station', hue='Source', data= df_together, palette=cols)

# %%
df_together = pd.merge(df_together, df_dOA_dT, on = ['station', 'Source'],)

# %%
df_together

# %% [markdown]
# ## Calculate dCOT/dT and dr_eff/dT

# %%
df_together['dCOT/dT'] =df_together['dOA/dT']*df_together['dCOT/dOA'] 
df_together['dr_eff/dT'] =df_together['dOA/dT']*df_together['dr_eff/dOA'] 


# %%

# %%

sns.barplot(y='dCOT/dT',x='station', hue='Source', data= df_together, palette=cols)

# %%

sns.barplot(y='dr_eff/dT',x='station', hue='Source', data= df_together, palette=cols)

# %%
df_together

# %%
fig, axs = plt.subplots(3,1, figsize=[5,6], sharex=True)
sns.barplot(y='dOA/dT', x='station', hue='Source', data= df_together, palette=cols, ax = axs[0], )
sns.barplot(y='dCOT/dOA', x='station', hue='Source', data= df_together, palette=cols, ax = axs[1])
sns.barplot(y='dCOT/dT',x='station', hue='Source', data= df_together, palette=cols, ax = axs[2])

l = axs[0].get_legend().draw_frame(False)


for ax in axs[1:]:
    ax.get_legend().set_visible(False)
    
for ax in axs:
    ax.grid(axis='y', linewidth=.2, alpha=.4)
    ax.set_xlabel('')
    
axs[0].set_title('Change in OA per change in temperature')
axs[1].set_title('Change in COT per change in OA')
axs[2].set_title('Change in COT per change in temperature')



fn = make_fn('COT2T',season_smr, season_atto,comment='whole_feedback_comparison')
print(fn)
fig.tight_layout()
sns.despine(fig)
fig.savefig(fn.with_suffix('.pdf'))
#axs[0].set_title('Organic aerosol per change in temperature')

# %%
fig, axs = plt.subplots(3,1, figsize=[5,6], sharex=True)
sns.barplot(y='dOA/dT', x='station', hue='Source', data= df_together, palette=cols, ax = axs[0], )
sns.barplot(y='dCOT/dOA', x='station', hue='Source', data= df_together, palette=cols, ax = axs[1])
sns.barplot(y='dCOT/dT',x='station', hue='Source', data= df_together, palette=cols, ax = axs[2])

l = axs[0].get_legend().draw_frame(False)


for ax in axs[1:]:
    ax.get_legend().set_visible(False)
    
for ax in axs:
    ax.grid(axis='y', linewidth=.2, alpha=.4)
    ax.set_xlabel('')
    
axs[0].set_title('Change in OA per change in temperature')
axs[1].set_title('Change in COT per change in OA')
axs[2].set_title('Change in COT per change in temperature')



fn = make_fn('COT2T',season_smr, season_atto,comment='whole_feedback_comparison')
print(fn)
fig.tight_layout()
sns.despine(fig)
fig.savefig(fn.with_suffix('.pdf'))
#axs[0].set_title('Organic aerosol per change in temperature')

# %%
fig, axs = plt.subplots(3,1, figsize=[5,6], sharex=True)
sns.barplot(y='dOA/dT', x='station', hue='Source', data= df_together, palette=cols, ax = axs[0], )
sns.barplot(y='dCOT/dOA', x='station', hue='Source', data= df_together, palette=cols, ax = axs[1])
sns.barplot(y='dCOT/dT',x='station', hue='Source', data= df_together, palette=cols, ax = axs[2])

l = axs[0].get_legend().draw_frame(False)


for ax in axs[1:]:
    ax.get_legend().set_visible(False)
    
for ax in axs:
    ax.grid(axis='y', linewidth=.2, alpha=.4)
    ax.set_xlabel('')
    
axs[0].set_title('Change in OA per change in temperature')
axs[1].set_title('Change in COT per change in OA')
axs[2].set_title('Change in COT per change in temperature')

axs[0].set_ylabel('dOA/dT [$\mu$gm$^{-3}$K$^{-1}$]')
axs[1].set_ylabel('dCOT/dOA [($\mu$gm$^{-3}$)$^{-1}$]')
axs[2].set_ylabel('dCOT/dT [K$^{-1}$]')


fn = make_fn('COT2T',season_smr, season_atto,comment='whole_feedback_comparison')
print(fn)
fig.tight_layout()
sns.despine(fig)
fig.savefig(fn.with_suffix('.pdf'))
#axs[0].set_title('Organic aerosol per change in temperature')

# %%
fig, axs = plt.subplots(3,1, figsize=[5,6], sharex=True)
g = sns.barplot(y='dOA/dT', x='station', hue='Source', 
            data= df_together, palette=cols, ax = axs[0], )
sns.barplot(y='dr_eff/dOA', x='station', hue='Source', 
            data= df_together, palette=cols, ax = axs[1])
sns.barplot(y='dr_eff/dT',x='station', hue='Source', 
            data= df_together, palette=cols, ax = axs[2])

#l = axs[0].get_legend().draw_frame(False)
axs[-1].legend( ncol = 2, frameon=False)

for ax in axs[:-1]:
    ax.get_legend().set_visible(False)

    
for ax in axs:
    ax.grid(axis='y', linewidth=.2, alpha=.4)
    ax.set_xlabel('')
    
axs[0].set_title('Change in OA per change in temperature')
axs[1].set_title('Change in COT per change in OA')
axs[2].set_title('Change in COT per change in temperature')

axs[0].set_ylabel('dOA/dT [$\mu$gm$^{-3}$K$^{-1}$]')
axs[1].set_ylabel('dCOT/dOA [($\mu$gm$^{-3}$)$^{-1}$]')
axs[2].set_ylabel('dr$_{\mathrm{eff}}$/dT [$\mu$mK$^{-1}$]')


fn = make_fn('r_eff2T',season_smr, season_atto,comment='whole_feedback_comparison')
print(fn)
fig.tight_layout()
sns.despine(fig)
fig.savefig(fn.with_suffix('.pdf'))
#axs[0].set_title('Organic aerosol per change in temperature')

# %%
df_together[df_together['station']==dic_station2nicename['SMR']]

# %%
fig, axs = plt.subplots(3,2, figsize=[7,6], sharex='col')
for st, ax in zip(['SMR','ATTO'],axs[0,:]):
    _df = df_together[df_together['station']==dic_station2nicename[st]]
    g = sns.barplot(y='dOA/dT', x='station', hue='Source', 
            data= _df, palette=cols, ax = ax, )
for st, ax in zip(['SMR','ATTO'],axs[1,:]):
    _df = df_together[df_together['station']==dic_station2nicename[st]]
    sns.barplot(y='dr_eff/dOA', x='station', hue='Source', 
            data= _df, palette=cols, ax = ax)
for st, ax in zip(['SMR','ATTO'],axs[2,:]):
    _df = df_together[df_together['station']==dic_station2nicename[st]]
    
    sns.barplot(y='dr_eff/dT',x='station', hue='Source', 
            data= _df, palette=cols, ax = ax)

#l = axs[0].get_legend().draw_frame(False)

for ax in axs.flatten():
    ax.get_legend().set_visible(False)

axs.flatten()[-1].legend( ncol = 2, frameon=False)
    
for ax in axs.flatten():
    ax.grid(axis='y', linewidth=.2, alpha=.4)
    ax.set_xlabel('')
    

axs[0,0].set_title('Change in OA per change in temperature')
axs[1,0].set_title('Change in $r_{\mathrm{eff}}$ per change in OA')
axs[2,0].set_title('Change in $r_{\mathrm{eff}}$ per change in temperature')

axs[0,0].set_ylabel('dOA/dT [$\mu$gm$^{-3}$K$^{-1}$]')
axs[1,0].set_ylabel('d$r_{\mathrm{eff}}$/dOA [($\mu$m ($\mu$gm$^{3}$)$^{-1}$]')
#axs[2,0].set_ylabel('d$r_{eff}$/dT [K$^{-1}$]')
axs[2,0].set_ylabel('dr$_{\mathrm{eff}}$/dT [$\mu$mK$^{-1}$]')

axs[0,1].set_ylabel('')
axs[1,1].set_ylabel('')
axs[2,1].set_ylabel('')







fn = make_fn('r_eff2T',season_smr, season_atto,comment='whole_feedback_comparison')
print(fn)
fig.tight_layout()
sns.despine(fig)
fig.savefig(fn.with_suffix('.pdf'))
#axs[0].set_title('Organic aerosol per change in temperature')

# %%
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
fig = plt.figure()

gs1 = GridSpec(3, 4,)# left=0.05, right=0.48, wspace=0.05)
ax1 = fig.add_subplot(gs1[0, 1:-1])
ax2 = fig.add_subplot(gs1[1, :2])
ax3 = fig.add_subplot(gs1[2, :2])
axs_left = [ax2,ax3]
ax4 = fig.add_subplot(gs1[1, 2:])
ax5 = fig.add_subplot(gs1[2, 2:])
axs_right = [ax4,ax5]
plt.tight_layout()


# %%
#fig, axs_sup = plt.subplots(3,2, figsize=[10,6], sharex=True)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=[10,6], )

gs1 = GridSpec(3, 4,)# left=0.05, right=0.48, wspace=0.05)
ax1 = fig.add_subplot(gs1[0, 1:-1])
ax2 = fig.add_subplot(gs1[1, :2])
ax3 = fig.add_subplot(gs1[2, :2])
axs_left = [ax1, ax2,ax3]
ax4 = fig.add_subplot(gs1[1, 2:])
ax5 = fig.add_subplot(gs1[2, 2:])
axs_right = [ax1,ax4,ax5]
plt.tight_layout()








axs = axs_left
g = sns.barplot(y='dOA/dT', x='station', hue='Source', 
            data= df_together, palette=cols, ax = ax1, )
sns.barplot(y='dCOT/dOA', x='station', hue='Source', 
            data= df_together, palette=cols, ax = axs[1])
sns.barplot(y='dCOT/dT',x='station', hue='Source', 
            data= df_together, palette=cols, ax = axs[2])

#l = axs[0].get_legend().draw_frame(False)
axs[-1].legend( ncol = 2, frameon=False)

for ax in axs[:-1]:
    ax.get_legend().set_visible(False)

    
for ax in axs:
    ax.grid(axis='y', linewidth=.2, alpha=.4)
    ax.set_xlabel('')
    
axs[0].set_title('Change in OA per change in temperature')
axs[1].set_title('Change in COT per change in OA')
axs[2].set_title('Change in COT per change in temperature')

axs[0].set_ylabel('dOA/dT [$\mu$gm$^{-3}$K$^{-1}$]')
axs[1].set_ylabel('dCOT/dOA [($\mu$gm$^{-3}$)$^{-1}$]')
axs[2].set_ylabel('dCOT/dT [K$^{-1}$]')




### SECOND ROW:
axs = axs_right

g = sns.barplot(y='dOA/dT', x='station', hue='Source', 
            data= df_together, palette=cols, ax = axs[0], )
sns.barplot(y='dr_eff/dOA', x='station', hue='Source', 
            data= df_together, palette=cols, ax = axs[1])
sns.barplot(y='dr_eff/dT',x='station', hue='Source', 
            data= df_together, palette=cols, ax = axs[2])

#l = axs[0].get_legend().draw_frame(False)
axs[-1].legend( ncol = 2, frameon=False)

for ax in axs[:]:
    ax.get_legend().set_visible(False)

    
for ax in axs:
    ax.grid(axis='y', linewidth=.2, alpha=.6)
    ax.set_xlabel('')
    
axs[0].set_title('Change in OA per change in temperature')
axs[1].set_title('Change in $r_{eff}$ per change in OA')
axs[2].set_title('Change in $r_{eff}$ per change in temperature')

axs[0].set_ylabel('dOA/dT [$\mu$gm$^{-3}$K$^{-1}$]')
axs[1].set_ylabel('d$r_{eff}$/dOA [($\mu$m ($\mu$gm$^{3}$)$^{-1}$]')
axs[2].set_ylabel('d$r_{eff}$/dT [K$^{-1}$]')










fn = make_fn('COTr_eff2T',season_smr, season_atto,comment='whole_feedback_comparison')
print(fn)
fig.tight_layout()
sns.despine(fig)
fig.savefig(fn.with_suffix('.pdf'))
#axs[0].set_title('Organic aerosol per change in temperature')

# %%

# %%

# %%

# %%

# %%
