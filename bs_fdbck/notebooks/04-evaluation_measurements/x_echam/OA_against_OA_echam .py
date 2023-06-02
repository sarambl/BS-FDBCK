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
fn1 = '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201207_emi_isop_bio.nc'
fn2 = '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201207_emi_monot_bio.nc'


# %%
ds1 = xr.open_dataset(fn1)

# %%
ds2 = xr.open_dataset(fn2)

# %%
ds1['emi_isop_bio'].groupby(ds1['time.hour']).mean().sel(lat=61.85, lon = 24.28, method='nearest').plot()

# %%
ds2['emi_monot_bio'].mean('time').plot(robust=True)

# %%
ds1['emi_isop_bio'].mean('time').plot(robust=True)

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
plot_path = Path('Plots')


# %%
def make_fn_eval(case,_type):
    #_x = v_x.split('(')[0]
    #_y = v_y.split('(')[0]
    f = f'evalOA_echam_{case}_{_type}.png'
    return plot_path /f


# %%
plot_path.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## EBAS OA timeseries:

    # %%
    download_link= 'http://ebas-data.nilu.no/DataSets.aspx?stations=FI0050R&InstrumentTypes=aerosol_mass_spectrometer&fromDate=1970-01-01&toDate=2021-12-31'

# %% [markdown] tags=[]
# ## Read in model data. 

# %%
model_lev_i=-1

# %%
models = ['ECHAM-SALSA','NorESM']

di_mod2cases = dict()
#for mod in models:
#    di_mod2cases[mod]=dict()

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
from_t = '2012-01-01'
to_t = '2015-01-01'

# %%
from_t2 = '2015-01-01'
to_t2 = '2019-01-01'

# %% [markdown] tags=[]
# ## Cases:

# %%
dic_mod_ca = dict()
dic_df_mod_case = dict()
dic_dfsm_mod_case = dict()

# %% [markdown] tags=[]
# ### LOAD ECHAM SALSA

# %%

# %% [markdown]
# ## Settings:

# %%
# %%
from_t = '2012-01-01'
to_t = '2019-01-01'


# %%



case_name = 'SALSA_BSOA_feedback'
case_name_echam = 'SALSA_BSOA_feedback'
from_time = '2012-01'
to_time = '2012-02'
time_res = 'hour'
space_res='locations'
model_name='ECHAM-SALSA'


case_mod = case_name#'OsloAero_intBVOC_f19_f19_mg17_fssp'
cases_echam = [case_name]
di_mod2cases[model_name]=cases_echam

# %%
 
log.ger.info(f'TIMES:****: {from_t} {to_t}')

# %% [markdown]
# ## Variables

# %%
varl =[
      'apm',
'geom',
'airdens',
'tempair',
'uw',
'vw',
'ccn02',
'ccn10',
'cod',
'cwp',
'ceff',
'ceff_ct',
'lcdnc',
'lcdnc_ct',
'clfr',
'cl_time',
'aot550nm',
'aot865nm',
'ang550865',
'up_sw',
'up_sw_cs',
'up_sw_noa',
'up_sw_cs_noa',
'up_lw',
'up_lw_cs',
'up_lw_noa',
'up_lw_cs_noa',
'mmrtrN500',
'mmrtrN250',
'mmrtrN200',
'mmrtrN100',
'mmrtrN50',
'mmrtrN3',
'oh_con',
'emi_monot_bio',
'emi_isop_bio',
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
'VBS1_gas',
'VBS10_gas',
'ORG_mass',
] 


# %%
for case_name in cases_echam:
    varlist = varl
    c = CollocateLONLATout(case_name, from_t, to_t,
                           True,
                           'hour',
                           model_name=model_name
                          # history_field=history_field
                          )
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
for ca in cases_echam:
    c = CollocateLONLATout(ca, from_t, to_t,
                           True,
                           'hour',
                           model_name=model_name
                          )
                          # history_field=history_field)
    ds = c.get_collocated_dataset(varl)
    if 'location' in ds.coords:
        ds = ds.rename({'location':'station'})
    dic_ds[ca]=ds.drop('station').rename(dict(locations='station'))

# %%
dic_mod_ca['ECHAM-SALSA'] = dic_ds.copy()

# %% [markdown]
# ## LOAD NORESM

# %% [markdown]
# ## Settings:

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
# ## Cases:

# %%
cases_noresm1 = ['OsloAero_intBVOC_f19_f19_mg17_full']
cases_noresm2 = ['OsloAero_intBVOC_f19_f19_mg17_ssp245']
# %%
case_mod = 'OsloAero_intBVOC_f19_f19_mg17_fssp'
case_noresm = 'OsloAero_intBVOC_f19_f19_mg17_fssp'
cases_noresm = [case_noresm]
di_mod2cases['NorESM'] = cases_noresm

# %% [markdown]
# ## Variables

# %%
varl =['N100','SOA_NA','SOA_A1','SO4_NA','DOD500','DOD440','ACTREL',#'TGCLDLWP',
       'H2SO4','SOA_LV','COAGNUCL','FORMRATE','FSNSC',
       'NUCLRATE','NCONC01','NCONC02','NCONC03','NCONC04','NCONC05','NCONC06','NCONC07',
       'NCONC08','NCONC09','NCONC10','NCONC11','NCONC12','NCONC13','NCONC14','SIGMA01',
       'SIGMA02','SIGMA03','SIGMA04','SIGMA05','SIGMA06','SIGMA07','SIGMA08','SIGMA09',
       'SIGMA10','SIGMA11','SIGMA12','SIGMA13','SIGMA14','NMR01','NMR02','NMR03','NMR04',
       'NMR05','NMR06','NMR07','NMR08','NMR09','NMR10','NMR11','NMR12','NMR13','NMR14', 
      'FSNS','FSDS_DRF','T','GR','GRH2SO4','GRSOA','TGCLDCWP','U','V', 'SO2','isoprene',
       'monoterp','GS_SO2', 'GS_H2SO4','GS_monoterp','GS_isoprene']


varl =['N100','DOD500','DOD440','ACTREL',#,'SOA_A1',
       'H2SO4','SOA_LV','COAGNUCL','FORMRATE','T','SOA_SV',
       'NCONC01','N50','N150','N200',#'DOD500',
       #'DOD500',
      'isoprene',
      'SFisoprene',
       'monoterp',
       'SFmonoterp',
      'SOA_NA','SOA_A1','OM_NI','OM_AI','OM_AC','SO4_NA','SO4_A1','SO4_A2','SO4_AC','SO4_PR',
      'BC_N','BC_AX','BC_NI','BC_A','BC_AI','BC_AC','SS_A1','SS_A2','SS_A3','DST_A2','DST_A3', 
      ] 


# %%
for case_name in cases_noresm1:
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
for case_name in cases_noresm2:
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
for ca in cases_noresm1:
    c = CollocateLONLATout(ca, from_t, to_t,
                           False,
                           'hour',
                           history_field=history_field)
    ds = c.get_collocated_dataset(varl)
    #ds2 = c.get_collocated_dataset(['DOD500'])
    if 'location' in ds.coords:
        ds = ds.rename({'location':'station'})
    dic_ds[ca]=ds

# %% tags=[]
#dic_ds = dict()
for ca in cases_noresm2:
    c = CollocateLONLATout(ca, from_t2, to_t2,
                           False,
                           'hour',
                           history_field=history_field)
    ds = c.get_collocated_dataset(varl)
    if 'location' in ds.coords:
        ds = ds.rename({'location':'station'})
    dic_ds[ca]=ds

# %%
case1 = cases_noresm1[0]
case2 = cases_noresm2[0]

ds1 = dic_ds[case1]
ds2 = dic_ds[case2]

st_y = from_t.split('-')[0]
mid_y_t = str(int(to_t.split('-')[0])-1)
mid_y_f = to_t.split('-')[0]
end_y = to_t2.split('-')[0]

print(st_y, mid_y_t, mid_y_f, end_y)

# %%
_ds1 = ds1.sel(time=slice(st_y, mid_y_t))
_ds2 = ds2.sel(time=slice(mid_y_f, end_y))
ds_comb = xr.concat([_ds1, _ds2], dim='time')#.sortby('time')

# %%
dic_ds = {case_mod: ds_comb}


# %%
dic_mod_ca['NorESM'] = dic_ds.copy()

# %%
ds_comb

# %% [markdown] tags=[]
# ## SELECT STATION:
#

# %%
for mod in dic_mod_ca.keys():
    for ca in dic_mod_ca[mod].keys():
        dic_mod_ca[mod][ca] = dic_mod_ca[mod][ca].sel(station='SMR')
        dic_mod_ca[mod][ca].load()

# %% [markdown] tags=[]
# # Functions:

# %%
R = 287.058
pressure = 1000. #hPa
kg2ug = 1e9
temperature = 273.15


# %% [markdown]
# ## ADJUST ECHAM

# %%
rn_dict_echam={
    'ORG_mass_conc' : 'OA',
    'tempair':'T',

    
}

    # %%
    from IPython import get_ipython

    # noinspection PyBroadException
    try:
        _ipython = get_ipython()
        _magic = _ipython.magic
        _magic('load_ext autoreload')
        _magic('autoreload 2')
    except:
        pass

# %%
from bs_fdbck.util.BSOA_datamanip import calculate_daily_median_summer,calculate_summer_median, mask4summer,ds2df_echam

# %%
standard_air_density = 100*pressure/(R*temperature)

# %%



# %%
import datetime


# %%
def fix_echam_time(dt):
    #a, b = divmod(round(dt.minute, -1), 60)
    tdelta = datetime.timedelta(minutes=dt.minute, seconds = dt.second)
    #nh = (dt.hour+a)%24
    ndt = datetime.datetime(dt.year, dt.month,dt.day, dt.hour)#dt - tdelta
    #dt_o = datetime.datetime(dt.year,dt.month, dt.day, (dt.hour + a) % 24,b)
    return ndt


# %%
dic_mod_ca['ECHAM-SALSA'][case_name_echam]

# %%

df, df_sm = ds2df_echam(dic_mod_ca['ECHAM-SALSA'][case_name_echam], take_daily_median=False, model_lev_i =model_lev_i)
df.index = df.reset_index()['time'].apply(fix_echam_time)
df

# %%


_di = {case_name_echam:df}
_dism = {case_name_echam:df_sm}

dic_df_mod_case['ECHAM-SALSA']= _di.copy()
dic_dfsm_mod_case['ECHAM-SALSA'] = _dism.copy()

# %% [markdown] tags=[]
# ### NorESM

# %%
dic_mod_ca['NorESM']['OsloAero_intBVOC_f19_f19_mg17_fssp']

# %%
index = dic_mod_ca['NorESM']['OsloAero_intBVOC_f19_f19_mg17_fssp'].to_dataframe().index#.get_level_values(0).name

# %%
index

# %%
model_lev_i

# %%
dic_mod_ca['NorESM']

# %%


dic_df_sm, dic_df = ds2df_inc_preprocessing(dic_mod_ca['NorESM'], model_lev_i=model_lev_i, 
                                            return_summer_median=True, take_daily_median=False)


dic_df_mod_case['NorESM'] = dic_df.copy()
dic_dfsm_mod_case['NorESM'] = dic_df_sm.copy()

# %% [markdown] tags=[]
# ## SHIFT TIME:

# %%
import datetime

# %% tags=[]
for mo in models:
    for ca in di_mod2cases[mo]:
        ind = dic_df_mod_case[mo][ca].index
        dic_df_mod_case[mo][ca].index = ind + datetime.timedelta(hours=3)

# %%
for mo in models:
    for ca in di_mod2cases[mo]:

        print(dic_df_mod_case[mo][ca].index[0:4])

# %% [markdown]
# ## Copy base case 
# %%
dic_df_pre = dict()#dic_df_mod_case.copy()#deep=True)
for mod in dic_df_mod_case.keys():
    dic_df_pre[mod] = dic_df_mod_case[mod].copy()

# %% [markdown] tags=[]
# ## Settings:

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
len(df_lii['Org'][df_lii['Org']<0])#.plot()

# %%
obs_hyy_s = df_lii[(df_lii.index.month==7) |(df_lii.index.month==8)]

# %% [markdown]
# ### mask anomaly

# %% [markdown]
# ## Set uo dic with all OA values from models

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
dic_df_pre['ECHAM-SALSA']['SALSA_BSOA_feedback'].resample('h').ffill()['OA'].plot()         
dic_df_pre['ECHAM-SALSA']['SALSA_BSOA_feedback']['OA'].plot()
plt.xlim(['2014-07','2014-08'])

# %% [markdown] tags=[]
# ## Merge with observations:

# %% tags=[]
# %%
for mod in dic_df_mod_case.keys():
    print(mod)
    for ca in dic_df_mod_case[mod].keys():
        df_mod = dic_df_pre[mod][ca].resample('h').ffill()
        _df_merged = pd.merge(df_mod, obs_hyy_s, right_on='time', left_on='time')
        _df_merged['year'] = _df_merged.index.year
        dic_df_mod_case[mod][ca]= _df_merged

# %%
dic_df_mod_case['ECHAM-SALSA']['SALSA_BSOA_feedback']#['OA']

# %%
mask_obs_OA = dic_df_mod_case[mod][ca]['Org'].notnull()

# %%
for mod in dic_df_mod_case.keys():
    print(mod)
    for ca in dic_df_mod_case[mod].keys():
        df_mod = dic_df_pre[mod][ca].resample('h').mean()
        _df_merged = pd.merge(df_mod, obs_hyy_s, right_on='time', left_on='time')
        _df_merged['year'] = _df_merged.index.year
        dic_df_mod_case[mod][ca]= _df_merged

# %%
mask_obs_OA = dic_df_mod_case[mod][ca]['Org'].notnull()

# %%
_df = obs_hyy_s['Org'].rename('Obs')

df_OA_all = pd.DataFrame(_df)
df_OAG_all = pd.DataFrame(_df)

for mod in dic_df_mod_case.keys():
    print(mod)
    for ca in dic_df_mod_case[mod].keys():
        if len(dic_df_mod_case[mod].keys())==1:
            use_name = mod
        else: 
            use_name = f'{mod}: {ca}'
        df_OA_all[use_name] = dic_df_mod_case[mod][ca]['OA']
        df_OAG_all[use_name] = dic_df_mod_case[mod][ca]['OAG']


df_OA_all = df_OA_all[df_OA_all[mod].notna()]
df_OA_all = df_OA_all[df_OA_all['Obs'].notna()]
df_OAG_all = df_OAG_all[df_OAG_all[mod].notna()]
df_OAG_all = df_OAG_all[df_OAG_all['Obs'].notna()]


# %%
mo = 'ECHAM-SALSA'
_df = dic_df_mod_case[mo][di_mod2cases[mo][0]]

fig, axs = plt.subplots(4, figsize=[10,7])
(_df['T']-273.15).groupby(_df.index.hour).mean().plot(ax =axs[0], label='temperature')
_df['OA'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='OA')
_df['OAG'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='OAG')
_df['mmrtrN500'].groupby(_df.index.hour).mean().plot(ax = axs[2], label='mmrtrN250')
_df['mmrtrN200'].groupby(_df.index.hour).mean().plot(ax = axs[3], label='mmrtrN200')
#_df['MYRC_gas'
for ax in axs:
    ax.legend()

# %%
_df['OAG'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='OA')

# %%
mo = 'NorESM'
_df = dic_df_mod_case[mo][di_mod2cases[mo][0]]

fig, axs = plt.subplots(4, figsize=[10,7])
(_df['T']-273.15).groupby(_df.index.hour).mean().plot(ax =axs[0], label='temperature')
_df['OA'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='OA')
_df['OAG'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='OA')
_df['SOA_SV'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='OAG')
#_df['VBS0_gas'].groupby(_df.index.hour).mean().plot(ax = axs[2], label='b-pinen')
#_df['VBS10_gas'].groupby(_df.index.hour).mean().plot(ax = axs[3], label='VBS10')
#_df['MYRC_gas'
for ax in axs:
    ax.legend()

# %%
mo = 'NorESM'
_df = dic_df_mod_case[mo][di_mod2cases[mo][0]]

fig, axs = plt.subplots(4, figsize=[10,7])
(_df['T']-273.15).groupby(_df.index.hour).mean().plot(ax =axs[0], label='temperature')
_df['OA'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='OA')
_df['OAG'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='OAG')
_df['SOA_SV'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='OAG')
#_df['VBS0_gas'].groupby(_df.index.hour).mean().plot(ax = axs[2], label='b-pinen')
#_df['VBS10_gas'].groupby(_df.index.hour).mean().plot(ax = axs[3], label='VBS10')
#_df['MYRC_gas'
for ax in axs:
    ax.legend()

# %%
mo = 'ECHAM-SALSA'
_df = dic_df_mod_case[mo][di_mod2cases[mo][0]]

fig, axs = plt.subplots(7, figsize=[10,10])
(_df['T']-273.15).groupby(_df.index.hour).mean().plot(ax =axs[0], label='temperature')
_df['OA'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='OA')
_df['N100'].groupby(_df.index.hour).mean().plot(ax = axs[2], label='N100')
_df['N500'].groupby(_df.index.hour).mean().plot(ax = axs[3], label='N500')
_df['VBS10_gas'].groupby(_df.index.hour).mean().plot(ax = axs[5], label='VBS10_gas')
_df['VBS0_gas'].groupby(_df.index.hour).mean().plot(ax = axs[4], label='VBS0_gas')
_df['APIN_gas'].groupby(_df.index.hour).mean().plot(ax = axs[6], label='APIN_gas')
_df['APIN_gas'].groupby(_df.index.hour).mean().plot(ax = axs[6], label='APIN_gas')

#_df['MYRC_gas'
for ax in axs:
    ax.legend()

# %%
mo = 'ECHAM-SALSA'
_df = dic_df_mod_case[mo][di_mod2cases[mo][0]]

fig, axs = plt.subplots(8, figsize=[10,10])
(_df['T']-273.15).groupby(_df.index.hour).mean().plot(ax =axs[0], label='temperature')
_df['OA'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='OA')
_df['mmrtrN3'].groupby(_df.index.hour).mean().plot(ax = axs[2], )#label='N100')
_df['mmrtrN50'].groupby(_df.index.hour).mean().plot(ax = axs[3], )#label='N100')
_df['mmrtrN100'].groupby(_df.index.hour).mean().plot(ax = axs[4],)# label='N500')
_df['mmrtrN200'].groupby(_df.index.hour).mean().plot(ax = axs[5], )#label='mmrtrN3')
_df['mmrtrN250'].groupby(_df.index.hour).mean().plot(ax = axs[5], )#label='VBS0_gas')
_df['mmrtrN500'].groupby(_df.index.hour).mean().plot(ax = axs[6],)# label='mmrtrN500')
_df['up_sw_cs'].groupby(_df.index.hour).mean().plot(ax = axs[7],)# label='mmrtrN500')

#_df['MYRC_gas'
for ax in axs:
    ax.legend()

# %%
mo = 'ECHAM-SALSA'
_df = dic_df_mod_case[mo][di_mod2cases[mo][0]]

fig, axs = plt.subplots(8, figsize=[10,10])
(_df['T']-273.15).groupby(_df.index.hour).mean().plot(ax =axs[0], label='temperature')
_df['airdens'].groupby(_df.index.hour).mean().plot(ax = axs[1],)# label='OA')
_df['aot865nm'].groupby(_df.index.hour).mean().plot(ax = axs[2], )#label='N100')
_df['aot550nm'].groupby(_df.index.hour).mean().plot(ax = axs[3], )#label='N100')
_df['VBS1_gas'].groupby(_df.index.hour).mean().plot(ax = axs[4],)# label='N500')
_df['ORG_mass'].groupby(_df.index.hour).mean().plot(ax = axs[5], )#label='mmrtrN3')
#_df['mmrtrN250'].groupby(_df.index.hour).mean().plot(ax = axs[5], )#label='VBS0_gas')
_df['SO2_gas'].groupby(_df.index.hour).mean().plot(ax = axs[6],)# label='mmrtrN500')
_df['up_sw_cs'].groupby(_df.index.hour).mean().plot(ax = axs[7],)# label='mmrtrN500')

#_df['MYRC_gas'
for ax in axs:
    ax.legend()

# %%

# %%
mo = 'ECHAM-SALSA'
_df = dic_df_mod_case[mo][di_mod2cases[mo][0]]

fig, axs = plt.subplots(5, figsize=[7,7], sharex=True, dpi=150)
(_df['T']-273.15).groupby(_df.index.hour).mean().plot(ax =axs[0], label='temperature')
ax = axs[0].twinx()
_df['up_sw_cs'].groupby(_df.index.hour).mean().plot(ax = ax, label='up_sw_cs', c='r')
ax.legend()

#_df['OA'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='OA')
#_df['VBS0_gas'].groupby(_df.index.hour).mean().plot(ax = axs[2], label='VBS0')
#_df['VBS1_gas'].groupby(_df.index.hour).mean().plot(ax = axs[3], label='VBS1')
#_df['APIN_gas'].groupby(_df.index.hour).mean().plot(ax = axs[4], label='APIN')
_df['ISOP_gas'].groupby(_df.index.hour).mean().plot(ax = axs[2], label='ISOP_gas')
_df['APIN_gas'].groupby(_df.index.hour).mean().plot(ax = axs[2], label='APIN_gas')
#_df['APIN_gas'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='APIN_gas')
_df['oh_con'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='oh_con')
#_df['emi_monot_bio'].groupby(_df.index.hour).mean().plot(ax = axs[3], label='emi monot')
_df['emi_isop_bio'].groupby(_df.index.hour).mean().plot(ax = axs[3], label='emi isop')
_df['emi_monot_bio'].groupby(_df.index.hour).mean().plot(ax = axs[3], label='emi mono')
_df['OAG'].groupby(_df.index.hour).mean().plot(ax = axs[4], label='OAG')
_df['OA'].groupby(_df.index.hour).mean().plot(ax = axs[4], label='OA')
for ax in axs:
    ax.legend()
    
fig.tight_layout()


# %%
ds

# %%
mo = 'NorESM'
_df = dic_df_mod_case[mo][di_mod2cases[mo][0]]

fig, axs = plt.subplots(5, figsize=[7,7], sharex=True, dpi=150)
(_df['T']-273.15).groupby(_df.index.hour).mean().plot(ax =axs[0], label='temperature')
#ax = axs[0].twinx()
#_df['up_sw_cs'].groupby(_df.index.hour).mean().plot(ax = ax, label='up_sw_cs', c='r')
#ax.legend()

_df['SFisoprene'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='SFisoprene')
_df['SFmonoterp'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='SFmonoterp')
#_df['VBS0_gas'].groupby(_df.index.hour).mean().plot(ax = axs[2], label='VBS0')
#_df['VBS1_gas'].groupby(_df.index.hour).mean().plot(ax = axs[3], label='VBS1')
#_df['APIN_gas'].groupby(_df.index.hour).mean().plot(ax = axs[4], label='APIN')
(_df['isoprene']*128./28).groupby(_df.index.hour).mean().plot(ax = axs[2], label='isoprene')
(_df['monoterp']*128./28).groupby(_df.index.hour).mean().plot(ax = axs[2], label='monoterp')
#_df['APIN_gas'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='APIN_gas')
#_df['oh_con'].groupby(_df.index.hour).mean().plot(ax = axs[1], label='oh_con')
_df['SOA_SV'].groupby(_df.index.hour).mean().plot(ax = axs[3], label='SOA_SV')
_df['SOA_LV'].groupby(_df.index.hour).mean().plot(ax = axs[3], label='SOA_LV')
#_df['emi_isop_bio'].groupby(_df.index.hour).mean().plot(ax = axs[3], label='emi isop')
#_df['emi_monot_bio'].groupby(_df.index.hour).mean().plot(ax = axs[3], label='emi mono')
_df['OAG'].groupby(_df.index.hour).mean().plot(ax = axs[4], label='OAG')
_df['OA'].groupby(_df.index.hour).mean().plot(ax = axs[4], label='OA')
for ax in axs:
    ax.legend()
    
fig.tight_layout()


# %%
_df_daily = _df - _df.resample('D').mean().resample('h').ffill()

# %%
mo = 'ECHAM-SALSA'
_df = dic_df_mod_case[mo][di_mod2cases[mo][0]]
n = 400
fig, ax = plt.subplots(1, figsize=[20,10])
ax2 = ax.twinx()
(_df_daily['T']).iloc[0:n].plot(ax = ax2, label='T_C', c='k', linestyle=':', linewidth=3)
(_df_daily['OA'].iloc[0:n]/_df_daily['OA'].iloc[0:n].max()).plot(ax = ax, label='OA', linewidth=2)
(_df_daily['VBS0_gas'].iloc[0:n]/_df_daily['VBS0_gas'].iloc[0:n].max()).plot(ax = ax, label='VBS0',linewidth=2)
#(_df['VBS1_gas'].iloc[0:n]/_df['VBS1_gas'].iloc[0:n].max()).plot(ax = ax, label='VBS1')
#(_df['VBS10_gas'].iloc[0:n]/_df['VBS10_gas'].iloc[0:n].max()).plot(ax = ax, label='VBS10')
ax.legend()

# %%
mo = 'ECHAM-SALSA'
_df = dic_df_mod_case[mo][di_mod2cases[mo][0]]
n = 400
fig, ax = plt.subplots(1, figsize=[20,10])
ax2 = ax.twinx()
(_df_daily['T']).iloc[0:n].plot(ax = ax2, label='T_C', c='k', linestyle=':', linewidth=3)
(_df_daily['OA'].iloc[0:n]/_df_daily['OA'].iloc[0:n].max()).plot(ax = ax, label='OA', linewidth=2)
(_df_daily['VBS1_gas'].iloc[0:n]/_df_daily['VBS1_gas'].iloc[0:n].max()).plot(ax = ax, label='VBS1',linewidth=2)
#(_df['VBS1_gas'].iloc[0:n]/_df['VBS1_gas'].iloc[0:n].max()).plot(ax = ax, label='VBS1')
#(_df['VBS10_gas'].iloc[0:n]/_df['VBS10_gas'].iloc[0:n].max()).plot(ax = ax, label='VBS10')
ax.legend()

# %%
mo = 'ECHAM-SALSA'
_df = dic_df_mod_case[mo][di_mod2cases[mo][0]]
n = 400
fig, ax = plt.subplots(1, figsize=[20,10])
ax2 = ax.twinx()
(_df_daily['T']).iloc[0:n].plot(ax = ax2, label='T_C', c='k', linestyle=':', linewidth=3)
(_df_daily['OA'].iloc[0:n]/_df_daily['OA'].iloc[0:n].max()).plot(ax = ax, label='OA', linewidth=2)
(_df_daily['VBS10_gas'].iloc[0:n]/_df_daily['VBS10_gas'].iloc[0:n].max()).plot(ax = ax, label='VBS10',linewidth=2)
#(_df['VBS1_gas'].iloc[0:n]/_df['VBS1_gas'].iloc[0:n].max()).plot(ax = ax, label='VBS1')
#(_df['VBS10_gas'].iloc[0:n]/_df['VBS10_gas'].iloc[0:n].max()).plot(ax = ax, label='VBS10')
ax.legend()
plt.show()

mo = 'ECHAM-SALSA'
_df = dic_df_mod_case[mo][di_mod2cases[mo][0]]
n = 400
fig, ax = plt.subplots(1, figsize=[20,10])
ax2 = ax.twinx()
(_df['T']).iloc[0:n].plot(ax = ax2, label='T_C', c='k', linestyle=':', linewidth=3)
(_df['OA'].iloc[0:n]/_df['OA'].iloc[0:n].max()).plot(ax = ax, label='OA',linewidth=2)
#(_df['VBS0_gas'].iloc[0:n]/_df['VBS0_gas'].iloc[0:n].max()).plot(ax = ax, label='VBS0')
(_df['VBS1_gas'].iloc[0:n]/_df['VBS1_gas'].iloc[0:n].max()).plot(ax = ax, label='VBS1',linewidth=2)
#(_df['VBS10_gas'].iloc[0:n]/_df['VBS10_gas'].iloc[0:n].max()).plot(ax = ax, label='VBS10')
ax.legend()
plt.show()

# %%
for v in _df.columns:
    if True:#'VBS' in v:
        print(v)

# %% [markdown]
# Small error due to time change in models but only 3 data points each summer. 
#

# %%
orgname={'NorESM' : 'OA',
         'ECHAM-SALSA': 'OA'}

# %% [markdown]
# ### Calculate anomaly from daily average

# %%

# %%

# %%
df_anom_OA = df_OA_all-df_OA_all.resample('D').mean().resample('h').ffill()
df_anom_OAG = df_OAG_all-df_OAG_all.resample('D').mean().resample('h').ffill()

# %%
linestyle_dic = {
    'Obs': '--',
    'NorESM':'dashdot',
    'ECHAM-SALSA':'-.'
}


# %%
fig, axs = plt.subplots(2,1, sharex=True, figsize=[5,5], dpi=100)
ax = axs[1]
#pl_obs = df_anom_OA['Obs'].groupby(df_anom_OA['obs'].index.hour).mean()
#pl_obs.plot(ax=ax,label='Observations', c='k')

for mod in df_anom_OA.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_anom_OA[mod].groupby(df_anom_OA.index.hour).mean().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.set_title("$OA'$: Average diurnal anomaly") 
#ax.legend(frameon=False)


ax = axs[0]


for mod in df_anom_OA.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_OA_all[mod].groupby(df_OA_all.index.hour).mean().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.legend(frameon=False)
ax.set_title("Organic aerosol and gas: Hourly average") 

ax.set_ylim([0,6])

ax = axs[1]
#hour = df_anom_OA[case_mod].index.hour
#anom = df_anom_OA[case_mod].groupby(hour).mean()
#hour = df_daily_mean[case_mod].index.hour
#daily = df_daily_mean[case_mod].groupby(hour).mean()
#nn = 2.74*daily + 1.86*anom
#nn.plot(label = "OsloAeroSec fit: \n  $y=2.74 \cdot \overline{OA}+ 1.86 \cdot OA'$       ")
#pl_obs = df_full_OA['obs'].groupby(df_full_OA['obs'].index.hour).mean()
#pl_obs.plot(ax=ax,label='__nolegend__', c='k')
ax.set_title("Hourly average deviation from daily average") 

plt.legend(frameon=False)
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
fig, axs = plt.subplots(2,1, sharex=True, figsize=[5,5], dpi=100)
ax = axs[1]
#pl_obs = df_anom_OA['Obs'].groupby(df_anom_OA['obs'].index.hour).mean()
#pl_obs.plot(ax=ax,label='Observations', c='k')

for mod in df_anom_OAG.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_anom_OAG[mod].groupby(df_anom_OAG.index.hour).mean().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.set_title("$OA'$: Average diurnal anomaly") 
#ax.legend(frameon=False)


ax = axs[0]


for mod in df_OAG_all.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_OAG_all[mod].groupby(df_OAG_all.index.hour).mean().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.legend(frameon=False)
ax.set_title("Organic aerosol and gas: Hourly average") 
ax.set_ylim([0,6])

ax = axs[1]
#hour = df_anom_OA[case_mod].index.hour
#anom = df_anom_OA[case_mod].groupby(hour).mean()
#hour = df_daily_mean[case_mod].index.hour
#daily = df_daily_mean[case_mod].groupby(hour).mean()
#nn = 2.74*daily + 1.86*anom
#nn.plot(label = "OsloAeroSec fit: \n  $y=2.74 \cdot \overline{OA}+ 1.86 \cdot OA'$       ")
#pl_obs = df_full_OA['obs'].groupby(df_full_OA['obs'].index.hour).mean()
#pl_obs.plot(ax=ax,label='__nolegend__', c='k')
ax.set_title("Hourly average deviation from daily average") 

plt.legend(frameon=False)
ax.set_xlabel('Time of day [h]')
for ax in axs: 
    ax.set_ylabel('OAG  [$\mu$gm$^{-3}$]')
plt.tight_layout()

ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,])

ax.set_xlim([0,23])
fn = make_fn_eval('_'.join(models), 'diurnal_mean')

fig.savefig(fn.with_suffix('.png'))
fig.savefig(fn.with_suffix('.pdf'))

# %%
fig, axs = plt.subplots(2,1, sharex=True, figsize=[5,5], dpi=100)
ax = axs[1]
#pl_obs = df_anom_OA['Obs'].groupby(df_anom_OA['obs'].index.hour).mean()
#pl_obs.plot(ax=ax,label='Observations', c='k')

for mod in df_anom_OA.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_anom_OA[mod].groupby(df_anom_OA.index.hour).median().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.set_title("$OA'$: Median diurnal anomaly") 
#ax.legend(frameon=False)


ax = axs[0]


for mod in df_anom_OA.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_OA_all[mod].groupby(df_OA_all.index.hour).median().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.legend(frameon=False)
ax.set_title("$\overline{OA}$: Median") 
ax.set_ylim([0,9])

ax = axs[1]
#hour = df_anom_OA[case_mod].index.hour
#anom = df_anom_OA[case_mod].groupby(hour).mean()
#hour = df_daily_mean[case_mod].index.hour
#daily = df_daily_mean[case_mod].groupby(hour).mean()
#nn = 2.74*daily + 1.86*anom
#nn.plot(label = "OsloAeroSec fit: \n  $y=2.74 \cdot \overline{OA}+ 1.86 \cdot OA'$       ")
#pl_obs = df_full_OA['obs'].groupby(df_full_OA['obs'].index.hour).mean()
#pl_obs.plot(ax=ax,label='__nolegend__', c='k')
ax.set_title("Median diurnal pattern") 

plt.legend(frameon=False)
ax.set_xlabel('Time of day [h]')
for ax in axs: 
    ax.set_ylabel('OA  [$\mu$gm$^{-3}$]')
plt.tight_layout()

ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,])

ax.set_xlim([0,23])
fn = make_fn_eval('_'.join(models), 'diurnal_median')

fig.savefig(fn.with_suffix('.png'))
fig.savefig(fn.with_suffix('.pdf'))


# %%
def make_cbar(fig, label):
    

    levs = [1000]+[np.round(dic_p[i]) for i in range(1,num_levs)] + [850]

    levs_bound = [(levs[i]+levs[i+1])/2 for i in np.arange(len(levs)-1)]

    lev_ticks = levs[1:-1][::-1]

    cmap= mpl.colors.ListedColormap(sns.color_palette('viridis_r',6))
    norm= mpl.colors.BoundaryNorm(levs_bound[::-1], len(levs_bound[::-1]))#, clip=True)

    #norm = mpl.colors.Normalize(vmin=993,vmax=886)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap), ticks = lev_ticks, label=label)
    #cb.ax.set_yticklabels(levs[::-1][:-1])
    cb.ax.invert_yaxis()



# %%
def make_cbar(fig, label):
    
    levs = [992.556095123291,
     976.325407391414,
     957.485479535535,
     936.1983984708786,
     912.644546944648,
     887.0202489197254]

    aa = [levs[0]+(levs[0]-levs[1])/2]
    for i in range(len(levs)-1):
        b = (levs[i] + levs[i+1])/2
        aa.append(b)

    aa.append( levs[-1]+(levs[-1]-levs[-2])/2   )

    a1 = aa[0]
    a2 = aa[-1]
    
    
    cmap = plt.get_cmap('plasma_r')

    norm = mpl.colors.Normalize(vmin=a2,vmax=a1)

    cols = [cmap(norm(min(levs, key=lambda x:abs(x-xx)))) for xx in np.linspace(a2,a1,256)]

    cmm = mpl.colors.ListedColormap(cols)
    

    #norm = mpl.colors.Normalize(vmin=993,vmax=886)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmm), ticks = levs, label=label)
    #cb.ax.set_yticklabels(levs[::-1][:-1])
    cb.ax.invert_yaxis()
    
    return norm, cmm


# %%
import matplotlib as mpl

# %%
mi = np.min(df_OA_all[df_OA_all['Obs']>0]['Obs'])*15
ma = np.max(df_OA_all[df_OA_all['Obs']>0]['Obs'])
bins_ = 10 ** np.linspace(np.log10(mi), np.log10(ma), 50)

# %%
df_OA_all

# %%
df_OA_all['Obs'].plot.hist(bins=bins_, alpha=0.5, 
                                     color='k',
                                     label='Observations'
                                    )
for mo in models:
    df_OA_all[mo].plot.hist(bins=bins_, alpha=0.5, 
                                     #color=None,
                                     label=mo
                                    )
plt.xscale('log')
#_mod_an.plot.hist(bins=bins_, alpha=0.5,label='OsloAero, SOA')

plt.xlabel('OA [$\mu$gm$^{-3}$]')

plt.title('Distribution July and August 2012-2018, Hyytiälä')

plt.legend()
fn = make_fn_eval('noresm_echam','hist')
plt.savefig(fn, dpi=300)
plt.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
for mo in models:
    (df_OA_all[mo]-df_OA_all['Obs']).plot.hist(#bins=bins_, 
        alpha=0.5, 
        bins=70,
                                     #color=None,
                                     label=mo
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
fig, axs = plt.subplots(1,2,sharey=True, figsize=[10,5], sharex=True)
ax = axs[0]
_df = df_OA_all
_df['hour'] = _df.index.hour
for mo, ax in zip(models,axs.flatten()):
    sns.scatterplot(x=mo, y='Obs',#orbins=bins_, alpha=0.5, 
                                     hue='hour', 
                #col = 'dir',
                ax=ax,
                alpha=0.4,
                palette='viridis',
                
                edgecolors=None,
                data = _df)
#ax.set_ylim([0,30])

#ax.set_xlim([0,15])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.1,30])
    ax.set_ylim([0.1,30])

    ax.set_xlabel(f'{mo} OA '+'[$\mu$gm$^{-3}$]')
    ax.set_ylabel('Observed OA [$\mu$gm$^{-3}$]')



    lims = ax.get_xlim()
    ax.plot(lims,lims,'k', linewidth=.5)


fn = make_fn_eval('_'.join(models),'scatt')
fig.savefig(fn, dpi=300)
fig.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
fig, axs = plt.subplots(1,2,sharey=True, figsize=[12,5], sharex=True)
ax = axs[0]
_df = df_OA_all
_df['hour'] = _df.index.hour
for mo, ax in zip(models,axs.flatten()):
    sns.histplot(x=mo, y='Obs',#orbins=bins_, alpha=0.5, 
                                    # hue='hour', 
                #col = 'dir',
                ax=ax,
                cmap = sns.color_palette("mako_r", as_cmap=True),
                log_scale=(True, True),
                     cbar=True, cbar_kws=dict(shrink=.75),
                
                edgecolors=None,
                 
                data = _df)


    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.1,30])
    ax.set_ylim([0.1,30])

    ax.set_xlabel(f'{mo} OA '+'[$\mu$gm$^{-3}$]')
    ax.set_ylabel('Observed OA [$\mu$gm$^{-3}$]')



    lims = ax.get_xlim()
    ax.plot(lims,lims,'k', linewidth=.5)


fn = make_fn_eval('_'.join(models),'scatt')
sns.despine(fig)
fig.savefig(fn, dpi=300)
fig.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
df_OA_all.T-df_OA_all['Obs']

# %%
_diff = (df_OA_all.drop(['hour','Obs'],axis=1).T - df_OA_all['Obs']).T

df_OA_incDiff = df_OA_all.copy()

for mo in _diff.columns:
    df_OA_incDiff[f'{mo}_diff'] = _diff[mo]


# %%
df_OA_incDiff

# %%
fig, axs = plt.subplots(1,2,sharey=True, figsize=[10,5])
ax = axs[0]
_df = df_OA_incDiff
_df['hour'] = _df.index.hour
for mo, ax in zip(models,axs.flatten()):
    sns.scatterplot(x=f'{mo}_diff', y=f'hour',#orbins=bins_, alpha=0.5, 
                                     hue=f'Obs', 
                #col = 'dir',
                ax=ax,
                alpha=0.4,
                palette='viridis',
                
                edgecolors=None,
                data = _df)

    #ax.set_yscale('log')
    # ax.set_xscale('symlog')
    ax.set_xlim([-15,6])
    #ax.set_ylim([0.1,30])

    #ax.set_xlabel(f'{mo} OA '+'[$\mu$gm$^{-3}$]')
    #ax.set_ylabel('Observed OA [$\mu$gm$^{-3}$]')



    lims = ax.get_xlim()
    #ax.plot(lims,lims,'k', linewidth=.5)


fn = make_fn_eval('_'.join(models),'scatt')
#fig.savefig(fn, dpi=300)
#fig.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%

# %%

# %%
