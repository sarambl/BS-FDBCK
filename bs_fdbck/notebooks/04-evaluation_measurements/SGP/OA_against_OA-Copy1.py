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

# %%
import xarray as xr

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



# %%
import datetime

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

stations = ['SMR', 'SGP']

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
to_t2 = '2015-02-01'

# %%
dic_mod_ca = dict()
dic_df_mod_case = dict()
dic_dfsm_mod_case = dict()

# %% [markdown] tags=[]
# ### Load echam-salsa data

# %%

# %% [markdown]
# #### Settings:

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
# #### Variables

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
# ### LOAD NORESM

# %% [markdown]
# #### Settings:

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
# #### Cases:

# %%
cases_noresm1 = ['OsloAero_intBVOC_f09_f09_mg17_full']
cases_noresm2 = ['OsloAero_intBVOC_f09_f09_mg17_ssp245']
# %%
case_mod = 'OsloAero_intBVOC_f19_f19_mg17_fssp'
case_noresm = 'OsloAero_intBVOC_f19_f19_mg17_fssp'
cases_noresm = [case_noresm]
di_mod2cases['NorESM'] = cases_noresm

# %% [markdown]
# #### Variables

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


# %% tags=[]
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
from_t2

# %%
to_t2

# %% tags=[]
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
# # SELECT STATION for models
#

# %%
dic_st_mod_ca = dict()

# %%
for st in stations:
    dic_st_mod_ca[st] = dict()
    for mod in dic_mod_ca.keys():
        dic_st_mod_ca[st][mod] = dict()
        
        for ca in dic_mod_ca[mod].keys():
            dic_st_mod_ca[st][mod][ca] = dic_mod_ca[mod][ca].sel(station=st)
            dic_st_mod_ca[st][mod][ca].load()


# %%

# %% [markdown] tags=[]
# # Define constants:

# %%
R = 287.058
pressure = 1000. #hPa
kg2ug = 1e9
temperature = 273.15

standard_air_density = 100*pressure/(R*temperature)


# %% [markdown] tags=[]
# ## Renaming  ECHAM-SALSA variables and fixing the time

# %%
rn_dict_echam={
    'ORG_mass_conc' : 'OA',
    'tempair':'T',

    
}


# %%
def fix_echam_time(dt):
    #a, b = divmod(round(dt.minute, -1), 60)
    tdelta = datetime.timedelta(minutes=dt.minute, seconds = dt.second)
    #nh = (dt.hour+a)%24
    ndt = datetime.datetime(dt.year, dt.month,dt.day, dt.hour)#dt - tdelta
    #dt_o = datetime.datetime(dt.year,dt.month, dt.day, (dt.hour + a) % 24,b)
    return ndt


# %% [markdown]
# ## Make dataframes

# %%
dic_df_st_mod = dict()
dic_df_st_mod_sm = dict()
for st in stations:
    dic_df_st_mod[st] = dict()    
    dic_df_st_mod_sm[st] = dict()

# %%

# %%
for st in stations:
    
    
    df, df_sm = ds2df_echam(dic_st_mod_ca[st]['ECHAM-SALSA'][case_name_echam], take_daily_median=False, model_lev_i =model_lev_i)
    df.index = df.reset_index()['time'].apply(fix_echam_time)
    df
    dic_df_st_mod[st]['ECHAM-SALSA']= df.copy()    
    dic_df_st_mod_sm[st]['ECHAM-SALSA']= df_sm

# %% [markdown]
# _di = {case_name_echam:df}
# _dism = {case_name_echam:df_sm}
#
# dic_df_mod_case['ECHAM-SALSA']= _di.copy()
# dic_dfsm_mod_case['ECHAM-SALSA'] = _dism.copy()

# %% [markdown] tags=[]
# ### NorESM

# %% [markdown]
# index = dic_st_mod_ca[st]['NorESM']['OsloAero_intBVOC_f19_f19_mg17_fssp'].to_dataframe().index#.get_level_values(0).name

# %% [markdown]
# index

# %%

for st in stations:
     
        
    dic_df_sm, dic_df = ds2df_inc_preprocessing(dic_st_mod_ca[st]['NorESM'], model_lev_i=model_lev_i, 
                                            return_summer_median=True, take_daily_median=False)

    dic_df_st_mod[st]['NorESM']= dic_df[case_noresm].copy()    
    dic_df_st_mod_sm[st]['NorESM']= dic_df_sm[case_noresm].copy()


# %% [markdown] tags=[]
# ## SHIFT TIME to local time for models

# %%
dic_station2time_offset = dict(SMR=3, SGP = -6)#time_shift

# %% tags=[]
for st in stations:
    for mo in models:
        ts = dic_station2time_offset[st]
        ind = dic_df_st_mod[st][mo].index
        dic_df_st_mod[st][mo].index = ind + datetime.timedelta(hours=ts)

# %% [markdown]
# ### Check: 

# %%
for mo in models:
    for st in stations: 
        print(dic_df_st_mod[st][mo].index[0:4])

# %% [markdown]
# # Copy base case into dictionary
# %%
dic_df_pre = dict()#dic_df_mod_case.copy()#deep=True)
for st in stations:
    dic_df_pre[st] = dict()
    for mod in models:
        dic_df_pre[st][mod] = dic_df_st_mod[st][mod].copy()

# %%
from bs_fdbck.constants import measurements_path
import pandas as pd


# %% [markdown]
# # Read ACSM data from Hyytiala

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
fn_liine = measurements_path / 'ACSM_DEFAULT.mat'

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
# ### READ FROM SGP

# %%
import pandas as pd

# %%

# %%
from bs_fdbck.constants import measurements_path

# %%
measurements_path

# %%
path_acsm_SGP = measurements_path/'ARM_data' /'watson'/'ACSM'


fl = list(path_acsm_SGP.glob('*.csv'))

fl.sort()

df_list = []
for f in fl:
    df_list.append(pd.read_csv(f, index_col=1))

_df_acsm_SGP = pd.concat(df_list)

_df_acsm_SGP.index = pd.to_datetime(_df_acsm_SGP.index)

df_acsm_SGP =  _df_acsm_SGP.resample('h').mean()

df_acsm_SGP.index = df_acsm_SGP.index.rename('time')

df_acsm_SGP

# %% [markdown] tags=[]
# ## Change from UTC to local: -6

# %%
ind = df_acsm_SGP.index 
df_acsm_SGP.index = ind + datetime.timedelta(hours=-6)

# %%
df_acsm_SGP

# %% [markdown]
# ## Select summer: 

# %%
JA_months = ((df_acsm_SGP.index.month==7 ) |  (df_acsm_SGP.index.month==8 ) )

df_acsm_SGP_sum = df_acsm_SGP[JA_months]

# %% [markdown] tags=[]
# ## Put observations in dic: 

# %% tags=[]
dic_df_obs_st = dict()
dic_df_obs_st['SMR'] = obs_hyy_s
dic_df_obs_st['SGP'] = df_acsm_SGP_sum

# %% [markdown]
# # Set up dic with all OA values from models

# %% tags=[]
dic_mod_oa = dict()
#dic_mod_soa = dict()
dic_mod_all = dict()

for st in stations:

    dic_mod_oa[st] = dict()
    #dic_mod_soa[mod] = dict()
    dic_mod_all[st] = dict()
    for mod in mod: 
        dic_mod_oa[st][mod] = dict()
        #dic_mod_soa[mod][ca] = dict()
        dic_mod_all[st][mod] = dict()
    
    
for st in stations:
    for mo in models:
        _df = dic_df_st_mod[st][mo]
        dic_mod_oa[st][mo] =_df['OA']
        #dic_mod_soa[mo][ca][i] =_df['SOA']
        dic_mod_all[st][mo] =_df
    



# %%
dic_df_pre['SMR']['ECHAM-SALSA'].resample('h').ffill()['OA'].plot()         
dic_df_pre['SMR']['ECHAM-SALSA']['OA'].plot()
plt.xlim(['2014-07','2014-08'])

# %% [markdown] tags=[]
# # Merge with observations in same dictionary!

# %%

# %%
dic_df_obs_relab_st = dict()
for st in stations:
    _df = dic_df_obs_st[st]
    df_obs_relabel = pd.DataFrame()
    for col in _df.columns:
        df_obs_relabel[f'{col}_obs'] = _df[col]
    dic_df_obs_relab_st[st] = df_obs_relabel.copy()

# %%
df_obs_relabel

# %%
for st in stations: 
    for mod in models:
        _df_mod = dic_df_obs_relab_st[st]['Org_obs']
        df_mod = dic_df_pre[st][mod].resample('h').ffill()
        
        _df_merged = pd.merge(df_mod, _df_mod, right_on='time', left_on='time')
        _df_merged['year'] = _df_merged.index.year
        dic_df_st_mod[st][mod]= _df_merged

# %%
dic_mask_obs_OA = dict()
for st in stations:
    mask_obs_OA = dic_df_st_mod[st][mod]['Org_obs'].notnull()
    dic_df_st_mod[st]['mask_obs_OA'] = mask_obs_OA
    dic_mask_obs_OA[st] = mask_obs_OA.copy()

# %%
dic_df_obs_st[st]

# %%
dic_df_OA_all = dict()
dic_df_OAG_all = dict()

for st in stations:
    _df_obs = dic_df_obs_st[st]['Org'].rename('Obs')

    df_OA_all = pd.DataFrame(_df_obs)
    df_OAG_all = pd.DataFrame(_df_obs)

    for mod in models:
        print(mod)
        df_OA_all[mod] = dic_df_st_mod[st][mod]['OA']
        df_OAG_all[mod] = dic_df_st_mod[st][mod]['OAG']


    df_OA_all = df_OA_all[df_OA_all[mod].notna()]
    df_OA_all = df_OA_all[df_OA_all['Obs'].notna()]
    df_OAG_all = df_OAG_all[df_OAG_all[mod].notna()]
    df_OAG_all = df_OAG_all[df_OAG_all['Obs'].notna()]
    dic_df_OA_all[st] = df_OA_all.copy()    
    dic_df_OAG_all[st] = df_OAG_all.copy()


# %%

# %%
orgname={'NorESM' : 'OA',
         'ECHAM-SALSA': 'OA'}

# %% [markdown]
# ### Calculate anomaly from daily average

# %%
dic_df_anom_OA =dict()
dic_df_anom_OAG =dict()
for st in stations:
    df_OA_all = dic_df_OA_all[st]
    df_OAG_all = dic_df_OAG_all[st]
    dic_df_anom_OA[st] = df_OA_all-df_OA_all.resample('D').mean().resample('h').ffill()
    dic_df_anom_OAG[st] = df_OAG_all-df_OAG_all.resample('D').mean().resample('h').ffill()

# %%
linestyle_dic = {
    'Obs': '--',
    'NorESM':'dashdot',
    'ECHAM-SALSA':'-.'
}


# %%
for st in stations:
    
    df_anom_OA = dic_df_anom_OA[st]
    df_anom_OA.plot()
    plt.ylim([-10,20])


# %%
_df = dic_df_OA_all['SGP'].copy()
_df['Obs'] = _df['Obs']#/10
_df.loc['2013':'2014'].plot()

# %%
_df = dic_df_OA_all['SGP'].copy()
_df['Obs'] = _df['Obs']#/100
_df.loc['2013-07-01':'2013-08-01'].plot()

# %%
_df = dic_df_OA_all['SMR'].copy()
_df['Obs'] = _df['Obs']
_df.loc['2013-07-01':'2013-08-01'].plot()

# %%
for y in ['2012','2013','2014', '2015','2016','2017','2018']:
    fy = y +'-07'
    ty = y +'-08'
    df_anom_OA.loc[fy:fy].plot()
    plt.show()

# %%
_df = dic_df_anom_OA['SGP'].dropna()

# %%
_df.groupby(_df.index.hour).mean().plot()

# %% [markdown] tags=[]
# ## Plots daily variability

# %%
for st in stations: 
    df_anom_OA = dic_df_anom_OA[st]
    df_OA_all = dic_df_OA_all[st]
    fig, axs = plt.subplots(2,1, sharex=True, figsize=[5,5], dpi=100)
    ax = axs[1]

    for mod in df_anom_OA.columns:
        if mod =='Obs': c = 'k'
        else: c=None
        ls = linestyle_dic[mod]
        df_anom_OA[mod].groupby(df_anom_OA.index.hour).mean().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
    ax.set_title(f"{st}: $OA'$: Average diurnal anomaly") 


    ax = axs[0]


    for mod in df_anom_OA.columns:
        if mod =='Obs': c = 'k'
        else: c=None
        ls = linestyle_dic[mod]
        df_OA_all[mod].groupby(df_OA_all.index.hour).mean().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
    ax.legend(frameon=False)
    ax.set_title(f"{st}: OA: Hourly average") 

    ax.set_ylim([0,5])

    ax = axs[1]
    ax.set_title("Hourly average deviation from daily average") 

    plt.legend(frameon=False)
    ax.set_xlabel('Time of day [h]')
    for ax in axs: 
        ax.set_ylabel('OA  [$\mu$gm$^{-3}$]')
    plt.tight_layout()

    ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,])

    ax.set_xlim([0,23])
    fn = make_fn_eval('_'.join(models), f'diurnal_mean_{st}')

    fig.savefig(fn.with_suffix('.png'))
    fig.savefig(fn.with_suffix('.pdf'))

# %%
for st in stations: 
    df_anom_OA = dic_df_anom_OAG[st]
    df_OA_all = dic_df_OAG_all[st]
    fig, axs = plt.subplots(2,1, sharex=True, figsize=[5,5], dpi=100)
    ax = axs[1]

    for mod in df_anom_OA.columns:
        if mod =='Obs': c = 'k'
        else: c=None
        ls = linestyle_dic[mod]
        df_anom_OA[mod].groupby(df_anom_OA.index.hour).mean().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
    ax.set_title(f"{st}: $OA'$: Average diurnal anomaly") 


    ax = axs[0]


    for mod in df_anom_OA.columns:
        if mod =='Obs': c = 'k'
        else: c=None
        ls = linestyle_dic[mod]
        df_OA_all[mod].groupby(df_OA_all.index.hour).mean().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
    ax.legend(frameon=False)
    ax.set_title(f"{st}: OAG: Hourly average") 

    ax.set_ylim([0,4])

    ax = axs[1]
    ax.set_title("Hourly average deviation from daily average") 

    plt.legend(frameon=False)
    ax.set_xlabel('Time of day [h]')
    for ax in axs: 
        ax.set_ylabel('OAG  [$\mu$gm$^{-3}$]')
    plt.tight_layout()

    ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,])

    ax.set_xlim([0,23])
    fn = make_fn_eval('_'.join(models), f'diurnal_mean_{st}')

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
df_OA_all = dic_df_OA_all['SMR'].dropna()

mi = np.min(df_OA_all[df_OA_all['Obs']>0]['Obs'])*15
ma = np.max(df_OA_all[df_OA_all['Obs']>0]['Obs'])
bins_ = 10 ** np.linspace(np.log10(mi), np.log10(ma), 50)


# %%
for st in stations: 
    #df_anom_OA = dic_df_anom_OA[st]
    df_OA_all = dic_df_OA_all[st].dropna()


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

    plt.title(f'Distribution July and August 2012-2018, {st}')

    plt.legend()
    fn = make_fn_eval('noresm_echam',f'hist_{st}')
    plt.savefig(fn, dpi=300)
    plt.savefig(fn.with_suffix('.pdf'), dpi=300)
    plt.show()

# %%
st = 'SGP'
df_OA_all = dic_df_OA_all[st].dropna()

df_OA_all.plot()

# %%
for st in stations: 
    #df_anom_OA = dic_df_anom_OA[st]
    df_OA_all = dic_df_OA_all[st].dropna()

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

    plt.title(f'Model anomaly, July and August 2012-2018, {st}')

    plt.legend()
    fn = make_fn_eval('diff_hist','hist')
    #plt.savefig(fn, dpi=300)
    #plt.savefig(fn.with_suffix('.pdf'), dpi=300)
    plt.show()

# %%
import numpy as np

# %%
for st in stations: 
    #df_anom_OA = dic_df_anom_OA[st]
    df_OA_all = dic_df_OA_all[st].dropna()
    fig, axs = plt.subplots(1,2,sharey=True, figsize=[12,5],dpi=150, sharex=True)
    ax = axs[0]
    _df = df_OA_all
    _df['hour'] = _df.index.hour
    for mo, ax in zip(models,axs.flatten()):
        sns.histplot(y=mo, x='Obs',#orbins=bins_, alpha=0.5, 
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

        ax.set_ylabel(f'{mo} OA '+'[$\mu$gm$^{-3}$]')
        ax.set_xlabel('Observed OA [$\mu$gm$^{-3}$]')



        lims = ax.get_xlim()
        ax.plot(lims,lims,'k', linewidth=.5)


    fn = make_fn_eval('_'.join(models),'scatt_OAG')
    sns.despine(fig)
    #fig.savefig(fn, dpi=300)
    #fig.savefig(fn.with_suffix('.pdf'), dpi=300)
    plt.show()

# %%
for st in stations: 
    #df_anom_OA = dic_df_anom_OA[st]
    df_OA_all = dic_df_OAG_all[st].dropna()
    fig, axs = plt.subplots(1,2,sharey=True, figsize=[12,5],dpi=150, sharex=True)
    ax = axs[0]
    _df = df_OA_all
    _df['hour'] = _df.index.hour
    for mo, ax in zip(models,axs.flatten()):
        sns.histplot(y=mo, x='Obs',#orbins=bins_, alpha=0.5, 
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

        ax.set_ylabel(f'{mo} OAG '+'[$\mu$gm$^{-3}$]')
        ax.set_xlabel('Observed OA [$\mu$gm$^{-3}$]')



        lims = ax.get_xlim()
        ax.plot(lims,lims,'k', linewidth=.5)


    fn = make_fn_eval('_'.join(models),'scatt_OAG')
    sns.despine(fig)
    #fig.savefig(fn, dpi=300)
    #fig.savefig(fn.with_suffix('.pdf'), dpi=300)
    plt.show()

# %%
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# %%
for st in stations: 
    #df_anom_OA = dic_df_anom_OA[st]
    df_OAG_all = dic_df_OAG_all[st].dropna()
    fig, axs = plt.subplots(1,2,sharey=True, figsize=[12,5],dpi=150, sharex=True)
    ax = axs[0]
    _df = df_OAG_all.resample('6H').median()
    _df['hour'] = _df.index.hour

    for mo, ax in zip(models,axs.flatten()):
        sns.histplot(y=mo, x='Obs',#orbins=bins_, alpha=0.5, 
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

        ax.set_ylabel(f'{mo} OAG '+'[$\mu$gm$^{-3}$]')
        ax.set_xlabel('Observed OA [$\mu$gm$^{-3}$]')



        lims = ax.get_xlim()
        ax.plot(lims,lims,'k', linewidth=.5)


    fn = make_fn_eval('_'.join(models),'scatt_OAG')
    sns.despine(fig)
    #fig.savefig(fn, dpi=300)
    #fig.savefig(fn.with_suffix('.pdf'), dpi=300)
    plt.show()

# %%
for st in stations: 
    #df_anom_OA = dic_df_anom_OA[st]
    df_OA_all = dic_df_OA_all[st].dropna()
    fig, axs = plt.subplots(1,2,sharey=True, figsize=[12,5],dpi=150, sharex=True)
    ax = axs[0]
    _df = df_OA_all.resample('6H').median()
    _df['hour'] = _df.index.hour

    for mo, ax in zip(models,axs.flatten()):
        sns.histplot(y=mo, x='Obs',#orbins=bins_, alpha=0.5, 
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

        ax.set_ylabel(f'{mo} OA '+'[$\mu$gm$^{-3}$]')
        ax.set_xlabel('Observed OA [$\mu$gm$^{-3}$]')



        lims = ax.get_xlim()
        ax.plot(lims,lims,'k', linewidth=.5)


    fn = make_fn_eval('_'.join(models),'scatt_OAG')
    sns.despine(fig)
    #fig.savefig(fn, dpi=300)
    #fig.savefig(fn.with_suffix('.pdf'), dpi=300)
    plt.show()

# %%
_diff = (df_OA_all.drop(['Obs'],axis=1).T - df_OA_all['Obs']).T

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
for st in stations:
    df_OA_all = dic_df_OA_all[st]
    _diff = (df_OA_all.drop(['Obs'],axis=1).T - df_OA_all['Obs']).T

    df_OA_incDiff = df_OA_all.copy()

    for mo in _diff.columns:
        print(mo)
        df_OA_incDiff[f'{mo}_diff'] = _diff[mo]
    fig, axs = plt.subplots(1,2,sharey=True, figsize=[10,5])
    ax = axs[0]
    _df = df_OA_incDiff
    _df['hour'] = _df.index.hour
    for mo, ax in zip(models,axs.flatten()):
        sns.boxenplot(y=f'{mo}_diff', x=f'hour',#orbins=bins_, alpha=0.5, 
                                     #hue=f'Obs', 
                #col = 'dir',
                ax=ax,
                #alpha=0.4,
                palette='viridis',
                
                #edgecolors=None,
                data = _df)


    lims = ax.get_xlim()

    ax.set_ylim([-15,15])
    plt.show()
#fig.savefig(fn, dpi=300)
#fig.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
for st in stations:
    df_OA_all = dic_df_OA_all[st]
    _diff = (df_OA_all.drop(['Obs'],axis=1).T - df_OA_all['Obs']).T

    df_OA_incDiff = df_OA_all.copy()

    for mo in _diff.columns:
        print(mo)
        df_OA_incDiff[f'{mo}_diff'] = _diff[mo]
    fig, axs = plt.subplots(1,2,sharey=True, figsize=[10,5])
    ax = axs[0]
    _df = df_OA_incDiff
    _df['hour'] = _df.index.hour
    for mo, ax in zip(models,axs.flatten()):
        sns.boxenplot(y=f'{mo}_diff', x=f'hour',#orbins=bins_, alpha=0.5, 
                                     #hue=f'Obs', 
                #col = 'dir',
                ax=ax,
                #alpha=0.4,
                palette='viridis',
                
                #edgecolors=None,
                data = _df)


    lims = ax.get_xlim()

    ax.set_ylim([-15,15])
    plt.show()
#fig.savefig(fn, dpi=300)
#fig.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%

# %%
df

# %%
plt.ylim([0,8])

# %%
len(df_lii['Org'][df_lii['Org']<0])#.plot()

# %%
obs_hyy_s = df_lii[(df_lii.index.month==7) |(df_lii.index.month==8)]

# %%
df = obs_hyy_s

# %%
df.columns

# %%
_df = df[['Org', 'SO4', 'NO3', 'NH4', 'Chl']].copy()


_df_q = _df.quantile([.05,.5,.95])
div = _df_q.loc[.95,:]-_df_q.loc[.05,:]

_df = (_df-_df_q.loc[.5,:])/div

_df= _df.dropna()

data = _df.values


N = 7

from sklearn.cluster import KMeans
km = KMeans(n_clusters=N, random_state = 1,init='random')
la = km.fit_predict(data)

df_wlab = df.copy().loc[_df.index]

df_wlab['lab'] = la

# %%
for x in ['SO4', 'NO3', 'NH4', 'Chl']:
    sns.scatterplot(data = df_wlab, hue='lab',x=x, y= 'Org',)
    plt.show()

# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

    # %%
    sns.scatterplot(data = df_wlab, hue='lab',x='Chl', y= 'Org',)
    plt.xlim([-.1,.2])
    plt.show()

    # %%
    sns.scatterplot(data = df_wlab, hue='lab',x='NH4', y= 'Org',)
    plt.show()

# %%
dfm = dic_st_mod_ca['SMR']['NorESM']['OsloAero_intBVOC_f19_f19_mg17_fssp'].drop(['station','lev']).to_dataframe()
dfmd = dfm.resample('d').median()

dfmdm= dfmd[(dfmd.index.month>=7 ) & (dfmd.index.month<=8 )]

sns.scatterplot(data = dfmdm, x='T',y= 'OA')
plt.ylim([0,7])

# %%
dfm = dic_st_mod_ca['SMR']['ECHAM-SALSA']['SALSA_BSOA_feedback'].isel(lev=-1).to_dataframe()
dfmd = dfm.resample('d').median()

dfmdm= dfmd[(dfmd.index.month>=7 ) & (dfmd.index.month<=8 )]

sns.scatterplot(data = dfmdm, x='tempair',y= 'OAG')
plt.ylim([0,7])

# %%
dfm = dic_st_mod_ca['SGP']['NorESM']['OsloAero_intBVOC_f19_f19_mg17_fssp'].drop(['station','lev']).to_dataframe()
dfmd = dfm.resample('d').median()

dfmdm= dfmd[(dfmd.index.month>=7 ) & (dfmd.index.month<=8 )]

sns.scatterplot(data = dfmdm, x='T',y= 'OA')
plt.ylim([0,7])
plt.ylim([0,7])
plt.xlim([290,315])

# %%
dfm = dic_st_mod_ca['SGP']['ECHAM-SALSA']['SALSA_BSOA_feedback'].isel(lev=-1).to_dataframe()
dfmd = dfm.resample('d').median()

dfmdm= dfmd[(dfmd.index.month>=7 ) & (dfmd.index.month<=8 )]

sns.scatterplot(data = dfmdm, x='tempair',y= 'OAG')
plt.ylim([0,7])
plt.ylim([0,7])
plt.xlim([290,315])

# %%
measurements_path

# %%
import xarray as xr
from pathlib import Path 

# %%
fn = measurements_path / 'ARM_data' / 'metdata_clean' / 'SGP_metdata_sgparmbeatmC1.c1_2013-2018.nc'
ds_T = xr.open_dataset(fn)

# %%
# %%
ds_T

# %%
ds_T_daily = ds_T[['temperature_sfc','relative_humidity_sfc','pressure_sfc','u_wind_sfc','v_wind_sfc']].resample({'time':'d'}).median()

# %%
ds_T_daily['temperature_sfc'].plot()

# %%
JA_month = (ds_T_daily['time.month']>=7) &(ds_T_daily['time.month']<=8)

# %%
ds_T_daily_sum = ds_T_daily.where(JA_month, drop=True)

# %%
df_T_daily_sum = ds_T_daily_sum.to_dataframe()
df_T_daily_sum

# %%

# %%
# %%
import pandas as pd

# %%

path_acsm_SGP =measurements_path / 'ARM_data' /'watson'/ 'ACSM'

#path_acsm_SGP = measurements_path/'ARM_data' /'watson'/'ACSM'


fl = list(path_acsm_SGP.glob('*.csv'))

fl.sort()

df_list = []
for f in fl:
    df_list.append(pd.read_csv(f, index_col=1))

_df_acsm_SGP = pd.concat(df_list)

_df_acsm_SGP.index = pd.to_datetime(_df_acsm_SGP.index)

df_acsm_SGP =  _df_acsm_SGP.resample('h').mean()

df_acsm_SGP.index = df_acsm_SGP.index.rename('time')

df_acsm_SGP

# %%
# %%
df_acsm_SGP['total'] =df_acsm_SGP.sum(axis=1) 
df_acsm_SGP['org_frac'] =df_acsm_SGP['Org']/ df_acsm_SGP['total']

# %%
df_acsm_SGP#['total']

# %%
# %%
import datetime

# %%
ind = df_acsm_SGP.index 
df_acsm_SGP.index = ind + datetime.timedelta(hours=-6)

# %%
import matplotlib.pyplot as plt

# %%
df_acsm_SGP_daily = df_acsm_SGP.resample('d').median()

# %%
# %%
JA_month = (df_acsm_SGP_daily.index.month>=7) &(df_acsm_SGP_daily.index.month<=8)

# %%

df_acsm_SGP_daily_sum = df_acsm_SGP_daily[JA_month]

# %%
df_acsm_SGP_daily_sum.head()

# %%
df_acsm_SGP_daily_sum.tail()

# %%
# %%
df = pd.concat([df_T_daily_sum,df_acsm_SGP_daily_sum],axis=1).dropna()

# %%
df.columns

# %%
# %%
for y in [#'temperature_sfc', 'relative_humidity_sfc', 'pressure_sfc',
       #'u_wind_sfc', 'v_wind_sfc', 
          'Org', 'NO3', 'SO4', 'NH4', 'Chl', 'total',
       'org_frac']:
    sns.scatterplot(data=df, x='Org',y=y)
    plt.show()

# %%

# %%
# %%
v = 'org_frac'

df[df[v]>=.3].plot.scatter(x='temperature_sfc', y='Org', ylim=[0,9], c=v, cmap='viridis', alpha=.5)

# %%
df.columns

# %%
# %%
_df.index

# %%
_df = df[['Org', 'NO3', 'SO4', 'NH4', 'Chl', 'total']].copy()
_df = ((_df.T/_df['total']).T).dropna()
data = _df.values


N = 7

from sklearn.cluster import KMeans
km = KMeans(n_clusters=N, random_state = 1,init='random')
la = km.fit_predict(data)

# %%
df

# %%

# %%
_df = df[['v_wind_sfc','u_wind_sfc','relative_humidity_sfc','pressure_sfc']].copy()


_df_q = _df.quantile([.05,.5,.95])
div = _df_q.loc[.95,:]-_df_q.loc[.05,:]

_df = (_df-_df_q.loc[.5,:])/div



data = _df.values


N = 7

from sklearn.cluster import KMeans
km = KMeans(n_clusters=N, random_state = 1,init='random')
la = km.fit_predict(data)

df_wlab = df.copy().loc[_df.index]

df_wlab['lab'] = la

# %%
import seaborn as sns

# %%
for la in df_wlab['lab'].unique():
    sns.scatterplot(data = df_wlab[df_wlab['lab']==la], hue='lab',x='temperature_sfc', y= 'Org',)
    plt.ylim([0,8])
    plt.show()

# %%
# %%
sns.scatterplot(data = df_wlab, hue='lab',y='Org', x= 'temperature_sfc',)
plt.ylim([0,7])
plt.xlim([290,315])

# %%
# %%
df_wlab[['Org','SO4','temperature_sfc']].cov()

# %%
# %%
sns.scatterplot(data = df_wlab, hue='lab',x='SO4', y= 'Org',)
plt.ylim([0,8])

# %%
# %%
sns.scatterplot(data = df_wlab, hue='lab',x='Chl', y= 'Org',)
plt.ylim([0,8])

# %%
# %%
sns.scatterplot(data = df_wlab, hue='lab',x='NH4', y= 'Org',)
plt.ylim([0,8])

# %%
# %%
df_wlab

# %%
# %%
sns.scatterplot(data = df_wlab, hue='lab',x='temperature_sfc', y= 'Org',)

# %%
# %%
df_wlab.plot.scatter(x='temperature_sfc', y= 'Org', c='lab')

# %%

# %%

# %%
df_wlab

# %%
# %%
fig, axs = plt.subplots(1,3, figsize=[15,5], sharex=True, sharey=True)
ax = axs[0]
sns.scatterplot(data = df_wlab,x='temperature_sfc', y= 'Org',ax = axs[0])
ax.set_ylim([0,7])
ax.set_xlim([290,315])
ax.set_title('OBS')
_df =df_wlab[['Org','temperature_sfc']]
print(_df.corr())
ax = axs[1]
dfm = dic_st_mod_ca['SGP']['NorESM']['OsloAero_intBVOC_f19_f19_mg17_fssp'].drop(['station','lev']).to_dataframe()
dfmd = dfm.resample('d').median()

dfmdm= dfmd[(dfmd.index.month>=7 ) & (dfmd.index.month<=8 )]

sns.scatterplot(data = dfmdm, x='T',y= 'OA', ax = ax)
ax.set_title('NorESM')

_df =dfmdm[['OA','T']]
print(_df.corr())
ax = axs[2]

dfm = dic_st_mod_ca['SGP']['ECHAM-SALSA']['SALSA_BSOA_feedback'].isel(lev=-1).to_dataframe()
dfmd = dfm.resample('d').median()

dfmdm= dfmd[(dfmd.index.month>=7 ) & (dfmd.index.month<=8 )]

sns.scatterplot(data = dfmdm, x='tempair',y= 'ORG_mass_conc', ax= ax)
ax.set_title('ECHAM')

_df =dfmdm[['ORG_mass_conc','tempair']]
print(_df.corr())
for ax in axs:
    ax.set_xlabel('Temperature [K]')

axs[0].set_ylabel('OA [$\mu$g/m$^3$]')

# %% jupyter={"outputs_hidden": true} tags=[]
dfmdm['ORG_mass_conc

# %%
_df =dfmdm[['ORG_mass_conc','tempair']]
print(_df.corr())

# %%
dfm = dic_st_mod_ca['SGP']['NorESM']['OsloAero_intBVOC_f19_f19_mg17_fssp'].drop(['station','lev']).to_dataframe()
dfmd = dfm.resample('d').median()

dfmdm= dfmd[(dfmd.index.month>=7 ) & (dfmd.index.month<=8 )]

sns.scatterplot(data = dfmdm, x='T',y= 'OA')
plt.ylim([0,7])
plt.ylim([0,7])
plt.xlim([290,315])

# %%
dfm = dic_st_mod_ca['SGP']['ECHAM-SALSA']['SALSA_BSOA_feedback'].isel(lev=-1).to_dataframe()
dfmd = dfm.resample('d').median()

dfmdm= dfmd[(dfmd.index.month>=7 ) & (dfmd.index.month<=8 )]

sns.scatterplot(data = dfmdm, x='tempair',y= 'OAG')
plt.ylim([0,7])
plt.ylim([0,7])
plt.xlim([290,315])
