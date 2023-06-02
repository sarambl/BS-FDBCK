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

# %% [markdown]
# # Nx versus T and OA

# %%
# %load_ext autoreload

# %autoreload 2


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

# %%
plot_path = Path('Plots')


# %%
def make_fn_eval(case,_type):
    #_x = v_x.split('(')[0]
    #_y = v_y.split('(')[0]
    f = f'evalNX_echam_noresm_{case}_{_type}.png'
    return plot_path /f


# %%

import numpy as np



# %%
plot_path = Path('Plots')


# %% pycharm={"name": "#%% \n"}
def make_fn_scat(case, v_x, v_y):
    _x = v_x.split('(')[0]
    _y = v_y.split('(')[0]
    f = f'scat_all_years_echam_noresm_{case}_{_x}_{_y}.png'
    return plot_path /f


# %%
plot_path.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# # Load observational data: 

# %%
import pandas as pd


# %%
from bs_fdbck.constants import path_measurement_data

# %%
fn = path_measurement_data / 'SourceData_Yli_Juuti2021.xls'

df_hyy_1 = pd.read_excel(fn, sheet_name=0, header=2, usecols=range(6))

df_hyy_1.head()

df_hyy_1['date'] = df_hyy_1.apply(lambda x: f'{x.year:.0f}-{x.month:02.0f}-{x.day:02.0f}', axis=1)

df_hyy_1['date'] = pd.to_datetime(df_hyy_1['date'] )



# %%

# %%
fn = path_measurement_data / 'SourceData_Yli_Juuti2021.xls'

df_hyy_2 = pd.read_excel(fn, sheet_name=2, header=2, usecols=range(6))

df_hyy_2.head()

df_hyy_2['date'] = df_hyy_2.apply(lambda x: f'{x.year:.0f}-{x.month:02.0f}-{x.day:02.0f}', axis=1)

df_hyy_2['date'] = pd.to_datetime(df_hyy_2['date'] )

df_hyy_2

df_hyy_2['date'] = df_hyy_2.apply(lambda x: f'{x.year:.0f}-{x.month:02.0f}-{x.day:02.0f}', axis=1)

df_hyy_2['date'] = pd.to_datetime(df_hyy_2['date'] )

df_hyy_2 = df_hyy_2.set_index('date')

# %%
df_hyy_2

# %%
df_hyy_1 = df_hyy_1.set_index('date')

# %%
df_hyy_1 = pd.merge(df_hyy_1,df_hyy_2[['AOD_340 nm','AOD_500 nm']], left_index=True, right_index=True,how="outer",)

# %%
df_hyy_1

# %%
import datetime

# %%
from bs_fdbck.util.EBAS_data import get_ebas_dataset_Nx_daily_JA_median_df, get_ebas_dataset_with_Nx

x_list = [50, 100, 200]


#ds_ebas_Nx = get_ebas_dataset_with_Nx()

ds_ebas_Nx= get_ebas_dataset_with_Nx(x_list=x_list)#x_list=x_list,station=station, path_ebas=path_ebas)#x_list = [90,100,110,120])

df_ebas_Nx = ds_ebas_Nx[[f'N{x:d}' for x in x_list]].to_dataframe().resample('h').mean()
df_ebas_Nx['JA'] = (df_ebas_Nx.index.month == 7) | (df_ebas_Nx.index.month == 8)
df_ebas_Nx = df_ebas_Nx[df_ebas_Nx['JA']] 

df_ebas_Nx.index = df_ebas_Nx.index + datetime.timedelta(hours=3)

# %%
fn = path_measurement_data / 'SourceData_Yli_Juuti2021.xls'

df_hyy_1y = pd.read_excel(fn, sheet_name=0, header=2, usecols=range(7,12),nrows=7)

df_hyy_1y.head()
df_hyy_1y= df_hyy_1y.rename({'year.1':'year',
                            'T (degree C).1':'T (degree C)',
                             'OA (microgram m^-3).1':'OA (microgram m^-3)',
                             'N100 (cm^-3).1':'N100 (cm^-3)'
                            }, axis=1)
#df_hyy_1y['year'] = pd.to_datetime(df_hyy_1y['year'].apply(x:str(x)))

df_hyy_1y

# %%
df_hyy_1

# %%
df_hyy_1y['year'] = df_hyy_1y['year'].apply(lambda x:f'{x:.0f}')

df_hyy_1y['date'] = df_hyy_1y['year']
df_hyy_1y = df_hyy_1y.set_index('date')

# %%
df_hyy_1['date'] = df_hyy_1.apply(lambda x: f'{x.year:.0f}-{x.month:02.0f}-{x.day:02.0f}', axis=1)

df_hyy_1['date'] = pd.to_datetime(df_hyy_1['date'] )


df_hyy_1 = df_hyy_1.set_index('date')

# %%
df_hyy_1.index = df_hyy_1.index.rename('time') 

# %%
df_hyy_1['N100 (cm^-3)'].plot.hist(bins=50, alpha=0.4, label='obs')

plt.show()



# %% [markdown] tags=[]
# ## Why is my method 20% off their method? Is it integration?

# %%

df_joint_hyy = pd.merge(df_ebas_Nx, df_hyy_1, left_index=True, right_index=True)
(df_joint_hyy['N100']).loc['2014-07':'2014-09'].plot(label='mine')
(df_joint_hyy['N100 (cm^-3)']).loc['2014-07':'2014-09'].plot(label='orig')
plt.legend()
plt.show()



print(df_joint_hyy['N100'][df_joint_hyy['N100 (cm^-3)'].notnull()].mean()/df_joint_hyy['N100 (cm^-3)'].mean())
# %% [markdown]
# # Read in model data:

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
models = ['ECHAM-SALSA','NorESM']

di_mod2cases = dict()
#for mod in models:
#    di_mod2cases[mod]=dict()

# %%
dic_mod_ca = dict()
dic_df_mod_case = dict()
dic_dfsm_mod_case = dict()

# %% [markdown] tags=[]
# ### LOAD ECHAM SALSA

# %%



case_name = 'SALSA_BSOA_feedback'
case_name_echam = 'SALSA_BSOA_feedback'
time_res = 'hour'
space_res='locations'
model_name='ECHAM-SALSA'


case_mod = case_name#'OsloAero_intBVOC_f19_f19_mg17_fssp'
cases_echam = [case_name]
di_mod2cases[model_name]=cases_echam

# %% [markdown]
# ## Settings:

# %%
# %%
from_t = '2012-01-01'
to_t = '2019-01-01'


# %%
case_mod = case_name#'OsloAero_intBVOC_f19_f19_mg17_fssp'
cases_echam = [case_name]

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
       'H2SO4','SOA_LV','COAGNUCL','FORMRATE','T',
       'NCONC01','N50','N150','N200',#'DOD500',
       #'DOD500',
      
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
case_mod


# %%
dic_ds = {case_mod: ds_comb}


# %%
dic_mod_ca['NorESM'] = dic_ds.copy()

# %% [markdown]
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
model_lev_i=-1

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
from bs_fdbck.util.BSOA_datamanip import calculate_daily_median_summer,calculate_summer_median

# %%
from bs_fdbck.util.BSOA_datamanip import calculate_daily_median_summer,calculate_summer_median, mask4summer

# %%
standard_air_density = 100*pressure/(R*temperature)


# %%
def ds2df_echam(ds_st, model_lev_i=-1, take_daily_median=True):
    
    

    # N50, N100 etc:
    nvars = ['mmrtrN3','mmrtrN50','mmrtrN100','mmrtrN200','mmrtrN250','mmrtrN500']
    for v in nvars:
        if v in ds_st:
            if ds_st[v].attrs['units']=='kg-1':
                v_new = v[5:]
                print(v_new)
                # /kg_air --> /m3_air by multiplying by density kg_air/m3_air
                # then from /m3-->/cm3: multiply by 1e-6
                ds_st[v_new] = ds_st[v]*standard_air_density*1e-6
                ds_st[v_new].attrs['units'] = 'm-3'
                long_name = ds_st[v].attrs['long_name'] 
                ds_st[v_new].attrs['long_name'] = 'number concentration ' + long_name.split('_')[-1]
                ds_st[v_new].attrs['description'] = 'number concentration ' + long_name.split('_')[-1]
                
    vars_kg2kg = ['ORG_mass', ]
    
    for v in vars_kg2kg:
        
        if v in ds_st:
            if ds_st[v].attrs['units']=='kg kg-1':
                
                v_new = v + '_conc'
                # kg_aero/kg_air --> kg_m3: multiply by density kg_air/m3_air
                # kg_aero/m3_air --> ug/m3_air: multiply by 
                ds_st[v_new] = ds_st[v]*standard_air_density*1e9
                
                ds_st[v_new].attrs['units'] = 'kg/m-3'
                long_name = ds_st[v].attrs['long_name'] 
                ds_st[v_new].attrs['long_name'] = 'number concentration ' + long_name.split('_')[-1]
                ds_st[v_new].attrs['description'] = 'number concentration ' + long_name.split('_')[-1]

    print(v_new)
    rn_sub ={k:rn_dict_echam[k] for k in rn_dict_echam if ((k in ds_st.data_vars) & (rn_dict_echam[k] not in ds_st.data_vars))}
    ds_st = ds_st.rename(rn_sub)
    ds_st_ilev = ds_st.isel(lev = model_lev_i)
    
    if 'T' in ds_st_ilev:
        ds_st_ilev['T_C'] = ds_st_ilev['T']- 273.15


    if take_daily_median:
        df = calculate_daily_median_summer(ds_st_ilev)
        df_sm = calculate_summer_median(df)
    else:
        _ds = mask4summer(ds_st_ilev)
        df = _ds.to_dataframe()
        df = df[df['isSummer'].notnull()]
        df_sm = None
    
    
    return df, df_sm



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
df, df_sm = ds2df_echam(dic_mod_ca['ECHAM-SALSA'][case_name_echam], take_daily_median=False, model_lev_i = -1)

# %%
df

# %%
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
from bs_fdbck.util.BSOA_datamanip import calculate_daily_median_summer,calculate_summer_median, mask4summer

# %%



dic_df_sm, dic_df = ds2df_inc_preprocessing(dic_mod_ca['NorESM'], model_lev_i=-1, 
                                            return_summer_median=True, take_daily_median=False)


dic_df_mod_case['NorESM'] = dic_df.copy()
dic_dfsm_mod_case['NorESM'] = dic_df_sm.copy()


# %% [markdown]
# ## Merge with observations:

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
dic_df_pre = dict()#dic_df_mod_case.copy()#deep=True)
for mod in dic_df_mod_case.keys():
    dic_df_pre[mod] = dic_df_mod_case[mod].copy()

# %%
rn_dic_obs = { x:f'{x}_obs' for x in df_ebas_Nx.columns}

# %%
df_ebas_Nx

pd.merge(dic_df_pre[mod][ca], df_ebas_Nx.rename(rn_dic_obs,axis=1), right_on='time', left_on='time')

# %%
for mod in dic_df_mod_case.keys():
    print(mod)
    for ca in dic_df_mod_case[mod].keys():
        _df_mod = dic_df_pre[mod][ca]
        _df_obs = df_ebas_Nx.rename(rn_dic_obs,axis=1)
        dic_df_mod_case[mod][ca] = pd.merge(_df_mod, _df_obs, right_on='time', left_on='time')
        dic_df_mod_case[mod][ca]['year'] = dic_df_mod_case[mod][ca].index.year

# %%
dic_df_mod_case[mod][ca]#['year']


# %%
def add_log(df, varl=None):
    if varl is None:
        varl = ['OA','N100', 'OA (microgram m^-3)','N100 (cm^-3)','N50','N150','N200']
    
    #
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
        
df_joint_hyy = add_log(df_joint_hyy)

# %% [markdown] tags=[]
# # Plots

# %%
dic_df_Nx = dict()
for x in x_list:
    vn = f'N{x:d}'
    print(vn)
    _df = pd.DataFrame(df_ebas_Nx[vn].rename('Obs')) 
    for mo in models:
        for ca in di_mod2cases[mo]:
            if len(di_mod2cases[mo])==1:
                keyname = mo
            else:
                keyname = f'{mo}: {ca}'
            df_mod = dic_df_pre[mo][ca]
            _df[keyname] = df_mod[vn]
        _df = _df[_df[mo].notna()]
    _df = _df[_df['Obs'].notna()]
    
    dic_df_Nx[vn] = _df.copy()
    
            

# %% [markdown]
# Small error due to time change in models but only 3 data points each summer. 
#

# %% [markdown]
# ### Calculate anomaly from daily average

# %%

dic_df_Nx_anom = dict()
for x in x_list:
    vn = f'N{x:d}'
    print(vn)
    _df = dic_df_Nx[vn] 
    _df_anom = _df - _df.resample('D').mean().resample('h').ffill()
    dic_df_Nx_anom[vn] = _df_anom.copy()



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
vn = 'N50'

df_plot = dic_df_Nx_anom[vn]

df_plot2 = dic_df_Nx[vn]

for mod in df_plot.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_plot[mod].groupby(df_plot.index.hour).mean().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.set_title(f"{vn}$'$: Average diurnal anomaly") 

#ax.legend(frameon=False)


ax = axs[0]


for mod in df_plot2.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_plot2[mod].groupby(df_plot2.index.hour).mean().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.legend(frameon=False)
ax.set_title("$\overline{%s}$: Average"%vn) 
#ax.set_ylim([0,9])'

fn = make_fn_eval('_'.join(models)+vn, 'diurnal_var')

fig.savefig(fn.with_suffix('.png'))
fig.savefig(fn.with_suffix('.pdf'))

# %%
fig, axs = plt.subplots(2,1, sharex=True, figsize=[5,5], dpi=100)
ax = axs[1]
#pl_obs = df_anom_OA['Obs'].groupby(df_anom_OA['obs'].index.hour).mean()
#pl_obs.plot(ax=ax,label='Observations', c='k')
vn = 'N50'

df_plot = dic_df_Nx_anom[vn]

df_plot2 = dic_df_Nx[vn]

for mod in df_plot.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_plot[mod].groupby(df_plot.index.hour).median().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.set_title(f"{vn}$'$: Median diurnal anomaly") 

#ax.legend(frameon=False)


ax = axs[0]


for mod in df_plot2.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_plot2[mod].groupby(df_plot2.index.hour).median().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.legend(frameon=False)
ax.set_title("$\overline{%s}$: Median"%vn) 
#ax.set_ylim([0,9])'

fn = make_fn_eval('_'.join(models)+vn, 'diurnal_var')

fig.savefig(fn.with_suffix('.png'))
fig.savefig(fn.with_suffix('.pdf'))

# %%
fig, axs = plt.subplots(2,1, sharex=True, figsize=[5,5], dpi=100)
ax = axs[1]
#pl_obs = df_anom_OA['Obs'].groupby(df_anom_OA['obs'].index.hour).mean()
#pl_obs.plot(ax=ax,label='Observations', c='k')
vn = 'N100'

df_plot = dic_df_Nx_anom[vn]

df_plot2 = dic_df_Nx[vn]

for mod in df_plot.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_plot[mod].groupby(df_plot.index.hour).mean().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.set_title(f"{vn}$'$: Average diurnal anomaly") 
#ax.legend(frameon=False)


ax = axs[0]


for mod in df_plot2.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_plot2[mod].groupby(df_plot2.index.hour).mean().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.legend(frameon=False)
ax.set_title("$\overline{%s}$: Average"%vn) 
#ax.set_ylim([0,9])'

fn = make_fn_eval('_'.join(models)+vn, 'diurnal_var')

fig.savefig(fn.with_suffix('.png'))
fig.savefig(fn.with_suffix('.pdf'))

# %%
fig, axs = plt.subplots(2,1, sharex=True, figsize=[5,5], dpi=100)
ax = axs[1]
#pl_obs = df_anom_OA['Obs'].groupby(df_anom_OA['obs'].index.hour).mean()
#pl_obs.plot(ax=ax,label='Observations', c='k')
vn = 'N100'

df_plot = dic_df_Nx_anom[vn]

df_plot2 = dic_df_Nx[vn]

for mod in df_plot.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_plot[mod].groupby(df_plot.index.hour).median().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.set_title(f"{vn}$'$: Median diurnal anomaly") 
#ax.legend(frameon=False)


ax = axs[0]


for mod in df_plot2.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_plot2[mod].groupby(df_plot2.index.hour).median().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.legend(frameon=False)
ax.set_title("$\overline{%s}$: Median"%vn) 
#ax.set_ylim([0,9])'

fn = make_fn_eval('_'.join(models)+vn, 'diurnal_var_median')

fig.savefig(fn.with_suffix('.png'))
fig.savefig(fn.with_suffix('.pdf'))

# %%
fig, axs = plt.subplots(2,1, sharex=True, figsize=[5,5], dpi=100)
ax = axs[1]
#pl_obs = df_anom_OA['Obs'].groupby(df_anom_OA['obs'].index.hour).mean()
#pl_obs.plot(ax=ax,label='Observations', c='k')
vn = 'N200'

df_plot = dic_df_Nx_anom[vn]

df_plot2 = dic_df_Nx[vn]

for mod in df_plot.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_plot[mod].groupby(df_plot.index.hour).mean().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.set_title(f"{vn}$'$: Average diurnal anomaly") 

#ax.legend(frameon=False)


ax = axs[0]


for mod in df_plot2.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_plot2[mod].groupby(df_plot2.index.hour).mean().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.legend(frameon=False)
ax.set_title("$\overline{%s}$: Average"%vn) 
#ax.set_ylim([0,9])'

fn = make_fn_eval('_'.join(models)+vn, 'diurnal_var')

fig.savefig(fn.with_suffix('.png'))
fig.savefig(fn.with_suffix('.pdf'))

# %%
fig, axs = plt.subplots(2,1, sharex=True, figsize=[5,5], dpi=100)
ax = axs[1]
#pl_obs = df_anom_OA['Obs'].groupby(df_anom_OA['obs'].index.hour).mean()
#pl_obs.plot(ax=ax,label='Observations', c='k')
vn = 'N200'

df_plot = dic_df_Nx_anom[vn]

df_plot2 = dic_df_Nx[vn]

for mod in df_plot.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_plot[mod].groupby(df_plot.index.hour).median().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.set_title(f"{vn}$'$: Median diurnal anomaly") 

#ax.legend(frameon=False)


ax = axs[0]


for mod in df_plot2.columns:
    if mod =='Obs': c = 'k'
    else: c=None
    ls = linestyle_dic[mod]
    df_plot2[mod].groupby(df_plot2.index.hour).median().plot(ax=ax,c=c,
                                                               linestyle=ls,
                                                               label=mod)#'OsloAeroSec',)# c='k')
ax.legend(frameon=False)
ax.set_title("$\overline{%s}$: Median"%vn) 
#ax.set_ylim([0,9])'

fn = make_fn_eval('_'.join(models)+vn, 'diurnal_var_median')

fig.savefig(fn.with_suffix('.png'))
fig.savefig(fn.with_suffix('.pdf'))

# %%
     
(df_plot.loc['2014-08-15':'2014-09-01']/df_plot.loc['2014-08-15':'2014-09-01'].max()).plot()

# %%
     
f, ax = plt.subplots(figsize=[20,10])
df_plot.loc['2014-08-15':'2014-09-01'].plot(ax = ax)
ax.set_ylim([-100,100])

# %%
vn

# %%
     
f, ax = plt.subplots(figsize=[20,10])
df_plot2.loc['2014-08-15':'2014-09-01'].plot(ax = ax)
ax.set_ylim([0,200])

# %%
vn = 'N50'

df_plot = dic_df_Nx[vn]




mi = np.min(df_plot[df_plot['Obs']>0]['Obs'])*10
ma = np.max(df_plot[df_plot['Obs']>0]['Obs'])*1
bins_ = 10 ** np.linspace(np.log10(mi), np.log10(ma), 50)

df_plot['Obs'].plot.hist(bins=bins_, alpha=0.5, 
                                     color='k',
                                     label='Observations'
                                    )
for mo in models:
    df_plot[mo].plot.hist(bins=bins_, alpha=0.5, 
                                     #color=None,
                                     label=mo
                                    )
plt.xscale('log')
#_mod_an.plot.hist(bins=bins_, alpha=0.5,label='OsloAero, SOA')

plt.xlabel('%s [#m$^{-3}$]'%vn)

plt.title('Distribution July and August 2012-2018, Hyytiälä')

plt.legend()
fn = make_fn_eval('noresm_echam_'+vn,'hist')
plt.savefig(fn, dpi=300)
plt.savefig(fn.with_suffix('.pdf'), dpi=300)
plt.show()

# %%
vn = 'N50'

df_plot = dic_df_Nx[vn]



for mo in models:
    (df_plot[mo]-df_plot['Obs']).plot.hist(#bins=bins_, 
        alpha=0.5, 
        bins=100,
                                     #color=None,
                                     label=mo
                                    )
#plt.xscale('log')
#_mod_an.plot.hist(bins=bins_, alpha=0.5,label='OsloAero, SOA')
plt.xlim([-4e3,4e3])
plt.xlabel('%s [#m$^{-3}$]'%vn)

plt.title('Model anomaly, July and August 2012-2018, Hyytiälä')

plt.legend()
fn = make_fn_eval('diff_hist_'+vn,'hist')
plt.savefig(fn, dpi=300)
plt.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
vn = 'N100'

df_plot = dic_df_Nx[vn]




mi = np.min(df_plot[df_plot['Obs']>0]['Obs'])*1
ma = np.max(df_plot[df_plot['Obs']>0]['Obs'])*1
bins_ = 10 ** np.linspace(np.log10(mi), np.log10(ma), 50)

df_plot['Obs'].plot.hist(bins=bins_, alpha=0.5, 
                                     color='k',
                                     label='Observations'
                                    )
for mo in models:
    df_plot[mo].plot.hist(bins=bins_, alpha=0.5, 
                                     #color=None,
                                     label=mo
                                    )
plt.xscale('log')
#_mod_an.plot.hist(bins=bins_, alpha=0.5,label='OsloAero, SOA')

plt.xlabel('%s [#m$^{-3}$]'%vn)

plt.title('Distribution July and August 2012-2018, Hyytiälä')

plt.legend()
fn = make_fn_eval('noresm_echam_'+vn,'hist')
plt.savefig(fn, dpi=300)
plt.savefig(fn.with_suffix('.pdf'), dpi=300)
plt.show()

# %%
vn = 'N100'

df_plot = dic_df_Nx[vn]



for mo in models:
    (df_plot[mo]-df_plot['Obs']).plot.hist(#bins=bins_, 
        alpha=0.5, 
        bins=100,
                                     #color=None,
                                     label=mo
                                    )
#plt.xscale('log')
#_mod_an.plot.hist(bins=bins_, alpha=0.5,label='OsloAero, SOA')
plt.xlim([-2e3,2e3])
plt.xlabel('%s [#m$^{-3}$]'%vn)

plt.title('Model anomaly, July and August 2012-2018, Hyytiälä')

plt.legend()
fn = make_fn_eval('diff_hist_'+vn,'hist')
plt.savefig(fn, dpi=300)
plt.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
vn = 'N200'

df_plot = dic_df_Nx[vn]




mi = np.min(df_plot[df_plot['Obs']>0]['Obs'])*2e-1
ma = np.max(df_plot[df_plot['Obs']>0]['Obs'])*1
bins_ = 10 ** np.linspace(np.log10(mi), np.log10(ma), 50)

df_plot['Obs'].plot.hist(bins=bins_, alpha=0.5, 
                                     color='k',
                                     label='Observations'
                                    )
for mo in models:
    df_plot[mo].plot.hist(bins=bins_, alpha=0.5, 
                                     #color=None,
                                     label=mo
                                    )
plt.xscale('log')
#_mod_an.plot.hist(bins=bins_, alpha=0.5,label='OsloAero, SOA')

plt.xlabel('%s [#m$^{-3}$]'%vn)

plt.title('Distribution July and August 2012-2018, Hyytiälä')

plt.legend()
fn = make_fn_eval('noresm_echam_'+vn,'hist')
plt.savefig(fn, dpi=300)
plt.savefig(fn.with_suffix('.pdf'), dpi=300)
plt.show()

# %%
vn = 'N200'

df_plot = dic_df_Nx[vn]



for mo in models:
    (df_plot[mo]-df_plot['Obs']).plot.hist(#bins=bins_, 
        alpha=0.5, 
        bins=100,
                                     #color=None,
                                     label=mo
                                    )
#plt.xscale('log')
#_mod_an.plot.hist(bins=bins_, alpha=0.5,label='OsloAero, SOA')
#plt.xlim([-2.5e2,2.5e2])
plt.xlabel('%s [#m$^{-3}$]'%vn)

plt.title('Model anomaly, July and August 2012-2018, Hyytiälä')

plt.legend()
fn = make_fn_eval('diff_hist_'+vn,'hist')
plt.savefig(fn, dpi=300)
plt.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
import numpy as np

# %%
import seaborn as sns

# %%
fig, axs = plt.subplots(1,2,sharey=True, figsize=[12,5], sharex= True)

ax = axs[0]
#_df = df_OA_all
vn = 'N50'

_df = dic_df_Nx[vn]


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
#ax.set_ylim([0,30])

#ax.set_xlim([0,15])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([1e2,6e3])
    ax.set_ylim([1e2,6e3])

    ax.set_xlabel(f'{mo} {vn} '+'[$\mu$gm$^{-3}$]')
    ax.set_ylabel(f'Observed {vn}'+' [$\mu$gm$^{-3}$]')



    lims = ax.get_xlim()
    ax.plot(lims,lims,'k', linewidth=.5)


fn = make_fn_eval('_'.join(models),'scatt_'+vn)
fig.savefig(fn, dpi=300)
fig.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
fig, axs = plt.subplots(1,2,sharey=True, figsize=[12,5], sharex= True)
ax = axs[0]
#_df = df_OA_all
vn = 'N100'

_df = dic_df_Nx[vn]


_df['hour'] = _df.index.hour
for mo, ax in zip(models,axs.flatten()):
    sns.histplot(x=mo, y='Obs',#orbins=bins_, alpha=0.5, 
                                     #hue='hour', 
                #col = 'dir',
                ax=ax,
                #alpha=0.4,
                cmap = sns.color_palette("mako_r", as_cmap=True),
                                      cbar=True, cbar_kws=dict(shrink=.75),

                log_scale=(True, True),
                
                #edgecolors=None,
                data = _df)
#ax.set_ylim([0,30])

#ax.set_xlim([0,15])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([2e1,7e3])
    ax.set_ylim([2e1,7e3])
    #ax.set_ylim([0.1,30])

    ax.set_xlabel(f'{mo} {vn} '+'[$\mu$gm$^{-3}$]')
    ax.set_ylabel(f'Observed {vn}'+' [$\mu$gm$^{-3}$]')



    lims = ax.get_xlim()
    ax.plot(lims,lims,'k', linewidth=.5)


fn = make_fn_eval('_'.join(models),'scatt_'+vn)
fig.savefig(fn, dpi=300)
fig.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
fig, axs = plt.subplots(1,2,sharey=True, figsize=[12,5], sharex= True)
ax = axs[0]
#_df = df_OA_all
vn = 'N200'

_df = dic_df_Nx[vn]


_df['hour'] = _df.index.hour
for mo, ax in zip(models,axs.flatten()):
    sns.histplot(x=mo, y='Obs',#orbins=bins_, alpha=0.5, 
                                     #hue='hour', 
                #col = 'dir',
                ax=ax,
                #alpha=0.4,
                #palette='viridis',
                cmap = sns.color_palette("mako_r", as_cmap=True),
                     cbar=True, cbar_kws=dict(shrink=.75),

                 log_scale=(True, True),
                
                #edgecolors=None,
                data = _df)
#ax.set_ylim([0,30])

#ax.set_xlim([0,15])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([2e0,1e3])
    ax.set_ylim([2e0,1e3])
    #ax.set_ylim([0.1,30])

    ax.set_xlabel(f'{mo} {vn} '+'[$\mu$gm$^{-3}$]')
    ax.set_ylabel(f'Observed {vn}'+' [$\mu$gm$^{-3}$]')



    lims = ax.get_xlim()
    ax.plot(lims,lims,'k', linewidth=.5)


fn = make_fn_eval('_'.join(models),'scatt_'+vn)
fig.savefig(fn, dpi=300)
fig.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
vn_list = [f'N{x:d}' for x in x_list]

# %%
dic_df_source = {so: pd.DataFrame() for so in dic_df_Nx[vn].columns}

# %%
for mo in dic_df_Nx[vn].columns:
    for vn in dic_df_Nx.keys():
        dic_df_source[mo][vn] = dic_df_Nx[vn][mo]

# %%
fig, axs = plt.subplots(1,3,sharey=True, figsize=[17,5], sharex= True)

ax = axs[0]
#_df = df_OA_all
vn1 = 'N200'
vn2 = 'N50'
so = 'Obs'


_df['hour'] = _df.index.hour
for so, ax in zip(dic_df_source.keys(), axs):
    _df = dic_df_source[so]
    #cmap = sns.cubehelix_palette(start=1, light=1, as_cmap=True)
    cmap = sns.color_palette("mako_r", as_cmap=True)

    sns.histplot(x=vn1, y=vn2,
                        #hue='hour', 
                ax=ax,
                #alpha=0.4,
                palette='viridis',
                log_scale=(True, True),
                edgecolors=None,
             cmap=cmap, 
                     cbar=True, cbar_kws=dict(shrink=.75),

                fill=True,
        #clip=([1e1, 1e4],[1e1, 1e4],), 
                #cut=10,
        #thresh=0.1, levels=15,
        #ax=ax,

                data = _df)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    

    ax.set_xlim([2,1e3])
    ax.set_ylim([.7e2,7e3])
    #ax.set_ylim([0.1,30])

    ax.set_xlabel(f'{so} {vn1} '+'[$\mu$gm$^{-3}$]')
    ax.set_ylabel(f'{so} {vn2}'+' [$\mu$gm$^{-3}$]')


    lims = ax.get_xlim()
    #ax.plot(lims,lims,'k', linewidth=.5)


fn = make_fn_eval('_'.join(models),'scatt_'+vn)
#fig.savefig(fn, dpi=300)
#fig.savefig(fn.with_suffix('.pdf'), dpi=300)

sns.despine(fig)


# %%
fig, axs = plt.subplots(1,3,sharey=True, figsize=[17,5], sharex= True)

ax = axs[0]
#_df = df_OA_all
vn1 = 'N200'
vn2 = 'N100'
so = 'Obs'


_df['hour'] = _df.index.hour
for so, ax in zip(dic_df_source.keys(), axs):
    _df = dic_df_source[so]
    #cmap = sns.cubehelix_palette(start=1, light=1, as_cmap=True)
    cmap = sns.color_palette("mako_r", as_cmap=True)

    sns.histplot(x=vn1, y=vn2,
                        #hue='hour', 
                ax=ax,
                #alpha=0.4,
                palette='viridis',
                log_scale=(True, True),
                edgecolors=None,
             cmap=cmap, 
                     cbar=True, cbar_kws=dict(shrink=.75),

                fill=True,
        #clip=([1e1, 1e4],[1e1, 1e4],), 
                #cut=10,
        #thresh=0.1, levels=15,
        #ax=ax,

                data = _df)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    

    ax.set_xlim([2,1e3])
    ax.set_ylim([1e1,7e3])
    #ax.set_ylim([0.1,30])

    ax.set_xlabel(f'{so} {vn1} '+'[$\mu$gm$^{-3}$]')
    ax.set_ylabel(f'{so} {vn2}'+' [$\mu$gm$^{-3}$]')


    lims = ax.get_xlim()
    #ax.plot(lims,lims,'k', linewidth=.5)


fn = make_fn_eval('_'.join(models),'scatt_'+vn)
#fig.savefig(fn, dpi=300)
#fig.savefig(fn.with_suffix('.pdf'), dpi=300)

sns.despine(fig)


# %%
fig, axs = plt.subplots(1,3,sharey=True, figsize=[17,5], sharex= True)

ax = axs[0]
#_df = df_OA_all
vn1 = 'N50'
vn2 = 'N100'
so = 'Obs'


_df['hour'] = _df.index.hour
for so, ax in zip(dic_df_source.keys(), axs):
    _df = dic_df_source[so]
    #cmap = sns.cubehelix_palette(start=1, light=1, as_cmap=True)
    cmap = sns.color_palette("mako_r", as_cmap=True)

    sns.histplot(x=vn1, y=vn2,
                        #hue='hour', 
                ax=ax,
                #alpha=0.4,
                palette='viridis',
                log_scale=(True, True),
                edgecolors=None,
             cmap=cmap, 
                     cbar=True, cbar_kws=dict(shrink=.75),

                fill=True,
        #clip=([1e1, 1e4],[1e1, 1e4],), 
                #cut=10,
        #thresh=0.1, levels=15,
        #ax=ax,

                data = _df)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    

    ax.set_xlim([5e1,1e4])
    ax.set_ylim([1e1,7e3])
    #ax.set_ylim([0.1,30])

    ax.set_xlabel(f'{so} {vn1} '+'[$\mu$gm$^{-3}$]')
    ax.set_ylabel(f'{so} {vn2}'+' [$\mu$gm$^{-3}$]')


    lims = ax.get_xlim()
    #ax.plot(lims,lims,'k', linewidth=.5)


fn = make_fn_eval('_'.join(models),'scatt_'+vn)
#fig.savefig(fn, dpi=300)
#fig.savefig(fn.with_suffix('.pdf'), dpi=300)

sns.despine(fig)


# %%
