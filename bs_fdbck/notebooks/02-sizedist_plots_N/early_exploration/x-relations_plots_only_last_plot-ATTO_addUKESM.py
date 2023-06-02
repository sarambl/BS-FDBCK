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
# # Nx versus T and OA

# %%
# %load_ext autoreload

# %autoreload 2


# %%
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

# %%

import numpy as np



# %%
plot_path = Path('Plots')


# %% pycharm={"name": "#%% \n"}
def make_fn_scat(case, v_x, v_y):
    _x = v_x.split('(')[0]
    _y = v_y.split('(')[0]
    f = f'scat_all_years_echam_noresm_{case}_{_x}_{_y}-ATTO_ukesm.png'
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
fn = path_measurement_data /'SourceData_Yli_Juuti2021.xls'

df_hyy_1 = pd.read_excel(fn, sheet_name=0, header=2, usecols=range(6))

df_hyy_1.head()

df_hyy_1['date'] = df_hyy_1.apply(lambda x: f'{x.year:.0f}-{x.month:02.0f}-{x.day:02.0f}', axis=1)

df_hyy_1['date'] = pd.to_datetime(df_hyy_1['date'] )



# %%

# %%
fn = path_measurement_data /'SourceData_Yli_Juuti2021.xls'

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

# %%
df_hyy_1.index = df_hyy_1.index.rename('time')

# %%
df_hyy_1.head()

# %%
from bs_fdbck.util.EBAS_data import get_ebas_dataset_Nx_daily_JA_median_df



#ds_ebas_Nx = get_ebas_dataset_with_Nx()

df_ebas_Nx, ds_ebas_Nx = get_ebas_dataset_Nx_daily_JA_median_df()#x_list = [90,100,110,120])


# %%
fn = path_measurement_data /'SourceData_Yli_Juuti2021.xls'

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
df_hyy_1y['year'] = df_hyy_1y['year'].apply(lambda x:f'{x:.0f}')

df_hyy_1y['date'] = df_hyy_1y['year']
df_hyy_1y = df_hyy_1y.set_index('date')



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

# %%

# %% [markdown]
# ### Some definitions:

# %%
models = ['NorESM','ECHAM-SALSA','UKESM']

# %%
dic_mod2case={

}

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

dic_mod2case[model_name] = case_name_echam

# %% [markdown]
# ### Settings:

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
# ### Variables

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


# %% tags=[]
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

# %% [markdown] tags=[]
# ## LOAD UKESM
#

# %%



case_name = 'CRES'
case_name_ukesm = 'CRES'

time_res = 'hour'
space_res='locations'
model_name='UKESM'

dic_mod2case[model_name] = case_name_ukesm

# %% [markdown]
# ### Settings:

# %%
from_t = '2012-01-01'
to_t = '2015-01-01'


# %%
case_mod = case_name#'OsloAero_intBVOC_f19_f19_mg17_fssp'
cases_ukesm = [case_name]

# %%

log.ger.info(f'TIMES:****: {from_t} {to_t}')

# %% [markdown]
# ### Variables

# %%
varl =[
    'tas',
    'sfmmroa',
      ]


# %% tags=[]
for case_name in cases_ukesm:
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
for ca in cases_ukesm:
    c = CollocateLONLATout(ca, from_t, to_t,
                           True,
                           'hour',
                           model_name=model_name
                          )
                          # history_field=history_field)
    ds = c.get_collocated_dataset(varl)
    if 'location' in ds.coords:
        ds = ds.rename({'location':'station'})
    dic_ds[ca]=ds.copy()

# %%
ds


# %%
dic_mod_ca['UKESM'] = dic_ds.copy()

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
cases_noresm1 = ['OsloAero_intBVOC_f09_f09_mg17_full']
cases_noresm2 = ['OsloAero_intBVOC_f09_f09_mg17_ssp245']
# %%
case_mod = 'OsloAero_intBVOC_f09_f09_mg17_fssp'
case_noresm = 'OsloAero_intBVOC_f09_f09_mg17_fssp'

# %%
model_name = 'NorESM'

dic_mod2case[model_name] = case_noresm

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
       'monoterp','GS_SO2', 'GS_H2SO4','GS_monoterp','GS_isoprene',
      ]


varl =['N100','DOD500','DOD440','ACTREL',#,'SOA_A1',
       'H2SO4','SOA_LV','COAGNUCL','FORMRATE','T',
       'NCONC01','N50','N150','N200',#'DOD500',
       #'NCONC01',
       #'SFisoprene',
       #'SFmonoterp',
       #'DOD500',
      'SFmonoterp','SFisoprene',
      
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
    print(mod)
    for ca in dic_mod_ca[mod].keys():
        print(ca)
        dic_mod_ca[mod][ca] = dic_mod_ca[mod][ca].sel(station='ATTO')
        dic_mod_ca[mod][ca].load()

# %%
_ds =dic_mod_ca['ECHAM-SALSA']['SALSA_BSOA_feedback']


# %%
_ds =dic_mod_ca['NorESM'][case_noresm]


# %%
_ds['SFisoprene'].plot()

# %% [markdown]
# # Adjust models

# %% [markdown] tags=[]
# ## ECHAM-SALSA

# %%
rn_dict_echam={
    'ORG_mass_conc' : 'OA',
    'tempair':'T',

    
}

# %%
model_lev_i=-1

# %%
from bs_fdbck.util.BSOA_datamanip import calculate_daily_median_summer,calculate_summer_median

# %%
dic_mod_ca['ECHAM-SALSA'][case_name_echam]

# %%

df, df_sm = ds2df_echam(dic_mod_ca['ECHAM-SALSA'][case_name_echam], summer_months=range(13))

# %%
df.columns


# %%

_di = {case_name_echam:df}
_dism = {case_name_echam:df_sm}

dic_df_mod_case['ECHAM-SALSA']= _di.copy()
dic_dfsm_mod_case['ECHAM-SALSA'] = _dism.copy()

# %% [markdown] tags=[]
# ## UKESM

# %%
from bs_fdbck.util.BSOA_datamanip import calculate_daily_median_summer,calculate_summer_median, ds2df_ukesm

# %%
dic_mod_ca['UKESM']

# %%
df, df_sm = ds2df_ukesm(dic_mod_ca['UKESM'][case_name_ukesm],summer_months=range(13))

# %%

_di = {case_name_ukesm:df}
_dism = {case_name_ukesm:df_sm}

dic_df_mod_case['UKESM']= _di.copy()
dic_dfsm_mod_case['UKESM'] = _dism.copy()

# %% [markdown]
# ## NorESM

# %%


dic_df_sm, dic_df = ds2df_inc_preprocessing(dic_mod_ca['NorESM'], model_lev_i=-1, return_summer_median=True, summer_months=range(13))


dic_df_mod_case['NorESM'] = dic_df.copy()
dic_dfsm_mod_case['NorESM'] = dic_df_sm.copy()


# %% [markdown]
# ## Merge with observations:

# %%
dic_df_pre = dict()#dic_df_mod_case.copy()#deep=True)
for mod in dic_df_mod_case.keys():
    dic_df_pre[mod] = dic_df_mod_case[mod].copy()

# %%
for mod in dic_df_mod_case.keys():
    print(mod)
    for ca in dic_df_mod_case[mod].keys():
        dic_df_mod_case[mod][ca] = pd.merge(dic_df_pre[mod][ca], df_hyy_1, right_on='time', left_on='time', how='outer')
        dic_df_mod_case[mod][ca]['year'] = dic_df_mod_case[mod][ca].index.year


# %%
def add_log(df, varl=None):
    if varl is None:
        varl = ['OA','N100', 'OA (microgram m^-3)','N100 (cm^-3)','N50','N150','N200']
    var_exist = df.columns
    
    varl_f = set(varl).intersection(var_exist)
    print(varl_f)
    for v in varl_f:
        df[f'log10({v})'] = np.log10(df[v])
    return df


for mod in dic_df_mod_case.keys():
    for c in dic_df_mod_case[mod].keys():
    
        dic_df_mod_case[mod][c] = add_log(dic_df_mod_case[mod][c])
        dic_dfsm_mod_case[mod][c] = add_log(dic_dfsm_mod_case[mod][c])
        
df_joint_hyy = add_log(df_joint_hyy)


# %%
mod='NorESM'

# %%
ca = case_mod

# %%
mask_obs_N = dic_df_mod_case[mod][ca]['N100 (cm^-3)'].notnull()
mask_obs_OA = dic_df_mod_case[mod][ca]['OA (microgram m^-3)'].notnull()

# %% [markdown] tags=[]
# # Plots

# %%
from bs_fdbck.util.plot.BSOA_plots import cdic_model

# %%
import seaborn as sns

# %%
cdic_model[mod]

# %%
from matplotlib import pyplot as plt, gridspec as gridspec

# %%
from bs_fdbck.util.plot.BSOA_plots import make_cool_grid2, make_cool_grid3

import scipy

# %% [markdown]
# ### Fit funcs

# %%
from bs_fdbck.util.BSOA_datamanip.fits import *

# %% [markdown] tags=[]
# ### season to month

# %%
season2month = dict(WET       = [1,2,3,4,5,6,7,          12],
                    WET_mid   =   [2,3,4,5],
                    WET_early = [1,2,3,                  12],
                    WET_late  =       [4,5,6,7],
                    DRY       =               [8,9,10,11],
                    DRY_early =           [6,7,8], 
                    DRY_late  =                 [9,10,11,12],
                   
                   
                  )


# %%
def select_months(df, season = None, month_list=None):
    if season is not None: 
        month_list = season2month[season]
    

    df['month'] = df.index.month
    return df['month'].isin(month_list)

# %%
from bs_fdbck.util.plot.BSOA_plots import cdic_model

# %% [markdown] tags=[]
# ## DRY:

# %% [markdown]
# ## Redo surface emissions: 

# %%
_df = dic_df_mod_case['ECHAM-SALSA'][case_name_echam]

# %%
_df['emi_isop_bio'].plot()

# %% [markdown]
# #### Linear

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
# fig, ax, daxs, axs_extra = make_cool_grid3()
fig, ax, daxs, axs_extra = make_cool_grid3(ncols_extra=2, nrows_extra=3,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.flatten()
#ax = axsaxs_extra

## Settings
alpha_scatt = 0.6

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


xlims = [22,40]
ylims = [0,40]


season='DRY'






# OBS: 

v_y = 'OA (microgram m^-3)'
v_x = 'T (degree C)'

ca ='OBS'
mo = 'Observations'


df_s = df_joint_hyy#.loc['2012':'2014']
"""
mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],
                    zorder=-10,

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__'
               )
popt, pov, label, func = get_exp_fit_weight(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
#ax.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}, eq. weight', linestyle='--')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)

"""

# NORESM: 
v_x = 'T_C'
v_y = 'OA'

for mo, ax_ex in zip(models, axs_extra[1:]):
    ca= dic_mod2case[mo]
    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    #popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )


    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models:
    ca= dic_mod2case[mo]

    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()

    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)



for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)

fn = make_fn_scat(f'exp_fit1_{season}', v_x, v_y)
ax.legend(frameon=False)

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %% [markdown]
# #### Exponential 

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
# fig, ax, daxs, axs_extra = make_cool_grid3()
fig, ax, daxs, axs_extra = make_cool_grid3(ncols_extra=2, nrows_extra=3,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.T.flatten()
#ax = axsaxs_extra

## Settings
alpha_scatt = 0.4

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


xlims = [22,40]
ylims = [0,40]


season='DRY'






# OBS: 

v_y = 'OA (microgram m^-3)'
v_x = 'T (degree C)'

ca ='OBS'
mo = 'Observations'


df_s = df_joint_hyy#.loc['2012':'2014']
"""
mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],
                    zorder=-10,

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__'
               )
popt, pov, label, func = get_exp_fit_weight(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
#ax.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}, eq. weight', linestyle='--')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)

"""

# NORESM: 
v_x = 'T_C'
v_y = 'OA'

for mo, ax_ex in zip(models, axs_extra[1:]):
    ca= dic_mod2case[mo]
    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo],
                    
                    
                   )

    #popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )


    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models:
    ca= dic_mod2case[mo]

    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()

    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)



for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)

fn = make_fn_scat(f'exp_fit1_{season}', v_x, v_y)
ax.legend(frameon=False)

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
# fig, ax, daxs, axs_extra = make_cool_grid3()
fig, ax, daxs, axs_extra = make_cool_grid3(ncols_extra=2, nrows_extra=3,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.T.flatten()
#ax = axsaxs_extra

## Settings
alpha_scatt = 0.4

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


xlims = [22,40]
ylims = [0,40]


season='DRY_early'






# OBS: 

v_y = 'OA (microgram m^-3)'
v_x = 'T (degree C)'

ca ='OBS'
mo = 'Observations'


df_s = df_joint_hyy#.loc['2012':'2014']
"""
mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],
                    zorder=-10,

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__'
               )
popt, pov, label, func = get_exp_fit_weight(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
#ax.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}, eq. weight', linestyle='--')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)

"""

# NORESM: 
v_x = 'T_C'
v_y = 'OA'

for mo, ax_ex in zip(models, axs_extra[1:]):
    ca= dic_mod2case[mo]
    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo],
                    
                    
                   )

    #popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )


    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models:
    ca= dic_mod2case[mo]

    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()

    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)



for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)

fn = make_fn_scat(f'exp_fit1_{season}', v_x, v_y)
ax.legend(frameon=False)

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
# fig, ax, daxs, axs_extra = make_cool_grid3()
fig, ax, daxs, axs_extra = make_cool_grid3(ncols_extra=2, nrows_extra=3,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.T.flatten()
#ax = axsaxs_extra

## Settings
alpha_scatt = 0.4

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


xlims = [22,40]
ylims = [0,40]


season='DRY_late'






# OBS: 

v_y = 'OA (microgram m^-3)'
v_x = 'T (degree C)'

ca ='OBS'
mo = 'Observations'


df_s = df_joint_hyy#.loc['2012':'2014']
"""
mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],
                    zorder=-10,

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__'
               )
popt, pov, label, func = get_exp_fit_weight(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
#ax.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}, eq. weight', linestyle='--')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)

"""

# NORESM: 
v_x = 'T_C'
v_y = 'OA'

for mo, ax_ex in zip(models, axs_extra[1:]):
    ca= dic_mod2case[mo]
    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo],
                    
                    
                   )

    #popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )


    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models:
    ca= dic_mod2case[mo]

    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()

    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)



for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)

fn = make_fn_scat(f'exp_fit1_{season}', v_x, v_y)
ax.legend(frameon=False)

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %% [markdown]
# ## Wet season

# %% [markdown]
# #### Linear

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
# fig, ax, daxs, axs_extra = make_cool_grid3()
fig, ax, daxs, axs_extra = make_cool_grid3(ncols_extra=2, nrows_extra=3,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.T.flatten()
#ax = axsaxs_extra

## Settings
alpha_scatt = 0.2

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


xlims = [22,40]
ylims = [0,30]


season='WET_early'






# OBS: 

v_y = 'OA (microgram m^-3)'
v_x = 'T (degree C)'

ca ='OBS'
mo = 'Observations'


df_s = df_joint_hyy#.loc['2012':'2014']
"""
mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],
                    zorder=-10,

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__'
               )
popt, pov, label, func = get_exp_fit_weight(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
#ax.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}, eq. weight', linestyle='--')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)

"""

# NORESM: 
v_x = 'T_C'
v_y = 'OA'

for mo, ax_ex in zip(models, axs_extra[1:]):
    ca= dic_mod2case[mo]
    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    #popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )


    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models:
    ca= dic_mod2case[mo]

    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()

    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)



for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)

fn = make_fn_scat(f'lin_fit1_{season}', v_x, v_y)
ax.legend(frameon=False)

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
# fig, ax, daxs, axs_extra = make_cool_grid3()
fig, ax, daxs, axs_extra = make_cool_grid3(ncols_extra=2, nrows_extra=3,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.T.flatten()
#ax = axsaxs_extra

## Settings
alpha_scatt = 0.2

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


xlims = [22,40]
ylims = [0,30]


season='WET_late'






# OBS: 

v_y = 'OA (microgram m^-3)'
v_x = 'T (degree C)'

ca ='OBS'
mo = 'Observations'


df_s = df_joint_hyy#.loc['2012':'2014']
"""
mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],
                    zorder=-10,

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__'
               )
popt, pov, label, func = get_exp_fit_weight(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
#ax.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}, eq. weight', linestyle='--')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)

"""

# NORESM: 
v_x = 'T_C'
v_y = 'OA'

for mo, ax_ex in zip(models, axs_extra[1:]):
    ca= dic_mod2case[mo]
    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    #popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )


    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models:
    ca= dic_mod2case[mo]

    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()

    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)



for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)

fn = make_fn_scat(f'lin_fit1_{season}', v_x, v_y)
ax.legend(frameon=False)

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
# fig, ax, daxs, axs_extra = make_cool_grid3()
fig, ax, daxs, axs_extra = make_cool_grid3(ncols_extra=2, nrows_extra=3,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.T.flatten()
#ax = axsaxs_extra

## Settings
alpha_scatt = 0.2

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


xlims = [22,40]
ylims = [0,30]


season='WET'






# OBS: 

v_y = 'OA (microgram m^-3)'
v_x = 'T (degree C)'

ca ='OBS'
mo = 'Observations'


df_s = df_joint_hyy#.loc['2012':'2014']
"""
mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],
                    zorder=-10,

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__'
               )
popt, pov, label, func = get_exp_fit_weight(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
#ax.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}, eq. weight', linestyle='--')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)

"""

# NORESM: 
v_x = 'T_C'
v_y = 'OA'

for mo, ax_ex in zip(models, axs_extra[1:]):
    ca= dic_mod2case[mo]
    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    #popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )


    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models:
    ca= dic_mod2case[mo]

    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()

    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)



for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)

fn = make_fn_scat(f'lin_fit1_{season}', v_x, v_y)
ax.legend(frameon=False)

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
# fig, ax, daxs, axs_extra = make_cool_grid3()
fig, ax, daxs, axs_extra = make_cool_grid3(ncols_extra=2, nrows_extra=3,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.T.flatten()
#ax = axsaxs_extra

## Settings
alpha_scatt = 0.2

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


xlims = [22,40]
ylims = [.1,30]


season='WET'






# OBS: 

v_y = 'OA (microgram m^-3)'
v_x = 'T (degree C)'

ca ='OBS'
mo = 'Observations'


df_s = df_joint_hyy#.loc['2012':'2014']
"""
mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],
                    zorder=-10,

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__'
               )
popt, pov, label, func = get_exp_fit_weight(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
#ax.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}, eq. weight', linestyle='--')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)

"""

# NORESM: 
v_x = 'T_C'
v_y = 'OA'

for mo, ax_ex in zip(models, axs_extra[1:]):
    ca= dic_mod2case[mo]
    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    #popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )


    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models:
    ca= dic_mod2case[mo]

    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()

    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

#ax.set_ylim(ylims)
#ax.set_xlim(xlims)
ax.set_yscale('log')


for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)

fn = make_fn_scat(f'lin_fit1_logscale{season}', v_x, v_y)
ax.legend(frameon=False)

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %% [markdown]
# #### Exponential fit

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
# fig, ax, daxs, axs_extra = make_cool_grid3()
fig, ax, daxs, axs_extra = make_cool_grid3(ncols_extra=2, nrows_extra=3,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.flatten()
#ax = axsaxs_extra

## Settings
alpha_scatt = 0.3

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


xlims = [22,40]
ylims = [0,25]


season='WET'






# OBS: 

v_y = 'OA (microgram m^-3)'
v_x = 'T (degree C)'

ca ='OBS'
mo = 'Observations'


df_s = df_joint_hyy#.loc['2012':'2014']
"""
mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],
                    zorder=-10,

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__'
               )
popt, pov, label, func = get_exp_fit_weight(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
#ax.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}, eq. weight', linestyle='--')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)

"""

# NORESM: 
v_x = 'T_C'
v_y = 'OA'

for mo, ax_ex in zip(models, axs_extra[1:]):
    ca= dic_mod2case[mo]
    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    #popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )


    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models:
    ca= dic_mod2case[mo]

    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()

    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)



for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)

fn = make_fn_scat(f'exp_fit1_{season}', v_x, v_y)
ax.legend(frameon=False)

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %% [markdown]
# ## WET_early

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
# fig, ax, daxs, axs_extra = make_cool_grid3()
fig, ax, daxs, axs_extra = make_cool_grid3(ncols_extra=2, nrows_extra=3,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.flatten()
#ax = axsaxs_extra

## Settings
alpha_scatt = 0.3

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


xlims = [22,40]
ylims = [0,25]


season='WET_early'






# OBS: 

v_y = 'OA (microgram m^-3)'
v_x = 'T (degree C)'

ca ='OBS'
mo = 'Observations'


df_s = df_joint_hyy#.loc['2012':'2014']
"""
mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],
                    zorder=-10,

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__'
               )
popt, pov, label, func = get_exp_fit_weight(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
#ax.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}, eq. weight', linestyle='--')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)

"""

# NORESM: 
v_x = 'T_C'
v_y = 'OA'

for mo, ax_ex in zip(models, axs_extra[1:]):
    ca= dic_mod2case[mo]
    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    #popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )


    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models:
    ca= dic_mod2case[mo]

    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()

    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)



for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)

fn = make_fn_scat(f'exp_fit1_{season}', v_x, v_y)
ax.legend(frameon=False)

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %% [markdown]
# ## WET_late

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
# fig, ax, daxs, axs_extra = make_cool_grid3()
fig, ax, daxs, axs_extra = make_cool_grid3(ncols_extra=2, nrows_extra=3,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.flatten()
#ax = axsaxs_extra

## Settings
alpha_scatt = 0.3

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


xlims = [22,40]
ylims = [0,25]


season='WET_late'






# OBS: 

v_y = 'OA (microgram m^-3)'
v_x = 'T (degree C)'

ca ='OBS'
mo = 'Observations'


df_s = df_joint_hyy#.loc['2012':'2014']
"""
mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],
                    zorder=-10,

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__'
               )
popt, pov, label, func = get_exp_fit_weight(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
#ax.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}, eq. weight', linestyle='--')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)

"""

# NORESM: 
v_x = 'T_C'
v_y = 'OA'

for mo, ax_ex in zip(models, axs_extra[1:]):
    ca= dic_mod2case[mo]
    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    #popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )


    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models:
    ca= dic_mod2case[mo]

    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()

    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)



for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)

fn = make_fn_scat(f'exp_fit1_{season}', v_x, v_y)
ax.legend(frameon=False)

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %% [markdown]
# ## Dry_early

# %% tags=[]
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
# fig, ax, daxs, axs_extra = make_cool_grid3()
fig, ax, daxs, axs_extra = make_cool_grid3(ncols_extra=2, nrows_extra=3,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.flatten()
#ax = axsaxs_extra

## Settings
alpha_scatt = 0.3

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


xlims = [22,40]
ylims = [0,25]


season='DRY_early'






# OBS: 

v_y = 'OA (microgram m^-3)'
v_x = 'T (degree C)'

ca ='OBS'
mo = 'Observations'


df_s = df_joint_hyy#.loc['2012':'2014']
"""
mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],
                    zorder=-10,

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__'
               )
popt, pov, label, func = get_exp_fit_weight(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
#ax.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}, eq. weight', linestyle='--')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)

"""

# NORESM: 
v_x = 'T_C'
v_y = 'OA'

for mo, ax_ex in zip(models, axs_extra[1:]):
    ca= dic_mod2case[mo]
    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    #popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )


    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models:
    ca= dic_mod2case[mo]

    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()

    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)



for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)

fn = make_fn_scat(f'exp_fit1_{season}', v_x, v_y)
ax.legend(frameon=False)

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %% [markdown]
# ## Dry_late

# %% tags=[]
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
# fig, ax, daxs, axs_extra = make_cool_grid3()
fig, ax, daxs, axs_extra = make_cool_grid3(ncols_extra=2, nrows_extra=3,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.flatten()
#ax = axsaxs_extra

## Settings
alpha_scatt = 0.3

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


xlims = [22,40]
ylims = [0,25]


season='DRY_late'






# OBS: 

v_y = 'OA (microgram m^-3)'
v_x = 'T (degree C)'

ca ='OBS'
mo = 'Observations'


df_s = df_joint_hyy#.loc['2012':'2014']
"""
mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],
                    zorder=-10,

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__'
               )
popt, pov, label, func = get_exp_fit_weight(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)
    
#ax.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}, eq. weight', linestyle='--')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)

"""

# NORESM: 
v_x = 'T_C'
v_y = 'OA'

for mo, ax_ex in zip(models, axs_extra[1:]):
    ca= dic_mod2case[mo]
    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    #popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    popt, pov, label, func = get_exp_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )


    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models:
    ca= dic_mod2case[mo]

    #mask_obs_ind_su = mask_obs_ind[mask_obs_ind.isin(dic_df_mod_case[mo][ca].index)]
    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()

    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)



for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)

fn = make_fn_scat(f'exp_fit1_{season}', v_x, v_y)
ax.legend(frameon=False)

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %%
case_mod

# %%
mo= 'NorESM'
ca = case_mod
df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

(df_s['SOA']/df_s['OA']).plot()

# %%
mo= 'NorESM'
ca = case_mod
df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

(df_s['OA']).plot()

# %%
mo= 'ECHAM-SALSA'
ca = case_name_echam
df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

#(df_s['SOA']/df_s['OA']).plot()

df_s['OA'].plot()
df_s['OAG'].plot(alpha=.5)

# %%
df_s.columns

# %%
mo= 'ECHAM-SALSA'
ca = case_name_echam
df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]

#(df_s['SOA']/df_s['OA']).plot()

df_s['OA'].plot()
df_s['OAG'].plot(alpha=.5)

# %%

# %%

# %%
mo= 'NorESM'
ca = case_mod
df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]




seas = 'WET_early'
month_ma = select_months(df_s, season=seas)
_df = df_s[month_ma]
(_df['SOA']/_df['OA']).plot(label=seas, alpha=.4)


seas = 'WET_late'
month_ma = select_months(df_s, season=seas)
_df = df_s[month_ma]
(_df['SOA']/_df['OA']).plot(label=seas, alpha=.4)


seas = 'DRY_early'
month_ma = select_months(df_s, season=seas)
_df = df_s[month_ma]
(_df['SOA']/_df['OA']).plot(label=seas, alpha=.4)


seas = 'DRY_late'
month_ma = select_months(df_s, season=seas)
_df = df_s[month_ma]
(_df['SOA']/_df['OA']).plot(label=seas, alpha=.4)

plt.title('SOA/total OA')
plt.legend()

# %%
mo= 'NorESM'
ca = case_mod
df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]




seas = 'WET_early'
month_ma = select_months(df_s, season=seas)
_df = df_s[month_ma]
(_df['OA']).plot(label=seas, alpha=.4)


seas = 'WET_late'
month_ma = select_months(df_s, season=seas)
_df = df_s[month_ma]
(_df['OA']).plot(label=seas, alpha=.4)


seas = 'DRY_early'
month_ma = select_months(df_s, season=seas)
_df = df_s[month_ma]
(_df['OA']).plot(label=seas, alpha=.4)


seas = 'DRY_late'
month_ma = select_months(df_s, season=seas)
_df = df_s[month_ma]
(_df['OA']).plot(label=seas, alpha=.4)
plt.title('OA')

plt.legend()

# %%
mo= 'NorESM'
ca = case_mod
df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind_su]




seas = 'WET_early'
month_ma = select_months(df_s, season=seas)
_df = df_s[month_ma]
(_df['SOA']).plot(label=seas, alpha=.4)


seas = 'WET_late'
month_ma = select_months(df_s, season=seas)
_df = df_s[month_ma]
(_df['SOA']).plot(label=seas, alpha=.4)


seas = 'DRY_early'
month_ma = select_months(df_s, season=seas)
_df = df_s[month_ma]
(_df['SOA']).plot(label=seas, alpha=.4)


seas = 'DRY_late'
month_ma = select_months(df_s, season=seas)
_df = df_s[month_ma]
(_df['SOA']).plot(label=seas, alpha=.4)


plt.legend()
plt.title('SOA')

# %%
month_ma = select_months(df_s, season='WET_early')

_df = df_s[month_ma]
(_df['SOA']/_df['OA']).plot()

# %%
month_ma = select_months(df_s, season='WET')

_df = df_s[month_ma]
(_df['SOA']/_df['OA']).plot()

# %%
month_ma = select_months(df_s, season='DRY')

_df = df_s[month_ma]
(_df['SOA']/_df['OA']).plot()

# %% [markdown]
# ### Residuals of fit

# %%
import scipy

# %% [markdown]
# ## N50

# %% [markdown]
# # Testing different fits

# %% [markdown]
# ### Residuals of fit

# %% [markdown]
# # OA to number: 

# %% [markdown]
# ## Dry season

# %% [markdown] tags=[]
# ### N50

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
fig, ax, daxs, axs_extra = make_cool_grid3()
#ax = axs

## Settings
alpha_scatt = 0.2

ylab = r'N$_{50}$  [cm$^{-3}$]'
xlab = r'OA [$\mu m^{-3}$]'

season = 'DRY'
xlims = [.1,50]

ylims = [30,12000]

# OBS: 
v_x = 'OA (microgram m^-3)'
v_y = 'N50'

ca ='OBS'
mo = 'Observations'
"""
df_s = df_joint_hyy#.loc['2012':'2014']

mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)

ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)
"""

# NORESM: 
v_x = 'OA'

for mo, ax_ex in zip(models[:-1], axs_extra[1:]):
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max()
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim, num=200)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )

    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )

    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

    
    
    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models[:-1]:
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)
ax.legend(frameon=True)
ax.set_yscale('log')
ax.set_xscale('log')

for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)
    ax_e.set_yscale('log')
    ax_e.set_xscale('log')

#ax.set_xscale('log')
fn = make_fn_scat(f'log_fit_{season}', v_x, v_y)

fig.tight_layout()

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)




plt.show()

# %% [markdown]
# ### N100

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
fig, ax, daxs, axs_extra = make_cool_grid3()
#ax = axs

## Settings
alpha_scatt = 0.2

ylab = r'N$_{100}$  [cm$^{-3}$]'
xlab = r'OA [$\mu m^{-3}$]'

season = 'DRY'
xlims = [.1,50]

ylims = [30,10000]

# OBS: 
v_x = 'OA (microgram m^-3)'
v_y = 'N100'

ca ='OBS'
mo = 'Observations'
"""
df_s = df_joint_hyy#.loc['2012':'2014']

mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)

ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)
"""

# NORESM: 
v_x = 'OA'

for mo, ax_ex in zip(models[:-1], axs_extra[1:]):
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max()
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim, num=200)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )

    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )

    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

    
    
    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models[:-1]:
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)
ax.legend(frameon=True)
ax.set_yscale('log')
ax.set_xscale('log')

for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)
    ax_e.set_yscale('log')
    ax_e.set_xscale('log')

#ax.set_xscale('log')
fn = make_fn_scat(f'log_fit_{season}', v_x, v_y)

fig.tight_layout()

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)




plt.show()

# %% [markdown]
# ### N200

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
fig, ax, daxs, axs_extra = make_cool_grid3()
#ax = axs

## Settings
alpha_scatt = 0.2

ylab = r'N$_{200}$  [cm$^{-3}$]'
xlab = r'OA [$\mu m^{-3}$]'

season = 'DRY'
xlims = [.1,50]

ylims = [1,10000]

# OBS: 
v_x = 'OA (microgram m^-3)'
v_y = 'N200'

ca ='OBS'
mo = 'Observations'
"""
df_s = df_joint_hyy#.loc['2012':'2014']

mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)

ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)
"""

# NORESM: 
v_x = 'OA'

for mo, ax_ex in zip(models[:-1], axs_extra[1:]):
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max()
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim, num=200)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )

    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )

    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

    
    
    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models[:-1]:
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)
ax.legend(frameon=True)
ax.set_yscale('log')
ax.set_xscale('log')

for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)
    ax_e.set_yscale('log')
    ax_e.set_xscale('log')

#ax.set_xscale('log')
fn = make_fn_scat(f'log_fit_{season}', v_x, v_y)

fig.tight_layout()

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)




plt.show()

# %% [markdown]
# ## WET season

# %% [markdown]
# ### N50

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
fig, ax, daxs, axs_extra = make_cool_grid3()
#ax = axs

## Settings
alpha_scatt = 0.2

ylab = r'N$_{50}$  [cm$^{-3}$]'
xlab = r'OA [$\mu m^{-3}$]'

season = 'WET'
xlims = [.1,50]

ylims = [30,12000]

# OBS: 
v_x = 'OA (microgram m^-3)'
v_y = 'N50'

ca ='OBS'
mo = 'Observations'

"""
df_s = df_joint_hyy#.loc['2012':'2014']

mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)

ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)
"""

# NORESM: 
v_x = 'OA'

for mo, ax_ex in zip(models[:-1], axs_extra[1:]):
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max()
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim, num=200)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )

    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )

    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

    
    
    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models[:-1]:
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)
ax.legend(frameon=True)
ax.set_yscale('log')
ax.set_xscale('log')

for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)
    ax_e.set_yscale('log')
    ax_e.set_xscale('log')

#ax.set_xscale('log')
fn = make_fn_scat(f'log_fit_{season}', v_x, v_y)

fig.tight_layout()

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)




plt.show()

# %% [markdown]
# ### N100

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
fig, ax, daxs, axs_extra = make_cool_grid3()
#ax = axs

## Settings
alpha_scatt = 0.2

ylab = r'N$_{100}$  [cm$^{-3}$]'
xlab = r'OA [$\mu m^{-3}$]'

season = 'WET'
xlims = [.1,50]

ylims = [30,10000]

# OBS: 
v_x = 'OA (microgram m^-3)'
v_y = 'N100'

ca ='OBS'
mo = 'Observations'
"""
df_s = df_joint_hyy#.loc['2012':'2014']

mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)

ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)
"""

# NORESM: 
v_x = 'OA'

for mo, ax_ex in zip(models[:-1], axs_extra[1:]):
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max()
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim, num=200)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )

    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )

    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

    
    
    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models[:-1]:
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)
ax.legend(frameon=True)
ax.set_yscale('log')
ax.set_xscale('log')

for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)
    ax_e.set_yscale('log')
    ax_e.set_xscale('log')

#ax.set_xscale('log')
fn = make_fn_scat(f'log_fit_{season}', v_x, v_y)

fig.tight_layout()

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)




plt.show()

# %% [markdown]
# ### N200

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
fig, ax, daxs, axs_extra = make_cool_grid3()
#ax = axs

## Settings
alpha_scatt = 0.2

ylab = r'N$_{200}$  [cm$^{-3}$]'
xlab = r'OA [$\mu m^{-3}$]'

season = 'WET'
xlims = [0,15]

ylims = [0,450]

# OBS: 
v_x = 'OA (microgram m^-3)'
v_y = 'N200'

ca ='OBS'
mo = 'Observations'
"""
df_s = df_joint_hyy#.loc['2012':'2014']

mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)

ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)
"""

# NORESM: 
v_x = 'OA'

for mo, ax_ex in zip(models[:-1], axs_extra[1:]):
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    popt, pov, label, func = get_exp_fit_wc(df_s,v_x,v_y, return_func=True)
    x = np.linspace(*xlims)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )

    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )

    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

    
    
    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models[:-1]:
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)
ax.legend(frameon=True)


for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)

#ax.set_xscale('log')
fn = make_fn_scat(f'log_fit_{season}', v_x, v_y)

fig.tight_layout()

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
fig, ax, daxs, axs_extra = make_cool_grid3()
#ax = axs

## Settings
alpha_scatt = 0.2

ylab = r'N$_{200}$  [cm$^{-3}$]'
xlab = r'OA [$\mu m^{-3}$]'

season = 'DRY'
xlims = [0,50]

ylims = [0,1000]

# OBS: 
v_x = 'OA (microgram m^-3)'
v_y = 'N200'

ca ='OBS'
mo = 'Observations'
"""
df_s = df_joint_hyy#.loc['2012':'2014']

mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)

ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)
"""

# NORESM: 
v_x = 'OA'

for mo, ax_ex in zip(models[:-1], axs_extra[1:]):
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    x = np.linspace(*xlims)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )

    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )

    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

    
    
    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models[:-1]:
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)
ax.legend(frameon=True)


for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)

#ax.set_xscale('log')
fn = make_fn_scat(f'log_fit_{season}', v_x, v_y)

fig.tight_layout()

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %%

# %% [markdown]
# ## Dry season early

# %% [markdown] tags=[]
# ### N50

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
fig, ax, daxs, axs_extra = make_cool_grid3()
#ax = axs

## Settings
alpha_scatt = 0.2

ylab = r'N$_{50}$  [cm$^{-3}$]'
xlab = r'OA [$\mu m^{-3}$]'

season = 'DRY_early'
xlims = [.1,50]

ylims = [30,12000]

# OBS: 
v_x = 'OA (microgram m^-3)'
v_y = 'N50'

ca ='OBS'
mo = 'Observations'
"""
df_s = df_joint_hyy#.loc['2012':'2014']

mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)

ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)
"""

# NORESM: 
v_x = 'OA'

for mo, ax_ex in zip(models[:-1], axs_extra[1:]):
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max()
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim, num=200)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )

    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )

    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

    
    
    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models[:-1]:
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)
ax.legend(frameon=True)
ax.set_yscale('log')
ax.set_xscale('log')

for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)
    ax_e.set_yscale('log')
    ax_e.set_xscale('log')

#ax.set_xscale('log')
fn = make_fn_scat(f'log_fit_{season}', v_x, v_y)

fig.tight_layout()

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)




plt.show()

# %% [markdown]
# ### N100

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
fig, ax, daxs, axs_extra = make_cool_grid3()
#ax = axs

## Settings
alpha_scatt = 0.2

ylab = r'N$_{100}$  [cm$^{-3}$]'
xlab = r'OA [$\mu m^{-3}$]'

season = 'DRY_early'
xlims = [.1,50]

ylims = [30,10000]

# OBS: 
v_x = 'OA (microgram m^-3)'
v_y = 'N100'

ca ='OBS'
mo = 'Observations'
"""
df_s = df_joint_hyy#.loc['2012':'2014']

mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)

ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)
"""

# NORESM: 
v_x = 'OA'

for mo, ax_ex in zip(models[:-1], axs_extra[1:]):
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max()
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim, num=200)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )

    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )

    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

    
    
    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models[:-1]:
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)
ax.legend(frameon=True)
ax.set_yscale('log')
ax.set_xscale('log')

for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)
    ax_e.set_yscale('log')
    ax_e.set_xscale('log')

#ax.set_xscale('log')
fn = make_fn_scat(f'log_fit_{season}', v_x, v_y)

fig.tight_layout()

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)




plt.show()

# %% [markdown]
# ### N200

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
fig, ax, daxs, axs_extra = make_cool_grid3()
#ax = axs

## Settings
alpha_scatt = 0.2

ylab = r'N$_{200}$  [cm$^{-3}$]'
xlab = r'OA [$\mu m^{-3}$]'

season = 'DRY_early'
xlims = [.1,50]

ylims = [1,10000]

# OBS: 
v_x = 'OA (microgram m^-3)'
v_y = 'N200'

ca ='OBS'
mo = 'Observations'
"""
df_s = df_joint_hyy#.loc['2012':'2014']

mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)

ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)
"""

# NORESM: 
v_x = 'OA'

for mo, ax_ex in zip(models[:-1], axs_extra[1:]):
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max()
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim, num=200)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )

    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )

    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

    
    
    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models[:-1]:
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)
ax.legend(frameon=True)
ax.set_yscale('log')
ax.set_xscale('log')

for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)
    ax_e.set_yscale('log')
    ax_e.set_xscale('log')

#ax.set_xscale('log')
fn = make_fn_scat(f'log_fit_{season}', v_x, v_y)

fig.tight_layout()

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)




plt.show()

# %% [markdown]
# ## Dry late

# %% [markdown] tags=[]
# ### N50

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
fig, ax, daxs, axs_extra = make_cool_grid3()
#ax = axs

## Settings
alpha_scatt = 0.2

ylab = r'N$_{50}$  [cm$^{-3}$]'
xlab = r'OA [$\mu m^{-3}$]'

season = 'DRY_late'
xlims = [.1,50]

ylims = [30,12000]

# OBS: 
v_x = 'OA (microgram m^-3)'
v_y = 'N50'

ca ='OBS'
mo = 'Observations'
"""
df_s = df_joint_hyy#.loc['2012':'2014']

mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)

ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)
"""

# NORESM: 
v_x = 'OA'

for mo, ax_ex in zip(models[:-1], axs_extra[1:]):
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max()
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim, num=200)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )

    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )

    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

    
    
    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models[:-1]:
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)
ax.legend(frameon=True)
ax.set_yscale('log')
ax.set_xscale('log')

for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)
    ax_e.set_yscale('log')
    ax_e.set_xscale('log')

#ax.set_xscale('log')
fn = make_fn_scat(f'log_fit_{season}', v_x, v_y)

fig.tight_layout()

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)




plt.show()

# %% [markdown]
# ### N100

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
fig, ax, daxs, axs_extra = make_cool_grid3()
#ax = axs

## Settings
alpha_scatt = 0.2

ylab = r'N$_{100}$  [cm$^{-3}$]'
xlab = r'OA [$\mu m^{-3}$]'

season = 'DRY_late'
xlims = [.1,50]

ylims = [30,10000]

# OBS: 
v_x = 'OA (microgram m^-3)'
v_y = 'N100'

ca ='OBS'
mo = 'Observations'
"""
df_s = df_joint_hyy#.loc['2012':'2014']

mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)

ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)
"""

# NORESM: 
v_x = 'OA'

for mo, ax_ex in zip(models[:-1], axs_extra[1:]):
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max()
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim, num=200)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )

    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )

    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

    
    
    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models[:-1]:
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)
ax.legend(frameon=True)
ax.set_yscale('log')
ax.set_xscale('log')

for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)
    ax_e.set_yscale('log')
    ax_e.set_xscale('log')

#ax.set_xscale('log')
fn = make_fn_scat(f'log_fit_{season}', v_x, v_y)

fig.tight_layout()

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)




plt.show()

# %% [markdown]
# ### N200

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
fig, ax, daxs, axs_extra = make_cool_grid3()
#ax = axs

## Settings
alpha_scatt = 0.2

ylab = r'N$_{200}$  [cm$^{-3}$]'
xlab = r'OA [$\mu m^{-3}$]'

season = 'DRY_late'
xlims = [.1,50]

ylims = [1,10000]

# OBS: 
v_x = 'OA (microgram m^-3)'
v_y = 'N200'

ca ='OBS'
mo = 'Observations'
"""
df_s = df_joint_hyy#.loc['2012':'2014']

mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)

ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)
"""

# NORESM: 
v_x = 'OA'

for mo, ax_ex in zip(models[:-1], axs_extra[1:]):
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max()
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim, num=200)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )

    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )

    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

    
    
    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models[:-1]:
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)
ax.legend(frameon=True)
ax.set_yscale('log')
ax.set_xscale('log')

for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)
    ax_e.set_yscale('log')
    ax_e.set_xscale('log')

#ax.set_xscale('log')
fn = make_fn_scat(f'log_fit_{season}', v_x, v_y)

fig.tight_layout()

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)




plt.show()

# %%

# %% [markdown]
# ## WET early

# %% [markdown] tags=[]
# ### N50

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
fig, ax, daxs, axs_extra = make_cool_grid3()
#ax = axs

## Settings
alpha_scatt = 0.2

ylab = r'N$_{50}$  [cm$^{-3}$]'
xlab = r'OA [$\mu m^{-3}$]'

season = 'WET_early'
xlims = [.1,50]

ylims = [30,12000]

# OBS: 
v_x = 'OA (microgram m^-3)'
v_y = 'N50'

ca ='OBS'
mo = 'Observations'
"""
df_s = df_joint_hyy#.loc['2012':'2014']

mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)

ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)
"""

# NORESM: 
v_x = 'OA'

for mo, ax_ex in zip(models[:-1], axs_extra[1:]):
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max()
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim, num=200)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )

    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )

    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

    
    
    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models[:-1]:
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)
ax.legend(frameon=True)
ax.set_yscale('log')
ax.set_xscale('log')

for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)
    ax_e.set_yscale('log')
    ax_e.set_xscale('log')

#ax.set_xscale('log')
fn = make_fn_scat(f'log_fit_{season}', v_x, v_y)

fig.tight_layout()

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)




plt.show()

# %% [markdown]
# ### N100

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
fig, ax, daxs, axs_extra = make_cool_grid3()
#ax = axs

## Settings
alpha_scatt = 0.2

ylab = r'N$_{100}$  [cm$^{-3}$]'
xlab = r'OA [$\mu m^{-3}$]'

season = 'DRY'
xlims = [.1,50]

ylims = [30,10000]

# OBS: 
v_x = 'OA (microgram m^-3)'
v_y = 'N100'

ca ='OBS'
mo = 'Observations'
"""
df_s = df_joint_hyy#.loc['2012':'2014']

mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)

ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)
"""

# NORESM: 
v_x = 'OA'

for mo, ax_ex in zip(models[:-1], axs_extra[1:]):
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max()
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim, num=200)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )

    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )

    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

    
    
    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models[:-1]:
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)
ax.legend(frameon=True)
ax.set_yscale('log')
ax.set_xscale('log')

for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)
    ax_e.set_yscale('log')
    ax_e.set_xscale('log')

#ax.set_xscale('log')
fn = make_fn_scat(f'log_fit_{season}', v_x, v_y)

fig.tight_layout()

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)




plt.show()

# %% [markdown]
# ### N200

# %%
#fig, axs = plt.subplots(1,1,dpi=150, figsize=[5,5])
fig, ax, daxs, axs_extra = make_cool_grid3()
#ax = axs

## Settings
alpha_scatt = 0.2

ylab = r'N$_{200}$  [cm$^{-3}$]'
xlab = r'OA [$\mu m^{-3}$]'

season = 'DRY'
xlims = [.1,50]

ylims = [1,10000]

# OBS: 
v_x = 'OA (microgram m^-3)'
v_y = 'N200'

ca ='OBS'
mo = 'Observations'
"""
df_s = df_joint_hyy#.loc['2012':'2014']

mask_obs_ind = df_s[[v_x,v_y]].notna().index

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = ax,
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

sns.scatterplot(x=v_x,
                y=v_y, 
                data = df_s, 
                color=cdic_model[mo], 
                alpha=alpha_scatt, 
                ax = axs_extra[0],
                facecolor='none',
                edgecolor=cdic_model[mo],

                label='__nolegend__')

popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
x = np.linspace(*xlims)

ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    
ax.plot(x, func(x, *popt), c=cdic_model[mo], linewidth=2,label=f'{mo}: {label}')
axs_extra[0].plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}')

axs_extra[0].plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}')

    
    

#fig.suptitle('Observations')
sns.kdeplot(
    x= df_s[v_x], 
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['x'],
)

sns.kdeplot(
    y=df_s[v_y],
    color=cdic_model[mo], 
    label=mo,
    ax = daxs['y'],
)
"""

# NORESM: 
v_x = 'OA'

for mo, ax_ex in zip(models[:-1], axs_extra[1:]):
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt, 
                    label='__nolegend__',
                    ax = ax,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )
    sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    facecolor='none',
                    edgecolor=cdic_model[mo]
                    
                    
                   )

    popt, pov, label, func = get_linear_fit(df_s,v_x,v_y, return_func=True)
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max()
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim, num=200)
    ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    
    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{mo}: {label}')
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=2,label=f'{mo}: {label}',
             )

    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
              )

    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))
    

    
    
    
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
fig.suptitle(f'ATTO, {season} season, 2012-2018')


# 
for mo in models[:-1]:
    ca= dic_mod2case[mo]

    df_s = dic_df_mod_case[mo][ca]#.loc[mask_obs_ind]

    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()


    sns.kdeplot(#x=v_x,
                    x= df_s[v_x], 
                    color=cdic_model[mo], 
                    label=mo,
                    ax = daxs['x'],
                    
                   )

    sns.kdeplot(#x=v_x,
                    
        y=df_s[v_y],
        #vertical=True,
                    color=cdic_model[mo], 
                    #alpha=alpha_scatt, 
                    label=mo,
                    ax = daxs['y'],
                    
                   )

ax.set_ylim(ylims)
ax.set_xlim(xlims)
ax.legend(frameon=True)
ax.set_yscale('log')
ax.set_xscale('log')

for ax_e in axs_extra:
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_ylim(ax.get_ylim())
    ax_e.set_xlim(ax.get_xlim())
    #ax_e.set_xticklabels
    ax_e.axes.xaxis.set_ticklabels([])
    ax_e.axes.yaxis.set_ticklabels([])
    #ax_e.axes.yaxis.set_visible(False)

    sns.despine(ax = ax_e)
    ax_e.set_yscale('log')
    ax_e.set_xscale('log')

#ax.set_xscale('log')
fn = make_fn_scat(f'log_fit_{season}', v_x, v_y)

fig.tight_layout()

fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)




plt.show()

# %%

# %%

# %%

# %%

# %%

# %%
