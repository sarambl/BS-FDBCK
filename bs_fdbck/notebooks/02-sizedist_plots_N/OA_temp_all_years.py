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

from bs_fdbck.util.BSOA_datamanip import ds2df_inc_preprocessing
from bs_fdbck.util.collocate.collocateLONLAToutput import CollocateLONLATout
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
    f = f'scat_all_years_{case}_{_x}_{_y}.png'
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
from bs_fdbck.util.EBAS_data import get_ebas_dataset_Nx_daily_JA_median_df



#ds_ebas_Nx = get_ebas_dataset_with_Nx()

df_ebas_Nx, ds_ebas_Nx = get_ebas_dataset_Nx_daily_JA_median_df()#x_list = [90,100,110,120])


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
df_hyy_1y['year'] = df_hyy_1y['year'].apply(lambda x:f'{x:.0f}')

df_hyy_1y['date'] = df_hyy_1y['year']
df_hyy_1y = df_hyy_1y.set_index('date')

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
cases_orig1 = ['OsloAero_intBVOC_f19_f19_mg17_full']
cases_orig2 = ['OsloAero_intBVOC_f19_f19_mg17_ssp245']
# %%
case_mod = 'OsloAero_intBVOC_f19_f19_mg17_fssp'

# %%
 
log.ger.info(f'TIMES:****: {from_t} {to_t}')

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
       'NCONC01','N50','N150','N200',
      'CLDFREE',
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

# %%
ds1 = dic_ds[case1]
ds2 = dic_ds[case2]

# %%
from_t


# %%
st_y = from_t.split('-')[0]
mid_y_t = str(int(to_t.split('-')[0])-1)
mid_y_f = to_t.split('-')[0]
end_y = to_t2.split('-')[0]

# %%
print(st_y, mid_y_t, mid_y_f, end_y)

# %%
_ds1 = ds1.sel(time=slice(st_y, mid_y_t))
_ds2 = ds2.sel(time=slice(mid_y_f, end_y))
ds_comb = xr.concat([_ds1, _ds2], dim='time')#.sortby('time')

# %% [markdown]
# ## SELECT STATION: 
#

# %%
ds_comb = ds_comb.sel(station='SMR')

# %%
dic_ds = dict()
dic_ds[case_mod] = ds_comb

# %%
ds_comb.load()

# %%
dic_ds[case_mod]

# %% [markdown] tags=[]
# # Functions:

# %%
R = 287.058
pressure = 1000. #hPa
kg2ug = 1e9


# %%
case_mod

# %%


dic_df_sm, dic_df = ds2df_inc_preprocessing(dic_ds, model_lev_i=-1, return_summer_median=True)


# %% [markdown]
# ## Merge with observations: 

# %%
dic_df_pre = dic_df.copy()

# %%
for ca in dic_df.keys():
    dic_df[ca] = pd.merge(dic_df_pre[ca], df_hyy_1, right_on='time', left_on='time')
    dic_df[ca]['year'] = dic_df[ca].index.year

# %%
dic_df[case_mod]['N50']

# %%
df_hyy_1


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


for c in dic_df.keys():
    
    dic_df[c] = add_log(dic_df[c])
    dic_df_sm[c] = add_log(dic_df_sm[c])
        
df_joint_hyy = add_log(df_joint_hyy)
#df_joint_hyy = add_log(df_joint_hyy)
df_joint_hyy = add_log(df_joint_hyy)
df_hyy_1y = add_log(df_hyy_1y)


# %%
df_joint_hyy

# %%
ca = case_mod

# %%
mask_obs_N = dic_df[ca]['N100 (cm^-3)'].notnull()
mask_obs_OA = dic_df[ca]['OA (microgram m^-3)'].notnull()

# %%


# %%


# %%
fig, axs, cax = make_cool_grid()


# %% [markdown] tags=[]
# # Plots

# %%
df_s['log10(OA (microgram m^-3))']

# %%

#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid()

v_x = 'T_C'
v_y = 'OA'
ca = case_mod
df_s = dic_df[ca][mask_obs_N].loc['2012':]

df_sy = dic_df_sm[ca].loc['2012':]
ylims = [0,12]

xlims = [5,30]
ylab = 'OA  $\mu m^{-3}$)'

xlab = r'N$_{50}$ [cm$^{-3}$]'
xlab = r'T [$^\circ$C]'

fig, ax = plot_scatter(v_x, v_y, df_s, df_sy, ca, xlims=xlims,
                       figsize=[6,7], ax = axs[0],
                       ylims=ylims, xlab=xlab, ylab = ylab)
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title('OsloAero')


v_y = 'OA (microgram m^-3)'
v_x = 'T (degree C)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2019']

df_sy = df_hyy_1y#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x, v_y, df_s, df_sy, ca, ax = axs[1],
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab)
ax.set_title('Observations')
fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.savefig(fn, dpi=150)

plt.show()

# %% [markdown]
# ## Log scale

# %%
fig, axs, cax = make_cool_grid()

v_y = 'log10(OA)'

v_x = 'T_C'
ca = case_mod
df_s = df_joint_hyy[mask_obs_N].loc['2012':]
df_s = dic_df[ca][mask_obs_N].loc['2012':]


df_sy = dic_df_sm[ca].loc['2012':]
ylims = [-.75,1.25]
ylims =[-1,1.1]# [0,5000]


xlims = [5,30]
ylab = 'log(OA  $\mu m^{-3}$)'

xlab = r'T [$^\circ$C]'

fig, ax = plot_scatter(v_x, v_y, df_s, df_sy, ca, xlims=xlims,
                       figsize=[6,7], ax = axs[0],
                       ylims=ylims, xlab=xlab, ylab = ylab,
                       legend_loc='lower right')
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title('OsloAero')


v_y = 'log10(OA (microgram m^-3))'
v_x = 'T (degree C)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2019']

df_sy = df_hyy_1y#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x, v_y, df_s, df_sy, ca, ax = axs[1],
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab,
                       legend_loc='lower right')
ax.set_title('Observations')
fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.savefig(fn, dpi=150)

plt.show()

# %%

# %%

# %%

# %%

# %%

# %%

# %%
