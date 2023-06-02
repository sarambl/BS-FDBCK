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
    f = f'scat_all_years_echam_noresm_{case}_{_x}_{_y}.png'
    return plot_path /f


# %%
plot_path.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# # Load observational data: 

# %%
import pandas as pd


# %%
from bs_fdbck.constants import measurements_path

# %%
fn = measurements_path /'SourceData_Yli_Juuti2021.xls'

df_hyy_1 = pd.read_excel(fn, sheet_name=0, header=2, usecols=range(6))

df_hyy_1.head()

df_hyy_1['date'] = df_hyy_1.apply(lambda x: f'{x.year:.0f}-{x.month:02.0f}-{x.day:02.0f}', axis=1)

df_hyy_1['date'] = pd.to_datetime(df_hyy_1['date'] )



# %%

# %%
fn = measurements_path /'SourceData_Yli_Juuti2021.xls'

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
fn = measurements_path /'SourceData_Yli_Juuti2021.xls'

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
from_time = '2012-01'
to_time = '2012-02'
time_res = 'hour'
space_res='locations'
model_name='ECHAM-SALSA'



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
standard_air_density = 100*pressure/(R*temperature)


# %%
def ds2df_echam(ds_st, model_lev_i=-1):
    
    

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

    
    rn_sub ={k:rn_dict_echam[k] for k in rn_dict_echam if ((k in ds_st.data_vars) & (rn_dict_echam[k] not in ds_st.data_vars))}
    ds_st = ds_st.rename(rn_sub)
    ds_st_ilev = ds_st.isel(lev = model_lev_i)
    
    if 'T' in ds_st_ilev:
        ds_st_ilev['T_C'] = ds_st_ilev['T']- 273.15



    df = calculate_daily_median_summer(ds_st_ilev)
    df_sm = calculate_summer_median(df)
    
    
    return df, df_sm

df, df_sm = ds2df_echam(dic_mod_ca['ECHAM-SALSA'][case_name_echam])

# %%


_di = {case_name_echam:df}
_dism = {case_name_echam:df_sm}

dic_df_mod_case['ECHAM-SALSA']= _di.copy()
dic_dfsm_mod_case['ECHAM-SALSA'] = _dism.copy()

# %% [markdown]
# ### NorESM

# %%


dic_df_sm, dic_df = ds2df_inc_preprocessing(dic_mod_ca['NorESM'], model_lev_i=-1, return_summer_median=True)


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
        dic_df_mod_case[mod][ca] = pd.merge(dic_df_pre[mod][ca], df_hyy_1, right_on='time', left_on='time')
        dic_df_mod_case[mod][ca]['year'] = dic_df_mod_case[mod][ca].index.year

# %%
dic_df_mod_case[mod][ca]#['year']


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

# %%
ca = case_mod

# %%
mask_obs_N = dic_df_mod_case[mod][ca]['N100 (cm^-3)'].notnull()
mask_obs_OA = dic_df_mod_case[mod][ca]['OA (microgram m^-3)'].notnull()

# %%


# %%


# %%
fig, axs, cax = make_cool_grid()


# %% [markdown] tags=[]
# # Plots

# %%
def fix_ax_labs(axs, x=True, y=True):
    if len(axs.shape)==1:
        _axs = np.expand_dims(axs, axis=0)
    else:
        _axs = axs
        
    
    #print(axs)
    if x:
        for ax in _axs[-1,:]:
            
            print(ax)
            x_ticks_new = ax.get_xticklabels()#[1:-1]
            t = x_ticks_new[-1]
            nt = matplotlib.text.Text(t.get_position()[0],t.get_position()[1])
            x_ticks_new[-1] = nt
            ax.set_xticklabels(x_ticks_new)
    if y:
        for ax in _axs[:,0]:
            #t = ax.get_xticklabels()[0]
            #nt = matplotlib.text.Text(t.get_position()[0],t.get_position()[1])
            y_ticks_new = ax.get_yticklabels()#[1:-1]
            #x_ticks_new[0] = nt
            t = y_ticks_new[-1]
            nt = matplotlib.text.Text(t.get_position()[0],t.get_position()[1])
            y_ticks_new[-1] = nt
            ax.set_yticklabels(y_ticks_new)
    return
        #    x_ticks_new = ax.get_xticklabels()[1:-1]
#    ax.set_xticklabels(x_ticks_new)

#fix_ax_labs(axs, y=False)


# %% [markdown]
# ## AOT:

# %%
df_joint_hyy

# %%

# %%
obs_mask.index

# %%
df_s

# %%

# %%
df_s.loc['2012-07-20']

# %%
#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid(ncols=3)


xlims = [5,30]
ylims = [0,0.5]
xlab = r'T  [$^\circ$C]'

ylab = r'AOT'


# OBS: 
ax = axs[2]
v_y = 'AOD_500 nm'
v_x = 'T (degree C)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2014']

df_sy = None#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab, 
                       ax = ax,
                      add_cbar=False)
print(ax.get_xticklabels())

ax.set_title('Observations')
obs_mask = df_s[[v_x,v_y]].dropna()




v_x = 'T_C'
v_y = 'DOD500'


# NORESM: 
ax = axs[0]
mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca].loc['2012':]


df_s = df_s.loc[obs_mask.index]
fig, ax = plot_scatter(v_x,v_y, df_s, None, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title(mod)


# ECHAM:
mod = 'ECHAM-SALSA'
v_y = 'aot550nm'

ca = case_name_echam
ax = axs[1]

df_s = dic_df_mod_case[mod][ca].loc['2012':]

df_s = df_s.loc[obs_mask.index]

fig, ax = plot_scatter(v_x,v_y, df_s, None, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
ax.set_title( mod)


# OBS: 
ax = axs[2]
v_y = 'AOD_500 nm'
v_x = 'T (degree C)'

fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.show()
#fix_ax_labs(axs, y=False)
#fix_ax_labs(axs, y=False)

#fig.savefig(fn, dpi=150)

# %%
#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid(ncols=3)

xlab = r'OA [$\mu m^{-3}$]'

ylab = r'AOT 500'

#xlims = [5,30]
xlims = [0,12]

ylims = [0,0.5]


# OBS: 
ax = axs[2]
v_y = 'AOD_500 nm'
#v_x = 'T (degree C)'
v_x = 'OA (microgram m^-3)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2014']

df_sy = None#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab, 
                       ax = ax,
                      add_cbar=False)
print(ax.get_xticklabels())

ax.set_title('Observations')
obs_mask = df_s[[v_x,v_y]].dropna()



v_x = 'OA'
v_y = 'DOD500'


# NORESM: 
ax = axs[0]
mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca].loc['2012':]
df_s = df_s.loc[obs_mask.index]

df_sy = None#dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title(mod)


# ECHAM:
mod = 'ECHAM-SALSA'
v_y = 'aot550nm'

ca = case_name_echam
ax = axs[1]

df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]
df_s = df_s.loc[obs_mask.index]

df_sy =  None


fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
ax.set_title( mod)



fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.show()
#fix_ax_labs(axs, y=False)
#fix_ax_labs(axs, y=False)

#fig.savefig(fn, dpi=150)

# %% [markdown]
# ## OA:

# %% [markdown] tags=[]
# ## N50

# %%
#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid(ncols=3)

v_x = 'T_C'
v_y = 'log10(OA)'

xlab = r'T  [$^\circ$C]'

ylab = r'log10(OA) [$\mu m^{-3}$]'

xlims = [5,30]
ylims = [-1,1.5]

# NORESM: 
ax = axs[0]
mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title(mod)


# ECHAM:
mod = 'ECHAM-SALSA'
ca = case_name_echam
ax = axs[1]

df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
ax.set_title( mod)


# OBS: 
ax = axs[2]
v_y = 'log10(OA (microgram m^-3))'
v_x = 'T (degree C)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2014']

df_sy = None#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab, 
                       ax = ax,
                      add_cbar=False)
print(ax.get_xticklabels())

ax.set_title('Observations')

fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.show()
#fix_ax_labs(axs, y=False)
#fix_ax_labs(axs, y=False)

#fig.savefig(fn, dpi=150)

# %%
fix_ax_labs(axs, y=False)

fig.savefig(fn, dpi=150)
fig

# %%
axs[0].get_xticklabels()

# %%

#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid(ncols=3)

v_x = 'T_C'

v_y = 'OA'
xlab = r'T  [$^\circ$C]'

ylab = r'OA [$\mu m^{-3}$]'

xlims = [5,30]

ylims = [0,12]

# NORESM: 
ax = axs[0]
mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
ax.set_title(mod)

# ECHAM:
mod = 'ECHAM-SALSA'
ca = case_name_echam
ax = axs[1]

df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title( mod)


# OBS: 
ax = axs[2]
v_y = 'OA (microgram m^-3)'
v_x = 'T (degree C)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2014']

df_sy = None#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab, 
                       ax = ax,
                      add_cbar=False)
ax.set_title('Observations')

fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.savefig(fn, dpi=150)


plt.show()

# %%
fix_ax_labs(axs, y=False)

fig.savefig(fn, dpi=150)
fig

# %% [markdown]
# ## N50

# %%

#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid(ncols=3)

v_x = 'OA'
v_y = 'N50'

xlab = 'OA  [$\mu m^{-3}$]'
ylab = r'N$_{50}$ [cm$^{-3}$]'


xlims = [0,12]

ylims = [0,5000]

# NORESM: 
ax = axs[0]
mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
ax.set_title(mod)

# ECHAM:
mod = 'ECHAM-SALSA'
ca = case_name_echam
ax = axs[1]

df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title( mod)


# OBS: 
ax = axs[2]
v_x = 'OA (microgram m^-3)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2014']

df_sy = None#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab, 
                       ax = ax,
                      add_cbar=False)
ax.set_title('Observations')

fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.savefig(fn, dpi=150)


plt.show()

# %%

#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid(ncols=3)

v_x = 'T_C'

v_y = 'N50'
xlab = r'T [$^\circ$C]'

ylab = r'N$_{50}$ [cm$^{-3}$]'


xlims = [5,30]

ylims = [0,5000]


# NORESM: 
ax = axs[0]
mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
ax.set_title(mod)

# ECHAM:
mod = 'ECHAM-SALSA'
ca = case_name_echam
ax = axs[1]

df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title( mod)


# OBS: 
ax = axs[2]
v_x = 'T (degree C)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2014']

df_sy = None#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab, 
                       ax = ax,
                      add_cbar=False)
ax.set_title('Observations')

fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.savefig(fn, dpi=150)


plt.show()

# %% [markdown]
# ## Log scale

# %%

#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid(ncols=3)

v_x = 'T_C'

v_y = 'log10(N50)'
xlab = r'T [$^\circ$C]'

ylab = r'N$_{50}$ [cm$^{-3}$]'


xlims = [5,30]

ylims = [2.25,4]
# xlab = 'OA  $\mu m^{-3}$)'
xlab = r'T [$^\circ$C]'

ylab = r'log10(N$_{50}$ [cm$^{-3}$])'


# NORESM: 
ax = axs[0]
mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
ax.set_title(mod)

# ECHAM:
mod = 'ECHAM-SALSA'
ca = case_name_echam
ax = axs[1]

df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title( mod)


# OBS: 
ax = axs[2]
v_x = 'T (degree C)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2014']

df_sy = None#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab, 
                       ax = ax,
                      add_cbar=False)
ax.set_title('Observations')

fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.savefig(fn, dpi=150)


plt.show()

# %% [markdown] tags=[]
# ## N100

# %%

#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid(ncols=3)

v_x = 'OA'
v_y = 'N100'

xlab = 'OA  [$\mu m^{-3}$]'
ylab = r'N$_{100}$ [cm$^{-3}$]'


xlims = [0,12]

ylims =None# [0,5000]

# NORESM: 
ax = axs[0]
mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
ax.set_title(mod)

# ECHAM:
mod = 'ECHAM-SALSA'
ca = case_name_echam
ax = axs[1]

df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title( mod)


# OBS: 
ax = axs[2]
v_x = 'OA (microgram m^-3)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2014']

df_sy = None#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab, 
                       ax = ax,
                      add_cbar=False)
ax.set_title('Observations')

fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.savefig(fn, dpi=150)


plt.show()

# %%

#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid(ncols=3)

v_x = 'T_C'

v_y = 'N100'
xlab = r'T [$^\circ$C]'

ylab = r'N$_{100}$ [cm$^{-3}$]'


xlims = [5,30]

ylims = None#[0,5000]


# NORESM: 
ax = axs[0]
mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
ax.set_title(mod)

# ECHAM:
mod = 'ECHAM-SALSA'
ca = case_name_echam
ax = axs[1]

df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title( mod)


# OBS: 
ax = axs[2]
v_x = 'T (degree C)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2014']

df_sy = None#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab, 
                       ax = ax,
                      add_cbar=False)
ax.set_title('Observations')

fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.savefig(fn, dpi=150)


plt.show()

# %%

#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid(ncols=3)

v_x = 'OA'
v_y = 'N50'

xlab = 'OA  [$\mu m^{-3}$]'
ylab = r'N$_{50}$ [cm$^{-3}$]'


xlims = [0,12]

ylims = [0,5000]

# NORESM: 
ax = axs[0]
mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
ax.set_title(mod)

# ECHAM:
mod = 'ECHAM-SALSA'
ca = case_name_echam
ax = axs[1]

df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title( mod)


# OBS: 
ax = axs[2]
v_x = 'OA (microgram m^-3)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2014']

df_sy = None#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab, 
                       ax = ax,
                      add_cbar=False)
ax.set_title('Observations')

fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.savefig(fn, dpi=150)


plt.show()

# %%

#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid(ncols=3)

v_x = 'T_C'

v_y = 'N50'
xlab = r'T [$^\circ$C]'

ylab = r'N$_{50}$ [cm$^{-3}$]'


xlims = [5,30]

ylims = [0,5000]


# NORESM: 
ax = axs[0]
mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
ax.set_title(mod)

# ECHAM:
mod = 'ECHAM-SALSA'
ca = case_name_echam
ax = axs[1]

df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title( mod)


# OBS: 
ax = axs[2]
v_x = 'T (degree C)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2014']

df_sy = None#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab, 
                       ax = ax,
                      add_cbar=False)
ax.set_title('Observations')

fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.savefig(fn, dpi=150)


plt.show()

# %% [markdown]
# ## Log scale

# %%

#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid(ncols=3)

v_x = 'T_C'

v_y = 'log10(N100)'
xlab = r'T [$^\circ$C]'

ylab = r'N$_{100}$ [cm$^{-3}$]'


xlims = [5,30]

ylims = None#[2.25,4]
# xlab = 'OA  $\mu m^{-3}$)'
xlab = r'T [$^\circ$C]'

ylab = r'log10(N$_{100}$ [cm$^{-3}$])'


# NORESM: 
ax = axs[0]
mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
ax.set_title(mod)

# ECHAM:
mod = 'ECHAM-SALSA'
ca = case_name_echam
ax = axs[1]

df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title( mod)


# OBS: 
ax = axs[2]
v_x = 'T (degree C)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2014']

df_sy = None#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab, 
                       ax = ax,
                      add_cbar=False)
ax.set_title('Observations')

fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.savefig(fn, dpi=150)


plt.show()

# %% [markdown]
# ## N200

# %%

#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid(ncols=3)

v_x = 'OA'
v_y = 'N200'

xlab = 'OA  [$\mu m^{-3}$]'
ylab = r'N$_{200}$ [cm$^{-3}$]'


xlims = [0,12]

ylims = [0,700]

# NORESM: 
ax = axs[0]
mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
ax.set_title(mod)

# ECHAM:
mod = 'ECHAM-SALSA'
ca = case_name_echam
ax = axs[1]

df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title( mod)


# OBS: 
ax = axs[2]
v_x = 'OA (microgram m^-3)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2014']

df_sy = None#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab, 
                       ax = ax,
                      add_cbar=False)
ax.set_title('Observations')

fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.savefig(fn, dpi=150)


plt.show()

# %%

#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid(ncols=3)

v_x = 'T_C'

v_y = 'N200'
xlab = r'T [$^\circ$C]'

ylab = r'N$_{200}$ [cm$^{-3}$]'


xlims = [5,30]

ylims = [0,700]


# NORESM: 
ax = axs[0]
mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
ax.set_title(mod)

# ECHAM:
mod = 'ECHAM-SALSA'
ca = case_name_echam
ax = axs[1]

df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title( mod)


# OBS: 
ax = axs[2]
v_x = 'T (degree C)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2014']

df_sy = None#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab, 
                       ax = ax,
                      add_cbar=False)
ax.set_title('Observations')

fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.savefig(fn, dpi=150)


plt.show()

# %% [markdown]
# ## Log scale

# %%

#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid(ncols=3)

v_x = 'T_C'

v_y = 'log10(N200)'

xlims = [5,30]

ylims = [0.5,3]
# xlab = 'OA  $\mu m^{-3}$)'
xlab = r'T [$^\circ$C]'

ylab = r'log10(N$_{200}$ [cm$^{-3}$])'


# NORESM: 
ax = axs[0]
mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
ax.set_title(mod)

# ECHAM:
mod = 'ECHAM-SALSA'
ca = case_name_echam
ax = axs[1]

df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
#ax.hlines(2000, 5,30, color='k', linewidth=1)
ax.set_title( mod)


# OBS: 
ax = axs[2]
v_x = 'T (degree C)'

ca ='OBS'
df_s = df_joint_hyy#.loc['2012':'2014']

df_sy = None#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
#xlims = [5,30]
#ylims = [0,2000]
fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab, 
                       ax = ax,
                      add_cbar=False, 
                      legend_loc= 4)
ax.set_title('Observations')

fn = make_fn_scat(f'obs_{case_mod}', v_x, v_y)

fig.savefig(fn, dpi=150)


plt.show()

# %% [markdown]
# ## The end
#

# %%
import seaborn as sns

# %%
v_x = 'N50'
v_y = 'N200'

mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]


f, axs = plt.subplots(1,3, figsize=[15,5])
ax = axs[0]

sns.scatterplot(
    x=v_x,
    y=v_y,
    data=df_s,#~df_mod['OA_mid_range']].reset_index(),
    hue='OA',
    ax = ax,
    #palette=palette_OA,
)
ax.set_title(f'MODEL: {mod}')

mod = 'ECHAM-SALSA'
ca = case_name_echam
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]
ax = axs[1]
sns.scatterplot(
    x=v_x,
    y=v_y,
    data=df_s,#~df_mod['OA_mid_range']].reset_index(),
    hue='OA',
    ax = ax,
    #palette=palette_OA,
)
ax.set_title(f'MODEL: {mod}')


ax = axs[2]
sns.scatterplot(
    x=v_x,
    y=v_y,
    data=df_joint_hyy,
    hue='OA (microgram m^-3)',
    ax = ax,
    #palette=palette_OA,
)
ax.set_title('OBSERVATIONS')

#print(len(df[~df['OA_mid_range']]))

# %%
v_x = 'N100'
v_y = 'N200'


mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]


f, axs = plt.subplots(1,3, figsize=[15,5])
ax = axs[0]

sns.scatterplot(
    x=v_x,
    y=v_y,
    data=df_s,#~df_mod['OA_mid_range']].reset_index(),
    hue='OA',
    ax = ax,
    #palette=palette_OA,
)
ax.set_title(f'MODEL: {mod}')

mod = 'ECHAM-SALSA'
ca = case_name_echam
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]
ax = axs[1]
sns.scatterplot(
    x=v_x,
    y=v_y,
    data=df_s,#~df_mod['OA_mid_range']].reset_index(),
    hue='OA',
    ax = ax,
    #palette=palette_OA,
)
ax.set_title(f'MODEL: {mod}')


ax = axs[2]
sns.scatterplot(
    x=v_x,
    y=v_y,
    data=df_joint_hyy,
    hue='OA (microgram m^-3)',
    ax = ax,
    #palette=palette_OA,
)
ax.set_title('OBSERVATIONS')

#print(len(df[~df['OA_mid_range']]))

# %%
v_x = 'N50'
v_y = 'N100'

mod = 'NorESM'
ca = case_noresm
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]


f, axs = plt.subplots(1,3, figsize=[15,5])
ax = axs[0]

sns.scatterplot(
    x=v_x,
    y=v_y,
    data=df_s,#~df_mod['OA_mid_range']].reset_index(),
    hue='OA',
    ax = ax,
    #palette=palette_OA,
)
ax.set_title(f'MODEL: {mod}')

mod = 'ECHAM-SALSA'
ca = case_name_echam
df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]
ax = axs[1]
sns.scatterplot(
    x=v_x,
    y=v_y,
    data=df_s,#~df_mod['OA_mid_range']].reset_index(),
    hue='OA',
    ax = ax,
    #palette=palette_OA,
)
ax.set_title(f'MODEL: {mod}')


ax = axs[2]
sns.scatterplot(
    x=v_x,
    y=v_y,
    data=df_joint_hyy,
    hue='OA (microgram m^-3)',
    ax = ax,
    #palette=palette_OA,
)
ax.set_title('OBSERVATIONS')

#print(len(df[~df['OA_mid_range']]))

# %% [markdown]
# ## ALL

# %%
axs[0]

# %%
ylim_dic=dict(
    N50=[0,6000],
    N100 = [0,2600],
    N200 = [0,600],
    
)


# %%
def fix_ax_labs(axs):
    for ax in axs[:,0].flatten():
        #t = ax.get_yticklabels()[0]
        #nt = matplotlib.text.Text(t.get_position()[0],t.get_position()[1])
        x_ticks_new = ax.get_yticklabels()#[1:-1]
        #x_ticks_new[0] = nt
        t = ax.get_yticklabels()[-1]
        nt = matplotlib.text.Text(t.get_position()[0],t.get_position()[1])
        x_ticks_new[-1] = nt
        ax.set_yticklabels(x_ticks_new)
    for ax in axs[-1,:].flatten():
        t = ax.get_xticklabels()[0]
        nt = matplotlib.text.Text(t.get_position()[0],t.get_position()[1])
        x_ticks_new = ax.get_xticklabels()#[1:-1]
        x_ticks_new[0] = nt
        t = ax.get_xticklabels()[-1]
        nt = matplotlib.text.Text(t.get_position()[0],t.get_position()[1])
        x_ticks_new[-1] = nt
        ax.set_xticklabels(x_ticks_new)



# %%

#fig, axs = plt.subplots(1,2, figsize=[12,4], sharey=True,)
fig, axs, cax = make_cool_grid(ncols=3, nrows = 3, )#w_plot=3.5)

fontsize= 15

v_x = 'OA'
v_y = 'N200'

xlab = 'OA  [$\mu m^{-3}$]'
ylab = r'N$_{200}$ [cm$^{-3}$]'


xlims = [0,12]

ylims = None

# NORESM: 
for i, v_y in enumerate(['N50','N100','N200']):
    v_x = 'OA'
    ylims=ylim_dic[v_y]
    nu = v_y[1:]
    ylab = r'N$_{%s}$ [cm$^{-3}$]'%nu

    ax = axs[i, 0 ]
    mod = 'NorESM'
    ca = case_noresm
    df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

    df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

    fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                           ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
    ax.set_title(mod, fontsize=fontsize)

    # ECHAM:
    mod = 'ECHAM-SALSA'
    ca = case_name_echam
    ax = axs[i,1]

    df_s = dic_df_mod_case[mod][ca][mask_obs_OA].loc['2012':]

    df_sy = dic_dfsm_mod_case[mod][ca].loc['2012':]

    fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,xlims=xlims,
                       figsize=[6,7],ax = ax,
                       ylims=ylims, xlab=xlab, ylab = ylab,
                      add_cbar=False,
                      legend_loc=1)
    #ax.hlines(2000, 5,30, color='k', linewidth=1)
    
    ax.set_title( mod, fontsize=fontsize)


    # OBS: 
    ax = axs[i,2]
    v_x = 'OA (microgram m^-3)'

    ca ='OBS'
    df_s = df_joint_hyy#.loc['2012':'2014']

    df_sy = None#df_joint_hyy.loc['2012':'2014'] #f_hyy_1.resample('Y').median()
    #xlims = [5,30]
    #ylims = [0,2000]
    fig, ax = plot_scatter(v_x,v_y, df_s, df_sy, ca,
                       xlims=xlims, ylims=ylims, xlab=xlab, ylab = ylab, 
                       ax = ax,
                      add_cbar=False)
    ax.set_title('Observations', fontsize=fontsize)

for ax in axs.flatten():
    xlab = ax.get_xlabel()
    ax.set_xlabel(xlab, fontsize=fontsize)
    ylab = ax.get_ylabel()
    ax.set_ylabel(ylab, fontsize=fontsize)
for ax in axs[1:,:].flatten():
    ax.set_title('')
fn = make_fn_scat(f'obs_{case_noresm}_{case_name_echam}_all', v_x, v_y)
#fix_ax_labs(axs)

#fix_ax_labs(axs)

fig.savefig(fn, dpi=150)


plt.show()

# %%
fix_ax_labs(axs)
fig
fig.savefig(fn, dpi=150)
fig

# %%
import matplotlib.text

# %%
nt = matplotlib.text.Text(t.get_position()[0],t.get_position()[1])#.set_label('')

# %%
ax = axs[0,0]
ax.get_yticklabels()

# %%
