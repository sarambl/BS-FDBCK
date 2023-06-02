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

# %%
plot_path = Path('Plots')


# %%
def make_fn_scat(case, v_x, v_y):
    _x = v_x.split('(')[0]
    _y = v_y.split('(')[0]
    f = f'scat_{case}_{_x}_{_y}.png'
    return plot_path /f


# %%
plot_path.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Load observations: 

# %%
import pandas as pd

# %%

# %%
from bs_fdbck.constants import path_measurement_data

# %%
fn = path_measurement_data / 'SourceData_Yli_Juuti2021.xls'

df_hyy_1 = pd.read_excel(fn, sheet_name=4, header=1,)# usecols=range(7,12),nrows=7)

df_hyy_1.head()
#df_hyy_1y= df_hyy_1y.rename({'year.1':'year',
#                            'T (degree C).1':'T (degree C)',
#                             'OA (microgram m^-3).1':'OA (microgram m^-3)',
#                             'N100 (cm^-3).1':'N100 (cm^-3)'
#                            }, axis=1)
#df_hyy_1y['year'] = pd.to_datetime(df_hyy_1y['year'].apply(x:str(x)))

df_hyy_1

# %%
import pandas as pd

# %%
df_hyy_1['date'] = df_hyy_1.apply(lambda x: f'{x.year:.0f}-{x.month:02.0f}-{x.day:02.0f}', axis=1)

df_hyy_1['date'] = pd.to_datetime(df_hyy_1['date'] )


# %%
df_hyy_1 = df_hyy_1.set_index(['date','LAT','LON'])

# %%
df_hyy_1

# %%
df_hyy_1['OA (microgram m^-3)'].plot.hist(bins=50, alpha=0.4, label='obs')


# %% [markdown]
# ## load models:

# %% [markdown] tags=[]
# ## Read in model data. 

# %%

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

# %% [markdown]
# ## Settings:

# %%
nr_of_bins = 5
maxDiameter = 39.6  #    23.6 #e-9
minDiameter = 5.0  # e-9
history_field='.h1.'

# %%
from_t = '2011-01-01'
to_t = '2015-01-01'

# %% [markdown]
# ## Cases:

# %%
cases_sec = ['OsloAeroSec_intBVOC_f19_f19']#'SECTv21_ctrl_def','SECTv11_noresm2_ctrl', 'SECTv11_ctrl_fbvoc','SECTv11_noresm2_adj','SECTv11_noresm2_eq18']#'SECTv11_noresm2_NFHIST']#'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']#,'SECTv11_ctrl_fbvoc']#'SECTv11_ctrl']
cases_orig = ['OsloAero_intBVOC_f19_f19']#, 'noSECTv21_ox_ricc']#'noSECTv11_noresm2_ricc', 'noSECTv11_noresm2_ctrl', 'noSECTv11_ctrl_fbvoc','noSECTv11_ctrl']#'noSECTv11_noresm2_NFHIST']#'noSECTv11_ctrl_fbvoc'] #/no SECTv11_ctrl

# %%
case_mod = cases_orig[0]

# %%
 
log.ger.info(f'TIMES:****: {from_t} {to_t}')

# %%
varl =['N100','SOA_NA','SOA_A1','SO4_NA','DOD500','DOD440','A',#'TGCLDLWP',
       'H2SO4','SOA_LV','COAGNUCL','FORMRATE','FSNSC',
       'NUCLRATE','NCONC01','NCONC02','NCONC03','NCONC04','NCONC05','NCONC06','NCONC07',
       'NCONC08','NCONC09','NCONC10','NCONC11','NCONC12','NCONC13','NCONC14','SIGMA01',
       'SIGMA02','SIGMA03','SIGMA04','SIGMA05','SIGMA06','SIGMA07','SIGMA08','SIGMA09',
       'SIGMA10','SIGMA11','SIGMA12','SIGMA13','SIGMA14','NMR01','NMR02','NMR03','NMR04',
       'NMR05','NMR06','NMR07','NMR08','NMR09','NMR10','NMR11','NMR12','NMR13','NMR14', 
      'FSNS','FSDS_DRF','T','GR','GRH2SO4','GRSOA','TGCLDCWP','U','V', 'SO2','isoprene',
       'monoterp','GS_SO2', 'GS_H2SO4','GS_monoterp','GS_isoprene',
      
      
      ]


varl =['N100','DOD500','DOD440','ACTREL','ACTNL','TGCLDLWP', #,'SOA_A1',
       'H2SO4','SOA_LV','COAGNUCL','FORMRATE','T','FCTL',
       'TOT_CLD_VISTAU','TOT_ICLD_VISTAU','TGCLDCWP',
       'TAUTLOGMODIS',
       'LWPMODIS','CLWMODIS','REFFCLWMODIS','TAUTMODIS','TAUWMODIS',
      
      'SOA_NA','SOA_A1','OM_NI','OM_AI','OM_AC','SO4_NA','SO4_A1','SO4_A2','SO4_AC','SO4_PR',
      'BC_N','BC_AX','BC_NI','BC_A','BC_AI','BC_AC','SS_A1','SS_A2','SS_A3','DST_A2','DST_A3', 
      ] 


# %%
for case_name in cases_sec:
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
for case_name in cases_orig:
    varlist = varl# list_sized_vars_noresm
    c = CollocateLONLATout(case_name, from_t, to_t,
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
for ca in cases_orig + cases_sec:
    c = CollocateLONLATout(ca, from_t, to_t,
                           False,
                           'hour',
                           history_field=history_field)
    ds = c.get_collocated_dataset(varl)
    dic_ds[ca]=ds

# %% [markdown]
# ## Compute ACTNL_incld

# %% tags=[]
for ca in cases_orig + cases_sec:
    _ds = dic_ds[ca]
    _ds['ACTNL_incld'] = _ds['ACTNL']/_ds['FCTL']
    _ds['ACTREL_incld'] = _ds['ACTREL']/_ds['FCTL']
    _ds['TOT_ICLD_VISTAU_s']= _ds['TOT_ICLD_VISTAU'].sum('lev')
    _ds['TOT_CLD_VISTAU_s']= _ds['TOT_CLD_VISTAU'].sum('lev')

# %%
dic_ds[ca]['ACTNL_incld'].isel(station=0).isel(time=slice(0,1000)).plot()
dic_ds[ca]['ACTNL'].isel(station=0).isel(time=slice(0,1000)).plot()

# %% tags=[]
_ds.load()

# %%
R = 287.058
pressure = 1000. #hPa
kg2ug = 1e9


# %%

def get_dic_df_mod(model_lev_i=-1):
    dic_df = dict()
    dic_df_sm = dict()

    for ca in dic_ds.keys():
        ds = dic_ds[ca]
        ds_sel = ds.sel(station='SMR').isel( lev=model_lev_i)
        rho = pressure*100/(R*ds_sel['T'])
    
        ds_sel['rho'] = rho.load()
        df = ds_sel.to_dataframe()
        ls_so4 = [c for c in df.columns if 'SO4_' in c]#['SO4_NA']

        for s in ['SOA_NA','SOA_A1','OM_AC','OM_AI','OM_NI']+ls_so4:
            un = '$\micro$g/m3'
            if ds_sel[s].attrs['units']!=un:
                ds_sel[s] = ds_sel[s]*ds_sel['rho']*kg2ug
                ds_sel[s].attrs['units']=un

        df = ds_sel.to_dataframe()
        df = df.drop([co for co in df.columns if (('lat_' in co)|('lon_' in co))], 
                     axis=1)

        df['SOA'] = df['SOA_NA'] + df['SOA_A1']

        df['OA'] = df['SOA_NA'] + df['SOA_A1'] +df['OM_AC']+df['OM_AI']+df['OM_NI']
        df['POA'] = df['OM_AC']+df['OM_AI']+df['OM_NI']
    
        df['SO4']=0
        for s in ls_so4:
            print(s)
            
            print(df[s].mean())
            df['SO4'] = df['SO4'] + df[s]
    
    
        df_daily = df.resample('D').median()

        months = (df_daily.index.month==7 )|(df_daily.index.month==8  )

        df_s = df_daily[months]
        df_s.loc[:,'year'] = df_s.index.year.values

        df_s.loc[:,'T_C'] = df_s['T'].values-273.15
        df_s.index = df_s.index.rename('date')
        df_merge = df_s#pd.merge(df_s, df_hyy_1, right_on='date', left_on='date')
        
        df_merge['year'] = df_merge.index.year

        
        dic_df[ca] = df_merge
        print(ca)
    
        months = (df.index.month==7 )|(df.index.month==8  )

        df_s = df[months]
        df_ym = df_s.resample('Y').median()
        df_ym.loc[:,'year'] = df_ym.index.year.values

        df_ym.loc[:,'T_C'] = df_ym['T'].values-273.15
        
        dic_df_sm[ca] = df_ym
        print(ca)
    return dic_df_sm, dic_df


dic_df_sm, dic_df = get_dic_df_mod(model_lev_i=-1)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
import numpy as np

# %%
import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge

# %%
cols = [
    #'#ffff33',
    '#0074c3',
    '#eb4600',
    '#f8ae00',
    '#892893',
    '#66ae00',
    '#00c1f3',
    '#b00029',
]

# %%
from matplotlib import cm


# %%
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

my_cmap = ListedColormap(cols)

my_cmap

# %%

# %%
type(df_hyy_1['year'][0])

# %%
col_dic = {}
for y,c in zip(range(2012, 2019), cols):
    col_dic[y] =c

# %% [markdown]
# ## Try to reproduce plot from paper:

# %%
df_hyy_1['OA_low'] = df_hyy_1['OA (microgram m^-3)']<1.59
df_hyy_1['OA_high'] = df_hyy_1['OA (microgram m^-3)']>3.02

# %%

df_hyy_1=df_hyy_1.assign(OA_category= pd.NA)
df_hyy_1.loc[df_hyy_1['OA_high'], 'OA_category'] = 'OA high'
df_hyy_1.loc[df_hyy_1['OA_low'], 'OA_category'] = 'OA low'



# %%

# %%
df_hyy_1['CWP (g m^-2)'].min()

# %%
bins = pd.IntervalIndex.from_tuples([(60, 100), (100, 140), (140, 180), (180, 220), (220, 260), (260, 300), (300, 340)])

# %%
labels=[ 80, 120, 160, 200, 240, 280, 320]

# %%
df_hyy_1['CWP_cut']=pd.cut(df_hyy_1['CWP (g m^-2)'], bins=bins, labels=labels)
df_hyy_1['CWP_qcut']=pd.qcut(df_hyy_1['CWP (g m^-2)'], 6)#bins=bins, labels=labels)

# %%
df_hyy_1['CWP_qcutl'] = df_hyy_1['CWP_qcut'].apply(lambda x:x.mid)

df_hyy_1['CWP_cutl'] = df_hyy_1['CWP_cut'].apply(lambda x:x.mid)

# %%
df_hyy_1.head()

# %%
df_hyy_1.tail()

# %%
sns.displot(#x='TGCLDLWP', 
            x='COT',
            data=df_hyy_1[df_hyy_1['OA_category'].notna()],
            hue='OA_category',
           #kind='swarm'
           )
print(len(df_hyy_1[df_hyy_1['OA_category'].notna()]))

# %%
sns.displot(#x='TGCLDLWP', 
            x='CWP (g m^-2)',
            data=df_hyy_1[df_hyy_1['OA_category'].notna()],
            hue='OA_category',
           #kind='swarm'
           )
print(len(df_hyy_1[df_hyy_1['OA_category'].notna()]))

# %%

# %%
df_hyy_1_sub = df_hyy_1[(df_hyy_1['OA_low'] | df_hyy_1['OA_high'])]

# %%
sns.catplot(x='CWP_cutl', 
            y='COT',
            #data=df_mod.reset_index(),
            #data=df_mod[~df_mod['OA_mid_range']].reset_index(),
            data=df_hyy_1[df_hyy_1['OA_category'].notna()].reset_index(),

            hue='OA_category',
           kind='boxen',
            #order==['OA low','OA high'],
            hue_order=['OA low','OA high'],
            
           )
#plt.ylim([0,250])

# %%
sns.catplot(x='CWP_cutl', 
            y='COT',
            #data=df_mod.reset_index(),
            #data=df_mod[~df_mod['OA_mid_range']].reset_index(),
            data=df_hyy_1[df_hyy_1['OA_category'].notna()].reset_index(),

            hue='OA_category',
           kind='swarm',
            #order==['OA low','OA high'],
            hue_order=['OA low','OA high'],
            
           )
#plt.ylim([0,250])

# %%
sns.catplot(x='CWP_cutl', 
            #y='COT',
            y = 'CER (micrometer)',
            
            #data=df_mod.reset_index(),
            #data=df_mod[~df_mod['OA_mid_range']].reset_index(),
            data=df_hyy_1[df_hyy_1['OA_category'].notna()].reset_index(),

            hue='OA_category',
           kind='boxen',
            #order==['OA low','OA high'],
            hue_order=['OA low','OA high'],
            
           )
plt.ylim([0,25])

# %%
sns.catplot(x='CWP_cutl', 
            #y='COT',
            y = 'CER (micrometer)',
            
            #data=df_mod.reset_index(),
            #data=df_mod[~df_mod['OA_mid_range']].reset_index(),
            data=df_hyy_1[df_hyy_1['OA_category'].notna()].reset_index(),

            hue='OA_category',
           kind='swarm',
            #order==['OA low','OA high'],
            hue_order=['OA low','OA high'],
            
           )
plt.ylim([0,25])
