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
import xarray as xr

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from bs_fdbck.constants import path_extract_latlon_outdata
from dask.diagnostics import ProgressBar
import seaborn as sns


# %%
from bs_fdbck.util.BSOA_datamanip import compute_total_tau, broadcase_station_data, change_units_and_compute_vars, \
    get_dic_df_mod

# %%
def make_fn(case, v_x, v_y, comment=''):
    _x = v_x.split('(')[0]
    _y = v_y.split('(')[0]
    f = f'dist_plots_f09_f09_allyears_OA_{comment}_{case}_{_x}_{_y}.png'
    return plot_path /f



# %%
plot_path = Path('Plots')

# %%
xr.set_options(keep_attrs=True) 

# %% [markdown]
# ## Get observational data

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

# %% tags=[]
#df_hyy_1['OA_category']

df_hyy_1['OA_low'] = df_hyy_1['OA (microgram m^-3)']<2
df_hyy_1['OA_high']= df_hyy_1['OA (microgram m^-3)']>2
ddf_hyy_1=df_hyy_1.assign(OA_category= pd.NA)
df_hyy_1.loc[df_hyy_1['OA_high'], 'OA_category'] = 'OA high'
df_hyy_1.loc[df_hyy_1['OA_low'], 'OA_category'] = 'OA low'



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
df_hyy_1['OA (microgram m^-3)'][df_hyy_1['OA_low']].plot.hist(bins=50, alpha=0.4, label='obs')
df_hyy_1['OA (microgram m^-3)'][df_hyy_1['OA_high']].plot.hist(bins=50, alpha=0.4, label='obs')



# %% [markdown] tags=[]
# ## Get model data:

# %%
model_name_noresm = 'NorESM'
model_name_echam  = 'ECHAM-SALSA'

# %%
models =[model_name_noresm]

# %% [markdown] tags=[]
# ### Settings

# %%
lon_lims = [22.,30.]
lat_lims = [60.,66.]

lat_smr = 61.85
lon_smr = 24.28
model_lev_i=-1

# %%
temperature = 273.15  # K


from_time1 = '2012-01-01'
to_time1 = '2015-01-01'
from_time2 ='2015-01-01'
to_time2 ='2019-01-01'
# %% [markdown]
# ### NorEMS

# %%
case_name_noresm = 'OsloAero_intBVOC_f09_f09_mg17_fssp245'


# %% [markdown]
# #### Input files

# %%
fn_noresm = path_extract_latlon_outdata/ case_name_noresm/f'{case_name_noresm}.h1._{from_time1}-{to_time2}_concat_subs_22.0-30.0_60.0-66.0_lev1_final.nc'
fn_noresm_csv = path_extract_latlon_outdata/ case_name_noresm/f'{case_name_noresm}.h1._{from_time1}-{to_time2}_concat_subs_22.0-30.0_60.0-66.0_lev1_final_long_summer.csv'

# %%

cases_noresm = [case_name_noresm]

# %% [markdown] tags=[]
# #### Station variables

# %% tags=[]
varl_st_noresm = [      'SOA_NA','SOA_A1','OM_NI','OM_AI','OM_AC','SO4_NA','SO4_A1','SO4_A2','SO4_AC','SO4_PR',
      'BC_N','BC_AX','BC_NI','BC_A','BC_AI','BC_AC','SS_A1','SS_A2','SS_A3','DST_A2','DST_A3',
                 ]


varl_cl_noresm = ['TOT_CLD_VISTAU','TOT_ICLD_VISTAU','TGCLDCWP','TGCLDLWP','TGCLDIWP',
           'TOT_CLD_VISTAU_s','TOT_ICLD_VISTAU_s','optical_depth',
           'CLDFREE',
           'FCTL',
           'ACTREL','ACTNL','TGCLDLWP',
           'FSDSC','FSDSCDRF',
           'FCTI',
           'FCTL',
           'FLNS',
           'FLNSC',
           'FLNT',
           'FLNTCDRF',
           'FLNT_DRF',
           'FLUS',
           'FLUTC','FORMRATE',
           'FREQI',
           'FREQL',
           'FSDSCDRF',
           'FSDS_DRF',
           'FSNS',
           'FSNSC',
           'FSNT',
           'FSNTCDRF',
           'FSNT_DRF',
           'FSUS_DRF',
           'FSUTADRF',
           ]

# %% [markdown] tags=[]
# ### ECHAM-SALSA

# %%


# %%
ds_mod = xr.open_dataset(fn_noresm, chunks = {'time':48})
ds_mod['NCONC01'].isel(lat=0, lon=0).plot()

# %% [markdown]
# ## If file createad already, skip to here

# %% [markdown] tags=[]
# ### Select hyytiala grid cell:

# %% [markdown]
# We use only hyytiala for org etc, but all grid cells over finland for cloud properties

# %%

ds_all = xr.open_dataset(fn_noresm)



# %%
ds_all['NCONC01'].isel(lat=1, lon=1).plot()

# %%
dic_ds = dict()
dic_ds[model_name_noresm] =ds_all


# %% tags=[]
df_mod = pd.read_csv(fn_noresm_csv, index_col=[0,1,2] )
dic_df = dict()
dic_df[model_name_noresm] = df_mod
# %%
df_mod

# %% [markdown]
# df_modean data:

# %% [markdown]
# ### Remove gridcells that don't have a lot of cloud?

# %%
v_x = 'TGCLDCWP'
#ax = axs[1]
df_mod = dic_df[model_name_noresm]

_df = (df_mod[df_mod[v_x].notnull()])#[(df_mod['OA_category'].notna())])
_df = _df[_df[v_x]>50]
_df = _df[_df[v_x]<10000]
sns.histplot(
    x=v_x,
    y= 'CWP',
    data=_df,
    #hue=v_hue,
    #hue_order=hue_order,
    #ax = ax,
    legend=False,
    palette = 'Set1',
    edgecolor='w',
)

# %% [markdown]
# ### Remove grid vells with no cloud top liquid

# %% [markdown]
# More than 10% cloud top liquid fraction in grid cell and 90% of the cloud fraction is covered by liquid. 

# %%
mask_liq_cloudtop = (df_mod['FCTL']>0.1) & (df_mod['FCTL']/(df_mod['FCTL']+df_mod['FCTI'])>.8)


df_mod.loc[:,'mask_liq_cloudtop'] = mask_liq_cloudtop

# %% [markdown] tags=[]
# ## Cloud water path above 50

# %% tags=[]
rn_dic_echam = {
    'cwp'      : 'CWP',
    'cod'      : 'COT',
    'ceff_ct'  : 'r_eff',


}
rn_dic_noresm = {
    'TGCLDCWP_incld'         : 'CWP',
    'TOT_CLD_VISTAU_s_incld': 'COT',
    'ACTREL_incld'     : 'r_eff',
}
rn_dic_obs = {
    'CWP (g m^-2)'        : 'CWP',
    'CER (micrometer)'    : 'r_eff',
    'OA (microgram m^-3)' : 'OA',

}



# %%
dic_df['Observations'] = df_hyy_1#.keys()

# %%
for key, rn in zip([model_name_noresm,'Observations'], [rn_dic_noresm, rn_dic_obs]):
    dic_df[key] = dic_df[key].rename(rn, axis=1)

# %% [markdown] tags=[]
# ## Group by cloud water path

# %%
df_mod['CWP_qcut']=pd.qcut(df_mod['TGCLDCWP'],6)# bins=bins, labels=labels)§

# %%
for key in dic_df.keys():
    df_mod = dic_df[key]
    df_mod['CWP_qcut']=pd.qcut(df_mod['CWP'],6)# bins=bins, labels=labels)§
    df_mod['CWP_qcutl'] = df_mod['CWP_qcut'].apply(lambda x:x.mid)
    dic_df[key] = df_mod


# %%
dic_bins = dict()
dic_bins[model_name_noresm] = pd.IntervalIndex.from_breaks([   50,  80,  110, 140, 170, 200,230, 500])
dic_bins[model_name_echam] = pd.IntervalIndex.from_breaks([   50,  80,  110, 140, 170, 200,230, 500])


# %%
df_mod = dic_df[model_name_noresm]

v = 'CLDFREE'
df_mod[v].quantile(.99)

# %%
df_mod[v].mean()


# %%
df_mod[v].max()

# %%
df_mod[v].min()

# %%
for model_name in models:
    df_mod = dic_df[model_name]
    # Optical thickness > 50:
    df_mod = df_mod[df_mod['CWP']>=50]

    bins = dic_bins[model_name]
    df_mod['CWP_cut']=pd.cut(df_mod['CWP'], bins=bins)#, labels=labels)

    df_mod['CWP_cutl'] = df_mod['CWP_cut'].apply(lambda x:x.mid)
    df_mod['OA_low'] = df_mod['OA']<df_mod['OA'].quantile(.34)
    df_mod['OA_high']= df_mod['OA']>df_mod['OA'].quantile(.66)
    mid_range = ( df_mod['OA'].quantile(.34)<df_mod['OA']) & (df_mod['OA']<df_mod['OA'].quantile(.66))
    df_mod['OA_mid_range'] = mid_range
    df_mod=df_mod.assign(OA_category= pd.NA)
    df_mod.loc[df_mod['OA_high'], 'OA_category'] = 'OA high'
    df_mod.loc[df_mod['OA_low'], 'OA_category'] = 'OA low'

    dic_df[model_name] = df_mod

for model_name in [model_name_noresm]:
    df_mod = dic_df[model_name]
    v = 'CLDFREE'
    df_mod[f'{v}_low'] = df_mod[v]<df_mod[v].quantile(.34)
    df_mod[f'{v}_high']= df_mod[v]>df_mod[v].quantile(.66)
    mid_range = ( df_mod[v].quantile(.34)<df_mod[v]) & (df_mod[v]<df_mod[v].quantile(.66))
    df_mod[f'{v}_mid_range'] = mid_range
    df_mod=df_mod.assign(**{f'{v}_category': pd.NA})
    df_mod.loc[df_mod[f'{v}_high'], f'{v}_category'] = f'{v} high'
    df_mod.loc[df_mod[f'{v}_low'], f'{v}_category'] = f'{v} low'
    dic_df[model_name] = df_mod



# %% [markdown]
# ## Distribution plots:

# %%
palette = 'Set2'

# %%
import numpy as np

# %%
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# %%
#cmap = cm.get_cmap(name=palette, )
cmap_list = ['#441FE0','#BBE01F'][::-1]#cmap(a) for a in np.linspace(0,1,8)]

# %%
palette_OA = cmap_list[0:2]


# %%
fig, axs = plt.subplots(2,1, sharex=True, figsize =[6,6])
v_x = 'COT'
x_cut = 100
v_hue = 'OA_category'
hue_order=['OA low', 'OA high'][::-1]

_palette = palette_OA#cmap_list[0:2]


for key, ax in zip(dic_df.keys(), axs):
    _df = dic_df[key].copy()
    _df = _df[_df[v_x]<x_cut]

    sns.histplot(
        x=v_x,
        data=_df,
        hue=v_hue,
        hue_order=hue_order,
        palette=_palette,
        legend=False,
        edgecolor='w',
        ax = ax
    )
#plt.ylim([0,250])
    ax.set_title(key)#'Observations')



custom_lines = [Line2D([0], [0], color=cmap_list[0], lw=4),
                Line2D([0], [0], color=cmap_list[1], lw=4),
               # Line2D([0], [0], color=cmap(1.), lw=4)

               ]

leg_els = [

    Patch(edgecolor='w',alpha = .5, facecolor=_palette[1], label='OA low'),
    Patch(edgecolor=None, alpha = .5,facecolor=_palette[0], label='OA high'),

]

ax.legend(handles = leg_els, frameon=False)
ax.set_xlabel('Cloud optical thickness')
#plt.ylim([0,250])
print(len(_df))
sns.despine(fig)

fn = make_fn('echam_noresm', v_x,'obs',comment='distribution')

#fig.savefig(fn, dpi=150)
fig.tight_layout()


# %%
fig, axs = plt.subplots(3,1, sharex=True, figsize =[6,6])

v_x = 'CWP'
x_cut = 700
v_hue = 'OA_category'
hue_order=['OA low', 'OA high'][::-1]

_palette = palette_OA#cmap_list[0:2]


for key, ax in zip(dic_df.keys(), axs):
    _df = dic_df[key].copy()
    _df = _df[_df[v_x]<x_cut]

    sns.histplot(
        x=v_x,
        data=_df,
        hue=v_hue,
        hue_order=hue_order,
        palette=_palette,
        legend=False,
        edgecolor='w',
        ax = ax
    )
#plt.ylim([0,250])
    ax.set_title(key)#'Observations')



custom_lines = [Line2D([0], [0], color=cmap_list[0], lw=4),
                Line2D([0], [0], color=cmap_list[1], lw=4),
               # Line2D([0], [0], color=cmap(1.), lw=4)

               ]

leg_els = [

    Patch(edgecolor='w',alpha = .5, facecolor=_palette[1], label='OA low'),
    Patch(edgecolor=None, alpha = .5,facecolor=_palette[0], label='OA high'),

]

ax.legend(handles = leg_els, frameon=False)
ax.set_xlabel('Cloud water path(g $m^{-2}$)')

#plt.ylim([0,250])
print(len(_df))
sns.despine(fig)

fn = make_fn(model_name_noresm, v_x,'obs',comment='distribution')

fig.savefig(fn, dpi=150)



# %%
fig, axs = plt.subplots(2,1, sharex=True, figsize =[6,6])
v_x = 'CWP (g m^-2)'
x_cut = 10e10
v_hue = 'OA_category'
hue_order=['OA low', 'OA high'][::-1]

_palette = palette_OA#cmap_list[0:2]


_df = df_hyy_1
_df = _df[_df[v_x]<x_cut]

ax = axs[0]
sns.histplot(
    x=v_x,
    data=_df,
    hue=v_hue,
    hue_order=hue_order,
    palette=_palette,
    legend=False,
    edgecolor='w',
    ax = ax
)
print(len(_df))
ax.set_title('Observations')


v_x = 'TGCLDCWP'
ax = axs[1]

_df = (df_mod[(df_mod['OA_category'].notna())])
_df = _df[_df[v_x]<x_cut]
sns.histplot(
    x=v_x,
    data=_df,
    hue=v_hue,
    hue_order=hue_order,
    ax = ax,
    legend=False,
    palette = _palette,
    edgecolor='w',
)

ax.set_title('OsloAero')

custom_lines = [Line2D([0], [0], color=cmap_list[0], lw=4),
                Line2D([0], [0], color=cmap_list[1], lw=4),
               # Line2D([0], [0], color=cmap(1.), lw=4)

               ]

leg_els = [

    Patch(edgecolor='w',alpha = .5, facecolor=_palette[1], label='OA low'),
    Patch(edgecolor=None, alpha = .5,facecolor=_palette[0], label='OA high'),

]

ax.legend(handles = leg_els, frameon=False)
ax.set_xlabel('CWP (g m^-2)')
#plt.ylim([0,250])
print(len(_df))
sns.despine(fig)

fn = make_fn(model_name + case_name_noresm, v_x,'obs',comment='distribution')

fig.savefig(fn, dpi=150)



# %% [markdown]
# ### Fractional occurance of cloud top liquid

# %%
_df = df_mod#[(df_mod['OA_category'].notna()) & (df_mod['TOT_ICLD_VISTAU_s']>1)])
_df = _df#[_df['TGCLDCWP']<700]
sns.displot(#x='TGCLDLWP',
            x='FCTL',
            data=_df,
            hue='OA_category',
    palette=palette_OA,
           #kind='swarm'
           )
#plt.ylim([0,250])
print(len(df_mod[df_mod['OA_category'].notna()]))

# %% [markdown]
# ### Take only points where 30% of cloud top is liquid

# %% tags=[]
_df = df_mod
_df = _df[_df['FCTL']>.3]
sns.displot(
    x='CWP',
    data=_df,
    hue='OA_category',
    palette=palette_OA,
)
print(len(df_mod[df_mod['OA_category'].notna()]))

# %%
sns.displot(
    x='ACTNL_incld',
    data=df_mod[~df_mod['OA_mid_range']].reset_index(),
    hue='OA_category',
    palette=palette_OA,
)
print(len(df_mod[~df_mod['OA_mid_range']]))

# %% [markdown]
# ## Cloud optical thickness

# %% [markdown]
# ### Incloud

# %%
hue_order = ['OA low','OA high']
palette_OA_2 = palette_OA[::-1]

# %%
x_var = 'CWP_cutl'
y_var = 'COT'
hue_var = 'OA_category'
hue_labs = ['OA low', 'OA high']
ylim = [0,52]
figsize = [12,10]
_palette = palette_OA_2

#fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex=True)
fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col', dpi=100)

markersize= 2


for key,ax in zip(dic_df.keys(), axs[0,:]):
    _df = dic_df[key].copy()
    
    _df_lim =_df[(_df[y_var]<=ylim[1])& (_df[y_var]>=ylim[0])]

    _df_lim = _df_lim[_df_lim[x_var].notna()]
    _df_lim = _df_lim[_df_lim[y_var].notna()]
    _df_lim = _df_lim[_df_lim[hue_var].notna()]
    
    sns.swarmplot(
        x=x_var,
        y=y_var,
        data=_df_lim,
        hue_order=hue_order,
        hue=hue_var,
        palette=_palette,
        size = markersize,
        ax = ax,
    )
    ax.set_title(key)

for key,ax in zip(dic_df.keys(), axs[1,:]):
    _df = dic_df[key].copy()
    
    _df_lim =_df[(_df[y_var]<=ylim[1])& (_df[y_var]>=ylim[0])]
    _df_lim = _df_lim[_df_lim[x_var].notna()]
    _df_lim = _df_lim[_df_lim[y_var].notna()]
    _df_lim = _df_lim[_df_lim[hue_var].notna()]

    sns.boxenplot(
        x=x_var,
        y=y_var,
        data= _df_lim,
        hue_order=hue_order,#['OA low','OA high'],
        hue=hue_var,
        #kind='boxen',
        ax = ax,
        palette=_palette,
           )


    

    ## ADJUSTMENTS

for ax in axs.flatten():
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)


leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label=hue_labs[0]),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label=hue_labs[1]),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label=hue_labs[0]
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label=hue_labs[1]
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)

for ax in axs[:,0]:
    ax.set_ylabel('Cloud optical depth []')
for ax in axs[1,:]:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)



sns.despine(fig)

fn = make_fn('noresm_only', x_var,y_var, comment='binned_by_x')
fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg

# %%
x_var = 'CWP_cutl'
y_var = 'r_eff'
hue_var = 'OA_category'
hue_labs = ['OA low', 'OA high']
ylim = [0,25]
figsize = [12,10]
_palette = palette_OA_2

#fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex=True)
fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col', dpi=100)

markersize= 2


for key,ax in zip(dic_df.keys(), axs[0,:]):
    _df = dic_df[key].copy()
    
    _df_lim =_df[(_df[y_var]<=ylim[1])& (_df[y_var]>=ylim[0])]

    _df_lim = _df_lim[_df_lim[x_var].notna()]
    _df_lim = _df_lim[_df_lim[y_var].notna()]
    _df_lim = _df_lim[_df_lim[hue_var].notna()]
    
    sns.swarmplot(
        x=x_var,
        y=y_var,
        data=_df_lim,
        hue_order=hue_order,
        hue=hue_var,
        palette=_palette,
        size = markersize,
        ax = ax,
    )
    ax.set_title(key)

for key,ax in zip(dic_df.keys(), axs[1,:]):
    _df = dic_df[key].copy()
    
    _df_lim =_df[(_df[y_var]<=ylim[1])& (_df[y_var]>=ylim[0])]
    _df_lim = _df_lim[_df_lim[x_var].notna()]
    _df_lim = _df_lim[_df_lim[y_var].notna()]
    _df_lim = _df_lim[_df_lim[hue_var].notna()]

    sns.boxenplot(
        x=x_var,
        y=y_var,
        data= _df_lim,
        hue_order=hue_order,#['OA low','OA high'],
        hue=hue_var,
        #kind='boxen',
        ax = ax,
        palette=_palette,
           )


    

    ## ADJUSTMENTS

for ax in axs.flatten():
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)


leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label=hue_labs[0]),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label=hue_labs[1]),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label=hue_labs[0]
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label=hue_labs[1]
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)

for ax in axs[:,0]:
    ax.set_ylabel('Cloud optical depth []')
for ax in axs[1,:]:
    ax.set_xlabel('r$_{eff}$ [$\mu$m]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)



sns.despine(fig)

fn = make_fn('noresm_only', x_var,y_var, comment='binned_by_x')
fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg

# %%
x_mod = 'CWP_cutl'
x_obs = 'CWP_cutl'
y_mod = 'TOT_ICLD_VISTAU_s'
y_obs = 'COT'
ylim = [0,52]
figsize = [12,10]
figsize = [11,7]
_palette = palette_OA_2

#fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex=True)
fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1#df_mod[df_mod['OA_category'].notna()].reset_index()
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]


ax = axs[0,1]
sns.swarmplot(
    x=x_obs,
    y=y_obs,
    data=_df_obs_lim,
    hue_order=hue_order,
    hue='OA_category',
    palette=_palette,
    size = markersize,
    ax = ax,
)


ax = axs[1,1]
sns.boxenplot(
    x=x_obs,
    y=y_obs,
    data= _df_obs,
    hue_order=hue_order,#['OA low','OA high'],
    hue='OA_category',
    #kind='boxen',
    ax = ax,
    palette=_palette,
           )


## PLOT MODEL



_df_mod = df_mod[df_mod['OA_category'].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]

markersize=3
ax = axs[0,0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=hue_order,

    hue='OA_category',
    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1,0]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=hue_order,

    ax = ax,
    hue='OA_category',
    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label='OA low'),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label='OA high'),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label='OA low'
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label='OA high'
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)


for ax in axs[:,1]:
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)
for ax in axs[:,0]:
    ax.set_ylabel('Cloud optical depth []')
for ax in axs[1,:]:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)


axs[0,0].set_title('OsloAero')
axs[0,1].set_title('Observations')

sns.despine(fig)

fn = make_fn(case_name, x_mod,y_mod, comment='binned_by_x')
fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%
x_mod = 'CWP_cutl'
x_obs = 'CWP_cutl'
y_mod = 'TOT_CLD_VISTAU_s'
y_obs = 'COT'
ylim = [0,52]
figsize = [12,10]
figsize = [11,7]

_palette = palette_OA_2

fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col')


markersize= 2

_df_obs = df_hyy_1#df_mod[df_mod['OA_category'].notna()].reset_index()
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]


ax = axs[0,1]
sns.swarmplot(
    x=x_obs,
    y=y_obs,
    data=_df_obs_lim,
    hue_order=['OA low','OA high'],
    hue='OA_category',
    palette=_palette,
    size = markersize,
    ax = ax,
)


ax = axs[1,1]
sns.boxenplot(
    x=x_obs,
    y=y_obs,
    data= _df_obs,
    hue_order=['OA low','OA high'],
    hue='OA_category',
    #kind='boxen',
    ax = ax,
    palette=_palette,
           )


## PLOT MODEL



_df_mod = df_mod[df_mod['OA_category'].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]

markersize=3
ax = axs[0,0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=['OA low','OA high'],
    hue='OA_category',
    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1,0]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=['OA low','OA high'],
    ax = ax,
    hue='OA_category',
    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label='OA low'),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label='OA high'),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label='OA low'
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label='OA high'
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)


for ax in axs[:,1]:
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)
for ax in axs[:,0]:
    ax.set_ylabel('Cloud optical depth []')
for ax in axs[1,:]:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)

axs[0,0].set_title('OsloAero')
axs[0,1].set_title('Observations')


sns.despine(fig)

fn = make_fn(case_name, x_mod,y_mod, comment='binned_by_x')
fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%
x_mod = 'CWP_cutl'
x_obs = 'CWP_cutl'
y_mod = 'ACTREL_incld'
y_obs = 'CER (micrometer)'
ylim = [3,25]
figsize = [11,7]
_palette = palette_OA_2
fig, axs = plt.subplots(2,2,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1#df_mod[df_mod['OA_category'].notna()].reset_index()
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]




_df_mod = df_mod[df_mod['OA_category'].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]



# OBSERVATIONS PLOT

ax = axs[0,1]
sns.swarmplot(
    x=x_obs,
    y=y_obs,
    data=_df_obs_lim,
    hue_order=['OA low','OA high'],
    hue='OA_category',
    palette=_palette,
    size = markersize,
    ax = ax,
)


ax = axs[1,1]
sns.boxenplot(
    x=x_obs,
    y=y_obs,
    data= _df_obs,
    hue_order=['OA low','OA high'],
    hue='OA_category',
    #kind='boxen',
    ax = ax,
    palette=_palette,
           )


## PLOT MODEL

markersize=3
ax = axs[0,0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=['OA low','OA high'],
    hue='OA_category',
    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1,0]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=['OA low','OA high'],
    ax = ax,
    hue='OA_category',
    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label='OA low'),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label='OA high'),
          ]
axs[1,0].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label='OA low'
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label='OA high'
           ),
          ]
axs[0,0].legend(handles = leg_els, frameon=False)


for ax in axs[:,1]:
    ax.legend([],[], frameon=False)
    ax.set_ylabel(None)
for ax in axs[:,0]:
    ax.set_ylabel('Cloud effective radius [$\mu m$]')
for ax in axs[1,:]:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[0,:]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)


axs[0,0].set_title('OsloAero')
axs[0,1].set_title('Observations')

sns.despine(fig)

fn = make_fn(case_name, x_mod,y_mod, comment='binned_by_x')
fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%
x_mod = 'CWP_cutl'
y_mod = 'ACTNL_incld'
ylim = [0,150]
figsize = [9,7]
_palette = palette_OA_2

fig, axs = plt.subplots(2,1,figsize=figsize, sharey=True, sharex='col')

markersize= 2

_df_obs = df_hyy_1
_df_obs_lim =_df_obs[(_df_obs[y_obs]<=ylim[1])& (_df_obs[y_obs]>=ylim[0])]

_df_mod = df_mod[df_mod['OA_category'].notna()].reset_index()
_df_mod_lim =_df_mod[(_df_mod[y_mod]<=ylim[1])& (_df_mod[y_mod]>=ylim[0])]



## PLOT MODEL

markersize=3
ax = axs[0]
sns.swarmplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod_lim,
    hue_order=['OA low','OA high'],
    hue='OA_category',
    palette=_palette,
    ax = ax,
    size = markersize,
)


ax = axs[1]
sns.boxenplot(
    x=x_mod,
    y=y_mod,
    data=_df_mod,
    hue_order=['OA low','OA high'],
    ax = ax,
    hue='OA_category',
    palette=_palette,
)


## ADJUSTMENTS

leg_els = [
    Patch(edgecolor='k', alpha = .9,facecolor=_palette[0], label='OA low'),
    Patch(edgecolor='k',alpha = .9, facecolor=_palette[1], label='OA high'),
          ]
axs[1].legend(handles = leg_els, frameon=False)

leg_els = [
     Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[0], markersize=10,
            label='OA low'
           ),
         Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=_palette[1], markersize=10,
            label='OA high'
           ),
          ]
axs[0].legend(handles = leg_els, frameon=False)


#for ax in axs[:,1]:
#    ax.legend([],[], frameon=False)
#    ax.set_ylabel(None)
for ax in axs:
    ax.set_ylabel(r'Cloud droplet number conc. [cm$^{-3}$]')
for ax in axs:
    ax.set_xlabel('CWP [g m$^{-2}$]')
for ax in axs[:-1]:
    ax.set_xlabel(None)

for ax in axs.flatten():

    ax.set_ylim(ylim)

axs[0].set_title('OsloAero')
axs[1].set_title('Observations')


sns.despine(fig)

fn = make_fn(case_name, x_mod,y_mod, comment='binned_by_x')
fig.savefig(fn, dpi=150)
plt.show()

### Grid box avg


# %%

# %%

# %%
