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
import matplotlib as mpl
import xarray as xr

from pathlib import Path

from bs_fdbck.util.BSOA_datamanip import ds2df_inc_preprocessing
from bs_fdbck.util.collocate.collocateLONLAToutput import CollocateLONLATout
from bs_fdbck.util.collocate.collocate_echam_salsa import CollocateModelEcham
import useful_scit.util.log as log

from bs_fdbck.util.plot.BSOA_plots import make_cool_grid, plot_scatter, cdic_model
log.ger.setLevel(log.log.INFO)
import time
import xarray as xr
import matplotlib.pyplot as plt

cdic_model

from bs_fdbck.preprocess.launch_monthly_station_collocation import launch_monthly_station_output
from bs_fdbck.util.Nd.sizedist_class_v2.SizedistributionBins import SizedistributionStationBins
from bs_fdbck.util.collocate.collocateLONLAToutput import CollocateLONLATout
from bs_fdbck.data_info.variable_info import list_sized_vars_nonsec, list_sized_vars_noresm
import useful_scit.util.log as log
log.ger.setLevel(log.log.INFO)
import time

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge

# %% [markdown] tags=[]
# ## Cases:

# %%
cases_noresm1 = ['OsloAero_f19_f19_mg17_act']
case_name = 'OsloAero_f19_f19_mg17_act'
# %%
from_t = '2009-01-01'
to_t = '2011-01-01'

# %% [markdown] tags=[]
# ### Output filename:
#

# %%
fn_output_corr_height = Path(f'Data/{case_name}_corr_height_{from_t}-{to_t}.nc')
fn_output_corr_height_renamed = Path(f'Data/{case_name}_corr_height_rename_{from_t}-{to_t}.nc')
fn_units_desc = Path(f'Data/{case_name}_corr_height_rename_{from_t}-{to_t}/{case_name}_units_description.csv')

# %%
fn_units_desc.parent.mkdir(exist_ok=True)

# %% [markdown] tags=[]
# ## Cases:

# %% [markdown]
# ## Settings:

# %%
nr_of_bins = 5
maxDiameter = 39.6  #    23.6 #e-9
minDiameter = 5.0  # e-9
history_field='.h1.'

# %% [markdown]
# ## Variables

# %% [markdown]
# ddry_nucsol
# ddry_aitinsol
# ddry_aitsol
# ddry_accsol
# ddry_corsol
# n_nucsol
# n_aitinsol
# n_aitsol
# n_accsol
# n_corsol
# cdnc
# n100
# n70
# n50
# n30
# activation supersaturation
# vertical velocity
# cloud fraction
# temperature

# %%

varl =['N100','N50','N30','N70',
       'AWNC',
       #'AWNC_incld',
       'AREL', 
       'FREQL', 
       'FREQI', 
       #'ACTNL_incld',
       'ACTNL',
       'ACTREL', 
       'ACTREI', 
       'FCTL', 'FCTI',
       'Z3',
       'Smax_cldv',
       'Smax_cldv_supZero',
       'Smax_incld',
       'Smax_incld_supZero',
       'WSUB',
       'WTKE',
       'WSUBI',
       'T',
       'LCLOUD', # liquid cloud fraction used in stratus activation
       'CLDTOT',
       'CLOUD',
       'CLOUDCOVER_CLUBB',
       'CLOUDFRAC_CLUBB',
       
      ] 


# %%
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
# %% tags=[]
c = CollocateLONLATout(case_name, from_t, to_t,
                           False,
                           'hour',
                           history_field=history_field)
ds = c.get_collocated_dataset(varl)
if 'location' in ds.coords:
    ds = ds.rename({'location':'station'})


# %%
import pandas as pd

# %%
df_loc = pd.read_csv('locations.csv', index_col =0)

# %%
from bs_fdbck.util.imports.import_fields_xr_v2 import xr_import_NorESM


# %% [markdown]
# ## Import geopotential height: 
#

# %%
_ds = xr.Dataset()

for st in df_loc.columns:
    _lat = df_loc.loc['lat', st]
    _lon = df_loc.loc['lon', st]
    _ds[st] = ds['Z3'].sel(station=st).copy()
    
    print(_lat)
    print(st)

# %%

# %%
ds['Z3'].isel(lev=-2).load()-ds['Z3'].isel(lev=-1).load()

# %% [markdown]
# ## Z3 does not vary. with time:

# %%
ds['Z3'].isel(station=0).plot()

# %%
ds['Z3'] = ds['Z3'].isel(time=0)

# %%
df_loc

# %%
dic_ds_station = dict()
indx_dic = dict()
pres_dic = dict()

for st in df_loc.columns:
    print(st)
    _ds = ds.sel(station=st)
    height = df_loc.loc['height (masl)', st]
    print(height)
    _ds['Z3'].isel(lev=slice(-10,None)).plot()
    plt.show()
    da_diff = np.abs(_ds['Z3']-height)
    da_diff.isel(lev=slice(-10,None)).plot()
    plt.show()
    index = da_diff.argmin().compute()
    indx_dic[st] = index
    pres_dic[st] = _ds['lev'].isel(lev=index)
    print(index) 
    dic_ds_station[st] = _ds.drop('Z3').isel(lev=index).copy().drop('lev')

ls = [dic_ds_station[st] for st in dic_ds_station.keys()]

ds_corr_height = xr.concat(ls, dim ='station')

# %%

# %%
ds_sub = ds.sel( lev=slice(600,1000))

fig, axs = plt.subplots(11,5, figsize=[18,15], sharex=True, sharey=False)
for i,l  in enumerate(ds_sub.lev):
    for j, st in enumerate(ds_sub.station):
        ax = axs[i,j]
        _ds = ds_sub.sel(lev=l, station=st)
        _ds = _ds.where(_ds['T']>(-5+273))
        (100*_ds['Smax_cldv']).plot.hist(ax=ax, bins=np.linspace(.01,2))
        ax.set_title(f'lev:{l.values:.1f}, {st.values}')
        if i==10:
            ax.set_xlabel('Smax [%]')
        else:
            ax.set_xlabel('')
        if float(pres_dic[str(st.values)])==float(l):
            print('hey!!!')
            ax.set_title(f'lev:{l.values:.1f}, {st.values}', c='r')
            
fig.tight_layout()
fig.savefig('Smax_test_lev.png', dpi=200)

# %%
ds_sub = ds.sel( lev=slice(600,1000))

fig, axs = plt.subplots(11,5, figsize=[18,15], sharex=True, sharey=False)
for i,l  in enumerate(ds_sub.lev):
    for j, st in enumerate(ds_sub.station):
        ax = axs[i,j]
        _ds = ds_sub.sel(lev=l, station=st)
        _ds = _ds.where(_ds['T']>(-5+273))
        _ds = _ds.where(_ds['Smax_cldv_supZero']>0)
        #_ds = _ds.where(_ds['WSUB']!=0.2)
        (_ds['WSUB']).plot.hist(ax=ax, bins=np.linspace(.01,2), density=False)
        ax.set_title(f'lev:{l.values:.1f}, {st.values}')
        if i==10:
            ax.set_xlabel('updraft [m/s]')
        else:
            ax.set_xlabel('')
        if float(pres_dic[str(st.values)])==float(l):
            print('hey!!!')
            ax.set_title(f'lev:{l.values:.1f}, {st.values}', c='r')
            
fig.tight_layout()
fig.savefig('updraft_test_lev.png', dpi=200)

# %%
fig.tight_layout()
fig

# %%
ds_corr_height.drop(['CLOUDCOVER_CLUBB','ilev'])

# %% [markdown]
# ## Write to file: 

# %% tags=[]
ds_corr_height.drop(['CLOUDCOVER_CLUBB','ilev']).to_netcdf(fn_output_corr_height)

# %% [markdown]
# ## Tidy up data:

# %%
ds_corr_height = xr.open_dataset(fn_output_corr_height, decode_times=True)

ds_corr_height['station'] = ds_corr_height['station'].astype(str)

# %% [markdown]
# ### CDNC

# %%
ds_corr_height['AWNC_incld'] = ds_corr_height['AWNC']/ds_corr_height['FREQL']

# %%
ds_corr_height['ACTNL_incld'] = ds_corr_height['ACTNL']/ds_corr_height['FCTL']

# %%
plt.figure(dpi=200)
ds_corr_height['AWNC'].sel(station='Puijo').plot(marker='*', label='CDNC average grid box')

ds_corr_height['AWNC_incld'].sel(station='Puijo').plot(marker='o', alpha=.4, label='CDNC in-cloud')
ds_corr_height['ACTNL_incld'].sel(station='Puijo').plot(marker='o', alpha=.4, label='cloud top CDNC in-cloud')
#ds_corr_height['ACTNL'].sel(station='Puijo').plot(marker='o', alpha=.4)
plt.legend()

# %% [markdown]
# ### Smax

# %%
ds_corr_height['Smax_cldv'] = ds_corr_height['Smax_cldv'].where(ds_corr_height['Smax_cldv_supZero'])
ds_corr_height['Smax_incld'] = ds_corr_height['Smax_incld'].where(ds_corr_height['Smax_incld_supZero'])
for v in ['Smax_cldv','Smax_incld']:
    ds_corr_height[v].attrs['units'] = 1

# %%
fig, ax = plt.subplots(dpi=200)
ds_corr_height['Smax_cldv'].sel(station='Puijo').plot(marker='o', linewidth=0, label='vertically changing')
#ds_corr_height['FREQI'].sel(station='Puijo').plot(marker='d', alpha=.4)
#(ds_corr_height['FREQL']+ds_corr_height['FREQI']).sel(station='Puijo').plot(marker='*', alpha=0.5)

ds_corr_height['Smax_incld'].sel(station='Puijo').plot(marker='o', alpha=.4, linewidth=0, label='growing or regrowing')

plt.legend()

# %% [markdown] tags=[]
# ## Change units:

# %%
fig, ax = plt.subplots(dpi=200)
ds_corr_height['N30'].sel(station='Puijo').plot(marker='o', linewidth=0, label='N30')
#ds_corr_height['FREQI'].sel(station='Puijo').plot(marker='d', alpha=.4)
#(ds_corr_height['FREQL']+ds_corr_height['FREQI']).sel(station='Puijo').plot(marker='*', alpha=0.5)

ds_corr_height['N50'].sel(station='Puijo').plot(marker='o', alpha=.4, linewidth=0,label='N50')

plt.legend()

# %%
vars_cm3_to_m3 = ['N30','N50','N70','N100','AWNC_incld','ACTNL_incld']

# %%
cm3_to_m3 = 1e-6

# %%
for v in vars_cm3_to_m3:
    print(ds_corr_height[v].units)
    if 'cm'  in ds_corr_height[v].units:
        ds_corr_height[v] = ds_corr_height[v]/cm3_to_m3
        ds_corr_height[v].attrs['units'] = 'm-3'
        
    print(ds_corr_height[v].units)

# %%
fig, ax = plt.subplots(dpi=200)
ds_corr_height['N30'].sel(station='Puijo').plot(marker='o', linewidth=0, label='N30')
#ds_corr_height['FREQI'].sel(station='Puijo').plot(marker='d', alpha=.4)
#(ds_corr_height['FREQL']+ds_corr_height['FREQI']).sel(station='Puijo').plot(marker='*', alpha=0.5)

ds_corr_height['N50'].sel(station='Puijo').plot(marker='o', alpha=.4, linewidth=0,label='N50')

plt.legend()

# %% [markdown]
# ## Set unit:

# %%
ds_corr_height['LCLOUD'].attrs['units'] = 1

# %% [markdown]
# ### Rename

# %%
rename_dic = dict(
    N100='n100',
    N30='n30',
    N50='n50',
    N70='n70',
    WSUB='vertical_velocity',
    LCLOUD='cloud_fraction',
    AWNC_incld = 'cdnc',
    ACTNL_incld = 'cdnc_cloud_top',
    T='temperature',
    Smax_cldv='activation_supersaturation',
    
    
)

# %%
ds_corr_height_rn = ds_corr_height.rename(rename_dic)

# %% [markdown]
# ## Drop WT

# %%
ds_corr_height_rn = ds_corr_height_rn.drop('WTKE') 
#.drop(.rename(rename_dic)

# %% [markdown]
# ## TO netcdf 

# %%
ds_corr_height_rn.to_netcdf(fn_output_corr_height_renamed)

# %%

# %%

# %% [markdown]
# ## To csv

# %%
for st in ds_corr_height['station'].values:
    print(st)
    fn_out = fn_output_corr_height_renamed.parent/ (fn_output_corr_height_renamed.stem)/ (fn_output_corr_height_renamed.stem+ f'_{st}.csv')
    print(fn_out)
    df = ds_corr_height_rn.sel(station=st).to_dataframe()
    df.to_csv(fn_out)
    
#ds_corr_height_rn.to_dataframe()

# %%
df['n70_cm3'] = df['n70']*1e-6
df['n100_cm3'] = df['n100']*1e-6
df['cdnc_cm3'] = df['cdnc']*1e-6

# %%
df['CC

# %%
df.columns

# %%
sns.histplot(y='cdnc_cm3',x='n100_cm3', data = df, bins = (np.linspace(0,100),np.linspace(0,100),))

# %%
df['cdnc_cm3'].dropna()

# %%
fig, axs = plt.subplots(5,2, figsize=[7,15])
for i, st in enumerate(ds_corr_height['station'].values):
    print(st)
    df = ds_corr_height_rn.sel(station=st).to_dataframe()
    if len(df['cdnc'].dropna())==0:
        continue

    df['n70_cm3'] = df['n70']*1e-6
    df['n100_cm3'] = df['n100']*1e-6
    df['cdnc_cm3'] = df['cdnc']*1e-6
    _axs = axs[i,:]
    for v, ax in zip(['n70_cm3','n100_cm3'], _axs):
        sns.histplot(y='cdnc_cm3',x=v, data = df,
                    ax = ax,
                    bins = (np.logspace(0,3,20),np.logspace(0,3,20),), 
                    cmap='plasma'
                    #    alpha=.2,
                    )
        ax.set_xscale('log')
        ax.set_yscale('log')
        #ax.set_ylim([0,1000])
        ax.set_title(st)
        
        
fig.tight_layout()

# %%
sns.displot(y='cdnc_cm3',x='n70_cm3', data = df, bins = (np.linspace(0,40,10),np.linspace(0,40,10),), cmap='plasma')


# %%
sns.displot(y='cdnc_cm3',x='n100_cm3', data = df, bins = (np.linspace(0,40,10),np.linspace(0,40,10),), cmap='plasma')

# %%
sns.displot(y='cdnc_cm3',x='n70_cm3', data = df, bins = (np.linspace(0,40,10),np.linspace(0,40,10),))

# %% [markdown]
#
# ## Extract units

# %%
df_units = pd.DataFrame(index = ['long_name','unit'])
for v in ds_corr_height_rn.keys():
    print(v)
    unit = ds_corr_height_rn[v].units
    print(unit)
    long_name = ds_corr_height_rn[v].long_name
    print(long_name)
    df_units[v] = [long_name,unit]

# %%
df_unitsT = df_units.T

df_unitsT.index= df_unitsT.index.rename('variable_name')

df_unitsT.to_csv(fn_units_desc)

# %% [markdown]
# ## Extra plots: 

# %%
fig, ax = plt.subplots(dpi=200)
ds_corr_height['N30'].sel(station='Puijo').plot(marker='o', linewidth=0, )
#ds_corr_height['FREQI'].sel(station='Puijo').plot(marker='d', alpha=.4)
#(ds_corr_height['FREQL']+ds_corr_height['FREQI']).sel(station='Puijo').plot(marker='*', alpha=0.5)

ds_corr_height['N50'].sel(station='Puijo').plot(marker='o', alpha=.4, linewidth=0)

plt.legend()

# %%
(ds_corr_height['N70']*1e-6).plot.hist(bins = np.logspace(0,4,), xscale='log')

# %%
ds_corr_height['Smax_cldv'].to_dataframe().reset_index()

# %%
sns.histplot(x='Smax_cldv', hue='station',data = ds_corr_height['Smax_cldv'].to_dataframe().reset_index())

# %%
sns.histplot(x='Smax_cldv', hue='station',data = ds_corr_height['Smax_cldv'].to_dataframe().reset_index())

# %%
sns.histplot(x='Smax_incld', hue='station',data = ds_corr_height['Smax_incld'].to_dataframe().reset_index())

# %%
plt.figure(dpi=200)
ds_corr_height['AWNC'].sel(station='Puijo').plot(marker='*', label='CDNC average grid box')

ds_corr_height['AWNC_incld'].sel(station='Puijo').plot(marker='o', alpha=.4, label='CDNC in-cloud')
ds_corr_height['ACTNL_incld'].sel(station='Puijo').plot(marker='o', alpha=.4, label='cloud top CDNC in-cloud')
#ds_corr_height['ACTNL'].sel(station='Puijo').plot(marker='o', alpha=.4)
plt.legend()

# %%
plt.figure(dpi=200)
ds_corr_height['WSUB'].plot()

# %%
plt.figure(dpi=200)
ds_corr_height['WSUB'].sel(station='Puijo').plot(marker='*')

ds_corr_height['WTKE'].sel(station='Puijo').plot(marker='o', alpha=.4)

# %%
plt.figure(dpi=200)
ds_corr_height['WSUB'].sel(station='Puijo').plot(marker='*')

ds_corr_height['WTKE'].sel(station='Puijo').plot(marker='o', alpha=.4)

# %%
plt.figure(dpi=200)
ds_corr_height['CLOUD'].sel(station='Puijo').plot(marker='*')

ds_corr_height['LCLOUD'].sel(station='Puijo').plot(marker='o', alpha=.4)

# %%
plt.figure(dpi=200)
ds_corr_height['AWNC'].sel(station='Puijo').plot(marker='*')

(ds_corr_height['AWNC']/ds_corr_height['FREQL']).sel(station='Puijo').plot(marker='o', alpha=.4)

# %%
fig, ax = plt.subplots(dpi=200)
ds_corr_height['FREQL'].sel(station='Puijo').plot(marker='*')
#ds_corr_height['FREQI'].sel(station='Puijo').plot(marker='d', alpha=.4)
#(ds_corr_height['FREQL']+ds_corr_height['FREQI']).sel(station='Puijo').plot(marker='*', alpha=0.5)

ds_corr_height['LCLOUD'].sel(station='Puijo').plot(marker='o', alpha=.4, linewidth=0, ax=ax.twinx(), c='m')
#(ds_corr_height['FCTI']+ds_corr_height['FCTL']).sel(station='Puijo').plot(marker='o', alpha=.4, linewidth=0)

# %%
fig, ax = plt.subplots(dpi=200)
ds_corr_height['Smax_cldv'].sel(station='Puijo').plot(marker='*', linewidth=0)
#ds_corr_height['FREQI'].sel(station='Puijo').plot(marker='d', alpha=.4)
#(ds_corr_height['FREQL']+ds_corr_height['FREQI']).sel(station='Puijo').plot(marker='*', alpha=0.5)

ds_corr_height['Smax_incld'].sel(station='Puijo').plot(marker='o', alpha=.4, linewidth=0, ax=ax.twinx(), c='m')
#(ds_corr_height['FCTI']+ds_corr_height['FCTL']).sel(station='Puijo').plot(marker='o', alpha=.4, linewidth=0)

# %%
fig, ax = plt.subplots(dpi=200)
ds_corr_height['Smax_cldv'].where(ds_corr_height['Smax_cldv_supZero']>0).sel(station='Puijo').plot(marker='*', linewidth=0)
#ds_corr_height['FREQI'].sel(station='Puijo').plot(marker='d', alpha=.4)
#(ds_corr_height['FREQL']+ds_corr_height['FREQI']).sel(station='Puijo').plot(marker='*', alpha=0.5)

ds_corr_height['Smax_incld'].where(ds_corr_height['Smax_incld_supZero']>0).sel(station='Puijo').plot(marker='o', alpha=.4, linewidth=0, ax=ax.twinx(), c='m')
#(ds_corr_height['FCTI']+ds_corr_height['FCTL']).sel(station='Puijo').plot(marker='o', alpha=.4, linewidth=0)

# %%
ds_corr_height['WTKE'].plot()

# %%

# %%
ds_corr_height

# %%

# %%
ds_corr_height['CLOUD'].plot()

# %% [markdown] tags=[]
# ## Geopotential height to geometrical height: 
# Ref: https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.geopotential_to_height.html
#

# %%
Ravg  = 6.371000e6#meters
g = 9.81

# %%
Ravg/1000


# %%
def geop_to_geom(Z):
    return g*(Z*Ravg)/(g*Ravg-Z)


# %%
geop_to_geom(100)

# %%
Z = np.linspace(0,10000)
zgeom = geop_to_geom(Z)

# %%
plt.plot(Z)
plt.plot(zgeom)

# %%

# %%

# %%

# %%

# %%
