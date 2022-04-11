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

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from bs_fdbck.constants import measurements_path, path_outdata, path_extract_latlon_outdata

# %%
xr.set_options(keep_attrs=True) 

# %%
path_extract_latlon_outdata

# %%
lat_smr = 61.85
lon_smr = 24.28

# %%
case_name = 'OsloAero_intBVOC_pertSizeDist_f19_f19_mg17_full'

# %%
fn = path_extract_latlon_outdata/ case_name/f'{case_name}.h1._2012-01-01-2015-01-01_concat_subs_22.0-30.0_60.0-66.0.nc'
fn_comb_lev1 = path_extract_latlon_outdata/ case_name/f'{case_name}.h1._2012-01-01-2015-01-01_concat_subs_22.0-30.0_60.0-66.0_lev1.nc'

# %%

# %%
fn2 = fn.parent / f'{fn.stem}_sort.nc'

# %%

# %%


cases = [case_name]

# %%
from pathlib import Path

# %%
plot_path = Path('Plots')


# %%
def make_fn(case, v_x, v_y):
    _x = v_x.split('(')[0]
    _y = v_y.split('(')[0]
    f = f'scat_{case}_{_x}_{_y}.png'
    return plot_path /f


# %%
plot_path.mkdir(exist_ok=True, parents=True)

# %%
varl =['DOD500','DOD440','ACTREL','ACTNL','TGCLDLWP', #,'SOA_A1',
       'H2SO4','SOA_LV','COAGNUCL','FORMRATE','T','FCTL',
       'TOT_CLD_VISTAU','TOT_ICLD_VISTAU','TGCLDCWP',
       #'TAUTLOGMODIS',
       #'MEANTAU_ISCCP',
       #'LWPMODIS','CLWMODIS','REFFCLWMODIS',#'TAUTMODIS','TAUWMODIS',
      
      'SOA_NA','SOA_A1','OM_NI','OM_AI','OM_AC','SO4_NA','SO4_A1','SO4_A2','SO4_AC','SO4_PR',
      'BC_N','BC_AX','BC_NI','BC_A','BC_AI','BC_AC','SS_A1','SS_A2','SS_A3','DST_A2','DST_A3', 
      ] 


# %%
varl_st = [      'SOA_NA','SOA_A1','OM_NI','OM_AI','OM_AC','SO4_NA','SO4_A1','SO4_A2','SO4_AC','SO4_PR',
      'BC_N','BC_AX','BC_NI','BC_A','BC_AI','BC_AC','SS_A1','SS_A2','SS_A3','DST_A2','DST_A3']

# %% [markdown]
# ## Load observations: 

# %% [markdown] tags=[]
# ## Open model dataset: 
#

# %%
model_lev_i=-1

# %%
ds_mod = xr.open_dataset(fn, chunks = {'time':48})#[fn1,fn2])#.sortby('time')
#ds_mod2 = xr.open_dataset(fn2, chunks = {'time':48})

# %%
ds_mod['TOT_ICLD_VISTAU_s']= ds_mod['TOT_ICLD_VISTAU'].sum('lev')
ds_mod['TOT_CLD_VISTAU_s']= ds_mod['TOT_CLD_VISTAU'].sum('lev')

ds_mod = ds_mod.sortby('time')#.sel(time=slice('2012','2014'))



# %%
ds_mod = ds_mod.isel(lev = model_lev_i)

# %% [markdown]
# ds_mod1

# %%
import dask.array as da
from dask.diagnostics import ProgressBar


# %%
delayed_obj = ds_mod.to_netcdf(fn_comb_lev1, compute=False)
with ProgressBar(): 
    results = delayed_obj.compute()

# %%
