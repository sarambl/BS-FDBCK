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
# ## layer thickness lacking coordinates:

# %%
from bs_fdbck_clean.constants import get_input_datapath

# %%
path_ukesm_input = get_input_datapath(model='UKESM')
path_ukesm_input

# %%
f_layer_thickness = path_ukesm_input/'AEROCOMTRAJ'/'aerocom3_UKESM_GlobalTraj-CE_LayerThickness_fixed.nc'
f_ordinary = path_ukesm_input/'AEROCOMTRAJ'/'aerocom3_UKESM_GlobalTraj-CE_altitude_ModelLevel_fixed.nc'

# %%
f_layer_thickness_out = path_ukesm_input/'AEROCOMTRAJ'/'aerocom3_UKESM_GlobalTraj-CE_layer_thickness_fixed_smb.nc'


# %%
import xarray as xr

# %%
ds_layer = xr.open_dataset(f_layer_thickness, engine='netcdf4')

# %%
ds_ordinary = xr.open_dataset(f_ordinary)

# %%
ds_layer['layer'].isel(lev=0).plot()

# %%
ds_ordinary['altitude'].isel(lev=0).plot()

# %%
ds_layer['lat'] = ds_ordinary['lat']

# %%
ds_layer['lon'] = ds_ordinary['lon']

# %%

# %%
ds_layer['layer'].isel(lev=0).plot()

# %%
f_layer_thickness

# %%
ds_layer.to_netcdf(f_layer_thickness_out)

# %%
