import os

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from bs_fdbck_clean.util.practical_functions import make_folders
import useful_scit.util.log as log

from bs_fdbck_clean.constants import get_outdata_path
from bs_fdbck_clean.util.eusaar_data import subs_codes as subset_codes_eusaar
from bs_fdbck_clean.util.eusaar_data.distc_var import percs_in_eusaar_files
from bs_fdbck_clean.util.eusaar_data.flags import make_data_flags

# path_eusaar_outdata
log.ger.setLevel(log.log.INFO)

savepath_distc_model_ds = get_outdata_path('eusaar') + '/noresm/'  # distc_ds_noresm.nc'
print(savepath_distc_model_ds)


# %%
def compute_percentile_flag(ds, flag, quants=None):
    if quants is None:
        quants = np.array(percs_in_eusaar_files) / 100.
    # get's flags:
    # %%
    flags = make_data_flags()
    ds_w = ds.where(flags[flag])
    log.ger.info(f'Computing percentiles for {flag}')
    with ProgressBar():
        ds_w = ds_w.compute()
    da_percs = ds_w.quantile(quants, dim='time')
    # %%
    return da_percs


def compute_percentiles_flag(ds, flag, percs=percs_in_eusaar_files):
    quants = np.array(percs) / 100.
    ds_out = compute_percentile_flag(ds, flag, quants)
    pst = (ds_out['quantile'] * 100).astype(int).astype(str)
    ds_out['percentile'] = pst.str.replace(r'(.*)', r'\1th percentile')
    ds_out = ds_out.swap_dims({'quantile': 'percentile'})
    # ds_out['quantile'] = ['%sth percentile'%(int(100*q)) for q in ds_out['quantile']]
    return ds_out


def compute_all_subsets_percs_flag(ds: xr.Dataset, flags: list = subset_codes_eusaar) -> xr.Dataset:
    # %%
    flags = list(set(flags) - {'ECH'})
    ls = []
    for flag in flags:
        log.ger.info(f'Computing percentiles for flag {flag}:')
        ds_fl = compute_percentiles_flag(ds, flag)
        ds_fl = ds_fl.assign_coords({'subset': flag})
        ls.append(ds_fl)
    # %%
    ds_out = xr.concat(ls, dim='subset')
    return ds_out


# %%
def get_all_distc_noresm(case_name, from_t, to_t, ds=None, recompute=False):
    fn = get_savepath_distc_model_ds(case_name, from_t, to_t)
    if os.path.isfile(fn) and not recompute:
        return xr.open_dataset(fn)
    else:
        if ds is None:
            print('Could not find file %s, and no dataset supplied please compute it' % fn)
        else:
            # %%
            print('combining to total distribution:')
            ds = combine_to_total_dist(ds)
            # %%
            print('Computing all subsets and flags')
            ds: xr.Dataset
            ds_out = compute_all_subsets_percs_flag(ds)
            # %%
            ds_out['dNdlog10dp'] = np.log(10) * ds_out['dNdlogD']
            ds_out['dNdlog10dp_mod'] = np.log(10) * ds_out['dNdlogD_mod']
            if 'dNdlogD_sec' in ds:
                ds_out['dNdlog10dp_sec'] = np.log(10) * ds_out['dNdlogD_sec']
            del_o = ds_out.to_netcdf(fn, compute=False)
            with ProgressBar():
                results = del_o.compute()
            return ds_out


# %%
def combine_to_total_dist(ds):
    if 'dNdlogD_sec' in ds:
        ds['dNdlogD'] = ds['dNdlogD_mod'] + ds['dNdlogD_sec']
    else:
        ds['dNdlogD'] = ds['dNdlogD_mod']
    return ds


def get_savepath_distc_model_ds(case, from_t, to_t):
    fn = savepath_distc_model_ds + '/distc_%s_%s_%s.nc' % (case, from_t, to_t)
    make_folders(fn)
    return fn
