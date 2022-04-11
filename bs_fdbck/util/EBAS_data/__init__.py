from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from bs_fdbck.constants import path_EBAS_data


# %%


# %%

def get_station_ebas_data(station='SMR', path_ebas=None):
    if path_ebas is None:
        path_ebas = path_EBAS_data
    else:
        path_ebas = Path(path_ebas)

    _path = path_ebas / 'raw_data' / station

    filep = _path.glob('FI0050R.201[2345678]0101000000.*.nc')
    filelist = list(filep)
    filelist.sort()
    print('Importing files: ')
    print(filelist)

    list_ds = []
    for f in filelist:
        ds = xr.open_dataset(f)
        list_ds.append(ds)

    for i, ds in enumerate(list_ds):
        if 'particle_number_size_distribution_amean_qc_flags' in ds.dims:
            ds = ds.isel(particle_number_size_distribution_amean_qc_flags=0)
            list_ds[i] = ds
        if 'particle_number_size_distribution_prec1587_qc_flags' in ds.dims:
            ds = ds.isel(particle_number_size_distribution_prec1587_qc_flags=0)
            list_ds[i] = ds
        if 'particle_number_size_distribution_perc8413_qc_flags' in ds.dims:
            ds = ds.isel(particle_number_size_distribution_perc8413_qc_flags=0)
            list_ds[i] = ds

    ds = xr.concat(list_ds, dim='time')
    return ds
# %%

def compute_Nx_ebas(ds, x=100, v_dNdlog10D='particle_number_size_distribution_amean'):
    # It's in dNdlogD
    # %%
    # %%
    #ds =    get_station_ebas_data()
    #v_dNdlog10D='particle_number_size_distribution_amean'
    # define mid point and top and bottom
    mid_points = (ds['D'].values[0:-1] + ds['D'].values[1:]) / 2
    bottom = ds['D'].values[0] - (mid_points[0] - ds['D'].values[0])
    top = ds['D'].values[-1] + (mid_points[-1] - ds['D'].values[-2])
    d_lims = np.concatenate([np.array([bottom]), mid_points, np.array([top])])

    ds['bottom'] = xr.DataArray(d_lims[0:-1].transpose(), dims={'D': ds['D']})
    ds['top'] = xr.DataArray(d_lims[1:].transpose(), dims={'D': ds['D']})

    ds['diam_lims'] = ds[['bottom', 'top']].to_array(dim='limit')
    # compute dlogD:
    dlogD = (np.log10(ds['diam_lims'].sel(limit='top')) - np.log10(ds['diam_lims'].sel(limit='bottom')))
    ds['dlog10D'] = xr.DataArray(dlogD, dims={'D': ds['D']})
    ds['log10D'] = np.log10(ds['D'])
    # compute number of particles in each bin:
    ds['dN'] = ds[v_dNdlog10D] * ds['dlog10D']
    # get index for bottom greater than limit
    arg_gt_x = int(ds['D'].where(ds['diam_lims'].sel(limit='bottom') >= x).argmin().values)
    # get limits for grid box below
    d_below = ds['diam_lims'].isel(D=(arg_gt_x - 1)).sel(limit='bottom')
    d_above = ds['diam_lims'].isel(D=(arg_gt_x - 1)).sel(limit='top')
    # fraction of gridbox above limit:
    frac_ab = (d_above - x) / (d_above - d_below)
    # Include the fraction of the bin box above limit:
    Nx = ds['dN'].where(ds['D'] >= x).sum('D') + ds['dN'].isel(D=(arg_gt_x - 1)) * frac_ab

    ds[f'N{x}'] = Nx

    # %%
    return ds


def get_ebas_dataset_with_Nx(x_list=None, station='SMR', path_ebas=None):
    # station='SMR', path_EBAS=None
    if x_list is None:
        x_list = [50, 80, 100, 150, 200, 250, 300]
    ds = get_station_ebas_data(station=station, path_ebas=path_ebas)
    for x in x_list:
        ds = compute_Nx_ebas(ds, x=x, v_dNdlog10D='particle_number_size_distribution_amean')

    return ds


def get_ebas_dataset_Nx_daily_median(x_list=None, station='SMR', path_ebas=None):
    # %%
    ds = get_ebas_dataset_with_Nx(x_list=x_list,station=station, path_ebas=path_ebas)
    # ds_day = ds.where((ds['time.hour']>6) & ds['time.hour']<18)
    ds_median = ds.resample(dict(time='D')).median()
    ds_median['JA'] = (ds_median['time.month'] == 7) | (ds_median['time.month'] == 8)
    ds_median['season'] = ds_median['time.season']
    # %%
    return ds_median, ds


# %%
def get_ebas_dataset_Nx_daily_JA_median_df(x_list=None, station='SMR', path_ebas=None):
    if x_list is None:
        x_list = [50, 80, 100, 150, 200, 250, 300]
    # %%
    ds_median, ds = get_ebas_dataset_Nx_daily_median(x_list=x_list,station=station, path_ebas=path_ebas)
    varl = [f'N{x}' for x in x_list] + ['JA', 'season']
    #print(varl)
    df = ds_median[varl].to_dataframe()#.set_
    df_JA = df[df['JA']]
    # %%
    df_JA.mean()
    # %%

    print(df_JA.median())
    ds.groupby(ds['time.hour']).mean()['N100'].plot()
    plt.show()
    # %%
    return df_JA, ds_median