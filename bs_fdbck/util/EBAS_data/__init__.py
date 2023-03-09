from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from bs_fdbck.constants import path_EBAS_data, path_measurement_data
from scipy import integrate

from bs_fdbck.util.EBAS_data.sizedistribution_integration import calc_Nx_interpolate_first


# %%


def get_ATTO_sizedist_data(station='ATTO', path_ebas=None):
    if path_ebas is None:
        path_ebas = path_measurement_data / 'ATTO'/'sizedistrib'
    else:
        path_ebas = Path(path_ebas)

    filep = path_ebas / 'ATTO-SMPS-clean_stp-transm-2014-202009_60m_correction_2021_resample1hour.nc'

    #filep = _path.glob('FI0050R.201[2345678]0101000000.*.nc')
    #filelist = list(filep) 
    #filelist.sort()
    print('Importing file: ')
    print(filep)

    #list_ds = []
    #for f in filelist:
    ds = xr.open_dataset(filep)
    rn_dic = dict(diameter='D',
                  dNdlog10D='particle_number_size_distribution_amean',

                  )
    ds = ds.rename(rn_dic)
    ds = ds.transpose("D", "time")
    #    list_ds.append(ds)
    return ds


# %%

def get_station_ebas_data(station='SMR', path_ebas=None, return_raw =False):
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
    if return_raw:
        return ds,
    return ds
# %%


def compute_Nx_ebas(ds, x=100, v_dNdlog10D='particle_number_size_distribution_amean'):
    # It's in dNdlogD
    # %%
    # %%
    #ds =    get_station_ebas_data()
    #v_dNdlog10D='particle_number_size_distribution_amean'
    # define mid point and top and bottom
    v_diam = 'log10D'
    ds['log10D'] = np.log10(ds['D'])
    mid_points = (ds[v_diam].values[0:-1] + ds[v_diam].values[1:]) / 2
    bottom = ds[v_diam].values[0] - (mid_points[0] - ds[v_diam].values[0])
    top = ds[v_diam].values[-1] + (mid_points[-1] - ds[v_diam].values[-2])
    d_lims = np.concatenate([np.array([bottom]), mid_points, np.array([top])])
    d_lims = 10**d_lims
    ds['bottom'] = xr.DataArray(d_lims[0:-1].transpose(), dims={'D': ds['D']})
    ds['top'] = xr.DataArray(d_lims[1:].transpose(), dims={'D': ds['D']})

    ds['diam_lims'] = ds[['bottom', 'top']].to_array(dim='limit')
    # compute dlogD:
    dlog10D = (np.log10(ds['diam_lims'].sel(limit='top')) - np.log10(ds['diam_lims'].sel(limit='bottom')))
    ds['dlog10D'] = xr.DataArray(dlog10D, dims={'D': ds['D']})
    ds['log10D'] = np.log10(ds['D'])
    ds['w_log10D'] = np.log10(ds['D'])
    # compute number of particles in each bin:
    ds['dN'] = ds[v_dNdlog10D] * ds['dlog10D']
    # %%
    #np.trapz(ds[v_dNdlog10D].squeeze().values, x=ds['log10D'].squeeze().values, axis=1)
    # %%
    #arg_gt_x = int(ds['D'].where(ds['diam_lims'].sel(limit='bottom') >= x).argmin().values)
    # arg_gt_x = int(ds['D'].where(ds['diam_lims'].sel(limit='bottom') >= x).argmin().values)
    #
    # y = ds[v_dNdlog10D].where(ds['D']>=100).squeeze().values
    # #x = np.array([*list(ds['log10D'].squeeze().values), float(d_lims[-1])])
    # print(y.shape)
    # #print(x.shape)
    # Ntot = integrate.trapz(y[arg_gt_x:,:], dx=0.05, axis=0)
    # Ntot_simp = integrate.simps(y[arg_gt_x:,:], dx=0.05, axis=0)
    # %%
    #Ntot_2 = integrate.quad(y[arg_gt_x:,:], dx=0.05, axis=0)
    ds['dNdD'] = ds[v_dNdlog10D]/ds['D']/np.log(10)
    # %%
    # def func_square(D):
    #     v = ds['dNdD'].sel(D=D, method='nearest')
    #     return v.values
    # 
    # integrate.quad(func_square, 100,1000, )
    # %%
    #ds['dNdD'].sel(D=slice(100,None)).integrate('D').plot()
    #plt.show()
    # %%
    # get index for bottom greater than limit
    arg_gt_x = int(ds['D'].where(ds['diam_lims'].sel(limit='bottom') > x).argmin().values)
    # get limits for grid box below
    # In log space...
    d_below = np.log10(ds['diam_lims'].isel(D=(arg_gt_x - 1)).sel(limit='bottom'))
    d_above = np.log10(ds['diam_lims'].isel(D=(arg_gt_x - 1)).sel(limit='top'))
    # fraction of gridbox above limit:
    frac_ab = (d_above - np.log10(x)) / (d_above - d_below)
    # Include the fraction of the bin box above limit:
    add = ds['dN'].isel(D=(arg_gt_x - 1)) * frac_ab
    #print(x)
    #print(ds['D'].isel(D=(arg_gt_x-1)))
    #print(ds['D'].isel(D=(arg_gt_x)))
    Nx_orig = ds['dN'].isel(D=slice(arg_gt_x,None)).sum('D') + add
    #print(x)
    #print(ds['D'].isel(D=slice(arg_gt_x,None)))
    Nx = ds['dNdD'].isel(D=slice(arg_gt_x,None)).integrate('D') + add
    #print(ds[v_dNdlog10D].isel(D=slice(arg_gt_x,None)).squeeze().values.shape)
    # %%
    # HERE:
    dNdlog10D = ds[v_dNdlog10D].isel(D = slice(arg_gt_x,None)).squeeze()
    Nx_trap2 = integrate.trapz(dNdlog10D.values, dx = 0.05, axis=0)+add
    # HERE:
    Nx_sum = dNdlog10D.sum('D')*0.05 + add

    #print(Nx_trap2.mean().values, Nx_sum.mean().values)
    #print(f'trap: {Nx_trap2.mean().values}, Nx sum:{Nx_sum.mean().values}')

    # %%
    # HERE:
    A = dNdlog10D[{'time':30}]

    Nx_trap2 = integrate.trapz(A.values, dx = 0.05, axis=0)
    # HERE:
    Nx_sum = A.sum()*0.05

    print(f'Nx_trap2: {Nx_trap2.mean()},Nx_sum: {Nx_sum.mean().values}')



    # %%
    print('orig, xarray integrate (trapezoidal), constant dlogD, trap2:')
    print(Nx_orig.mean().values, Nx.mean().values ,Nx_sum.mean().values, np.nanmean(Nx_trap2))
    print(Nx_orig.mean().values/Nx.mean().values, Nx.mean().values/Nx.mean().values ,Nx_sum.mean().values/Nx.mean().values)
    #Nx = Ntot +  ds['dN'].isel(D=(arg_gt_x - 1)) * frac_ab
    #Ntot_add = Ntot + ds['dN'].isel(D=(arg_gt_x - 1)) * frac_ab
    ds[f'N{x}'] = Nx_orig



    # %%
    #plt.plot(Ntot, alpha=0.5, label = 'trap')

    #plt.plot(ds['dNdD'].sel(D=slice(100,None)).integrate('D'), alpha=0.5, label='xarr')
    # plt.plot(Ntot_add, alpha=0.5, label = 'trap_add')
    # plt.plot(Nx_orig, alpha=0.5, label = 'orig')
    # plt.plot(Nx_sum, alpha=0.5, label = 'orig, sum')
    # plt.plot(Nx, alpha=0.5, label = 'xarr_tog')
    #
    # plt.legend()
    # plt.show()
    #
    #
    # #print(np.nanmean(Nx_orig))
    # #print(np.nanmean(Nx))
    # print(np.nanmean(Nx_orig)/np.nanmean(Nx))
    # print(np.nanmean(Nx_sum)/np.nanmean(Nx))
    #
    # # %%
    # #print(np.nanmean(Ntot_add))
    # print(np.nanmean(Nx))
    # #print(np.nanmean(Ntot_add)/np.nanmean(Nx))
    #print(np.nanmean(Nx)/np.nanmean(Ntot_add))
    # 

    # %%
    return ds

# %%
def get_ebas_dataset_with_Nx(x_list = None,
                             station='SMR',
                             path_ebas=None,
                             ds = None,
                             ):
    """

    :param x_list:
    :param station:
    :param path_ebas:
    :return:
    """
    # station='SMR', path_EBAS=None
    # %%
    if x_list is None:
        x_list = [50, 80, 100, 150, 200, 250, 300]
    if ds is None:
        if station == 'ATTO':
            ds = get_ATTO_sizedist_data( path_ebas=path_ebas)
        else:
            ds = get_station_ebas_data(station=station, path_ebas=path_ebas)
    # %%
    for x in x_list:
        Nx = calc_Nx_interpolate_first(ds,
                                       x=x,
                                       var_diam='D',
                                       v_dNdlog10D='particle_number_size_distribution_amean' )
        ds[f'N{x}'] = Nx

    return ds

# %%
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
    #ds.groupby(ds['time.hour']).mean()['N100'].plot()
    #plt.show()
    # %%
    return df_JA, ds_median