import os

import numpy as np
import pandas as pd
import xarray as xr

from ukesm_bs_fdbck.constants import path_eusaar_data
from ukesm_bs_fdbck.util.eusaar_data import time_h, savepath_histc_flags, station_codes


def load_gd(station):
    """
    Load good data flag
    :param station:
    :return:
    """
    dr = path_eusaar_data + '/HISTC/'
    fp = dr + station + '_' + 'gd.dat'
    arr = (np.loadtxt(fp)[:, 0] == 1)
    return pd.Series(arr, index=time_h, name=station)


def load_dn(station):
    """
    Load day/night data flag
    :param station:
    :return:
    """
    dr = path_eusaar_data + '/HISTC/'
    fp = dr + station + '_' + 'gd.dat'
    arr = (np.loadtxt(fp)[:, 1] == 1)
    return pd.Series(arr, index=time_h, name=station)


def load_flags_allstations():
    if os.path.isfile(savepath_histc_flags):
        return xr.open_dataset(savepath_histc_flags)
    first = True
    for station in station_codes:
        a = load_gd(station).to_frame(name=station)
        if first:
            df = a  # .to_frame(name=station)
            first = False
        else:
            df[station] = a
    first = True
    for station in station_codes:
        a = load_dn(station).to_frame(name=station)
        if first:
            df_dn = a  # .to_frame(name=station)
            first = False
        else:
            df_dn[station] = a
    df_dn['KPO'][0:24]  # .plot()

    ds = df.to_xarray()
    ds_dn = df_dn.to_xarray()
    da = ds.to_array(dim='station', name='gd')
    da_dn = ds_dn.to_array(dim='station', name='dn')
    ds_flag = xr.merge([da_dn, da])
    ds_flag.to_netcdf(savepath_histc_flags)

    # %%


def make_data_flags():
    flags = load_flags_allstations()
    # day/night
    flags['NIG'] = flags['dn']
    flags['DAY'] = ~flags['NIG']

    # SEASONS:
    seas2monthn = dict(WIN=[12, 1, 2],
                       SPR=[3, 4, 5],
                       SUM=[6, 7, 8],
                       AUT=[9, 10, 11])

    # array of month number
    month = flags['time.month']
    for seas in seas2monthn.keys():
        flags[seas+'_c'] = xr.DataArray(np.in1d(month, seas2monthn[seas]), dims='time')
    flags['TOT'] = flags['gd']
    # make data_vars, not coordinates:
    for seas in seas2monthn.keys():
        flags[seas] = flags['TOT']&  flags[seas+'_c']
    #flags.reset_coords(['WIN', 'SUM', 'AUT', 'SPR'])
    return flags
    # %%

