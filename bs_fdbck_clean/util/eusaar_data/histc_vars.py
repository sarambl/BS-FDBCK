import os

import numpy as np
import pandas as pd
import xarray as xr

from bs_fdbck_clean.constants import path_eusaar_data
from bs_fdbck_clean.util.eusaar_data import time_h, station_codes, long_name_var_dic, standard_varlist_histc, \
    savepath_histc_vars
from bs_fdbck_clean.util.eusaar_data.flags import load_gd
from bs_fdbck_clean.util.practical_functions import make_folders


def load_data_timeseries(station, var):
    """
    Load data timeseries for variable
    :param station:
    :param var:
    :return: pandas.Series
    """
    dr = path_eusaar_data + '/HISTC/'
    fp = dr + station + '_' + var + '.dat'
    arr = np.loadtxt(fp)
    return pd.Series(arr, index=time_h, name=station)


# %%


def load_var_as_dtframe(var):
    """
    Load variable for all stations as dataframe
    :param var:
    :return:
    """
    df_o = pd.DataFrame()
    for station in station_codes:
        s = load_data_timeseries(station, var)
        s_gd = load_gd(station)
        df_o[station] = s.where(s_gd)
    return df_o


def load_var_as_xarray(var):
    """Loads variable list from HISTC and creates xarray dataarray
    with dims station and time
    :param var:
    :return: xr.DataArray
    """
    attrs = dict(
        units='cm-3',
    )
    if var in long_name_var_dic:
        attrs['long_name'] = long_name_var_dic[var]
        attrs['fancy_name'] = long_name_var_dic[var]

    df = load_var_as_dtframe(var)
    da = df.to_xarray().to_array(dim='station', name=var)
    for att in attrs:
        da.attrs[att] = attrs[att]
    return da


def load_vars_as_xarray(varl=None):
    """
    Loads variable list from HISTC and creates xarray dataset
    with dims station and time
    :param varl: list of variables
    :return:
    """
    if varl is None:
        varl = standard_varlist_histc
    xa_l = []
    for var in varl:
        xa_l.append(load_var_as_xarray(var))
    return xr.merge(xa_l)


def load_and_save_vars_as_xarray():
    ds = load_vars_as_xarray()
    make_folders(savepath_histc_vars)
    ds.to_netcdf(savepath_histc_vars)
    return ds


def get_histc_vars_xr():
    """
    get histc variables (N30, N50, N100, N250) for all years
    :return:
    """
    if os.path.isfile(savepath_histc_vars):
        return xr.load_dataset(savepath_histc_vars)
    else:
        return load_and_save_vars_as_xarray()