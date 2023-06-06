import re

import pandas as pd
from bs_fdbck_clean.util.naming_conventions import constants

path_var_overview = constants.path_var_overview

def df_var_overview():
    return pd.read_csv(path_var_overview, index_col=0)



def get_fancy_unit(var):
    if var in df_var_overview().index:

        return df_var_overview().loc[var,'nice_units']#.value
    else:
        return ''
def get_fancy_var_name(var):
    if var in df_var_overview().index:
        return df_var_overview().loc[var,'nice_name']
    else:
        return var
# %%

def get_fancy_unit_xr(xr_arr, var):
    fancy_unit = get_fancy_unit(var)
    if pd.isna(fancy_unit):
        fancy_unit = xr_arr.attrs['units']
    if len(fancy_unit)==0:
        if 'units' in xr_arr.attrs.keys():
            fancy_unit = xr_arr.attrs['units']
        else:
            fancy_unit=''
    return fancy_unit


def get_fancy_var_name_xr(xr_arr, var):
    varn = get_fancy_var_name(var)
    varn = get_fancylabel_Nd_from_varN(varn)
    if varn==var:
        if 'long_name' in xr_arr.attrs:
            varn = xr_arr.attrs['long_name']
    return  varn




####################################################################
def get_varname_Nd(fromNd, toNd):
    """
    returns variable name of N_x<dy.
    :param fromNd:
    :param toNd:
    :return:
    """
    if fromNd > 0:
        varNameN = 'N%d_d_%d' % (fromNd, toNd)
    else:
        varNameN = 'Nd_%d' % toNd
    return varNameN


def get_fancylabel_Nd(fromNd, toNd):
    """

    :param fromNd:
    :param toNd:
    :return:
    """
    if fromNd > 0:
        varNameN = 'N$_{%d<d<%d}$' % (fromNd, toNd)
    else:
        varNameN = 'N$_{d<%d}$' % toNd
    return varNameN


def get_fancylabel_Nd_from_varN(var):
    if bool(re.match('N\d+_d_\d+',var)):
       fromNd = var.split('_')[0][1:]
       toNd = var.split('_')[-1]
       return get_fancylabel_Nd(int(fromNd), int(toNd))
    else: return  var