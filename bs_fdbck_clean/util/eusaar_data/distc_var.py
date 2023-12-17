import os

import numpy as np
import pandas as pd
import xarray as xr

from bs_fdbck_clean.util.eusaar_data import p_gen, p_distc, station_codes, subs_codes, years_codes
from bs_fdbck_clean.constants import path_eusaar_outdata

savepath_distc_vars = path_eusaar_outdata / 'distc_vars.nc'
percs_in_eusaar_files = [5., 16., 50., 84., 95.]  # percentiles in files


def get_diameter_sized():
    p = p_gen / 'standard_sizes.dat'
    diameter = pd.Index(np.loadtxt(p), name='diameter')
    return diameter


d = get_diameter_sized()


# %%
def get_distc_dataframe(station, year='BOTH', subs='TOT'):
    rows = ['%.0fth percentile' % p for p in
            percs_in_eusaar_files]  # , '16th percentile', '50th percentile','84th percentile']
    diam = get_diameter_sized()
    p = p_distc / f'{station}_{subs}_DIST_{year}.dat'
    m = np.loadtxt(p)
    return pd.DataFrame(m.transpose(), index=diam, columns=rows)


def get_distc_xarray_station(station, year='BOTH', subs='TOT'):
    df = get_distc_dataframe(station, year=year, subs=subs)
    da = df.to_xarray().to_array(dim='percentile')
    da['diameter'].attrs['units'] = 'nm'
    da.name = 'dN/dlog10(d$_p$)'
    return da


def get_distc_xarray(year='BOTH', subs='TOT'):
    ls = []
    for station in station_codes:
        ls.append(get_distc_xarray_station(station, year=year, subs=subs))
    da_c = xr.concat(ls, dim='station')
    da_c['station'] = station_codes
    return da_c


def get_distc_xarray_allsubs(year='BOTH'):
    ls = []
    for subs in subs_codes:
        ls.append(get_distc_xarray(year=year, subs=subs))
    da_c = xr.concat(ls, dim='subset')
    da_c['subset'] = subs_codes
    return da_c


def get_distc_xarray_all(from_nc=True):
    if from_nc and os.path.isfile(savepath_distc_vars):
        return xr.open_dataset(savepath_distc_vars)
    ls = []
    for year in years_codes:
        ls.append(get_distc_xarray_allsubs(year=year))
    da_c = xr.concat(ls, dim='year')
    da_c['year'] = years_codes
    print(da_c)
    da_c = da_c.rename('dNdlog10dp')
    da_c.to_netcdf(savepath_distc_vars)
    return da_c
# %%
