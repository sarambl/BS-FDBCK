import numpy as np
import pandas as pd
import xarray as xr

from bs_fdbck.util.eusaar_data import p_gen, p_histc, station_codes


row_name_hists = ['N30-50 all time',
                  'N50    all time',
                  'N100   all time',
                  'N30-50 Winter',
                  'N30-50 Spring',
                  'N30-50 Summer',
                  'N30-50 Autumn',
                  'N50 Winter',
                  'N50 Spring',
                  'N50 Summer',
                  'N50 Autumn',
                  'N100 Winter',
                  'N100 Spring',
                  'N100 Summer',
                  'N100 Autumn',
                  'N30-50 Night',
                  'N30-50 Day',
                  'N50 Night',
                  'N50 Day',
                  'N100 Night',
                  'N100 Day',
                  'N30-50 ECHAM5 Sampling',
                  'N50 ECHAM5 Sampling',
                  'N100 ECHAM5 Sampling']

def open_hists_2dataframe(station):
    hist_c_means_p = p_gen+'hist_cons_means.dat'
    hist_c_means = np.loadtxt(hist_c_means_p)

    p = p_histc + station+'_hists.dat'
    hists = np.loadtxt(p)
    return pd.DataFrame(hists.transpose(), index=hist_c_means, columns=row_name_hists)


def open_hists2xarray_station(station):
    df = open_hists_2dataframe(station)
    ds = df.to_xarray().rename({'index':'number concentration'})
    ds['number concentration'].attrs['unit']='cm-3'
    ds['number concentration'].attrs['long_name'] = 'Number concentration'

    for var in row_name_hists:
        ds[var].attrs['units']='Absolute relative frequency'
    return ds
    # %%


def open_hists2xarray():
    ls = []
    for station in station_codes:
        ls.append(open_hists2xarray_station(station))

    ds_c = xr.concat(ls, dim='station')
    ds_c['station']=station_codes
    return ds_c

# %%
"""
import matplotlib.pyplot as plt
a = open_hists2xarray()
for station in ['ASP','BIR','PAL']:# station_codes[0:4]:
    a['N50    all time'].sel(station=station).plot(xscale='log', label=station)
plt.xlim([1,1e4])
##"plt.ylim([10,1e4])

plt.legend()
plt.show()
#plt.show()
"""
# %%