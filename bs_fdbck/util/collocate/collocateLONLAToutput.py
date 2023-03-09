import os
import re
import time

import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

from bs_fdbck.constants import collocate_locations
from bs_fdbck.util.collocate.collocate import CollocateModel


# %%

def make_st_dataset(ds, var, locations = None):
    if locations is None:
        _df = collocate_locations.transpose()
    else:
        _df = locations.transpose()
    _n = 'noresm_input_format'
    station = 'station'
    # _df = collocate_locations.transpose()
    _df.index.name = station
    _df = _df[[_n]].reset_index().set_index(_n)

    varl = [var + '_' + ext for ext in _df.index]
    if varl[0] not in ds.data_vars:
        # try noresm2:
        _n = 'noresm_input_format'
        _df = locations.transpose()
        _df.index.name = station
        _df = _df[[_n]].reset_index().set_index(_n)
        varl = [var + '_' + ext for ext in _df.index]
    da: xr.DataArray = ds[varl].squeeze().to_array(name=var, dim='orig_names')
    # copy attributes:
    _attrs_dum = ds[varl[0]].attrs

    for att in ['units', 'long_name', 'cell_methods']:
        if att in _attrs_dum.keys():
            da.attrs[att] = _attrs_dum[att]
    # nd = da['orig_names'].copy()
    nd = xr.DataArray(_df[station].values, dims='orig_names')
    da = da.assign_coords(**{'station': nd})
    da = da.swap_dims({'orig_names': 'station'})
    return da


def get_unique_vars(ds):
    vrs = list(ds.data_vars)  # .values
    vrs_cl = []
    # v = vrs[1]
    for v in vrs:
        if v.find('LON') > -1:
            v_c = v[:(v.find('LON') - 1)]
            if v_c not in vrs_cl:
                vrs_cl.append(v_c)
        elif bool(re.match('.*_[-+]?\d*\.\d+|\d+[ew]_[-+]?\d*\.\d+|\d+[ns]', v)):  # v.find('lon')>-1:
            # v_c = v[:(v.find('lon')-1)]
            v_c = '_'.join(v.split('_')[:-2])
            if v_c not in vrs_cl:
                vrs_cl.append(v_c)

    return vrs_cl


class CollocateLONLATout(CollocateModel):

    def __init__(self, *_vars, **kwargs):
        if 'history_field' not in kwargs.keys():
            kwargs['history_field'] = '.h1.'
            # del kwargs['history_field']#='False'
        # if 'use_pressure_coords' in kwargs.keys():
        kwargs['use_pressure_coords'] = False
        #kwargs['space_res'] = 'locations'
        print(kwargs)
        super().__init__(*_vars, **kwargs)  # , history_field='.h1.')

    """
    Read in raw data,
    - translate from var_name_LON_XX_LAT_XX(lev,time) --> var_name(station, lev,tim)
        ls_da = []
        - for lat_lon in [station_latlo]:
            var_na = var_name+latlon 
            ls_da.append(ds[var_na].rename(
    """

    def get_station_ds(self, varl, ds=None):
        ls_dataset = []
        if len(varl) == 1:
            return self.make_station_da(varl[0], ds=ds)
        for var in varl:
            ls_dataset.append(self.make_station_da(var, ds=ds))
        return xr.merge(ls_dataset, compat='override')

    def make_station_da(self, var, ds=None, recalculate=False):
        fn = self.savepath_coll_ds(var)
        print(fn)
        if os.path.isfile(fn) and not recalculate:
            return xr.open_dataset(fn)
        else:
            if ds is None:
                print('Loading dataset:')
                ds = self.load_raw_ds(None)  # chunks={'lev':2})
            print('making other dataset')
            da = make_st_dataset(ds, var, locations=self.locations)
            #ds = xr_fix(da.to_dataset(), model_name=self.model_name)
            # ds = xr_fix(da.to_dataset(), model_name=self.model_name)
            # da.to_netcdf(fn)
            print(fn)
            delayed_obj = ds.to_netcdf(fn, compute=False)  # , chunks={'diameter':1})
            print('Saving %s to %s' % (var, fn))

            with ProgressBar():
                results = delayed_obj.compute()
        return ds


    def make_station_data_all(self):
        time1 = time.time()
        ds = self.load_raw_ds(None, chunks={'time': 24 * 7})
        time2 = time.time()
        print('TIME TO LOAD RAW DATASET IN COLLOCATELATLON: %s s' % (time2 - time1))
        # get unique variables in code:
        vrs = get_unique_vars(ds)
        for var in vrs:
            print(var)
            print('making station dataset for %s' % var)
            self.make_station_da(var, ds=ds)
        return ds

    def make_station_data_merge_monthly(self, varlist):
        # varlist = list_sized_vars_noresm
        _dr = pd.date_range(self.from_time, self.to_time, freq='MS')[:-1]
        for d in _dr:
            ft = d.strftime(format='%Y-%m-%d')

            tt = (d + pd.DateOffset(months=1)).strftime(format='%Y-%m-%d')
            print('Running collocate montly')
            self.collocate_month(ft, tt, varlist)
            # self.collocate_month(from, varlist, year)
        for var in varlist:
            fn = self.savepath_coll_ds(var)
            print(fn)
            if os.path.isfile(fn):
                continue
            print(f'Concatinating months for {var}:')
            ds_conc = self.concatinate_months(var)

            delayed_obj = ds_conc.to_netcdf(fn, compute=False)  # , chunks={'diameter':1})
            print('Saving %s to %s' % (var, fn))

            with ProgressBar():
                results = delayed_obj.compute()
        return

    def collocate_month(self, from_t, to_t, varlist, location=None):
        c = CollocateLONLATout(self.case_name, from_t, to_t,
                               self.isSectional,
                               'hour',
                               history_field=self.history_field,
                               locations=self.locations)
        print(f'CHECKING if raw load necessary for {from_t}-{to_t}')
        if c.check_if_load_raw_necessary(varlist):
            time1 = time.time()
            a = c.make_station_data_all()
            time2 = time.time()
            print(f'Subset {from_t} to {to_t}  done')
            print('DONE : took {:.3f} s'.format((time2 - time1)))

    # noinspection PyTypeChecker
    def concatinate_months(self, var):
        ls = []
        _dr = pd.date_range(self.from_time, self.to_time, freq='MS')[:-1]
        for d in _dr:
            from_t = d.strftime(format='%Y-%m-%d')
            to_t = (d + pd.DateOffset(months=1)).strftime(format='%Y-%m-%d')
            c = CollocateLONLATout(self.case_name, from_t, to_t,
                                   self.isSectional,
                                   'hour',
                                   history_field=self.history_field,
                                   locations=self.locations)
            ls.append(c.make_station_da(var))
        ds_conc = xr.concat(ls, 'time')
        # remove duplicates in time:
        ds_conc: xr.Dataset
        ds_conc = ds_conc.sel(time=~ds_conc.indexes['time'].duplicated())
        return ds_conc

    def get_new_instance(self, *_vars, **kwargs):
        return CollocateLONLATout(*_vars, **kwargs)