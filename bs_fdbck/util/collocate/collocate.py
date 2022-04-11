import os
import time
from bs_fdbck.data_info import get_nice_name_case
from bs_fdbck.util.imports.fix_xa_dataset_v2 import xr_fix
from bs_fdbck.util.imports.get_pressure_coord_fields import get_pressure_coord_fields

from bs_fdbck.util.imports.import_fields_xr_v2 import xr_import_NorESM
from useful_scit.util import log

from bs_fdbck import constants
import xarray as xr
from bs_fdbck.util.practical_functions import make_folders
from dask.diagnostics import ProgressBar


class Collocate:
    pass


class CollocateModel(Collocate):
    """
    collocate a model to a list of locations
    """
    col_dataset = None
    input_dataset = None

    def __init__(self, case_name, from_time, to_time,
                 is_sectional,
                 time_res,
                 space_res='locations',
                 model_name='NorESM',
                 history_field='.h0.',
                 raw_data_path=constants.get_input_datapath(),
                 locations=constants.collocate_locations,
                 read_from_file=True,
                 chunks=None,
                 use_pressure_coords=False,
                 dataset=None,
                 savepath_root=constants.get_outdata_path('collocated')
                 ):
        """
        :param case_name:
        :param from_time:
        :param to_time:
        :param raw_data_path:
        :param is_sectional:
        :param time_res: 'month', 'year', 'hour'
        :param space_res: 'full', 'locations'
        :param model_name:
        """
        self.chunks = chunks
        self.read_from_file = read_from_file
        self.model_name = model_name
        # self.case_plotting_name = model_name
        self.dataset = None
        self.use_pressure_coords = use_pressure_coords
        self.case_name_nice = get_nice_name_case(case_name)
        self.case_name = case_name
        self.raw_data_path = raw_data_path
        self.from_time = from_time
        self.to_time = to_time
        self.time_resolution = time_res
        self.space_resolution = space_res
        self.history_field = history_field
        self.locations = locations
        self.isSectional = is_sectional
        self.dataset = dataset
        self.savepath_root = savepath_root

        self.attrs_ds = dict(raw_data_path=str(self.raw_data_path),
                             model=self.model_name, model_name=self.model_name,
                             case_name=self.case_name, case=self.case_name,
                             case_name_nice=self.case_name_nice,
                             isSectional=str(self.isSectional),
                             from_time=self.from_time,
                             to_time=self.to_time
                             )

    """
    def load_sizedist_dataset(self, dlim_sec, nr_bins=5):

        :param dlim_sec:
        :param nr_bins:
        :return:
        s = SizedistributionSurface(self.case_name, self.from_time, self.to_time,
                                    dlim_sec, self.isSectional, self.time_resolution,
                                    space_res=self.space_resolution, nr_bins=nr_bins,
                                    model_name=self.model_name, history_field=self.history_field,
                                    locations=self.locations,
                                    chunks=self.chunks, use_pressure_coords=False)
        ds = s.get_sizedist_var()
        self.input_dataset = ds
        return ds
    """

    def load_raw_ds(self, varlist, chunks=None):
        if chunks is None:
            chunks = {}
        if self.use_pressure_coords:
            print('Loading input from converted pressure coord files:')
            ds = get_pressure_coord_fields(self.case_name,
                                           varlist,
                                           self.from_time,
                                           self.to_time,
                                           self.history_field,
                                           model=self.model_name)
            self.input_dataset = ds
            return ds
        print('Loading input dataset from raw file:')
        ds = xr_import_NorESM(self.case_name, varlist, self.from_time, self.to_time,
                              model=self.model_name, history_fld=self.history_field, comp='atm', chunks=chunks)
        ds = xr_fix(ds)

        self.input_dataset = ds
        return ds

    def set_input_datset(self, ds):
        self.input_dataset = ds

    def get_collocated_dataset(self, var_names, chunks=None, parallel=True):
        """

        :param parallel:
        :param var_names:
        :param chunks:
        :return:
        """
        # if not self.isSectional:
        #    var_names = [D_NDLOG_D_MOD]
        # else:
        #    var_names = [D_ND LOG_D_MOD, D_NDLOG_D_SEC]
        if chunks is None:
            chunks = self.chunks
        fn_list = []
        if type(var_names) is not list:
            var_names = [var_names]
        for var_name in var_names:
            fn = str(self.savepath_coll_ds(var_name))
            print(fn)
            fn_list.append(fn)
        log.ger.info('Opening: [' + ','.join(fn_list) + ']')
        print(fn_list)
        if len(fn_list) > 1:
            # print('DROPPONG ORIG NAMES')
            ds = xr.open_mfdataset(fn_list, combine='by_coords', chunks=chunks, drop_variables='orig_names',
                                   parallel=parallel)
        else:
            ds = xr.open_dataset(fn_list[0], chunks=chunks)
        return ds

    def savepath_coll_ds(self, var_name):
        #print(self.space_resolution)
        sp = str(self.savepath_root)
        st = '/%s/%s/%s/%s_%s' % (sp, self.model_name, self.case_name,
                                  var_name, self.case_name)
        st = st + '_%s_%s' % (self.from_time, self.to_time)
        st = st + '_%s_%s' % (self.time_resolution, self.space_resolution)
        fn = st + '.nc'
        make_folders(fn)
        return fn

    def collocate_dataset_vars(self, var_names, redo=False):
        ds = xr.Dataset()
        for var in var_names:
            _ds = self.collocate_dataset(var, redo=redo)
            # print(_ds)
            if type(_ds) is xr.DataArray:
                _ds = _ds.to_dataset(name=var)
            ds[var] = _ds[var]
        return ds

    def collocate_dataset(self, var_name, redo=False):
        """

        :return:
        """
        fn = self.savepath_coll_ds(var_name)
        if redo and os.path.isfile(fn):
            print(fn)
            os.remove(fn)
        elif os.path.isfile(fn):
            return xr.open_dataset(fn)
        if self.input_dataset is None:
            raise Exception('Dataset not loaded or set. Use .set_input_dataset or .load_raw_ds')
        ds = collocate_dataset(var_name, self.input_dataset, locations=self.locations)
        ds.to_netcdf(fn)
        # ds.close()
        return ds

    def check_if_load_raw_necessary(self, varlist):
        for var in varlist:
            fn = self.savepath_coll_ds(var)
            # print(fn)
            if not os.path.isfile(fn):
                print(f'File {fn} not found')
                return True
        return False

    def make_station_data_all(self):
        time1 = time.time()
        ds = self.load_raw_ds(None, chunks={'time': 24 * 7})
        time2 = time.time()
        print('TIME TO LOAD RAW DATASET IN collocate.py : %s s' % (time2 - time1))
        # get unique variables in code:
        vrs = ds.data_vars
        for var in vrs:
            print(var)
            print('making station dataset for %s' % var)
            self.make_station_da(var, ds=ds)
        return ds

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
            da = collocate_dataset(var, ds, locations=self.locations)
            ds = da.to_dataset()
            # ds = xr_fix(da.to_dataset(), model_name=self.model_name)
            # ds = xr_fix(da.to_dataset(), model_name=self.model_name)
            # da.to_netcdf(fn)
            print(fn)
            delayed_obj = ds.to_netcdf(fn, compute=False)  # , chunks={'diameter':1})
            print('Saving %s to %s' % (var, fn))

            with ProgressBar():
                delayed_obj.compute()
        return ds


def collocate_dataset(var_name, ds, locations=constants.collocate_locations):
    """
    Collocate by method 'nearest' to locations
    :param var_name:
    :param ds:
    :param locations:
    :return:
    """

    da = ds[var_name]
    if 'lat' not in da.coords:
        return da
    if 'lon' not in da.coords:
        return da
    # print(da)
    ds_tmp = xr.Dataset()
    for loc in locations:
        lat = locations[loc]['lat']
        lon = locations[loc]['lon']

        ds_tmp[loc] = da.sel(lat=lat, lon=lon, method='nearest', drop=True)
    da_out = ds_tmp.to_array(dim='location', name=var_name)
    for at in da.attrs:
        if at not in da_out.attrs:
            da_out.attrs[at] = da.attrs[at]
    del ds_tmp

    return da_out
