import os

import xarray as xr
from dask.diagnostics import ProgressBar
from useful_scit.util import log

from bs_fdbck.constants import get_locations
from bs_fdbck.util.collocate.collocate import CollocateModel, collocate_dataset
from bs_fdbck.util.imports.fix_xa_dataset_v2 import xr_fix
from bs_fdbck.util.imports.import_fields_xr_echam import xr_import_ECHAM

log.ger.setLevel(log.log.INFO)


from IPython import get_ipython

# noinspection PyBroadException
try:
    _ipython = get_ipython()
    _magic = _ipython.magic
    _magic('load_ext autoreload')
    _magic('autoreload 2')
except:
    pass

# %%

# %%


# %%
class CollocateModelEcham(CollocateModel):
    """
    collocate a model to a list of locations
    """
    def __init__(self,*_vars, **kwargs):
        print(_vars, kwargs)
        kwargs['model_name'] = 'ECHAM-SALSA'
        super().__init__(*_vars,**kwargs)

    def load_raw_ds(self, varlist, chunks=None):
        if chunks is None:
            chunks = {}
        #if self.use_pressure_coords:
        #    print('Loading input from converted pressure coord files:')
        #    ds = get_pressure_coord_fields(self.case_name,
        #                                   varlist,
        #                                   self.from_time,
        #                                   self.to_time,
        #                                   self.history_field,
        #                                   model=self.model_name)
        #    self.input_dataset = ds
        #    return ds

        print('Loading input dataset from raw file:')
        ds = xr_import_ECHAM(self.case_name,
                             varlist,
                             self.from_time, self.to_time,
                             model=self.model_name,
                             history_fld=self.history_field,
                             comp=None,
                             chunks=chunks
                             )
        ds = xr_fix(ds, model_name=self.model_name)

        self.input_dataset = ds
        return ds

    def set_input_datset(self, ds):
        self.input_dataset = ds






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
        ds = collocate_dataset_echam(var_name, self.input_dataset, locations=self.locations)
        ds.to_netcdf(fn)
        # ds.close()
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
            da = collocate_dataset_echam(var, ds, locations=self.locations)
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
    def get_new_instance(self, *_vars, **kwargs):
        return CollocateModelEcham(*_vars, **kwargs)
# %%

def collocate_dataset_echam(var_name, ds, model_name='ECHAM-SALSA', locations = None

                            ):
    if locations is None:
        locations = get_locations(model_name)
    #var_name = 'SO2_gas'

    if 'ncells' not in ds[var_name].dims:
        print('ncells not found')
        if 'location' in ds[var_name].dims:
            # already converted
            print('Already converted')
            return ds[var_name]

        return collocate_dataset(var_name, ds, locations=locations)
        #da = collocate_dataset(var_name, ds, locations=locations)
    else:
        print(locations)
        #da = ds[var_name]
        locations.loc['grid_nr_echam'] = locations.loc['grid_nr_echam'].astype(int)
        print(locations)
        locations = locations.sort_values('grid_nr_echam', axis=1)
        locations.columns

        if 'locations' not in ds.coords:
            # %%
            co = xr.DataArray(list(locations.columns), dims='ncells', name='locations')
            # %%
            ds = ds.assign_coords({'locations' : co})
            # %%
        ds = ds.swap_dims({'ncells':'locations'})

        return ds[var_name]
# %%


    # %%

    # %%
