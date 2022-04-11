import os
import pathlib

import xarray as xr
from dask.diagnostics import ProgressBar

from bs_fdbck import constants
from bs_fdbck.util.filenames import get_filename_ng_field, get_filename_pressure_coordinate_field
from bs_fdbck.util.imports.fix_xa_dataset_v2 import xr_fix
from bs_fdbck.util.imports.get_pressure_coord_fields import get_pressure_coord_fields
from bs_fdbck.util.imports.import_fields_xr_v2 import xr_import_NorESM, import_constants
import pandas as pd

# %%


def get_field_fixed(case, varlist, from_time, to_time, raw_data_path=constants.get_input_datapath(),
                    pressure_adjust=True, model = 'NorESM', history_fld='.h0.', comp='atm', chunks=None,
                    get_constants=True):
    """
    Imports and fixes:
    :param history_fld:
    :param comp:
    :param chunks:
    :param get_constants:
    :param model:
    :param case:
    :param varlist:
    :param from_time:
    :param to_time:
    :param raw_data_path:
    :param pressure_adjust:
    :return:
    """

    # If pressure coordinate --> check if in pressure coordinates, else get_pressure_coordinate etc
    # IF LEV NOT DIM! JUST LOAD AND AVERAGE
    # If not pressure coordinates --> check outpaths['original_coords']=path_outdata + '/computed_fields_ng'
    #raw_data_path=constants.get_input_datapath()
    #pressure_adjust=True; model = 'NorESM'; history_fld='.h0.'; comp='atm'; chunks=None
    if type(from_time) is int: from_time='%s'%from_time
    if type(to_time) is int: to_time='%s'%to_time
    if len(to_time)==4:
        to_time = to_time+'-12'
    if len(from_time)==4: # if only year, add month
        from_time=from_time+'-01'


    if pressure_adjust:
        #log.ger.debug('Calling get_pressure_coord_fields...')
        ds = get_pressure_coord_fields(case,
                                       varlist,
                                       from_time,
                                       to_time,
                                       history_fld,
                                       comp=comp,
                                       model=model)
        if get_constants:
            _ds = import_constants(case)
            ds = xr.merge([ds, _ds])
        return ds
    else:
        ds = get_fields_hybsig(case, comp, varlist, from_time, to_time, history_fld, model, raw_data_path, chunks,
                               )
        return ds


def get_fields_hybsig(case, comp, varlist, from_time, to_time, history_fld='.h0.', model='NorESM', raw_data_path=constants.get_input_datapath(),
                      chunks=None, save_field=True):

    fl = []
    if varlist is not None:
        fl = []
        vl_lacking = []
        for var in varlist:
            fn = get_filename_ng_field(var, model, case, from_time, to_time)
            if os.path.isfile(fn):
                fl.append(fn)
            else:
                vl_lacking.append(var)
    else:
        vl_lacking = varlist
    # print(vl_lacking)
    ds = xr_import_NorESM(case, vl_lacking, from_time, to_time, path=raw_data_path,
                          model=model,
                          history_fld=history_fld,
                          comp=comp, chunks=chunks)
    if vl_lacking is None:
        vl_lacking = ds.data_vars
    ds = xr_fix(ds, model_name=model)
    for v in vl_lacking:
        da = ds[v]
        da['pressure_coords'] = 'False'
        if save_field:

            ds_save = da.to_dataset()
            for k in ds.attrs.keys():
                if k not in ds_save.attrs:
                    ds_save.attrs[k] = ds.attrs[k]
            fn = get_filename_ng_field(v, model, case, from_time, to_time)
            #make_folders(fn)
            fn = pathlib.Path(fn)
            fn.parent.mkdir(parents=True, exist_ok=True)
            ds_save.to_netcdf(fn)
            # save_pressure_coordinate_field(ds_save, var)
    if len(fl) > 0:
        print([f.stem for f in fl])
        print(fl[0].parent)
        ds_f_file = xr.open_mfdataset(fl, combine='by_coords')
        ds = xr.merge([ds, ds_f_file])
    return ds


def merge_monthly_flds(case, varlist, from_time, to_time, raw_data_path=constants.get_input_datapath(),
                       pressure_adjust=True, model = 'NorESM', history_fld='.h0.', comp='atm', chunks=None,
                       get_constants=True):



    _dr = pd.date_range(from_time, to_time, freq='MS')[:-1]
    ls_ds = []
    for d in _dr:
        ft = d.strftime(format='%Y-%m-%d')
        tt = (d + pd.DateOffset(months=1)).strftime(format='%Y-%m-%d')
        _ds = get_field_fixed(case, varlist, ft, tt, raw_data_path=raw_data_path,
                        pressure_adjust=pressure_adjust, model = model, history_fld=history_fld,
                        comp=comp, chunks=chunks,
                        get_constants=get_constants)
        ls_ds.append(_ds)
        #self.collocate_month(ft, tt, varlist)
        # self.collocate_month(from, varlist, year)


    ds_conc = concatinate_months(ls_ds)

    for var in varlist:
        if pressure_adjust:
            fn = get_filename_pressure_coordinate_field(var, model, case, from_time, to_time)
        else:
            fn = get_filename_ng_field(var, model, case, from_time, to_time)


            delayed_obj = ds_conc.to_netcdf(fn, compute=False)  # , chunks={'diameter':1})
            print('Saving %s to %s' % (var, fn))

            with ProgressBar():
                delayed_obj.compute()
        return



def concatinate_months(ls_ds):
    #ls = []
    ds_conc = xr.concat(ls_ds, 'time')
    # remove duplicates in time:
    ds_conc: xr.Dataset
    ds_conc = ds_conc.sel(time=~ds_conc.indexes['time'].duplicated())
    return ds_conc

