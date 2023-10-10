
import xarray as xr
import useful_scit.util.log as log

import ukesm_bs_fdbck.data_info.variable_info
from ukesm_bs_fdbck.util.imports.import_fields_xr_v2 import xr_import_NorESM
from ukesm_bs_fdbck import constants
from ukesm_bs_fdbck.util.imports.fix_xa_dataset_v2 import xr_fix
import os.path
from ukesm_bs_fdbck.util.imports.hybridsig2pressure import hybsig2pres_vars
from ukesm_bs_fdbck.util.filenames import get_filename_pressure_coordinate_field

from dask.diagnostics import ProgressBar



def get_pressure_coord_field(case, var, from_time, to_time, model='NorESM'):
    """
    Gets one pressure coordinate field
    :param case:
    :param var:
    :param from_time:
    :param to_time:
    :param model:
    :return:
    """
    fn = get_filename_pressure_coordinate_field(var, model, case, from_time, to_time)
    return xr.open_dataset(fn)


def get_pressure_coord_fields(case, varlist, from_time, to_time, history_field,
                              comp='atm', model='NorESM',
                              path_raw_data=constants.get_input_datapath(),
                              save_field=True):
    """
    Reads or calculates pressure coordinate fields
    :param save_field:
    :param case:
    :param varlist:
    :param from_time:
    :param to_time:
    :param history_field:
    :param comp:
    :param model:
    :param path_raw_data:
    :return:
    """
    varlist_get = list(set(varlist).union(set(ukesm_bs_fdbck.data_info.variable_info.import_always_include)))
    log.ger.debug(f'checking for pressure coord fields for {varlist_get}')
    fl, found_vars, not_found_vars = get_fl_pressure_coord_field(case, varlist_get, from_time, to_time, model=model)
    if len(fl) > 0:
        # if pres coordinate already computed
        log.ger.debug('Opening pressure coord files: ['+', '.join(fl)+']')
        ds = xr.open_mfdataset(fl, combine='by_coords')
    else:
        # make empty dataset
        log.ger.debug('no files found..')
        ds = xr.Dataset()
    # check if all files found:
    if len(set(varlist).intersection(set(not_found_vars))) != 0:

        log.ger.debug(f'Fields not found in pressure coordinates:  {comp}')
        log.ger.debug(set(varlist).intersection(set(not_found_vars)))
        ds_np = xr_import_NorESM(case, not_found_vars, from_time, to_time, path_raw_data, model=model,
                                 history_fld=history_field, comp=comp)
        log.ger.debug('Starting xr_fix:')
        ds_np = xr_fix(ds_np, model_name=model, comp=comp)
        log.ger.debug('Starting converting to hybrid sigma:')

        ds_pc = hybsig2pres_vars(ds_np,_vars=None, save_field=save_field)
        ds_out = xr.merge([ds, ds_pc])
        ds_np.close()
        ds_pc.close()
    else:
        ds_out = ds
    ds.close()
    return ds_out


def get_fl_pressure_coord_field(case, varlist, from_time, to_time, model='NorESM'):
    """
    Gets filelist for pressure coordinate fields
    :param case:
    :param varlist:
    :param from_time:
    :param to_time:
    :param model:
    :return:
    """
    fl = []
    found_vars = []
    not_found_vars = []
    for var in varlist:
        fn = get_filename_pressure_coordinate_field(var, model, case, from_time, to_time)
        if os.path.isfile(fn):
            fl.append(fn)
            found_vars.append(var)
        else:
            not_found_vars.append(var)
    return fl, found_vars, not_found_vars
