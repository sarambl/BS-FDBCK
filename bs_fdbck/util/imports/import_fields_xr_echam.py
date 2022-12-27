from datetime import datetime
from os import listdir

import numpy as np
import pandas as pd
import useful_scit.util.log as log
import xarray as xr

from bs_fdbck import constants
from bs_fdbck.constants import get_locations
from bs_fdbck.data_info import get_nice_name_case

xr.set_options(keep_attrs=True)


def xr_import_ECHAM(case,
                    varlist,
                    from_time,
                    to_time, path=None,
                    model='ECHAM-SALSA',
                    locations=None,
                    chunks=None,
                    ):
    """
    Imports raw data for
    :param locations:
    :param chunks:
    :param case: case bane
    :param varlist: list of vars to import
    :param path: path to raw data base
    :param from_time: string format e.g. '2008-01' or '2008-01-01'
    :param to_time: string format e.g. '2008-01' or '2008-01-01'
    :param model: default 'NorESM'
    :return: xr.Dataset
    """
    # %%
    if path is None:
        # %%
        path = constants.get_input_datapath(model=model)
        # %%
    # depricated kind of: if wish to use alternative name
    casename_nice = get_nice_name_case(case)

    # Find files: returns a list of files to be read
    pathfile_list = filelist_ECHAM(varlist, case, path, from_time, to_time,
                                   # history_field=history_fld,
                                   # comp=comp

                                   )
    log.ger.info('Path file list:')
    log.ger.info(pathfile_list)
    # %%
    # Read the files:
    try:
        ds_out = xr.open_mfdataset(pathfile_list,
                                   chunks=chunks,
                                   # decode_times=False,
                                   # drop_variables=drop_list,
                                   # combine='nested', concat_dim='time', chunks=chunks
                                   )
    except TypeError:
        ds_out = xr.open_mfdataset(pathfile_list, decode_cf=False)
        ds_out = decode_cf_echam(ds_out)

    # %%
    # %%

    # attributes to add to dataset:
    attrs_ds = dict(raw_data_path=str(path),
                    model=model, model_name=model,
                    case_name=case, case=case,
                    case_name_nice=casename_nice,
                    from_time=from_time,
                    to_time=to_time,
                    startyear=from_time[:4],
                    endyear=to_time[:4]
                    )
    # If specified which variables to load, adds the attributes
    if varlist is not None:
        for key in ds_out.data_vars:
            ds_out[key].attrs['pressure_coords'] = 'False'
            for att in attrs_ds:
                if att not in ds_out[key].attrs:
                    ds_out[key].attrs[att] = attrs_ds[att]
    # Adds attributes to dataset:
    for key in attrs_ds:
        # print(key)
        if not (key in ds_out.attrs):
            ds_out.attrs[key] = attrs_ds[key]
    log.ger.info('Returning raw dataset from import_fields_cr_v2.py')

    # %%
    # %%
    if 'ncells' in ds_out.dims:
        ds_out['ncells'] = ds_out['ncells'].values
        # %%
        if locations is None:
            locations = get_locations()
        locationsT = locations.T

        locationsT['grid_nr_echam'] = locationsT['grid_nr_echam'].astype(int)

        # %%
        loc_tr = locationsT.reset_index().set_index('grid_nr_echam')
        # %%
        new_dim = [loc_tr.loc[i, 'index'] for i in ds_out['ncells'].values]
        ds_out['ncells'] = new_dim
        # %%
        ds_out = ds_out.rename({'ncells': 'locations'})
        # %%
    # %%
    return ds_out


def decode_cf_echam(ds_out):
    for v in ds_out.data_vars:

        un = ds_out[v].attrs['units']
        if type(un) is not str:
            ds_out[v].attrs['units'] = str(un)
            # print(ds_out[v].attrs['units'])
            # print(un)
            # print(type(un))

    ds_out = xr.decode_cf(ds_out)
    return ds_out


# %%


def filelist_ECHAM(varl, case, path, from_time, to_time):
    """
    Picks out the files to load dependent on time etc
    :param varl:
    :param case:
    :param path:
    :param from_time:
    :param to_time:
    :return:
    """
    # %%

    # %%

    path_raw_data = path / case  # path to files
    # because code is old and does a lot of stuff:

    filelist_all = [f for f in listdir(str(path_raw_data) + '/') if
                    f[0] != '.' and f[-3:] == '.nc']  # list of filenames with correct req in path folder

    _n = len(case.split('_'))
    filelist_vars = ['_'.join(f.split('_')[(_n + 1):])[:-3] for f in filelist_all]
    filelist_vars_tf = pd.Series([v in varl for v in filelist_vars])
    ds_filelist_all = pd.Series(filelist_all)
    filelist_right_var = list(ds_filelist_all[pd.Series(filelist_vars_tf)])
    filelist_right_var.sort()

    filelist_time = [f.split('_')[_n] for f in filelist_right_var]
    form = '%Y%m'

    filelist_date = [datetime.strptime(f, form) for f in filelist_time]
    try:
        from_dt = datetime.strptime(from_time, '%Y-%m-%d')
        to_dt = datetime.strptime(to_time, '%Y-%m-%d')
    except ValueError:
        from_dt = datetime.strptime(from_time, '%Y-%m')
        to_dt = datetime.strptime(to_time, '%Y-%m')

    # %%
    tf = np.array([to_dt > filelist_date[i] >= from_dt for i in np.arange(len(filelist_right_var))])
    # %%
    import_list = np.array(filelist_right_var)[tf]
    pathfile_list = [path_raw_data / imp for imp in import_list]
    # %%
    return pathfile_list
