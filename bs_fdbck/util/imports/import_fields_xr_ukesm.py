import os
from datetime import datetime
from os import listdir

import dask
import numpy as np
import pandas as pd
import xarray as xr
import useful_scit.util.log as log

import bs_fdbck.data_info.variable_info
from bs_fdbck.data_info import get_nice_name_case
from bs_fdbck.data_info.variable_info import import_constants as import_constants_list
from bs_fdbck.util.filenames import get_filename_constants
from bs_fdbck import constants
from bs_fdbck.util.imports.fix_xa_dataset_v2 import xr_fix
from bs_fdbck.constants import latlon_path, get_locations
from bs_fdbck.util.imports.import_fields_xr_v2 import get_vars_for_computed_vars

xr.set_options(keep_attrs=True)
from pathlib import Path

default_varl_ukesm = [
    'tas',
    'sfmmroa',
]


# %%
# %%

def xr_import_ukesm(case,
                    varlist,
                    from_time,
                    to_time, path=None,
                    model='UKESM',
                    # case= 'CRES',
                    history_fld='.h0.',
                    comp='atm',
                    chunks=None,
                    locations=None,
                    ):
    """
    Imports raw data for
    :param chunks:
    :param case: case bane
    :param varlist: list of vars to import
    :param path: path to raw data base
    :param from_time: string format e.g. '2008-01' or '2008-01-01'
    :param to_time: string format e.g. '2008-01' or '2008-01-01'
    :param model: default 'NorESM'
    :param history_fld: e.g '.h1.'
    :param comp: e.g 'atm', 'lnd' etc
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
    pathfile_list = filelist_ukesm(varlist, case, path, from_time, to_time,
                                   # history_field=history_fld,
                                   # comp=comp

                                   )
    log.ger.info('Path file list:')
    log.ger.info(pathfile_list)
    # %%
    drop_list = None
    # Read the files:
    try:
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            ds_out = xr.open_mfdataset(pathfile_list,
                                   # decode_times=False,
                                   # drop_variables=drop_list,
                                   # combine='nested', concat_dim='time', chunks=chunks
                                   )
    except TypeError:
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):

            ds_out = xr.open_mfdataset(pathfile_list, decode_cf=False)
        # ds_out = decode_cf_echam(ds_out)

    # %%
    # %%
    # Find time resolution of input:
    # (decides if monthly or hourly)
    # time_resolution = _get_time_resolution(ds_out)
    # Decodes the times to cf time
    # ds_out = decode_NorESM_time(ds_out, time_resolution)
    # If combined variables in varList, created combined, if not returns unchanged
    # ds_out = create_computed_fields(ds_out, varlist, model)
    # varList = varNames_mod

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
    # Adds attributes to dataset:
    for key in attrs_ds:
        # print(key)
        if not (key in ds_out.attrs):
            ds_out.attrs[key] = attrs_ds[key]
    log.ger.info('Returning raw dataset from import_fields_cr_v2.py')




    return ds_out


def filelist_ukesm(varl, case, path, from_time, to_time, prefer_resolution='hourly'):
    """
    Picks out the files to load dependent on time etc
    :param case:
    :param path:
    :param from_time:
    :param to_time:
    :param history_field:
    :param comp:
    :return:
    """
    # %%

    if varl is None:
        varl = default_varl_ukesm

    path_raw_data = path / case  # path to files
    filelist_all = []
    # because code is old and does a lot of stuff:
    for v in varl:

        f_hourly = [path_raw_data/'hourly'/f for f in listdir(str(path_raw_data) + '/hourly/') if
                    (f[0] != '.' and f[-3:] == '.nc') and f.split('_')[0]==v]  # list of filenames with correct req in path folder

        f_6hourly =[path_raw_data/'6hourly'/f for f in listdir(str(path_raw_data) + '/6hourly/') if
                    (f[0] != '.' and f[-3:] == '.nc') and f.split('_')[0]==v]  # list of filenames with correct req in path folder

        f_daily = [path_raw_data/'daily'/f for f in listdir(str(path_raw_data) + '/daily/') if
                  (f[0] != '.' and f[-3:] == '.nc') and f.split('_')[0]==v]  # list of filenames with correct req in path folder
        if prefer_resolution =='hourly':
            priority_list = [f_hourly, f_6hourly, f_daily]
        else:
            priority_list = [f_daily, f_6hourly, f_hourly]
        f_add =[]
        for fl in priority_list[::-1]:
            # go in reverse order through priorities and use the last one that has files:
            if len(fl)>0:
                f_add = fl
        filelist_all = filelist_all + f_add


    # %%

    ds_filelist_all = pd.DataFrame(filelist_all, columns=['file_path'])

    ds_filelist_all['file_stem'] = ds_filelist_all['file_path'].apply(lambda x: x.stem)
    # %%
    ds_filelist_all['time_str'] = ds_filelist_all['file_stem'].apply(lambda x: x.split('_')[-1])
    # trim away hour from format
    ds_filelist_all['from_time_str'] = ds_filelist_all['time_str'].apply(lambda x: x.split('-')[0][:-4])
    ds_filelist_all['to_time_str'] = ds_filelist_all['time_str'].apply(lambda x: x.split('-')[1][:-4])

    form = '%Y%m%d'
    # %%
    ds_filelist_all.head()
    ds_filelist_all['from_time'] = ds_filelist_all['from_time_str'].apply(lambda x: datetime.strptime(x, form))
    ds_filelist_all['to_time'] = ds_filelist_all['to_time_str'].apply(lambda x: datetime.strptime(x, form))

    ds_filelist_all['to_year'] = ds_filelist_all['to_time'].dt.year
    ds_filelist_all['from_year'] = ds_filelist_all['from_time'].dt.year
    # %%
    ds_filelist_all = ds_filelist_all.sort_values('from_time')



    try:
        from_dt = datetime.strptime(from_time, '%Y-%m-%d')
        to_dt = datetime.strptime(to_time, '%Y-%m-%d')
    except ValueError:
        from_dt = datetime.strptime(from_time, '%Y-%m')
        to_dt = datetime.strptime(to_time, '%Y-%m')

    # %%
    ds_filelist_all['time_mask_1']= [(to_dt >= tt) & (tf >= from_dt) for tt, tf in zip(ds_filelist_all['to_time'],ds_filelist_all['from_time'])]
    ds_filelist_all['time_mask_add1']= [(to_dt <= tt) & (to_dt >= tf) for tt, tf in zip(ds_filelist_all['to_time'],ds_filelist_all['from_time'])]
    ds_filelist_all['time_mask_add2']= [(from_dt >= tf) & (from_dt <= tt) for tt, tf in zip(ds_filelist_all['to_time'],ds_filelist_all['from_time'])]
    ds_filelist_all['time_mask_add'] = ds_filelist_all['time_mask_add1']  |ds_filelist_all['time_mask_add2']
    ds_filelist_all['time_mask'] = ds_filelist_all['time_mask_1'] | ds_filelist_all['time_mask_add']
    # ds_filelist_all[['from_time','to_time','time_mask', 'time_mask_add']]
    ds_filelist = ds_filelist_all[ds_filelist_all['time_mask']]



    import_list = list(ds_filelist['file_path'])
    import_list
    # %%
    return import_list
