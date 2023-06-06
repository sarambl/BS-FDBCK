from datetime import datetime
from os import listdir

import dask
import pandas as pd
import useful_scit.util.log as log
import xarray as xr

from bs_fdbck_clean import constants
from bs_fdbck_clean.util.BSOA_datamanip.ukesm import ukesm_var_overview, get_rndic_ukesm

xr.set_options(keep_attrs=True)

default_varl_ukesm = [
    'tas',
    'sfmmroa',
]

"""
    case = 'AEROCOMTRAJ'
    from_time = '2012-01'
    to_time = '2012-02'
    varlist = [
        'Mass_Conc_OM_NS',
         'mmrtr_OM_NS',
         'nconcKI',
                'ddryCS',
                'Temp',
        # 'SFisoprene',
                ]
    path=None
    model='UKESM'
    chunks=None

    locations=None
"""


# %%


def xr_import_ukesm(case,
                    from_time,
                    to_time,
                    varlist,
                    path=None,
                    model='UKESM',
                    # comp='atm',
                    chunks=None,
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
    :return: xr.Dataset
    """

    # %%
    if path is None:
        path = constants.get_input_datapath(model=model)

    path_raw_data = path / case  # path to files
    from_time_dt = pd.to_datetime(from_time)
    to_time_dt = pd.to_datetime(to_time)
    pathfile_list, rename_dic,dic_varname2file = get_pathlist_and_rndic_ukesm(from_time_dt, path_raw_data, to_time_dt, varlist)

    # Find files: returns a list of files to be read
    log.ger.info('Path file list:')
    log.ger.info(pathfile_list)
    # Read the files:
    try:
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            ds_out = xr.open_mfdataset(pathfile_list,
                                       chunks=chunks,

                                       )
    except TypeError:
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):

            ds_out = xr.open_mfdataset(pathfile_list, decode_cf=False)
    ds_out = ds_out.rename(rename_dic)
    # %%
    # %%

    # attributes to add to dataset:
    attrs_ds = dict(raw_data_path=str(path),
                    model=model, model_name=model,
                    case_name=case, case=case,
                    case_name_nice=case,
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
    log.ger.info('Returning raw dataset from import_fields_xr_ukesm.py')
    # %%

    return ds_out


def get_pathlist_and_rndic_ukesm(from_time_dt, path_raw_data, to_time_dt, varlist):
    # %%
    #rename_dic = {}
    #dic_varname2file = {}
    pathfile_list = []
    rename_dic, dic_varname2file = get_rndic_ukesm(varlist)

    for v in varlist:
        if v in ukesm_var_overview.index:
            df_out = filelist_ukesm(v, from_time_dt, to_time_dt, path_raw_data)

            fl_open = list(df_out['path'])
            pathfile_list += fl_open
            #file_name_var = ukesm_var_overview.loc[v, 'var_name_infile']
            #var_in_filename = ukesm_var_overview.loc[v, 'orig_var_name_file']
            #new_var_name = v
            #rename_dic[file_name_var] = new_var_name
            #dic_varname2file[new_var_name] =var_in_filename
            print(v)
    # %%
    return pathfile_list, rename_dic, dic_varname2file

# %%

# %%



def filelist_ukesm(var, from_time_dt, to_time_dt, path_raw_data):
    file_name_var = ukesm_var_overview.loc[var, 'orig_var_name_file']
    relm = ukesm_var_overview.loc[var, 'relm']
    if type(relm) is not str:
        p = path_raw_data

        fl = list(p.glob(f'*{file_name_var}*.nc'))
        df_fl = pd.DataFrame(fl, columns=['path'])
        df_fl['filename'] = df_fl['path'].apply(lambda x: x.name)
        df_fl = df_fl.sort_index(ascending=True)
        df_out = df_fl.copy()
        return df_out

    p = path_raw_data / relm
    fl = list(p.glob(f'*{file_name_var}*.nc'))
    df_fl = pd.DataFrame(fl, columns=['path'])
    df_fl['filename'] = df_fl['path'].apply(lambda x: x.name)
    df_fl['yearmonth'] = df_fl['filename'].apply(lambda x: x.split('_')[-3])
    df_fl['year'] = df_fl['yearmonth'].apply(lambda x: x[0:3])
    df_fl['month'] = df_fl['yearmonth'].apply(lambda x: x[4:])
    df_fl['dt_time'] = df_fl['yearmonth'].apply(lambda x: pd.to_datetime(x, format='%Y%m'))
    df_fl = df_fl.sort_index(ascending=True)
    df_out = df_fl[(df_fl['dt_time'] >= from_time_dt) & (df_fl['dt_time'] <= to_time_dt)]
    return df_out


# %%

def old_filelist_ukesm(varl, case, path, from_time, to_time, prefer_resolution='hourly'):
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

    if varl is None:
        varl = default_varl_ukesm

    path_raw_data = path / case  # path to files
    filelist_all = []
    # because code is old and does a lot of stuff:
    for v in varl:

        f_hourly = [path_raw_data / 'hourly' / f for f in listdir(str(path_raw_data) + '/hourly/') if
                    (f[0] != '.' and f[-3:] == '.nc') and f.split('_')[
                        0] == v]  # list of filenames with correct req in path folder

        f_6hourly = [path_raw_data / '6hourly' / f for f in listdir(str(path_raw_data) + '/6hourly/') if
                     (f[0] != '.' and f[-3:] == '.nc') and f.split('_')[
                         0] == v]  # list of filenames with correct req in path folder

        f_daily = [path_raw_data / 'daily' / f for f in listdir(str(path_raw_data) + '/daily/') if
                   (f[0] != '.' and f[-3:] == '.nc') and f.split('_')[
                       0] == v]  # list of filenames with correct req in path folder
        if prefer_resolution == 'hourly':
            priority_list = [f_hourly, f_6hourly, f_daily]
        else:
            priority_list = [f_daily, f_6hourly, f_hourly]
        f_add = []
        for fl in priority_list[::-1]:
            # go in reverse order through priorities and use the last one that has files:
            if len(fl) > 0:
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
    ds_filelist_all['time_mask_1'] = [(to_dt >= tt) & (tf >= from_dt) for tt, tf in
                                      zip(ds_filelist_all['to_time'], ds_filelist_all['from_time'])]
    ds_filelist_all['time_mask_add1'] = [(to_dt <= tt) & (to_dt >= tf) for tt, tf in
                                         zip(ds_filelist_all['to_time'], ds_filelist_all['from_time'])]
    ds_filelist_all['time_mask_add2'] = [(from_dt >= tf) & (from_dt <= tt) for tt, tf in
                                         zip(ds_filelist_all['to_time'], ds_filelist_all['from_time'])]
    ds_filelist_all['time_mask_add'] = ds_filelist_all['time_mask_add1'] | ds_filelist_all['time_mask_add2']
    ds_filelist_all['time_mask'] = ds_filelist_all['time_mask_1'] | ds_filelist_all['time_mask_add']
    # ds_filelist_all[['from_time','to_time','time_mask', 'time_mask_add']]
    ds_filelist = ds_filelist_all[ds_filelist_all['time_mask']]

    import_list = list(ds_filelist['file_path'])
    # %%
    return import_list
