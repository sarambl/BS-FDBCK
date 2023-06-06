from datetime import datetime
import pandas as pd
import useful_scit.util.log as log
import xarray as xr
from bs_fdbck_clean import constants

xr.set_options(keep_attrs=True)


# %%

def xr_import_EC_earth(
        case,
        from_time,
        to_time,
        which='TM5',
        path=None,
        model='EC-Earth',
        chunks=None,
):
    """
    Imports raw data for
    :param which:
    :param chunks:
    :param case: case bane
    :param path: path to raw database
    :param from_time: string format e.g. '2008-01' or '2008-01-01'
    :param to_time: string format e.g. '2008-01' or '2008-01-01'
    :param model: default 'NorESM'
    :return: xr.Dataset
    """
    # %%
    # varlist = ['N_NUS','var130','var176']
    # from_time='2012-01'
    # to_time = '2012-02'
    # path=None
    # model='EC-Earth'
    # locations = None

    # %%
    if path is None:
        path = constants.get_input_datapath(model=model)
    # %%
    fl_TM5 = list(path.glob('*TM5*.nc'))
    fl_IFS_T = list(path.glob('T_IFS*.nc'))
    fl_IFS_GG = list(path.glob('IFS_GG*.nc'))
    fl_TM5.sort()
    fl_IFS_T.sort()
    fl_IFS_GG.sort()
    fl_TM5 = get_file_subset_EC_Earth(fl_TM5, from_time, to_time)
    fl_IFS_T = get_file_subset_EC_Earth(fl_IFS_T, from_time, to_time)
    fl_IFS_GG = get_file_subset_EC_Earth(fl_IFS_GG, from_time, to_time)
    if which == 'TM5':
        fl = fl_TM5
    elif which == 'IFS_T':
        fl = fl_IFS_T
    else:
        fl = fl_IFS_GG

    # %%
    # Find files: returns a list of files to be read
    log.ger.info('Path file list:')
    log.ger.info(fl)
    # %%
    # Read the files:
    ds_out = xr.open_mfdataset(fl, chunks=chunks, engine='netcdf4')
    # attributes to add to dataset:
    ds_out = add_various_extra_info(case, ds_out, from_time, model, path, to_time, )

    return ds_out


def xr_import_EC_earth_both(
        case,
        from_time,
        to_time,
        # which='TM5',
        path=None,
):
    ds_out_tm4 = xr_import_EC_earth(case,
                                    from_time,
                                    to_time,
                                    which='TM5',
                                    path=path,
                                    model='EC-Earth')
    ds_out_ifs = xr_import_EC_earth(case,
                                    from_time,
                                    to_time,
                                    which='IFS',
                                    path=path,
                                    model='EC-Earth')
    dic_out = {'IFS': ds_out_ifs,
               'TM5': ds_out_tm4
               }
    return dic_out


def add_various_extra_info(case, ds_out, from_time, model, path, to_time):
    attrs_ds = dict(raw_data_path=str(path),
                    model=model, model_name=model,
                    case_name=case, case=case,
                    # case_name_nice=casename_nice,
                    from_time=from_time,
                    to_time=to_time,
                    startyear=from_time[:4],
                    endyear=to_time[:4]
                    )
    # If specified which variables to load, adds the attributes
    # Adds attributes to dataset:
    for key in attrs_ds:
        # print(key)
        if not (key in ds_out.attrs):
            ds_out.attrs[key] = attrs_ds[key]

    return ds_out


# %%
def get_file_subset_EC_Earth(fl, from_time, to_time):
    df_fl = pd.DataFrame(fl, columns=['path'])
    df_fl['filename'] = df_fl['path'].apply(lambda x: x.stem)
    df_fl['time_str'] = df_fl['filename'].apply(lambda x: x.split('+')[-1])
    df_fl['time'] = df_fl['time_str'].apply(lambda x: datetime.strptime(x, '%Y%m'))

    if len(from_time.split('-')) > 2:

        form = '%Y-%m-%d'
    else:
        form = '%Y-%m'
    print(form)
    print(from_time)
    print(to_time)
    ft = datetime.strptime(from_time, form)
    tt = datetime.strptime(to_time, form)

    times = (df_fl['time'] >= ft) & (df_fl['time'] < tt)
    df_fl_want = df_fl[times]
    return list(df_fl_want['path'])


# %%


# %%
