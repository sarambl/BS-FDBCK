import os

import numpy as np
import xarray as xr


def make_folders(path):
    """
    Takes path and creates to folders
    :param path: Path you want to create (if not already existant)
    :return: nothing
    """
    path = extract_path_from_filepath(path)
    split_path = path.split('/')
    if path[0] == '/':

        path_inc = '/'
    else:
        path_inc = ''
    for ii in np.arange(len(split_path)):
        # if ii==0: path_inc=path_inc+split_path[ii]
        path_inc = path_inc + split_path[ii]
        if not os.path.exists(path_inc):
            os.makedirs(path_inc)
        path_inc = path_inc + '/'

    return


def append2dic(self, ds_append, ds_add):
    for key in ds_add.attrs.keys():
        if key not in ds_append.attrs:
            ds_append.attrs[key] = ds_add.attrs[key]
    return ds_append


def extract_path_from_filepath(file_path):
    """
    ex: 'folder/to/file.txt' returns 'folder/to/'
    :param file_path:
    :return:
    """

    st_ind = file_path.rfind('/')
    foldern = file_path[0:st_ind] + '/'
    return foldern


def save_dataset_to_netcdf(dtset, filepath):
    """

    :param dtset:
    :param filepath:
    :return:
    """
    dummy = dtset.copy()

    # dummy.time.encoding['calendar']='standard'
    go_through_list = list(dummy.coords)
    if isinstance(dummy, xr.Dataset):
        go_through_list = go_through_list + list(dummy.data_vars)
    for key in go_through_list:
        if 'Pres_addj' in dummy[key].attrs:
            if dummy[key].attrs['Pres_addj']:
                dummy[key].attrs['Pres_addj'] = 'True'
            else:
                dummy[key].attrs['Pres_addj'] = 'False'

    if 'Pres_addj' in dummy.attrs:
        if dummy.attrs['Pres_addj']:
            dummy.attrs['Pres_addj'] = 'True'
        else:
            dummy.attrs['Pres_addj'] = 'False'
    if 'time' in dummy.coords:
        if 'units' in dummy['time'].attrs:
            del dummy['time'].attrs['units']
        if 'calendar' in dummy['time'].attrs:
            del dummy['time'].attrs['calendar']
    print('Saving dataset to: ' + filepath)
    make_folders(filepath)
    dummy.load()
    dummy.to_netcdf(filepath, mode='w')  # ,encoding={'time':{'units':'days since 2000-01-01 00:00:00'}})
    del dummy
    return


# def get_varn_eusaar_comp(varn):


# print(check_dummy['time'].values)
def boolean_2_string(b):
    if type(b) is bool:
        if b:
            return 'True'
        else:
            return 'False'
    else:
        return b


def get_foldername_Nd(caseName, from_year, model_name, pressure_adjust, to_year, from_diam, to_diam):
    filename = get_filename_Nd(caseName, from_year, model_name, pressure_adjust, to_year, from_diam, to_diam)
    st_ind = filename.rfind('/')
    foldern = filename[0:st_ind]
    return foldern


def get_filename_Nd(caseName, from_year, model_name, pressure_adjust, to_year, from_diam, to_diam):
    if pressure_adjust:
        filen = dataset_path_Nd + '/' + model_name + '/%s_%s_%s_%s_dmin%d_maxd%d_press_adj.nc' % (
            model_name, caseName, from_year, to_year, from_diam, to_diam)
    else:
        filen = dataset_path_Nd + '/' + model_name + '/%s_%s_%s_%s_dmin%d_maxd%d.nc' % (
            model_name, caseName, from_year, to_year, from_diam, to_diam)
    return filen


def get_filename_Nd_from_varName(varName, caseName, from_year, model_name, pressure_adjust, to_year):
    """
    Get filename from varName for N_d variable
    :param varName:
    :param caseName:
    :param from_year:
    :param model_name:
    :param pressure_adjust:
    :param to_year:
    :return:
    """
    n_split = varName.split('_')
    if len(n_split) == 3:
        from_diam = int(n_split[0][1:])
        to_diam = int(n_split[-1])
    else:
        from_diam = 0
        to_diam = int(n_split[-1])
    filen = get_filename_Nd(caseName, from_year, model_name, pressure_adjust, to_year, from_diam, to_diam)
    return filen


dataset_path = 'Data/Avg_sizedist_datasets'
dataset_path_Nd = 'Data/Datasets_Nd'
