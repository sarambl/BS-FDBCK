import os
from datetime import datetime
from os import listdir
import numpy as np
import xarray as xr
import useful_scit.util.log as log
from useful_scit.util.make_folders import make_folders

import ukesm_bs_fdbck.data_info.variable_info
from ukesm_bs_fdbck.data_info import get_nice_name_case
from ukesm_bs_fdbck.data_info.variable_info import import_constants as import_constants_list
from ukesm_bs_fdbck.util.filenames import get_filename_constants
from ukesm_bs_fdbck import constants
from ukesm_bs_fdbck.util.imports.fix_xa_dataset_v2 import xr_fix
from ukesm_bs_fdbck.constants import latlon_path

xr.set_options(keep_attrs=True)


def xr_import_NorESM(case, varlist, from_time, to_time, path=constants.get_input_datapath(),
                     model='NorESM', history_fld='.h0.', comp='atm', chunks=None):
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

    # depricated kind of: if wish to use alternative name
    casename_nice = get_nice_name_case(  case)

    # Find files: returns a list of files to be read
    pathfile_list = filelist_NorESM(case, path, from_time, to_time, history_field=history_fld,
                                    comp=comp)
    log.ger.info('Path file list:')
    log.ger.info(pathfile_list)
    # Find modified variable list and get drop_list for open_mfdataset
    # I.e. some variables require the sum/ratio etc of other variables.
    if varlist is not None:
        # if varlist is None, imports everything
        var_names_mod = get_vars_for_computed_vars(varlist, model)
        # Open first file to get a list of variables to drop from your dataset:
        dummy = xr.open_dataset(pathfile_list[0])
        drop_list = list((set(dummy.data_vars.keys()) - set(var_names_mod)) - set(
            ukesm_bs_fdbck.data_info.variable_info.import_always_include))
        dummy.close()
    else:
        drop_list = None
    # Read the files:
    ds_out = xr.open_mfdataset(pathfile_list, decode_times=False, drop_variables=drop_list,
                               combine='nested', concat_dim='time', chunks=chunks)
    # Find time resolution of input:
    # (decides if monthly or hourly)
    time_resolution = _get_time_resolution(ds_out)
    # Decodes the times to cf time
    ds_out = decode_NorESM_time(ds_out, time_resolution)
    # If combined variables in varList, created combined, if not returns unchanged
    ds_out = create_computed_fields(ds_out, varlist, model)
    # varList = varNames_mod

    # attributes to add to dataset:
    attrs_ds = dict(raw_data_path=str(path),
                    model=model, model_name=model,
                    case_name=case, case=case,
                    case_name_nice=casename_nice,
                    isSectional='False',
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
    #ds_out['lat'] = np.around(ds_out['lat'].values, decimals=2)
    if comp=='lnd':
        ds_lat = xr.open_dataset(latlon_path)
        ds_out['lat'] = ds_lat['lat'].values
        ds_out['lon'] = ds_lat['lon'].values
    return ds_out


def _get_time_resolution(ds_out):
    """
    Checks the time resolution (monthly/hourly etc)
    :param ds_out:
    :return:
    """
    _diff = ds_out['time'].isel(time=1) - ds_out['time'].isel(time=0)
    if _diff.values > 1:
        time_resolution = 'month'
    elif _diff.values < 1:
        time_resolution = 'hour'
    else:
        log.ger.warning('_get_time_resolution: NOT SAFE for resolution day -- implement properly')
        time_resolution = 'day'
    return time_resolution


def decode_NorESM_time(ds, time_resolution):
    """
    Decode the NorESM time
    :param ds:
    :param time_resolution:
    :return:
    """
    # Quick and dirty fix for the fact that the output is dated at the last day of the month
    # making it vulnerable to choice of calendar...

    if time_resolution == 'month':
        with xr.set_options(keep_attrs=True):
            ds['time'] = ds['time'] - 15

    ds = xr.decode_cf(ds)

    ds = ds.sortby('time')
    return ds


def filelist_NorESM(case, path, from_time, to_time, history_field='.h0.', comp='atm'):
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
    if comp == 'lnd':
        model_lab = 'clm2'
    else:
        model_lab = 'cam'
    path_raw_data = path / case  / comp /'hist'  # path to files
    # because code is old and does a lot of stuff:
    path_raw_data = str(path_raw_data) + '/'
    filelist_d = [f for f in listdir(path_raw_data) if
                  ((history_field in f) and ((f[0] != '.')) and ('.nc_tmp' not in f))]  # list of filenames with correct req in path folder
    filelist_d.sort()
    _ln = len(case) + len(model_lab) + 5
    filelist_time = [
        f[_ln:-3] for f in filelist_d]
    _test = filelist_time[0]
    if len(_test.split('-')) > 2:
        filelist_time = ['-'.join(f.split('-')[:3]) for f in filelist_time]
        form = '%Y-%m-%d'
        # time_resolution='not_month'
    else:
        form = '%Y-%m'
    filelist_date = [datetime.strptime(f, form) for f in filelist_time]
    try:
        from_dt = datetime.strptime(from_time, '%Y-%m-%d')
        to_dt = datetime.strptime(to_time, '%Y-%m-%d')
    except ValueError:
        from_dt = datetime.strptime(from_time, '%Y-%m')
        to_dt = datetime.strptime(to_time, '%Y-%m')

    tf = np.array([to_dt >= filelist_date[i] >= from_dt for i in np.arange(len(filelist_d))])
    import_list = np.array(filelist_d)[tf]
    pathfile_list = [path_raw_data + imp for imp in import_list]
    return pathfile_list


# %%

def import_constants(case, include=import_constants_list,
                     path=constants.get_input_datapath(),
                     model='NorESM', reload=False):
    """
    Imports only constant values.
    :param reload:
    :param case:
    :param include:
    :param path:
    :param model:
    :return:
    """
    fn = get_filename_constants(case, model)
    if os.path.isfile(fn) and not reload:

        ds = xr.open_dataset(fn)
        print(f'found contants file {fn}')
        return ds
    from_time = '0001-01'
    to_time = '2100-02'
    fl = filelist_NorESM(case, path, from_time, to_time)
    file = fl[0]
    ds = xr.open_dataset(file)
    drop_l = list(set(ds.data_vars) - set(include)) + ['PS']
    ds = ds.drop_vars(drop_l)  # [include]

    ds = xr_fix(ds)
    ds = ds.isel(time=0)
    ds.squeeze()
    make_folders(fn)
    #ds['lat'] = np.around(ds['lat'].values, decimals=2)
    ds.to_netcdf(fn)

    return ds


def get_vars_for_computed_vars(varNames, model):
    """
    Terribly formatted function for adding needed fields to varlist for my self defined variables.
    :param varNames:
    :param model:
    :return:
    """
    varnames_mod = []
    if model == 'NorESM':
        for var in varNames:
            if 'SW_rest_Ghan' == var:
                varnames_mod = varnames_mod + ['FSNTCDRF']
            if 'LW_rest_Ghan' == var:
                varnames_mod = varnames_mod + ['FLNTCDRF']
            if 'DIR_Ghan' == var:
                varnames_mod = varnames_mod + ['FSNT_DRF', 'FSNT', 'FLNT_DRF', 'FLNT']
            if 'SWDIR_Ghan' == var:
                varnames_mod = varnames_mod + ['FSNT_DRF', 'FSNT']
            if 'LWDIR_Ghan' == var:
                varnames_mod = varnames_mod + ['FLNT_DRF', 'FLNT']
            if 'SWCF_Ghan' == var:
                varnames_mod = varnames_mod + ['FSNT_DRF', 'FSNTCDRF']
            elif var == 'LWCF_Ghan':
                varnames_mod = varnames_mod + ['FLNT_DRF', 'FLNTCDRF']
            elif var == 'NCFT_Ghan':
                varnames_mod = varnames_mod + ['FSNT_DRF', 'FSNTCDRF'] + ['FLNT_DRF', 'FLNTCDRF']
            elif var == 'condTend_SOA_total':
                varnames_mod = varnames_mod + ['SOA_NAcondTend', 'SOA_A1condTend']
            elif var == 'cb_SOA_dry':
                varnames_mod = varnames_mod + ['cb_SOA_NA', 'cb_SOA_A1']
            if var == 'AREL_incld':
                varnames_mod = varnames_mod + ['AREL', 'FREQL']
            if var == 'Smax_w':
                varnames_mod = varnames_mod + ['Smax', 'Smax_supZero']
            if var == 'AWNC_incld':
                varnames_mod = varnames_mod + ['AWNC', 'FREQL']
            if var == 'ACTNL_incld':
                varnames_mod = varnames_mod + ['ACTNL', 'FCTL']
            if var == 'ACTREL_incld':
                varnames_mod = varnames_mod + ['ACTREL', 'FCTL']
            if var in ['NACT%02.0f'%i for i in range(1,15)]:
                nr = var[-2:]
                varnames_mod = varnames_mod + [f'NCONC{nr}', f'NACT_FRAC{nr}']
            if var in ['SEC0%s' % (1 + ii) for ii in np.arange(5)]:
                nr = var[-2:]
                varnames_mod = varnames_mod + ['SO4_SEC%s' % nr, 'SOA_SEC%s' % nr]
            if var in ['nrSEC0%s' % (1 + ii) for ii in np.arange(5)]:
                nr = var[-2:]
                varnames_mod = varnames_mod + ['nrSO4_SEC%s' % nr, 'nrSOA_SEC%s' % nr]
            if var == 'nrSOA_SEC_tot':
                varnames_mod = varnames_mod + ['nrSOA_SEC0%s' % (1 + ii) for ii in np.arange(5)]
            if var == 'nrSO4_SEC_tot':
                varnames_mod = varnames_mod + ['nrSO4_SEC0%s' % (1 + ii) for ii in np.arange(5)]
            if var == 'nrSEC_tot':
                varnames_mod = varnames_mod + ['nrSOA_SEC0%s' % (1 + ii) for ii in np.arange(5)] + [
                    'nrSO4_SEC0%s' % (1 + ii) for ii in np.arange(5)]
            if var == 'N_secmod':
                varl_SEC = get_vars_for_computed_vars(['nrSEC_tot'], model)
                varnames_mod = varnames_mod + varl_SEC + ['N_AER']
            if var == 'SOA_SEC_tot':
                varnames_mod = varnames_mod + ['SOA_SEC0%s' % (1 + ii) for ii in np.arange(5)]
            if var == 'SO4_SEC_tot':
                varnames_mod = varnames_mod + ['SO4_SEC0%s' % (1 + ii) for ii in np.arange(5)]
            if var == 'SEC_tot':
                varnames_mod = varnames_mod + ['SOA_SEC0%s' % (1 + ii) for ii in np.arange(5)] + [
                    'SO4_SEC0%s' % (1 + ii) for ii in np.arange(5)]
            if var == 'cb_SEC_tot':
                varnames_mod = varnames_mod + ['cb_SOA_SEC0%s' % (1 + ii) for ii in np.arange(5)] + [
                    'cb_SO4_SEC0%s' % (1 + ii) for ii in np.arange(5)]
            if var == 'cb_SOA_SEC_tot':
                varnames_mod = varnames_mod + ['cb_SOA_SEC0%s' % (1 + ii) for ii in
                                               np.arange(5)]  # + ['cb_SO4_SEC0%s'%(1+ii) for ii in np.arange(5)]
            if var == 'cb_SO4_SEC_tot':
                varnames_mod = varnames_mod + ['cb_SO4_SEC0%s' % (1 + ii) for ii in np.arange(5)]
            if var == 'cb_NA':
                varnames_mod = varnames_mod + ['cb_SO4_NA', 'cb_SOA_NA']

            lss_exts = ['DDF', 'SFWET', 'coagTend', 'clcoagTend']
            if (var[-9:] == '_totLossR') or (var[-9:] == '_lifetime'):
                # %%
                trac_n = var[:-9]
                addvars = [f'{trac_n}{lt}' for lt in lss_exts] + [f'cb_{trac_n}']
                varnames_mod = varnames_mod + addvars

            else:
                varnames_mod.append(var)

    elif model == 'EC-Earth':
        for var in varNames:
            if var == 'condTend_SOA_total':
                varnames_mod = varnames_mod + ['p_svoc2D', 'p_elvoc2D']
            else:
                varnames_mod.append(var)
    elif model == 'ECHAM':
        varnames_mod = varNames

    return varnames_mod


def create_computed_fields(xr_ds, varNames, model):
    """
    Terribly formatted function for computing fields in varlist for my self defined variables.
    :param xr_ds:
    :param varNames:
    :param model:
    :return:
    """

    if varNames is None:
        return xr_ds
    if model == 'NorESM':
        for var in varNames:
            if 'SW_rest_Ghan' == var:
                xr_ds[var] = xr_ds['FSNTCDRF']
            if 'LW_rest_Ghan' == var:
                xr_ds[var] = xr_ds['FLNTCDRF']
            if ('SWDIR_Ghan' == var) or ('DIR_Ghan' == var):
                xr_ds['SWDIR_Ghan'] = xr_ds['FSNT'] - xr_ds['FSNT_DRF']
                xr_ds['SWDIR_Ghan'].attrs['units'] = xr_ds['FSNT_DRF'].attrs['units']
            if ('LWDIR_Ghan' == var) or ('DIR_Ghan' == var):
                xr_ds['LWDIR_Ghan'] = -(xr_ds['FLNT'] - xr_ds['FLNT_DRF'])
                xr_ds['LWDIR_Ghan'].attrs['units'] = xr_ds['FLNT_DRF'].attrs['units']
            if 'DIR_Ghan' == var:
                xr_ds['DIR_Ghan'] = xr_ds['LWDIR_Ghan'] + xr_ds['SWDIR_Ghan']
                xr_ds['DIR_Ghan'].attrs['units'] = xr_ds['LWDIR_Ghan'].attrs['units']
            if 'SWCF_Ghan' == var:
                xr_ds['SWCF_Ghan'] = xr_ds['FSNT_DRF'] - xr_ds['FSNTCDRF']
                xr_ds[var].attrs['units'] = xr_ds['FSNT_DRF'].attrs['units']
            if 'LWCF_Ghan' == var:
                xr_ds[var] = -(xr_ds['FLNT_DRF'] - xr_ds['FLNTCDRF'])
                xr_ds[var].attrs['units'] = xr_ds['FLNT_DRF'].attrs['units']
            elif var == 'NCFT_Ghan':
                xr_ds[var] = xr_ds['FSNT_DRF'] - xr_ds['FSNTCDRF'] - (xr_ds['FLNT_DRF'] - xr_ds['FLNTCDRF'])
                xr_ds[var].attrs['units'] = xr_ds['FLNT_DRF'].attrs['units']
            elif var == 'condTend_SOA_total':
                xr_ds[var] = xr_ds['SOA_NAcondTend'] + xr_ds['SOA_A1condTend']
                xr_ds[var].attrs['units'] = xr_ds['SOA_NAcondTend'].attrs['units']
            elif var == 'cb_SOA_dry':
                xr_ds[var] = xr_ds['cb_SOA_NA'] + xr_ds['cb_SOA_A1']
                xr_ds[var].attrs['units'] = xr_ds['cb_SOA_NA'].attrs['units']

            if var == 'AREL_incld':
                xr_ds[var] = xr_ds['AREL'] / xr_ds['FREQL']
                xr_ds[var] = xr_ds[var].where((xr_ds['FREQL'] != 0))
                xr_ds[var].attrs['units'] = xr_ds['AREL'].attrs['units']
            if var == 'AWNC_incld':
                xr_ds[var] = xr_ds['AWNC'] / xr_ds['FREQL'] * 1.e-6
                xr_ds[var] = xr_ds[var].where((xr_ds['FREQL'] != 0))
                xr_ds[var].attrs['units'] = '#/cm$^3$'
            if var == 'ACTNL_incld':
                xr_ds[var] = xr_ds['ACTNL'] / xr_ds['FCTL'] * 1.e-6
                xr_ds[var] = xr_ds[var].where((xr_ds['FCTL'] != 0))
                xr_ds[var].attrs['units'] = '#/cm$^3$'
            if var == 'ACTREL_incld':
                xr_ds[var] = xr_ds['ACTREL'] / xr_ds['FCTL']  # *1.e-6
                xr_ds[var] = xr_ds[var].where((xr_ds['FCTL'] != 0))
                xr_ds[var].attrs['units'] = xr_ds['ACTREL'].attrs['units']

            if var in ['SEC0%s' % (1 + ii) for ii in np.arange(5)]:
                nr = var[-2:]
                xr_ds[var] = xr_ds['SO4_SEC%s' % nr] + xr_ds['SOA_SEC%s' % nr]
                xr_ds[var].attrs['units'] = xr_ds['SO4_SEC%s' % nr].attrs['units']
            if var in ['nrSEC0%s' % (1 + ii) for ii in np.arange(5)]:
                nr = var[-2:]
                xr_ds[var] = xr_ds['nrSO4_SEC%s' % nr] + xr_ds['nrSOA_SEC%s' % nr]
                xr_ds[var].attrs['units'] = xr_ds['nrSO4_SEC%s' % nr].attrs['units']
            if var == 'nrSOA_SEC_tot':
                xr_ds[var] = xr_ds['nrSOA_SEC01'].copy()
                for ii in np.arange(2, 6):
                    xr_ds[var] = xr_ds[var] + xr_ds[('nrSOA_SEC0%s' % ii)]
                xr_ds[var].attrs['units'] = xr_ds['nrSOA_SEC01'].attrs['units']
            if var == 'SOA_SEC_tot':
                xr_ds[var] = xr_ds['SOA_SEC01'].copy()
                for ii in np.arange(2, 6):
                    xr_ds[var] = xr_ds[var] + xr_ds[('SOA_SEC0%s' % ii)]
                xr_ds[var].attrs['units'] = xr_ds['SOA_SEC01'].attrs['units']
            if var == 'nrSO4_SEC_tot':
                xr_ds[var] = xr_ds['nrSO4_SEC01'].copy()
                for ii in np.arange(2, 6):
                    xr_ds[var] = xr_ds[var] + xr_ds['nrSO4_SEC0%s' % ii]
                xr_ds[var].attrs['units'] = xr_ds['nrSO4_SEC01'].attrs['units']
            if var == 'SO4_SEC_tot':
                xr_ds[var] = xr_ds['SO4_SEC01'].copy()
                for ii in np.arange(2, 6):
                    xr_ds[var] = xr_ds[var] + xr_ds['SO4_SEC0%s' % ii]
                xr_ds[var].attrs['units'] = xr_ds['SO4_SEC01'].attrs['units']
            if var == 'nrSEC_tot':
                xr_ds[var] = xr_ds['nrSO4_SEC01'].copy() + xr_ds['nrSOA_SEC01']
                for ii in np.arange(2, 6):
                    xr_ds[var] = xr_ds[var] + xr_ds['nrSO4_SEC0%s' % ii] + xr_ds['nrSOA_SEC0%s' % ii]
                xr_ds[var].attrs['units'] = xr_ds['nrSO4_SEC01'].attrs['units']
            if var == 'SEC_tot':
                xr_ds[var] = xr_ds['SO4_SEC01'].copy() + xr_ds['SOA_SEC01']
                for ii in np.arange(2, 6):
                    xr_ds[var] = xr_ds[var] + xr_ds['SO4_SEC0%s' % ii] + xr_ds['SOA_SEC0%s' % ii]
                xr_ds[var].attrs['units'] = xr_ds['SO4_SEC01'].attrs['units']
            if var == 'cb_SEC_tot':
                xr_ds[var] = xr_ds['cb_SO4_SEC01'].copy() + xr_ds['cb_SOA_SEC01']
                for ii in np.arange(2, 6):
                    xr_ds[var] = xr_ds[var] + xr_ds['cb_SO4_SEC0%s' % ii] + xr_ds['cb_SOA_SEC0%s' % ii]
                xr_ds[var].attrs['units'] = xr_ds['cb_SO4_SEC01'].attrs['units']
            if var == 'cb_SO4_SEC_tot':
                xr_ds[var] = xr_ds['cb_SO4_SEC01'].copy()  # +xr_ds['cb_SOA_SEC01']
                for ii in np.arange(2, 6):
                    xr_ds[var] = xr_ds[var] + xr_ds['cb_SO4_SEC0%s' % ii]  # + xr_ds['cb_SOA_SEC0%s'%ii]
                xr_ds[var].attrs['units'] = xr_ds['cb_SO4_SEC01'].attrs['units']
            if var == 'cb_SOA_SEC_tot':
                xr_ds[var] = xr_ds['cb_SOA_SEC01']
                for ii in np.arange(2, 6):
                    xr_ds[var] = xr_ds[var] + xr_ds['cb_SOA_SEC0%s' % ii]
                xr_ds[var].attrs['units'] = xr_ds['cb_SOA_SEC01'].attrs['units']
            if var == 'cb_NA':
                xr_ds[var] = xr_ds['cb_SOA_NA'] + xr_ds['cb_SO4_NA']
                xr_ds[var].attrs['units'] = xr_ds['cb_SOA_NA'].attrs['units']
            if var == 'N_secmod':
                sec_tot = create_computed_fields(xr_ds, ['nrSEC_tot'], model)['nrSEC_tot']
                xr_ds[var] = xr_ds['N_AER'] * 1e6 + sec_tot  # ups, N_AER in #/cm3, sec in #/m3
                xr_ds[var].attrs['units'] = xr_ds['nrSEC_tot'].attrs['units']
            if var in ['NACT%02.0f'%i for i in range(1,15)]:
                nr = var[-2:]
                ncon = f'NCONC{nr}'
                fr = f'NACT_FRAC{nr}'
                xr_ds[var] = xr_ds[ncon]*xr_ds[fr]
                xr_ds[var].attrs['units'] = xr_ds[ncon].attrs['units']
            lss_exts = ['DDF', 'SFWET', 'coagTend', 'clcoagTend']
            if (var[-9:] == '_totLossR') or (var[-9:] == '_lifetime'):
                # or (var[-9:] =='_lifetime'):
                trac_n = var[:-9]
                addvars = [f'{trac_n}{lt}' for lt in lss_exts]  # + [f'cb_{trac_n}']
                tot_lossR = _comp_total_lossR(addvars, trac_n, xr_ds)
                if var[-9:] == '_lifetime':
                    xr_ds[var] = xr_ds[f'cb_{trac_n}'] / xr_ds[tot_lossR]

    if model == 'EC-Earth':
        for var in varNames:
            if 'condTend_SOA_total' == var:
                xr_ds[var] = xr_ds['p_svoc2D'] + xr_ds['p_elvoc2D']
                xr_ds[var].attrs['units'] = xr_ds['p_svoc2D'].attrs['units']  # +xr_ds['p_elvoc2D']
    return xr_ds


def _comp_total_lossR(addvars, trac_n, xr_ds):
    tot_lossR = f'{trac_n}_totLossR'
    xr_ds[tot_lossR] = xr_ds[addvars[0]].copy()
    for ext in addvars[1:]:
        xr_ds[tot_lossR] = xr_ds[tot_lossR] + xr_ds[ext]
    return tot_lossR
