import numpy as np
import useful_scit.util.log as log
import xarray as xr

import bs_fdbck_clean.data_info.variable_info
from bs_fdbck_clean.util.slice_average import area_mod

xr.set_options(keep_attrs=True)

def xr_fix(dtset, model_name='NorESM', comp='atm'):
    """

    :param comp:
    :param dtset:
    :param model_name:
    :return:
    """
    # print('xr_fix: Doing various fixes for %s' % model_name)
    log.ger.debug('xr_fix: Doing various fixes for %s' % model_name)
    # Rename stuff:
    # if (model_name != 'NorESM'):
    #    for key in dtset.variables:
    #        print(key)
    #        if (not sizedistribution or key not in constants.list_sized_vars_noresm):
    #            var_name_noresm = translate_var_names.model2NorESM(key, model_name)
    #
    #            if 'orig_name' not in dtset[key].attrs:
    #                dtset[key].attrs['orig_name'] = key
    #            if (len(var_name_noresm) > 0):
    #                print('Translate %s to %s ' % (key, var_name_noresm))
    #                dtset = dtset.rename({key: var_name_noresm})

    ############################
    # NorESM:
    ############################
    if model_name == 'NorESM':
        # print('So far not much to do')
        # time = dtset['time'].values  # do not cast to numpy array yet
        #
        # if isinstance(time[0], float):
        #    time_unit = dtset['time'].attrs['units']
        #    time_convert = num2date(time[:] - 15, time_unit, dtset.time.attrs['calendar'])
        #    dtset.coords['time'] = time_convert
        dtset = noresm_fix(dtset)
    elif model_name=='ECHAM-SALSA':

        dtset = echam_salsa_fix(dtset)
    elif model_name=='EC-Earth':

        print('No fixes made for EC-Earth')

    elif model_name=='UKESM':
        print('No fixes made for UKESM')
    # get weights:
    if 'lat' in dtset:
        if 'gw' in dtset.data_vars:
            dtset['lat_wg'] = dtset['gw']
        else:
            wgts_ = area_mod.get_wghts_v2(dtset)
            dtset['lat_wg'] = xr.DataArray(wgts_, coords=[dtset.coords['lat']], dims=['lat'], name='lat_wg')
    if 'lon' in dtset:
        if np.min(dtset['lon'].values) >= 0:
            log.ger.debug('xr_fix: shifting lon to -180-->180')
            dtset.coords['lon'] = (dtset['lon'] + 180) % 360 - 180
            if not 'ncells' in dtset.dims and not 'locations' in dtset.dims:
                dtset = dtset.sortby('lon')





    return dtset


def echam_salsa_fix(ds):

    return ds

def ec_earth_fix(ds):

    return ds

def noresm_fix(dtset):
    NCONC_noresm = bs_fdbck_clean.data_info.variable_info.sized_varListNorESM['NCONC']
    for nconc in NCONC_noresm:
        typ = 'numberconc'
        if nconc in dtset:
            # if (dtset[nconc].attrs['units'] = '#/m3'):
            _ch_unit(dtset, typ, nconc)
        nr = nconc[-2:]
        nact = f'NACT{nr}'
        if nact in dtset:
            print(nact)
            _ch_unit(dtset, typ, nact)
    NMR_noresm = bs_fdbck_clean.data_info.variable_info.sized_varListNorESM['NMR']
    for nmr in NMR_noresm:
        typ = 'NMR'
        if nmr in dtset:
            if dtset[nmr].attrs['units'] == 'm':
                _ch_unit(dtset, typ, nmr)
    if 'NNAT_0' in dtset.data_vars:
        dtset['SIGMA00'] = dtset['NNAT_0'] * 0 + 1.6  # Kirkevag et al 2018
        dtset['SIGMA00'].attrs['units'] = '-'  # Kirkevag et al 2018
        dtset['NMR00'] = dtset['NNAT_0'] * 0 + 62.6  # nm Kirkevag et al 2018
        dtset['NMR00'].attrs['units'] = 'nm'  # nm Kirkevag et al 2018
        dtset['NCONC00'] = dtset['NNAT_0']
    for cvar in ['AWNC']:
        if cvar in dtset:
            if dtset[cvar].units == 'm-3':
                dtset[cvar].values = 1.e-6 * dtset[cvar].values
                dtset[cvar].attrs['units'] = '#/cm^3'
    for cvar in ['ACTNI', 'ACTNL']:
        if cvar in dtset:
            if dtset[cvar].units != '#/cm^3':
                dtset[cvar].values = 1.e-6 * dtset[cvar].values
                dtset[cvar].attrs['units'] = '#/cm^3'
    for svar in ['Smax_w', 'Smax']:
        if svar in dtset.data_vars:
            _ch_unit(dtset, 'percent', svar)
    for mod in range(1, 15):
        fvar = 'NACT_FRAC%02.0f' % mod
        if fvar in dtset.data_vars:
            _ch_unit(dtset, 'percent', fvar)
    # while cont:
    for i in range(10):

        typ = 'numberconc'
        varSEC = 'nrSO4_SEC%02.0f' % i
        if varSEC in dtset.data_vars:
            _ch_unit(dtset, typ, varSEC)
    for i in range(10):
        varSEC = 'nrSOA_SEC%02.0f' % i
        typ = 'numberconc'
        if varSEC in dtset.data_vars:
            _ch_unit(dtset, typ, varSEC)
    # for mm_var in ['SOA_NA','SO4_NA','SOA_A1','SO4_A1']:
    #    typ='mixingratio'
    #    if mm_var in dtset.data_vars:
    #        _ch_unit(dtset,typ,mm_var)
    for sec_var in ['N_secmod', 'nrSO4_SEC_tot', 'nrSOA_SEC_tot', 'nrSEC_tot'] + ['nrSEC%02.0f' % ii for ii in
                                                                                  range(1, 6)]:
        typ = 'numberconc'
        if sec_var in dtset:
            if dtset[sec_var].attrs['units'] == 'unit':
                _ch_unit(dtset, typ, sec_var)
    for ii in np.arange(1, 6):
        typ = 'numberconc'
        sec_nr = 'nrSOA_SEC%02.0f' % ii
        if sec_nr in dtset:
            if dtset[sec_nr].attrs['units'] == 'unit':
                _ch_unit(dtset, typ, sec_nr)

        sec_nr = 'nrSO4_SEC%02.0f' % ii
        if sec_nr in dtset:
            typ = 'numberconc'
            if dtset[sec_nr].attrs['units'] == 'unit':
                _ch_unit(dtset, typ, sec_nr)
                # dtset[sec_nr].values = dtset[sec_nr].values * 1e-6
                # dtset[sec_nr].attrs['units'] = 'cm-3'
    return dtset


def _ch_unit(dtset, typ, var):
    fac = bs_fdbck_clean.data_info.variable_info.default_units[typ]['factor']
    un = bs_fdbck_clean.data_info.variable_info.default_units[typ]['units']
    exceptions = bs_fdbck_clean.data_info.variable_info.default_units[typ]['exceptions']
    if 'units' in dtset[var].attrs['units']:
        orig_unit = dtset[var].attrs['units']
    else:
        orig_unit = ''
    attrs_orig = dtset[var].attrs
    if orig_unit != un and var not in exceptions:
        log.ger.debug('converting %s unit from %s to %s with fac %s' % (var, orig_unit, un, fac))
        dtset[var] = dtset[var] * fac  # m-3 --> cm-3
        for att in attrs_orig:
            dtset[var].attrs[att]=attrs_orig[att]
        dtset[var].attrs['units'] = un
