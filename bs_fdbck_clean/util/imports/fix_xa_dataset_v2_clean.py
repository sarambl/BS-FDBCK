import numpy as np
import xarray as xr

# %%




xr.set_options(keep_attrs=True)


sized_varListNorESM = \
    {'NCONC': ['NCONC01', 'NCONC02', 'NCONC04', 'NCONC05', 'NCONC06', 'NCONC07', 'NCONC08',
               'NCONC09', 'NCONC10', 'NCONC12', 'NCONC14'],
     'SIGMA': ['SIGMA01', 'SIGMA02', 'SIGMA04', 'SIGMA05', 'SIGMA06', 'SIGMA07', 'SIGMA08',
               'SIGMA09', 'SIGMA10', 'SIGMA12', 'SIGMA14'],
     'NMR': ['NMR01', 'NMR02', 'NMR04', 'NMR05', 'NMR06', 'NMR07', 'NMR08',
             'NMR09', 'NMR10', 'NMR12', 'NMR14'],
     }
sized_varlist_SOA_SEC = ['nrSOA_SEC01', 'nrSOA_SEC02', 'nrSOA_SEC03', 'nrSOA_SEC04', 'nrSOA_SEC05']

sized_varlist_SO4_SEC = ['nrSO4_SEC01', 'nrSO4_SEC02', 'nrSO4_SEC03', 'nrSO4_SEC04', 'nrSO4_SEC05']

list_sized_vars_noresm = sized_varListNorESM['NCONC'] + \
                         sized_varListNorESM['SIGMA'] + \
                         sized_varListNorESM['NMR'] + \
                         sized_varlist_SOA_SEC + \
                         sized_varlist_SO4_SEC
list_sized_vars_nonsec = sized_varListNorESM['NCONC'] + sized_varListNorESM['SIGMA'] + sized_varListNorESM['NMR']

import_always_include = ['P0', 'area', 'LANDFRAC', 'hyam', 'hybm', 'PS', 'gw', 'LOGR',
                         'hyai', 'hybi', 'ilev', 'slon', 'slat']  # , 'date',  'LANDFRAC','Press_surf',
import_constants = ['P0', 'GRIDAREA', 'landfrac', 'hyam', 'hybm', 'gw', 'LOGR',
                    'hyai', 'hybi', 'ilev', 'LANDFRAC', 'slon', 'slat']
not_pressure_coords = ['P0', 'hyam', 'hybm', 'PS', 'gw', 'LOGR', 'aps',
                       'hyai', 'hybi', 'ilev', 'date']
vars_time_not_dim = ['P0', 'area', ]

default_units = dict(
    numberconc={
        'units': '#/cm3',
        'factor': 1.e-6,
        'exceptions': ['N_AER']
    },
    NMR={
        'units': 'nm',
        'factor': 1.e9,
        'exceptions': []
    },
    mixingratio={
        'units': '$\mu$g/kg',
        'factor': 1e9,
        'exceptions': []
    },

    percent={
        'units': '%',
        'factor': 1e2,
        'exceptions': []
    }

)


def xr_fix(dtset, model_name='NorESM', comp='atm'):
    """

    :param comp:
    :param dtset:
    :param model_name:
    :return:
    """
    # print('xr_fix: Doing various fixes for %s' % model_name)
    print('xr_fix: Doing various fixes for %s' % model_name)
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

        print('NEEED TO IMPLEMENT!!!!')

    elif model_name=='UKESM':
        print('NEEED TO IMPLEMENT!!!!')
    # get weights:
    if 'lat' in dtset:
        if 'gw' in dtset.data_vars:
            dtset['lat_wg'] = dtset['gw']
        else:
            wgts_ = get_wghts_v2(dtset)
            dtset['lat_wg'] = xr.DataArray(wgts_, coords=[dtset.coords['lat']], dims=['lat'], name='lat_wg')
    if 'lon' in dtset:
        if np.min(dtset['lon'].values) >= 0:
            print('xr_fix: shifting lon to -180-->180')
            dtset.coords['lon'] = (dtset['lon'] + 180) % 360 - 180
            if not 'ncells' in dtset.dims and not 'locations' in dtset.dims:
                dtset = dtset.sortby('lon')





    return dtset


def echam_salsa_fix(ds):

    return ds

def ec_earth_fix(ds):

    return ds

def noresm_fix(dtset):
    NCONC_noresm = sized_varListNorESM['NCONC']
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
    NMR_noresm = sized_varListNorESM['NMR']
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
    fac = default_units[typ]['factor']
    un = default_units[typ]['units']
    exceptions = default_units[typ]['exceptions']
    if 'units' in dtset[var].attrs['units']:
        orig_unit = dtset[var].attrs['units']
    else:
        orig_unit = ''
    attrs_orig = dtset[var].attrs
    if orig_unit != un and var not in exceptions:
        print('converting %s unit from %s to %s with fac %s' % (var, orig_unit, un, fac))
        dtset[var] = dtset[var] * fac  # m-3 --> cm-3
        for att in attrs_orig:
            dtset[var].attrs[att] = attrs_orig[att]
        dtset[var].attrs['units'] = un




def get_wghts_v2(ds):
    """
    get latitude weights for gaussian grid.
    :param ds:
    :return:
    """
    if 'gw' in ds:
        return ds['gw'].values
    lat = ds['lat'].values
    latr = np.deg2rad(lat)  # convert to radians
    weights = np.cos(latr)  # calc weights
    return weights