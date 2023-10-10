import datetime

import numpy as np
import pandas as pd
import xarray as xr

from ukesm_bs_fdbck.constants import path_data_info
from ukesm_bs_fdbck.util.BSOA_datamanip import calculate_daily_median_summer, calculate_summer_median, mask4summer

sec_in_timestep = 60 * 60 * 3
fn_data_info_ec_earth = path_data_info / 'ec_earth_info' / 'ec_earth_var_overview.csv'

df_varinfo = pd.read_csv(fn_data_info_ec_earth, index_col=0)

kg2ug = 1e9

R = 287.058
pressure_default = 100000.  # Pa
temperature_default = 273.15
standard_air_density = pressure_default / (R * temperature_default)

rad_vars = ['RDRY_NUS',
            'RDRY_AIS',
            'RDRY_ACS',
            'RDRY_COS',
            'RWET_AII',
            'RWET_ACI',
            'RWET_COI',
            ]
num_vars = ['N_NUS',
            'N_AIS',
            'N_ACS',
            'N_COS',
            'N_AII',
            'N_ACI',
            'N_COI',
            ]
diam_vars = [f'D{v[1:]}' for v in rad_vars]

modes_dic = {
    'NUS': dict(NUM='N_NUS', RAD='RDRY_NUS', DIAM='DDRY_NUS', SIG=1.59),
    'AIS': dict(NUM='N_AIS', RAD='RDRY_AIS', DIAM='DDRY_AIS', SIG=1.59),
    'ACS': dict(NUM='N_ACS', RAD='RDRY_ACS', DIAM='DDRY_ACS', SIG=1.59),
    'COS': dict(NUM='N_COS', RAD='RDRY_COS', DIAM='DDRY_COS', SIG=2.),
    'AII': dict(NUM='N_AII', RAD='RWET_AII', DIAM='DWET_AII', SIG=1.59),
    'ACI': dict(NUM='N_ACI', RAD='RWET_ACI', DIAM='DWET_ACI', SIG=1.59),
    'COI': dict(NUM='N_COI', RAD='RWET_COI', DIAM='DWET_COI', SIG=2.),
}

rn_dic_ec_earth_cloud = {

}

rn_dict_ec_earth = {
    # 'ORG_mass_conc': 'OA',
    'temp': 'T',

}

OA_components = [
    'M_SOANUS',
    'M_POMAIS',
    'M_SOAAIS',
    'M_POMACS',
    'M_SOAACS',
    'M_POMCOS',
    'M_SOACOS',
    'M_POMAII',
    'M_SOAAII',
]
SOA_components = [
    'M_SOANUS',
    'M_SOAAIS',
    'M_SOAACS',
    #    'M_SOACOS',
    'M_SOAAII'
]
POM_components = [
    'M_POMAIS',
    'M_POMACS',
    #    'M_POMCOS',
    'M_POMAII'
]

cwp_kg_to_g = [
    'tcw',
    'tcwv',
    'tclw',
    'tciw',
    'CWP',

]

# TODO: Add variables to cloud list and station list.
# TODO: add necessary functions to read data.
# TODO: make dataset for station data EC-Earth
# TODO: compute N50,N100 and N200


varlist_tm5 = [
    'CCN0.20',
    'CCN1.00',
    'M_SO4NUS',
    'M_SOANUS',
    'M_BCAIS',
    'M_POMAIS',
    'M_SOAAIS',
    'M_SO4ACS',
    'M_BCACS',
    'M_POMACS',
    'M_SSACS',
    'M_DUACS',
    'M_SOAACS',
    'M_SO4COS',
    'M_BCCOS',
    'M_POMCOS',
    'M_SSCOS',
    'M_DUCOS',
    'M_SOACOS',
    'M_BCAII',
    'M_POMAII',
    'M_SOAAII',
    'M_DUACI',
    'M_DUCOI',
    'N_NUS',
    'N_AIS',
    'N_ACS',
    'N_COS',
    'N_AII',
    'N_ACI',
    'N_COI',
    'GAS_O3',
    'GAS_SO2',
    'GAS_TERP',
    'GAS_OH',
    'GAS_ISOP',
    'RWET_NUS',
    'RWET_AIS',
    'RWET_ACS',
    'RWET_COS',
    'RWET_AII',
    'RWET_ACI',
    'RWET_COI',
    'RDRY_NUS',
    'RDRY_AIS',
    'RDRY_ACS',
    'RDRY_COS',
    'loadoa',
    'od550aer',
    'od550oa',
    'od550soa',
    'od440aer',
    'od870aer',
    'od350aer',
    'loadsoa',
    'emiterp',
    'emiisop'
]
varlist_ifs_gg = [
    'var68',
    'var69',
    'var70',
    'var71',
    'var72',
    'var73',
    'var74',
    'var75',
    'var176',
    'var177',
    'var178',
    'var179',
    'var208',
    'var209',
    'var210',
    'var211',
    'var136',
    'var137',
    'var78',
    'var79',
    'var164',
    'var20',
    'var130',
    'var131',
    'var132',
    'var167',
    'var248',
    'var54',
]
varlist_ifs_t = [
    'var130',
]

varlist_dic = {
    'TM5': varlist_tm5,
    'IFS_T': varlist_ifs_t,
    'IFS_GG': varlist_ifs_gg
}

from scipy.stats import lognorm


def ds2df_ec_earth(ds_st, model_lev_i=-1,
                   take_daily_median=True,
                   summer_months=None,
                   mask_summer=False,
                   pressure=None,
                   temperature=None,
                   air_density=None
                   ):
    # N50, N100 etc:

    ds_st = change_units_and_compute_vars_ec_earth(ds_st,
                                                   # air_density=air_density,
                                                   # pressure = pressure,
                                                   # temperature=temperature

                                                   )

    # ifs_vars = varlist_dic['IFS_T'] + varlist_dic['IFS_GG']
    # ifs_vars = list(set(ifs_vars).intersection(set(ds_st.data_vars)))
    # ds_ifs = ds_st[ifs_vars]
    # ds_tm5 = ds_st.drop(ifs_vars)

    if 'lev' in ds_st.dims:
        model_lev_i_tm5 = -model_lev_i - 1
        ds_st_ilev = ds_st.isel(lev=model_lev_i_tm5)
    elif 'lev_ifs' in ds_st.dims:
        ds_st_ilev = ds_st.isel(lev=model_lev_i)

    else:
        ds_st_ilev = ds_st

    if take_daily_median:
        df = calculate_daily_median_summer(ds_st_ilev, summer_months=summer_months)
        df_sm = calculate_summer_median(df)
    else:
        if mask_summer:
            _ds = mask4summer(ds_st_ilev, months=summer_months)
        else:
            _ds = ds_st_ilev
        _ds['is_JJA'] = _ds['time.month'].isin([6, 7, 8])
        _ds['is_JA'] = _ds['time.month'].isin([7, 8])
        _ds['isSummer'] = _ds['time.month'].isin([7, 8])
        df = _ds.to_dataframe()
        df_sm = None

    return df, df_sm


def fix_timestamp_ec_earth(ds):
    if float(ds['time.minute'].isel(time=0)) == 0:
        return ds
    ti = ds['time']
    timedelta = pd.Timedelta(30, 'minutes')
    ti_new = ti - timedelta
    ds = ds.rename({'time': 'time_orig'})
    ds = ds.assign_coords(time=ti_new.rename({'time': 'time_orig'}))
    ds = ds.swap_dims({'time_orig': 'time'})
    return ds


def change_units_and_compute_vars_ec_earth(ds_st,  # air_density = None,
                                           # pressure = pressure_default,
                                           # temperature = temperature_default
                                           ):
    """

    :param ds_st:
    :return:
    """
    # if air_density is None:
    #    if pressure is None:
    #        pressure = pressure_default
    #    if temperature is None:
    #        temperature = temperature_default
    #    air_density =  pressure / (R * temperature)

    ds_st = fix_units_ec_earth(ds_st)
    ds_st = add_Nx_to_dataset(ds_st, x_list=None, add_to_500=True)
    ds_st = compute_OAs(ds_st)

    rn_sub = {k: rn_dict_ec_earth[k] for k in rn_dict_ec_earth if
              ((k in ds_st.data_vars) & (rn_dict_ec_earth[k] not in ds_st.data_vars))}
    ds_st = ds_st.rename(rn_sub)
    if 'T' in ds_st:
        ds_st['T_C'] = ds_st['T'] - 273.15
    return ds_st


def extract_cloud_top(ds):
    if 'lev_orig' not in ds.coords:
        ds = make_dummy_lev(ds)
    ds = ds.sortby('lev', ascending=False)
    ds['cloud_time_norm'] = ds['liq_cloud_time'] / sec_in_timestep

    ds['cumsum'] = (ds['cl_frac_where_cltime_pos']
                    .fillna(0)
                    .where(ds['cl_time_liq_norm'] > 0)
                    .cumsum('lev')
                    .where(ds['cl_frac_where_cltime_pos'] > 0.2)
                    .where(ds['cloud_time_norm'] == 1.)
                    )  #
    # ds['cumsum'] = ds['cumsum'].where(np.abs(ds['re_liq']-4) > 0.01)

    ds['argmax'] = ds['cumsum'].fillna(0).argmax('lev', )

    #ds['re_liq_incld_cltop'] = ds['re_liq_incld'].isel(lev=ds['argmax']).where(ds['argmax'] > 0)
    ds['cdnc_incld_cltop'] = (ds['cdnc_incld']
                              .where(ds['cl_frac_where_cltime_pos'] > 0)
                              .where(ds['liq_cloud_time'] > 0)
                              .isel(lev=ds['argmax'])
                              .where(ds['argmax'] > 0)
                              )

    ds['re_liq_cltop'] = (ds['re_liq']
                          .where(ds['cl_frac_where_cltime_pos'] > 0)
                          .where(ds['liq_cloud_time'] > 0)
                          .isel(lev=ds['argmax'])
                          .where(ds['argmax'] > 0)
                          )
    ds['cc_cltop'] = (ds['cc_all']
                      .where(ds['cl_frac_where_cltime_pos'] > 0)
                      .where(ds['liq_cloud_time'] > 0)
                      .isel(lev=ds['argmax'])
                      .where(ds['argmax'] > 0)
                      )

    ds = ds.sortby('lev', ascending=True)
    ds['argmax'] = -ds['argmax'] - 1

    return ds


def calculate_incld_values_warmclouds(ds):
    # 3 hourly data:
    sec_in_timestep = (3 * 60 * 60)
    re_ex = ('re_liq' in ds.data_vars)
    liq_cloud_time_ex = ('liq_cloud_time' in ds.data_vars)
    cdnc_incld_ex = ('cdnc' in ds.data_vars)
    tciw_ex = ('tciw' in ds.data_vars)
    tclw_ex = ('tclw' in ds.data_vars)
    ttc_ex = ('ttc' in ds.data_vars)
    cc_ex = ('cc' in ds.data_vars)
    if (re_ex and liq_cloud_time_ex):
        print('calculating re_liq_incld')
        ds['re_liq_incld'] = ((ds['re_liq'] / ds['liq_cloud_time'])
                              .where(ds['liq_cloud_time'] > 0)
                              .where(ds['re_liq'] > 0)
                              * sec_in_timestep
                              )
    if (cdnc_incld_ex and liq_cloud_time_ex):
        print('calculating cdnc_incld')
        ds['cdnc_incld'] = ((ds['cdnc'] / ds['liq_cloud_time'])
                            .where(ds['liq_cloud_time'] > 0)
                            .where(ds['cdnc'] > 0)
                            # *(60*60*3)
                            )
    if liq_cloud_time_ex:
        print('calculating cl_time_liq_norm')
        ds['cl_time_liq_norm'] = ds['liq_cloud_time'].where(ds['liq_cloud_time'] > 0) / 10800
    if 'cc' in ds.data_vars:
        print('Calulating cc_all')
        ds['cc_all'] = ds['cc'].where(ds['cc'] > 0)
    if tciw_ex and tclw_ex:
        print('Calulating liq_frac_cwp')

        ds['liq_frac_cwp'] = ds['tclw'] / (ds['tclw'] + ds['tciw'])
    if cc_ex and liq_cloud_time_ex:
        print('calculating cl_frac_where_cltime_post')
        ds['cl_frac_where_cltime_pos'] = ds['cc_all'].where(ds['cl_time_liq_norm'] > 0)
    if ttc_ex and tclw_ex:
        print('calculating cwp_incld')
        ds['cwp_incld'] = ds['tclw'] / ds['ttc']
    return ds


def make_dummy_lev(ds):
    if 'lev_orig' not in ds.coords:
        ds = ds.rename({'lev': 'lev_orig'})
        # ds = ds.rename(lev=
        ds['lev'] = (1e-2 * (ds['hyam'] + ds['hybm'] * 1e5)).swap_dims({'nhym': 'lev_orig'})
        ds = ds.swap_dims({'lev_orig': 'lev'})
    return ds


def add_diameter_mode(ds):
    for r in rad_vars:
        if r in ds:
            d = f'D{r[1:]}'
            ds[d] = ds[r] * 2
            ds[d].attrs['units'] = ds[r].attrs['units']
            ln_orig = ds[r].attrs['long_name']
            ds[d].attrs['long_name'] = ln_orig[:3] + f'diameter of {r[5:]}'
    return ds


def fix_units_ec_earth(ds):
    for v in num_vars:
        if v in ds:
            if ds[v].attrs['units'] == '1 m-3':
                print(f'Converting {v} from m-3 to cm-3')
                with xr.set_options(keep_attrs=True):
                    ds[v] = ds[v] * 1e-6
                ds[v].attrs['units'] = 'cm-3'
    for v in rad_vars:
        if v in ds:
            if ds[v].attrs['units'] == 'm':
                print(f'Converting {v} from m to nm')
                with xr.set_options(keep_attrs=True):
                    ds[v] = ds[v] * 1e9
                ds[v].attrs['units'] = 'nm'
    for v in diam_vars:
        if v in ds:
            if ds[v].attrs['units'] == 'm':
                print(f'Converting {v} from m to nm')
                with xr.set_options(keep_attrs=True):
                    ds[v] = ds[v] * 1e9
                ds[v].attrs['units'] = 'nm'

    ds = add_diameter_mode(ds)

    for v in OA_components + ['OA', 'POM', 'SOA']:
        if v in ds:
            if ds[v].attrs['units'] == 'kg m-3':
                print(f'Converting {v} from kg/m3 to ug/m3')
                with xr.set_options(keep_attrs=True):
                    ds[v] = ds[v] * kg2ug
                ds[v].attrs['units'] = 'ug m-3'
    for v in cwp_kg_to_g:
        if v in ds:
            if ds[v].attrs['units'] == 'kg m-2':
                print(f'Converting {v} from kg/m2 to g/m2')
                with xr.set_options(keep_attrs=True):
                    ds[v] = ds[v] * 1000
                ds[v].attrs['units'] = 'g m-2'

    ds = rename_ifs_vars(ds)
    return ds


def compute_OAs(ds):
    ds['OA'] = 0
    ds['POM'] = 0

    # whats better, this:
    ds['SOA'] = 0
    for v in SOA_components:
        ds['SOA'] = ds['SOA'] + ds[v]
    for v in POM_components:
        ds['POM'] = ds['POM'] + ds[v]
    ds['OA'] = ds['SOA'] + ds['POM']
    ## or this:
    ds['SOA2'] = ds[SOA_components].to_array(dim='comps', name='SOA').sum('comps')

    ds['SOA'].attrs['units'] = ds[SOA_components[0]].attrs['units']
    ds['OA'].attrs['units'] = ds[SOA_components[0]].attrs['units']
    ds['POM'].attrs['units'] = ds[POM_components[0]].attrs['units']
    return ds


def get_Nx_ec_earth_mod(ds, mod, fromx=100, tox=100000):
    _di = modes_dic[mod]
    vnum = _di['NUM']
    vrad = _di['RAD']
    vdiam = _di['DIAM']
    sig = _di['SIG']
    num = ds[vnum]
    rad = ds[vrad]
    diam = ds[vdiam]

    Nx = num * (lognorm.cdf(tox, np.log(sig), scale=diam)) - num * (
        lognorm.cdf(fromx, np.log(sig), scale=diam))
    return Nx


def get_Nx_ec_earth(ds, fromx=100, tox=100000):
    Nx = 0
    for mod in modes_dic.keys():
        Nxmod = get_Nx_ec_earth_mod(ds, mod, fromx=fromx, tox=tox)
        Nx += Nxmod
    return Nx


def add_Nx_to_dataset(ds, x_list=None, add_to_500=True):
    if x_list is None:
        x_list = [50, 70, 100, 150, 200, 500]
    ds = fix_units_ec_earth(ds)

    for x in x_list:
        ds[f'N{x}'] = get_Nx_ec_earth(ds, fromx=x)
    if add_to_500 and (500 in x_list):
        for x in x_list:
            if x == 500: continue
            nva = f'N{x}'
            nva2 = f'N{x}-500'
            ds[nva2] = ds[nva] - ds['N500']
    return ds


def rename_ifs_vars(ds):
    for v in ds.data_vars:
        if v in df_varinfo.index:
            new_varn = df_varinfo.loc[v, 'varname']
            print(f'renaming {v} to {new_varn}')
            ds[v].attrs['long_name'] = df_varinfo.loc[v, 'long_name']
            ds[v].attrs['units'] = df_varinfo.loc[v, 'units']
            ds = ds.rename({v: new_varn})
    return ds
