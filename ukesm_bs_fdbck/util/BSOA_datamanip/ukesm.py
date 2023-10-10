import pandas as pd
from scipy.stats import lognorm

import numpy as np

from ukesm_bs_fdbck.util.BSOA_datamanip import calculate_daily_median_summer, calculate_summer_median, mask4summer
import xarray as xr
from ukesm_bs_fdbck.constants import path_data_info

f = path_data_info / 'ukesm_info' / 'variable_overview_ukesm.csv'

kg2ug = 1e9

ukesm_var_overview = pd.read_csv(f, index_col=0)

rn_dict_ukesm = {
    'Temp': 'T',

}
# %%
varlist = [
    'Mass_Conc_OM_NS',
    'Mass_Conc_OM_KS',
    'Mass_Conc_OM_KI',
    'Mass_Conc_OM_AS',
    'Mass_Conc_OM_CS',
    'mmrtr_OM_NS',
    'mmrtr_OM_KS',
    'mmrtr_OM_KI',
    'mmrtr_OM_AS',
    'mmrtr_OM_CS',
    'nconcNS',
    'nconcKS',
    'nconcKI',
    'nconcAS',
    'nconcCS',
    'ddryNS',
    'ddryKS',
    'ddryKI',
    'ddryAS',
    'ddryCS',
    'Temp',
    # 'SFisoprene',
    # 'SFterpene',
]

mode_names = [
    'NS',  # nucleation soluble
    'KS',  # aitken soluble
    'KI',  # Aitken insoluble
    'AS',  # Accumulation soluble
    'CS',  # coarse soluble
]

modes_dic = {
    'NUS': dict(NUM='nconcNS', DIAM='ddryNS', SIG=1.59),
    'AIS': dict(NUM='nconcKS', DIAM='ddryKS', SIG=1.59),
    'AII': dict(NUM='nconcKI', DIAM='ddryKI', SIG=1.59),
    'ACS': dict(NUM='nconcAS', DIAM='ddryAS', SIG=1.4),
    'COS': dict(NUM='nconcCS', DIAM='ddryCS', SIG=2.),
}

num_vars = [
    f'nconc{mo}' for mo in mode_names
]
diam_vars = [
    f'ddry{mo}' for mo in mode_names
]

modes_in_PM1 = ['NS', 'KS', 'KI', 'AS']

rename_dic_ukesm = {
    'm01s38i493': 'Mass_Conc_OM_NS',
    'm01s38i494': 'Mass_Conc_OM_KS',
    'm01s38i497': 'Mass_Conc_OM_KI',
    'm01s38i495': 'Mass_Conc_OM_AS',
    'm01s38i496': 'Mass_Conc_OM_CS',
}

mass_in_OA = [
    f'Mass_Conc_OM_{s}' for s in modes_in_PM1
]

# %% [markdown] tags=[]
# #### Read in EC-Earth:

# %% tags=[]
case_name = 'AEROCOMTRAJ'
from_t = '2012-01-01'
to_t = '2012-02-01'
time_res = 'hour'
space_res = 'locations'
model_name = 'UKESM'

print()


# %%
def ds2df_ukesm(ds_st, model_lev_i=-1,
                take_daily_median=True,
                summer_months=None,
                mask_summer=False,
                pressure=None,
                temperature=None,
                air_density=None
                ):
    ds_st = change_units_and_compute_vars_ukesm(ds_st,
                                                # air_density=air_density,
                                                # pressure = pressure,
                                                # temperature=temperature

                                                )
    if take_daily_median:
        df = calculate_daily_median_summer(ds_st, summer_months=summer_months)
        df_sm = calculate_summer_median(df)
    else:
        if mask_summer:
            _ds = mask4summer(ds_st, months=summer_months)
        else:
            _ds = ds_st
        _ds['is_JJA'] = _ds['time.month'].isin([6, 7, 8])
        _ds['is_JA'] = _ds['time.month'].isin([7, 8])
        _ds['isSummer'] = _ds['time.month'].isin([7, 8])
        df = _ds.to_dataframe()
        df_sm = None

    return df, df_sm


# %%

def change_units_and_compute_vars_ukesm(ds_st,
                                        # air_density = None,
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
    ds_st = fix_units_ukesm(ds_st)
    if 'nconcNS' in ds_st:
        print('HEY')
        ds_st = add_Nx_to_dataset(ds_st,
                                  x_list=None, add_to_500=True)
    ds_st = compute_OAs(ds_st)

    rn_sub = {k: rn_dict_ukesm[k] for k in rn_dict_ukesm if
              ((k in ds_st.data_vars) & (rn_dict_ukesm[k] not in ds_st.data_vars))}
    ds_st = ds_st.rename(rn_sub)
    rn_dic2, dic_varname2file = get_rndic_ukesm(ds_st.data_vars)
    print(list(ds_st.data_vars))
    print(rn_dic2)
    print(dic_varname2file)
    rn_dic2 = {k: rn_dic2[k] for k in rn_dic2 if
              (k in ds_st.data_vars) }
    ds_st = ds_st.rename(rn_dic2)

    if 'T' in ds_st:
        ds_st['T_C'] = ds_st['T'] - 273.15


    return ds_st


def fix_units_ukesm(ds):
    for v in num_vars:
        if v in ds:
            if ds[v].attrs['units'] == 'm-3':
                print(f'Converting {v} from m-3 to cm-3')
                with xr.set_options(keep_attrs=True):
                    ds[v] = ds[v] * 1e-6
                ds[v].attrs['units'] = 'cm-3'

    for v in diam_vars:
        if v in ds:
            if ds[v].attrs['units'] == 'm':
                print(f'Converting {v} from m to nm')
                with xr.set_options(keep_attrs=True):
                    ds[v] = ds[v] * 1e9
                ds[v].attrs['units'] = 'nm'

    for v in mass_in_OA + ['OA', ]:
        if v in ds:
            if 'units' not in ds[v].attrs:
                ds[v].attrs['units'] = 'kg m-3'
                print(f'Converting {v} from kg/m3 to ug/m3')
                with xr.set_options(keep_attrs=True):
                    ds[v] = ds[v] * kg2ug
                ds[v].attrs['units'] = 'ug m-3'
    """            
    for v in cwp_kg_to_g:
        if v in ds:
            if ds[v].attrs['units'] == 'kg m-2':
                print(f'Converting {v} from kg/m2 to g/m2')
                with xr.set_options(keep_attrs=True):
                    ds[v] = ds[v] * 1000
                ds[v].attrs['units'] = 'g m-2'

    ds = rename_ifs_vars(ds)
    """
    return ds


def compute_OAs(ds):
    ds['OA'] = 0
    all_there = True
    for v in mass_in_OA:
        if v not in ds.data_vars:
            all_there = False
    if not all_there:
        return ds
    for v in mass_in_OA:
        ds['OA'] = ds['OA'] + ds[v]
    ## or this:
    ds['OA'].attrs['units'] = ds[mass_in_OA[0]].attrs['units']
    return ds


def get_Nx_ukesm_mod(ds, mod, fromx=100, tox=100000):
    _di = modes_dic[mod]
    vnum = _di['NUM']
    vdiam = _di['DIAM']
    sig = _di['SIG']
    num = ds[vnum]
    diam = ds[vdiam]

    Nx = num * (lognorm.cdf(tox, np.log(sig), scale=diam)) - num * (
        lognorm.cdf(fromx, np.log(sig), scale=diam))
    return Nx


def get_Nx_ukesm(ds, fromx=100, tox=100000):
    Nx = 0
    for mod in modes_dic.keys():
        Nxmod = get_Nx_ukesm_mod(ds, mod, fromx=fromx, tox=tox)
        Nx += Nxmod
    return Nx


def add_Nx_to_dataset(ds, x_list=None, add_to_500=True):
    if x_list is None:
        x_list = [50, 70, 100, 150, 200, 500]
    ds = fix_units_ukesm(ds)

    for x in x_list:
        ds[f'N{x}'] = get_Nx_ukesm(ds, fromx=x)
    if add_to_500 and (500 in x_list):
        for x in x_list:
            if x == 500: continue
            nva = f'N{x}'
            nva2 = f'N{x}-500'
            ds[nva2] = ds[nva] - ds['N500']
    return ds


def extract_2D_cloud_time_ukesm(ds_all):
    # Calculate r_eff:
    if ('Reff_2d_distrib_x_weight' in ds_all) and ('weight_Reff_2d_distrib' in ds_all):
        ds_all['r_eff'] = ds_all['Reff_2d_distrib_x_weight'] / ds_all['weight_Reff_2d_distrib']

    # Calculate r_eff alternative:
    if ('Reff_2d_x_weight_warm_cloud' in ds_all) and ('weight_Reff_2d' in ds_all):
        ds_all['r_eff2'] = ds_all['Reff_2d_x_weight_warm_cloud'] / ds_all['weight_Reff_2d']
    # Calculate maximum extent in cloud column
    if 'area_cloud_fraction_in_each_layer' in ds_all:
        ds_all['max_cloud_cover'] = ds_all['area_cloud_fraction_in_each_layer'].max('model_level')
    # Calculate maximum volume fraction in cloud column
    if 'liq_cloud_fraction_in_each_layer' in ds_all:
        ds_all['max_cloud_fraction'] = ds_all['liq_cloud_fraction_in_each_layer'].max('model_level')
    # Calculate total liquid water path and total ice water path.
    ds_all['lwp'] = ds_all['ls_lwp'] + ds_all['conv_lwp']
    ds_all['iwp'] = ds_all['ls_iwp'] + ds_all['conv_iwp']
    for v in ['lwp', 'iwp']:
        if ds_all[v].units == 'kg m-2':
            print(f'converting units {v}')
            ds_all[v] = ds_all[v] * 1000
            ds_all[v].attrs['units'] = 'g m-2'

    if 'max_cloud_cover' in ds_all:
        if 'lwp' in ds_all:
            ds_all['lwp_incld'] = ds_all['lwp']/ds_all['max_cloud_cover']
    if ('rho' in ds_all) and ('qcl' in ds_all) and ('layer' in ds_all):
        ds_all['computed_lwp_sum'] = 1000*(ds_all['rho']*ds_all['qcl']*ds_all['layer']).sum('model_level')
        ds_all['computed_lwp_sum'].attrs['standard_name'] = 'Computen lwp from sum qcl'
        ds_all['computed_lwp_sum'].attrs['long_name'] = 'Computen lwp from sum qcl'
        ds_all['computed_lwp_sum'].attrs['units'] = 'g m-3'
        if 'max_cloud_cover' in ds_all:
            ds_all['computed_lwp_sum_incld'] = ds_all['computed_lwp_sum'] / ds_all['max_cloud_fraction']
            ds_all['computed_lwp_sum_incld'].attrs['standard_name'] = 'Computen lwp from sum qcl divided by cloud ' \
                                                                      'cover fraction'
            ds_all['computed_lwp_sum_incld'].attrs['long_name'] = 'Computen lwp from sum qcl divided by cloud cover ' \
                                                                  'fraction'
            ds_all['computed_lwp_sum_incld'].attrs['units'] = 'g m-3'

    if ('rho' in ds_all) and ('qcf' in ds_all) and ('layer' in ds_all):
        ds_all['computed_iwp_sum'] = 1000*(ds_all['rho']*ds_all['qcf']*ds_all['layer']).sum('model_level')
        ds_all['computed_iwp_sum'].attrs['standard_name'] = 'Computen lwp from sum qcl'
        ds_all['computed_iwp_sum'].attrs['long_name'] = 'Computen lwp from sum qcl'
        ds_all['computed_iwp_sum'].attrs['units'] = 'g m-3'
        if 'max_cloud_cover' in ds_all:
            ds_all['computed_iwp_sum_incld'] = ds_all['computed_iwp_sum'] / ds_all['max_cloud_fraction']
            ds_all['computed_iwp_sum_incld'].attrs['standard_name'] = 'Computen iwp from sum qcf divided by cloud ' \
                                                                      'cover fraction'
            ds_all['computed_iwp_sum_incld'].attrs['long_name'] = 'Computen iwp from sum qcf divided by cloud cover ' \
                                                                  'fraction'
            ds_all['computed_iwp_sum_incld'].attrs['units'] = 'g m-3'

    if ('computed_iwp_sum' in ds_all) and 'computed_lwp_sum' in ds_all:
        ds_all['liq_frac_cwp'] = ds_all['computed_lwp_sum'] / (ds_all['computed_iwp_sum'] + ds_all['computed_lwp_sum'])
        # ds_all['liq_frac_cwp'] = ds_all['lwp'] / (ds_all['lwp'] + ds_all['iwp'])


    return ds_all


def get_rndic_ukesm(varlist):
    rename_dic = {}
    dic_varname2file = {}
    for v in varlist:
        if v in ukesm_var_overview.index:
            file_name_var = ukesm_var_overview.loc[v, 'var_name_infile']
            var_in_filename = ukesm_var_overview.loc[v, 'orig_var_name_file']

            new_var_name = v
            rename_dic[file_name_var] = new_var_name
            dic_varname2file[new_var_name] = var_in_filename
        elif v in list(ukesm_var_overview['var_name_infile']):
            _tf = (ukesm_var_overview['var_name_infile']==v)
            file_name_var = ukesm_var_overview[_tf].iloc[0].name
            rename_dic[v] = file_name_var


    return rename_dic, dic_varname2file
