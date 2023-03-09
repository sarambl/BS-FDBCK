import pandas as pd
from scipy.stats import lognorm

import numpy as np

from bs_fdbck.util.BSOA_datamanip import calculate_daily_median_summer, calculate_summer_median, mask4summer
from bs_fdbck.util.collocate.collocateLONLAToutput import CollocateLONLATout
import xarray as xr
from bs_fdbck.constants import path_data_info

f = path_data_info / 'ukesm_info' / 'variable_overview_ukesm.csv'

kg2ug = 1e9

ukesm_var_overview = pd.read_csv(f, index_col=0)
ukesm_var_overview

rn_dict_ukesm = {
    'Temp':'T',


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
import time

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
    ds_st = add_Nx_to_dataset(ds_st, x_list=None, add_to_500=True)
    ds_st = compute_OAs(ds_st)

    rn_sub = {k: rn_dict_ukesm[k] for k in rn_dict_ukesm if
              ((k in ds_st.data_vars) & (rn_dict_ukesm[k] not in ds_st.data_vars))}
    ds_st = ds_st.rename(rn_sub)
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
