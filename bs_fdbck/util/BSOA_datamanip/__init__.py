import datetime
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

R = 287.058
pressure_default = 101325  # Pa
kg2ug = 1e9
temperature_default = 273.15
standard_air_density = pressure_default / (R * temperature_default)

lat_smr = 61.85
lon_smr = 24.28
model_lev_i = -1

rn_dic_echam_cloud = {
    'cwp_incld': 'CWP',
    # 'cwp_incld':'CWP',
    'cod': 'COT',
    # 'ceff_ct': 'r_eff',
    'ceff_ct_incld': 'r_eff',
    'lcdnc_ct_cm3_incld': 'CDNC',
}
rn_dic_ec_earth_cloud = {
    'cwp_incld': 'CWP',
    # 'TOT_CLD_VISTAU_s_incld': 'COT',
    're_liq_cltop': 'r_eff',
    'cdnc_incld_cltop': 'CDNC'
}

rn_dic_noresm_cloud = {
    'TGCLDLWP_incld': 'CWP',
    'TOT_CLD_VISTAU_s_incld': 'COT',
    'ACTREL_incld': 'r_eff',
    'ACTNL_incld': 'CDNC',
}
rn_dic_obs_cloud = {
    'CWP (g m^-2)': 'CWP',
    'CER (micrometer)': 'r_eff',
    'OA (microgram m^-3)': 'OA',

}

rn_dict_echam = {
    'ORG_mass_conc': 'OA',
    'tempair': 'T',

}

varl_st_default = ['SOA_NA', 'SOA_A1', 'OM_NI', 'OM_AI', 'OM_AC', 'SO4_NA', 'SO4_A1', 'SO4_A2', 'SO4_AC', 'SO4_PR',
                   'BC_N', 'BC_AX', 'BC_NI', 'BC_A', 'BC_AI', 'BC_AC', 'SS_A1', 'SS_A2', 'SS_A3', 'DST_A2', 'DST_A3',
                   'N100', 'N150', 'N50', 'N200',
                   'N100_STP', 'N150_STP', 'N50_STP', 'N200_STP',
                   'OA',
                   'OA_amb',
                   'OA_STP',
                   # ECHAM:
                   'mmrtrN200',
                   'mmrtrN100',
                   'mmrtrN50',
                   'emi_monot_bio',
                   'emi_isop_bio',
                   'ORG_mass',
                   'T',
                   'T_C',
                   'N500',
                   'N100', 'DOD500', 'DOD440',
                   'H2SO4', 'SOA_LV', 'COAGNUCL', 'FORMRATE', 'T',
                   'N500',
                   'NCONC01', 'N50', 'N150', 'N200',  # 'DOD500',

                   # 'NCONC01',
                   # 'SFisoprene',
                   # 'SFmonoterp',
                   # 'DOD500',
                   'SFmonoterp', 'SFisoprene',
                   'PS',

                   'SOA_NA', 'SOA_A1', 'OM_NI', 'OM_AI', 'OM_AC', 'SO4_NA', 'SO4_A1', 'SO4_A2', 'SO4_AC', 'SO4_PR',
                   'BC_N', 'BC_AX', 'BC_NI', 'BC_A', 'BC_AI', 'BC_AC', 'SS_A1', 'SS_A2', 'SS_A3', 'DST_A2', 'DST_A3',
                   # EC-Earth
                   'CCN0.20', 'CCN1.00', 'M_BCACS', 'M_BCAII', 'M_BCAIS', 'M_BCCOS', 'M_DUACI', 'M_DUACS',
                   'M_DUCOI', 'M_DUCOS', 'M_POMACS', 'M_POMAII', 'M_POMAIS', 'M_POMCOS', 'M_SO4ACS', 'M_SO4COS',
                   'M_SO4NUS',
                   'M_SOAACS', 'M_SOAAII', 'M_SOAAIS', 'M_SOACOS', 'M_SOANUS', 'M_SSACS', 'M_SSCOS', 'SOA',
                   'N_ACI', 'N_ACS', 'N_AII', 'N_AIS', 'N_COI', 'N_COS', 'N_NUS', 'RDRY_ACS', 'RDRY_AIS',
                   'RDRY_COS', 'RDRY_NUS', 'RWET_ACI', 'RWET_ACS', 'RWET_AII', 'RWET_AIS', 'RWET_COI', 'RWET_COS',
                   'RWET_NUS',
                   'emiisop', 'emiterp', 'T', 'DDRY_NUS', 'DDRY_AIS', 'DDRY_ACS', 'DDRY_COS', 'DWET_AII',
                   'DWET_ACI', 'DWET_COI', 'N50', 'N70', 'N100', 'N150', 'N200', 'N500', 'N50-500',
                   'N70-500', 'N100-500', 'N150-500', 'N200-500', 'POM', 'SOA', 'SOA2', 'T_C',
                   'N70-500_STP', 'N100-500_STP', 'N150-500_STP', 'N200-500_STP', 'POM', 'SOA', 'SOA2', 'T_C',

                   'POM',

                   ]

varl_cl_default = [
    'TGCLDCWP',
    'TGCLDCWP_incld',
    'TGCLDLWP',
    'TGCLDLWP_incld',
    'TGCLDIWP',
    'TGCLDIWP_incld',
    'TOT_CLD_VISTAU',
    'TOT_ICLD_VISTAU',
    'TOT_CLD_VISTAU_s',
    'TOT_ICLD_VISTAU_s',
    'TOT_CLD_VISTAU_s_incld',
    'TOT_ICLD_VISTAU_s_incld',
    'optical_depth',
    'CLDFREE',
    'CLDTOT',
    'FCTL',
    'ACTREL',
    'ACTREL_incld',
    'ACTNL',
    'ACTNL_incld',
    'FSDSC',
    'FSDSCDRF',
    'FCTI',
    'FCTL',
    'FLNS',
    'FLNSC',
    'FLNT',
    'FLNTCDRF',
    'FLNT_DRF',
    'FLUS',
    'FLUTC',
    'FREQI',
    'FREQL',
    'FSDSCDRF',
    'FSDS_DRF',
    'FSNS',
    'FSNSC',
    'FSNT',
    'FSNTCDRF',
    'FSNT_DRF',
    'FSUS_DRF',
    'FSUTADRF',
    'optical_depth',
    'transmittance',

    # ECHAM:
    'CWP',
    'COT',
    'r_eff',
    'cwp',
    'cod',
    'ceff_ct',
    'cod_incld',
    'cwp_incld',
    'ceff',
    'ceff_um',
    'ceff_um_incld',
    'ceff_ct',
    'ceff_ct_incld',
    'lcdnc',
    'lcdnc_cm3',
    'lcdnc_cm3_incld',
    'lcdnc_ct',
    'lcdnc_ct_cm3',
    'lcdnc_ct_cm3_incld',
    'cl_time_ct',
    'clfr',
    'cl_time',
    'cl_time',
    'lcdnc_ct_incld',
    'tempair_ct',
    'min_cl_tempair',
    'min_cl_T',
    'T_ct',
    'clfr_lev_sum23-38',
    'clfr_mask',

    # EC-Earth:
    'tcw',
    'tcwv',
    'tclw',
    'tciw',
    'ttc',
    'cdnc',
    'cdnc_incld',
    'cdnc_incld_cltop',
    're_liq',
    're_liq_incld',
    're_liq_incld_cltop',
    'liq_cloud_time',
    'cc_cltop',
    'cl_frac_where_cltime_pos',
    'argmax',
    'liq_frac_cwp',
    'cc_liq',
    'cc_all',
    'liq_frac_cwp',

]


def fix_echam_time(dt):
    """
    Datetime has an inconvenient timestamp which is misleading so we correct it to left aligned.
    :param dt:
    :return:
    """
    # a, b = divmod(round(dt.minute, -1), 60)
    # tdelta = datetime.timedelta(minutes=dt.minute, seconds = dt.second)
    # nh = (dt.hour+a)%24
    ndt = datetime.datetime(dt.year, dt.month, dt.day, dt.hour)  # dt - tdelta
    # dt_o = datetime.datetime(dt.year,dt.month, dt.day, (dt.hour + a) % 24,b)
    return ndt


def compute_total_tau(ds_mod):
    if 'TOT_ICLD_VISTAU' in ds_mod:
        ds_mod['TOT_ICLD_VISTAU_s'] = ds_mod['TOT_ICLD_VISTAU'].sum('lev')
    if 'TOT_CLD_VISTAU' in ds_mod:
        ds_mod['TOT_CLD_VISTAU_s'] = ds_mod['TOT_CLD_VISTAU'].sum('lev')

    return ds_mod


def broadcase_station_data(ds_all, varl_st=None, lat=lat_smr, lon=lon_smr):
    if varl_st is None:
        varl_st = varl_st_default

    print(varl_st)
    ds_sel = ds_all.sel(lat=lat, lon=lon, method='nearest')
    print(ds_sel)
    ds_all = broadcast_vars_in_ds_sel(ds_all, ds_sel, varl_st)
    return ds_all


def broadcast_vars_in_ds_sel(ds_all, ds_sel, varl_st, only_already_in_ds=True):
    ds_1, ds_2 = xr.broadcast(ds_sel, ds_all)
    for v in varl_st:
        print(v)
        if v in ds_all or not only_already_in_ds:
            if v not in ds_1:
                print(f'Did not find {v}. Skipping')
                continue
            ds_all[v] = ds_1[v]
            print(f'replacing {v} ')

    return ds_all


def compute_optical_thickness(ds):
    if not (('FSDS_DRF' in ds) and ('FSDSCDRF' in ds)):
        return ds
    SW_down_surf = ds['FSDS_DRF']

    SW_down_surf_cs = ds['FSDSCDRF']

    transmittance = SW_down_surf / SW_down_surf_cs

    opt_depth = - np.log(transmittance)

    ds['optical_depth'] = opt_depth
    ds['transmittance'] = transmittance
    return ds


def ds2df_echam(ds_st, model_lev_i=-1,
                take_daily_median=True,
                summer_months=None,
                mask_summer=False,
                pressure=None,
                temperature=None,
                air_density=None
                ):
    # N50, N100 etc:
    ds_st = change_units_and_compute_vars_echam(ds_st,
                                                air_density=air_density,
                                                pressure=pressure,
                                                temperature=temperature
                                                )

    if 'lev' in ds_st.dims:
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
        # df = df[df['isSummer'].notnull()]
        df_sm = None

    return df, df_sm


def ds2df_ec_earth(ds_st, model_lev_i=-1,
                   take_daily_median=True,
                   summer_months=None,
                   mask_summer=False,
                   pressure=None,
                   temperature=None,
                   air_density=None
                   ):
    # N50, N100 etc:
    # ds_st = change_units_and_compute_vars_echam(ds_st,
    #                                            air_density=air_density,
    #                                            pressure = pressure,
    #                                            temperature=temperature
    #                                            )

    if 'lev' in ds_st.dims:
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
        # df = df[df['isSummer'].notnull()]
        df_sm = None

    return df, df_sm


def change_units_and_compute_vars_echam(ds_st, air_density=None,
                                        pressure=pressure_default,
                                        temperature=temperature_default):
    """

    :param ds_st:
    :return:
    """
    if air_density is None:
        if pressure is None:
            pressure = pressure_default
        if temperature is None:
            temperature = temperature_default
        air_density = pressure / (R * temperature)

    nvars = ['mmrtrN3', 'mmrtrN50', 'mmrtrN100', 'mmrtrN200', 'mmrtrN250', 'mmrtrN500']
    for v in nvars:
        if v in ds_st:
            if ds_st[v].attrs['units'] == 'kg-1':
                v_new = v[5:]
                print(f'Computing {v_new} from {v}')
                print(v_new)
                # /kg_air --> /m3_air by multiplying by density kg_air/m3_air
                # then from /m3-->/cm3: multiply by 1e-6
                ds_st[v_new] = ds_st[v] * air_density * 1e-6
                ds_st[v_new].attrs['units'] = 'm-3'
                long_name = ds_st[v].attrs['long_name']
                ds_st[v_new].attrs['long_name'] = 'number concentration ' + long_name.split('_')[-1]
                ds_st[v_new].attrs['description'] = 'number concentration ' + long_name.split('_')[-1]
    vars_kg2kg = ['ORG_mass', 'VBS1_gas', 'VBS0_gas', 'VBS10_gas']
    for v in vars_kg2kg:

        if v in ds_st:
            if ds_st[v].attrs['units'] == 'kg kg-1':
                v_new = v + '_conc'
                print(f'Computing {v_new} from {v} and convertign units')
                # kg_aero/kg_air --> kg_m3: multiply by density kg_air/m3_air
                # kg_aero/m3_air --> ug/m3_air: multiply by
                ds_st[v_new] = ds_st[v] * air_density * kg2ug

                ds_st[v_new].attrs['units'] = 'kg/m-3'
                long_name = ds_st[v].attrs['long_name']
                ds_st[v_new].attrs['long_name'] = 'number concentration ' + long_name.split('_')[-1]
                ds_st[v_new].attrs['description'] = 'number concentration ' + long_name.split('_')[-1]
                print(v_new)
    ls_oa = [
        'VBS0_gas_conc',
        'VBS1_gas_conc',
        'VBS10_gas_conc',
        'ORG_mass_conc',
    ]
    all_there = True
    for v in ls_oa:
        if v not in ds_st.data_vars:
            all_there = False
    if all_there:
        ds_st['OAG'] = 0  # ['VBS0_gas_conc', 'VBS1_gas_conc', 'VBS10_gas_conc','OA']
        print('Computing OAG')
        for v in ls_oa:
            ds_st['OAG'] = ds_st['OAG'] + ds_st[v]
        ds_st['OAG'].attrs['units'] = ds_st['ORG_mass_conc'].attrs['units']

    vars_kg2g_perm2 = ['cwp', 'cwp_incld']
    for v in vars_kg2g_perm2:
        un = 'g m $^{-2}$'
        if v in ds_st:
            if ds_st[v].attrs['units'] != un:
                print(f'Computing {v} by ')
                v_new = v + '_orig'
                ds_st = ds_st.rename({v: v_new})
                # kg_aero/kg_air --> kg_m3: multiply by density kg_air/m3_air
                # kg_aero/m3_air --> ug/m3_air: multiply by
                ds_st[v] = ds_st[v_new] * 1000.  # kg--> g
                # standard_air_density * 1e9

                ds_st[v].attrs['units'] = un
                long_name = ds_st[v_new].attrs['long_name']
                ds_st[v].attrs['long_name'] = long_name
                ds_st[v].attrs['description'] = long_name
                print(v)

    if 'ceff' in ds_st:
        if ds_st['ceff'].attrs['units'] == 'm':
            ds_st['ceff_um'] = ds_st['ceff'] * 1e6
            ds_st['ceff_um'].attrs['units'] = 'um'
    if 'lcdnc' in ds_st:
        if ds_st['lcdnc'].attrs['units'] == 'm-3':
            ds_st['lcdnc_cm3'] = ds_st['lcdnc'] * 1e-6
            ds_st['lcdnc_cm3'].attrs['units'] = 'cm-3'
    if 'lcdnc_ct' in ds_st:
        if ds_st['lcdnc_ct'].attrs['units'] == 'cm-3':
            print(f'Converting units from lcdnc_ct')

            ds_st['lcdnc_ct_cm3'] = ds_st['lcdnc_ct'] * 1e-6
            ds_st['lcdnc_ct_cm3'].attrs['units'] = 'cm^{-3}'
    if 'lcdnc_ct_incld' in ds_st:
        if ds_st['lcdnc_ct_incld'].attrs['units'] == 'cm-3':
            print(f'Converting units from lcdnc_ct_incld')

            ds_st['lcdnc_ct_cm3_incld'] = ds_st['lcdnc_ct_incld'] * 1e-6
            ds_st['lcdnc_ct_cm3_incld'].attrs['units'] = 'cm^{-3}'

    rn_sub = {k: rn_dict_echam[k] for k in rn_dict_echam if
              ((k in ds_st.data_vars) & (rn_dict_echam[k] not in ds_st.data_vars))}
    ds_st = ds_st.rename(rn_sub)
    if 'T' in ds_st:
        ds_st['T_C'] = ds_st['T'] - 273.15
    return ds_st


def change_units_and_compute_vars(ds,
                                  air_density=None,
                                  temperature=temperature_default,
                                  pressure=pressure_default,
                                  ):
    """
    Changing units and computing necessary variables.
    :param ds:
    :return:
    """
    if air_density is None:
        if pressure is None:
            pressure = pressure_default
        if temperature is None:
            temperature = temperature_default
        air_density = pressure / (R * temperature)

    # NorESM
    for v in ['TGCLDLWP', 'TGCLDIWP', 'TGCLDCWP']:
        if v in ds.data_vars:
            if ds[v].attrs['units'] == 'kg/m2':
                ds[v] = ds[v] * 1000
                ds[v].attrs['units'] = 'g/m2'
    ds['rho'] = air_density
    for v in ['ACTNL', 'AWNC']:
        if v not in ds:
            continue
        if ds[v].attrs['units'] == 'm-3':
            ds[v] = ds[v] * 1e-6  # m-3--> cm-3
            ds[v].attrs['units'] = 'cm$^{-3}$'
    if 'FCTL' in ds:
        for v in ['ACTNL', 'ACTREL']:
            if v not in ds:
                continue
            ds[f'{v}_incld'] = ds[v] / ds['FCTL']
            # ds['ACTREL_incld'] = ds['ACTREL'] / ds['FCTL']
    if 'T' in ds:
        ds['T_C'] = ds['T'] - 273.15

    # ds = ds.where(ds['CLDFREE'] < 1)
    ls_mol2mol_oa = ['SOA_LV', 'SOA_SV']
    for s in ls_mol2mol_oa:
        un = 'kg/kg'
        kg2kmol_oa = 168.2  # kg/kmol
        kg2kmol_air = 28.97  # kg/kmol
        molmix_masmix = kg2kmol_oa / kg2kmol_air
        if s not in ds:
            continue
        if ds[s].attrs['units'] != un:
            ds[s] = ds[s] * molmix_masmix
            ds[s].attrs['units'] = un

    ls_so4 = [c for c in ds.data_vars if 'SO4_' in c]  # ['SO4_NA']

    ls_oa = ['SOA_NA', 'SOA_A1', 'OM_AC', 'OM_AI', 'OM_NI']
    ls_soa = ['SOA_NA', 'SOA_A1']
    ls_oag = ['SOA_NA', 'SOA_A1', 'OM_AC', 'OM_AI', 'OM_NI', 'SOA_LV', 'SOA_SV']
    ls_poa = ['OM_AC', 'OM_AI', 'OM_NI']
    for s in ls_oag + ls_so4:
        un = '$\mu$g/m3'
        if s not in ds:
            continue
        if ds[s].attrs['units'] != un:
            ds[s] = ds[s] * ds['rho'] * kg2ug
            ds[s].attrs['units'] = un
    if set(ls_soa).issubset(set(ds.data_vars)):
        ds['SOA'] = ds['SOA_NA'] + ds['SOA_A1']
    if set(ls_oag).issubset(set(ds.data_vars)):
        ds['OAG'] = 0  # ds['SOA_NA'] + ds['SOA_A1']
        for v in ls_oag:
            ds['OAG'] = ds['OAG'] + ds[v]
        ds['OAG'].attrs['units'] = ds['SOA_NA'].attrs['units']

    if set(ls_oa).issubset(set(ds.data_vars)):
        ds['OA'] = ds['SOA_NA'] + ds['SOA_A1'] + ds['OM_AC'] + ds['OM_AI'] + ds['OM_NI']
    if set(ls_poa).issubset(set(ds.data_vars)):
        ds['POA'] = ds['OM_AC'] + ds['OM_AI'] + ds['OM_NI']

    if set(ls_so4).issubset(set(ds.data_vars)):
        ds['SO4'] = 0

        for s in ls_so4:
            print(s)

            # print(ds[s].mean())
            ds['SO4'] = ds['SO4'] + ds[s]
    # NorESM
    if 'CLDTOT' in ds:
        for v in ['TGCLDLWP', 'TGCLDCWP', 'TOT_CLD_VISTAU_s']:
            if v not in ds:
                continue
            ds[f'{v}_incld'] = ds[v] / (ds['CLDTOT'])
            # ONLY USE WHERE CLOUD FREE IS LESS THAN 99%
            ds[f'{v}_incld'] = ds[f'{v}_incld'].where(ds['CLDTOT'] > .01)
            print(f'masking {v} where cloud covers at least 1% of the gridbox')
            print(f'to avoid funny values ')

    ds = compute_optical_thickness(ds)
    return ds


def get_dic_df_mod(dic_ds,
                   select_hours_clouds=False,
                   mask_cloud_values=True,
                   from_hour=8,
                   to_hour=14,
                   varl_cl=None,
                   varl_st=None,
                   take_daily_median=True,
                   mask_summer=True,
                   return_summer_median=False,
                   summer_months=None,
                   kwrgs_mask_clouds=None
                   ):
    """
    Calculates daily median for data, masking cloud data to daytime values if requested.
    :param return_summer_median:
    :param take_daily_median:
    :param summer_months:
    :param dic_ds:
    :param select_hours_clouds:
    :param from_hour:
    :param to_hour:
    :param varl_cl:
    :return:
    """
    if varl_cl is None:
        varl_cl = varl_cl_default  # + varl_st_default
    if varl_st is None:
        varl_st = varl_st_default
    if kwrgs_mask_clouds is None:
        kwrgs_mask_clouds = dict()
    dic_df = dict()
    dic_df_y = dict()

    for ca in dic_ds.keys():

        ds = dic_ds[ca].copy()
        # (ds['r_eff']
        # .resample(time='D').median().count('time').plot()
        # )
        # plt.show()

        # ds = change_units_and_compute_vars(ds)
        ds['hour'] = ds['time.hour']

        if select_hours_clouds:
            ds = extract_hours_for_satellite_vars(ds, from_hour, select_hours_clouds, to_hour, varl_cl + varl_st)
            print('hours')
            # (ds['r_eff']
            # .where(ds['time.month'].isin([7,8]))
            # .resample(time='D').median().count('time').plot()
            # )
            # plt.show()

        ds_cl = ds[list(set(ds.data_vars).intersection(varl_cl))]
        # print(ds_cl.data_vars)

        if mask_cloud_values:
            print(kwrgs_mask_clouds)
            ds_cl = mask_values_clouds(ds_cl, **kwrgs_mask_clouds)
            print('cloud and resampled')
            # (ds_cl['r_eff']
            # .where(ds_cl['time.month'].isin([7,8]))
            # .resample(time='D').median()
            # .count('time').plot()
            # )
            plt.show()

        if take_daily_median:
            # print(ds_cl)
            ds_cl_dm = calculate_daily_median_summer(ds_cl, summer_months=summer_months, return_pandas=False)
            print('cloud_avg median')
            # (ds_cl_dm['r_eff']
            # .where(ds_cl_dm['time.month'].isin([7,8]))
            # #.resample(time='D').median()
            # .count('time').plot()
            # )
            # plt.show()

            ds_st = ds[list(set(ds.data_vars).intersection(varl_st))]

            # print(ds_st)
            ds_st_dm = calculate_daily_median_summer(ds_st, summer_months=summer_months, return_pandas=False)
            print('station')
            if 'OA' in ds_st_dm:
                _v = 'OA'
            else:
                _v = 'OA_STP'
            (ds_st_dm[_v]
             .where(ds_st_dm['time.month'].isin([7, 8]))
             .resample(time='D').median()
             .count('time').plot()
             )
            plt.show()

            ds_comb = xr.merge([ds_cl_dm, ds_st_dm])
            print('merge')

            with ProgressBar():
                df = ds_comb.to_dataframe()
        else:
            if mask_summer:
                _ds = mask4summer(ds, months=summer_months)
            else:
                _ds = ds
            _ds['is_JJA'] = _ds['time.month'].isin([6, 7, 8])
            _ds['is_JA'] = _ds['time.month'].isin([7, 8])
            _ds['isSummer'] = _ds['time.month'].isin([7, 8])
            df = _ds.to_dataframe()
            if mask_summer:
                df = df[df['isSummer'].notnull()]

            # df_sm = None

        df = df.drop([co for co in df.columns if (('lat_' in co) | ('lon_' in co))],
                     axis=1)
        print('df1')
        _dds = df.to_xarray()

        # months = (df.index.get_level_values(0).month==7 )|(df.index.get_level_values(0).month==8  )

        # df.loc[:,'year'] = df.index.get_level_values(0).year.values
        # if type(df.index) is
        if type(df.index) is pd.MultiIndex:
            df['year'] = df.index.get_level_values(0).year.values
        else:
            df['year'] = df.index.year.values
        # print(df['year'])

        dic_df[ca] = df
        # print(dic_df[ca]['year'])

        if return_summer_median:
            df_ym = calculate_summer_median(df)
            dic_df_y[ca] = df_ym
        # print(dic_df[ca]['year'])
    if return_summer_median:
        # print(dic_df[ca]['year'])
        return dic_df_y, dic_df

    return dic_df


def calculate_summer_median(df):
    df_ym = df.resample('Y').median()
    df_ym.loc[:, 'year'] = df_ym.index.year.values
    if ('T_C' not in df_ym) and ('T' in df_ym):
        df_ym.loc[:, 'T_C'] = df_ym['T'].values - 273.15
    return df_ym


def calculate_daily_median_summer(ds, summer_months=None,
                                  st_variables=varl_st_default,
                                  return_pandas=True):
    # st_vars = set(st_variables).union(set(ds.data_vars))

    with ProgressBar():
        ds_sel_median = ds.resample({'time': 'D'}).median('time')
    with ProgressBar():
        ds_sel_median = mask4summer(ds_sel_median, months=summer_months)
    if not return_pandas:
        return ds_sel_median
    with ProgressBar():
        df = ds_sel_median.to_dataframe()
    return df


def mask4summer(ds_sel_median, months=None):
    if months is None:
        months = list(np.arange(1, 13))
    ds_sel_median['month'] = ds_sel_median['time.month']
    # df_s['ACTNL'].plot()
    for m in months:
        ds_sel_median['ismonth%d' % m] = (ds_sel_median['month'] == m)
    ds_sel_median['isJuly'] = ds_sel_median['month'] == 7
    ds_sel_median['isAug'] = ds_sel_median['month'] == 8

    ds_sel_median['isJA'] = (ds_sel_median['isJuly'] | ds_sel_median['isAug'])
    ds_sel_median['isSummer'] = False
    for m in months:
        ds_sel_median['isSummer'] = ds_sel_median['isSummer'] | ds_sel_median['ismonth%d' % m]
    # ds_sel_median = ds_sel_median.where(ds_sel_median['month']
    ds_sel_median = ds_sel_median.where(ds_sel_median['isSummer'])  # ['month']
    return ds_sel_median


def extract_hours_for_satellite_vars(ds, from_hour, select_hours_clouds, to_hour, varl_cl):
    ds['hour'] = ds['time.hour']
    hours_we_want = (ds['hour'] >= from_hour) & (ds['hour'] <= to_hour)
    if select_hours_clouds:
        for v in varl_cl:
            if v in ds.data_vars:
                ds[v] = ds[v].where(hours_we_want)
    return ds


def mask_values_clouds(ds,
                       tau_bounds=None,
                       min_cwp=50,
                       min_reff=5,
                       min_temp=-15,
                       ):
    if tau_bounds is None:
        tau_bounds = [5, 50]
    if 'COT' in ds.data_vars:
        print(f'Masking with {tau_bounds[0]}<COT<{tau_bounds[1]}!')

        tau = ds['COT']
        ma = (tau >= tau_bounds[0]) & (tau <= tau_bounds[1])
        ds = ds.where(ma)
    else:
        print('WARNING: COT NOT FOUND, not masking these values!')
    if 'CWP' in ds.data_vars:
        cwp = ds['CWP']
        ma = (cwp >= min_cwp)
        print(f'Masking with {min_cwp}<CWP!')

        ds = ds.where(ma)
    else:
        print('WARNING: CWP NOT FOUND, not masking these values!')

    if 'r_eff' in ds.data_vars:
        r_eff = ds['r_eff']
        ma = (r_eff >= min_reff)
        print(f'Masking with r_eff>{min_reff}!')

        ds = ds.where(ma)
    else:
        print('WARNING: r_eff NOT FOUND, not masking these values!')
    # if 'min_cl_tempair' in ds.data_vars:
    #    tempair_ct = ds['min_cl_tempair']
    #    ma = (tempair_ct>min_temp+273.15)
    #    print(f'Masking with temp>{min_temp} for min_cl_tempair!')
    #    ds = ds.where(ma)
    # elif 'min_cl_T' in ds.data_vars:
    #    tempair_ct = ds['min_cl_T']
    #    ma = (tempair_ct>min_temp+273.15)
    #    print(f'Masking with temp>{min_temp} for min_cl_T!')
    #    ds = ds.where(ma)
    if 'tempair_ct' in ds.data_vars:
        tempair_ct = ds['tempair_ct']
        ma = (tempair_ct > min_temp + 273.15)
        print(f'Masking with temp>{min_temp}!')
        ds = ds.where(ma)
    elif 'T_ct' in ds.data_vars:
        tempair_ct = ds['T_ct']
        ma = (tempair_ct > min_temp + 273.15)
        print(f'Masking with temp>{min_temp}!')
        ds = ds.where(ma)

    elif 'T_ct' in ds.data_vars:
        tempair_ct = ds['T_ct']
        ma = (tempair_ct > min_temp + 273.15)
        print(f'Masking with temp>{min_temp}!')
        ds = ds.where(ma)
    else:
        print('WARNING: tempair_ct NOT FOUND, not masking these values!')
    if False:  # 'clfr' in ds.data_vars:

        ma = ds['clfr_mask']  # .sel(lev=slice(23,38)).sum('lev')<10
        print(f'Masking where sum of cloud fraction above liquid clouds are below 10!')
        ds = ds.where(ma)
    else:
        print('WARNING: clfr NOT FOUND, not masking these values!')

    return ds


def ds2df_inc_preprocessing(dic_ds,
                            temperature=temperature_default,
                            pressure=pressure_default,
                            mask_summer=False,
                            air_density=None,
                            return_summer_median=True,
                            model_lev_i=-1,
                            select_model_layer=True,
                            select_hours_clouds=False,
                            do_broadcase_station_data=False,
                            varl_st=None, lat=lat_smr, lon=lon_smr,
                            from_hour=8,
                            to_hour=14,
                            mask_cloud_values=False,
                            varl_cl=None,
                            take_daily_median=True,
                            summer_months=None,
                            ):
    for ca in dic_ds.keys():

        print('hey')
        ds = dic_ds[ca]
        # print(ds)

        # print(ds)
        ds = compute_total_tau(ds)
        ds = compute_optical_thickness(ds)

        # print(ds)
        if select_model_layer:
            if 'lev' in ds.dims:
                print(model_lev_i)
                ds = ds.isel(lev=model_lev_i)

        ds = change_units_and_compute_vars(ds,
                                           air_density=air_density,
                                           temperature=temperature,
                                           pressure=pressure)
        if do_broadcase_station_data:
            ds = broadcase_station_data(ds, varl_st=varl_st, lat=lat, lon=lon)
        dic_ds[ca] = ds
    return get_dic_df_mod(dic_ds,
                          select_hours_clouds=select_hours_clouds,
                          from_hour=from_hour,
                          to_hour=to_hour,
                          varl_cl=varl_cl,
                          mask_summer=mask_summer,
                          mask_cloud_values=mask_cloud_values,
                          take_daily_median=take_daily_median,
                          return_summer_median=return_summer_median,
                          summer_months=summer_months)


def extract_2D_cloud_time_echam(ds, cloud_top_var='ceff_ct', cloud_var='ceff_um',
                                epsilon=1e-10):
    fill_value = 9999
    if ('ceff_um' not in ds) and (cloud_var == 'ceff_um'):
        ds['ceff_um'] = ds['ceff'] * 1e6
    ds['ceff_um'].attrs['units'] = 'um'

    ds[cloud_var] = ds[cloud_var].where(ds[cloud_var] > 0)
    ds[cloud_top_var] = ds[cloud_top_var].where(ds[cloud_top_var] > 0)

    ds['cl_time'].isel(lon=0, lat=0).plot(y='lev', ylim=[50, 0])
    plt.show()

    a_ct, b_ic = xr.broadcast(ds[cloud_top_var], ds[cloud_var], )
    diff = a_ct - b_ic

    diff.where(diff < 9999).isel(lat=2, lon=2).plot(y='lev', ylim=[50, 0], robust=True)
    plt.show()

    a_ct.plot(alpha=.3)
    b_ic.plot(alpha=.3)
    plt.legend()
    plt.show()

    diff = np.fabs(diff.fillna(fill_value))
    # print(diff.argmin(['lev']))
    diff.load()

    diff_argmin = diff.argmin(['lev'], skipna=True)

    diff_all_null = (diff == fill_value).all(dim='lev')

    if type(diff_argmin) is dict:
        diff_argmin = diff_argmin['lev']

    for _v in ['T', 'tempair']:
        if _v in ds:
            print(f'computing {_v}_incld')
            ds[f'{_v}_ct'] = ds[_v].isel(lev=diff_argmin).where(~diff_all_null)
            ds[f'min_cl_{_v}'] = ds[_v].where(ds['clfr'] > 0.9).min('lev')
    if ('clfr' in ds) and ('cl_time' in ds):
        # get clfr where no cloud in cl_time
        clfr_ma_cltm = ds.where((ds['clfr'] - ds['cl_time']) > 0)['clfr']
        lev_sum = clfr_ma_cltm.sum('lev')  # <10
        ds['clfr_lev_sum'] = lev_sum
        ds['clfr_mask'] = lev_sum < 10

    ds['cl_time_ct'] = ds['cl_time'].isel(lev=diff_argmin).where(~diff_all_null)
    ds['cl_time_ct'] = ds['cl_time_ct'].where(ds['cl_time_ct'] > epsilon)

    ds['diff_v'] = diff
    ds['diff_v_argminlev'] = diff.argmin('lev')

    if 'ceff_ct' in ds:
        ds['ceff_ct_incld'] = ds['ceff_ct'] / ds['cl_time_ct']
    if 'lcdnc_ct' in ds:
        ds['lcdnc_ct_incld'] = ds['lcdnc_ct'] / ds['cl_time_ct']

    if 'lcdnc_ct_cm3' in ds:
        ds['lcdnc_ct_cm3_incld'] = ds['lcdnc_ct_cm3'] / ds['cl_time_ct']
    ds['cl_time_max'] = ds['cl_time'].max('lev')
    ds['cl_clfr_max'] = ds['clfr'].max('lev')
    if 'cwp' in ds:
        ds['cwp_incld'] = ds['cwp'] / ds['cl_time_max']
    if 'cod' in ds:
        ds['cod_incld'] = ds['cod'] / ds['cl_time_max']

    return ds
