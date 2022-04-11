# %%

from useful_scit.imps import *

from bs_fdbck.data_info.variable_info import list_sized_vars_noresm
from bs_fdbck.util.imports import get_averaged_fields
from bs_fdbck.util.imports.get_fld_fixed import get_field_fixed

log.ger.setLevel(log.log.DEBUG)

# load and autoreload
from IPython import get_ipython

# noinspection PyBroadException
try:
    _ipython = get_ipython()
    _magic = _ipython.magic
    _magic('load_ext autoreload')
    _magic('autoreload 2')
except:
    pass
# %%
model = 'NorESM'

startyear = '2012-01'
endyear = '2012-12'
#startyear = '0003-01'
#endyear = '0003-12'
history_field = '.h1.'
#startyear = '0004-01'
#endyear = '0008-12'

p_level=1013.
pmin = 850.  # minimum pressure level
avg_over_lev = True  # True # True#False#True
pressure_adjust = True  # Can only be false if avg_over_lev false. Plots particular hybrid sigma lev
p_levels = [1013.,900., 800., 700., 600.]  # used if not avg
# %%
cases_sec = [
]
cases_orig = [

    'OsloAero_intBVOC_f19_f19_mg17_full',

        ]

cases = cases_sec + cases_orig
# %%

vars_extra_ns = ['AWNC', 'FREQL', 'AREL', 'ACTNL', 'FCTL', 'ACTREI', 'FCTI', 'ACTREL', 'CDNUMC', 'CLDTOT', 'TOT_CLD_VISTAU', 'TOT_ICLD_VISTAU', 'TGCLDIWP', 'TGCLDLWP', 'TGCLDCWP',  'FSNTCDRF', 'FSNT_DRF', 'FSNT', 'FLNT_DRF', 'FLNT', 'FSNT_DRF', 'FSNT', 'FLNT_DRF', 'FLNT', 'FSNT_DRF', 'FSNTCDRF', 'FLNT_DRF', 'FLNTCDRF', 'FSNT_DRF', 'FSNTCDRF', 'FLNT_DRF', 'FLNTCDRF', 'FLUS', 'FSDSCDRF', 'FSDS_DRF', 'FSUS_DRF', 'FSUTADRF', 'FSNS', 'FLNSC', 'FSNSC', 'PMTOT', 'PM2P5', 'PM25', 'SO4_NA', 'SO4_A1', 'SO4_A2', 'SO4_AC', 'SO4_PR', 'SOA_NA', 'SOA_A1', 'BC_N', 'BC_AX', 'BC_NI', 'BC_A', 'BC_AI', 'BC_AC', 'OM_NI', 'OM_AI', 'OM_AC', 'DST_A2', 'DST_A3', 'SS_A1', 'SS_A2', 'SS_A3', 'SOA_LV', 'SOA_SV', 'H2SO4', 'SO2', 'monoterp', 'isoprene', 'DMS', 'NUCLRATE', 'FORMRATE', 'GRH2SO4', 'GRSOA', 'GR', 'COAGNUCL', 'SFisoprene', 'SFmonoterp', 'CCN1', 'CCN2', 'CCN3', 'CCN4', 'CCN5', 'CCN6', 'CCN7', 'CCN_B',  'MEANTAU_ISCCP', 'TAUTMODIS', 'REFFCLWMODIS', 'CLHMODIS', 'CLIMODIS', 'CLLMODIS', 'CLMMODIS', 'CLMODIS', 'CLRIMODIS', 'CLRLMODIS', 'CLTMODIS', 'CLWMODIS', 'IWPMODIS', 'LWPMODIS', 'PCTMODIS', 'REFFCLIMODIS', 'REFFCLWMODIS', 'TAUILOGMODIS', 'TAUIMODIS', 'TAUTLOGMODIS', 'TAUTMODIS', 'TAUWLOGMODIS', 'TAUWMODIS', 'NCONC01', 'NCONC02', 'NCONC03', 'NCONC04', 'NCONC05', 'NCONC06', 'NCONC07', 'NCONC08', 'NCONC09', 'NCONC10', 'NCONC11', 'NCONC12', 'NCONC13', 'NCONC14', 'SIGMA01', 'SIGMA02', 'SIGMA03', 'SIGMA04', 'SIGMA05', 'SIGMA06', 'SIGMA07', 'SIGMA08', 'SIGMA09', 'SIGMA10', 'SIGMA11', 'SIGMA12', 'SIGMA13', 'SIGMA14', 'NNAT_0', 'NMR01', 'NMR02', 'NMR03', 'NMR04', 'NMR05', 'NMR06', 'NMR07', 'NMR08', 'NMR09', 'NMR10', 'NMR11', 'NMR12', 'NMR13', 'NMR14', 'AOD_VIS', 'ABSVIS', 'CABSVIS', 'CAODVIS', 'CLDFREE', 'CDOD550', 'CDOD440', 'CDOD870','DOD500',  'DOD550', 'DOD440', 'DOD870', 'AEROD_v', 'CABS550', 'CABS550A', 'OD550DRY', 'RHW', 'MMR_AH2O', 'MMRPM2P5', 'GRIDAREA', 'EC550AER', 'DAERH2O', 'D500_POM', 'DER', 'DERGT05', 'DERLT05', 'D500_BC', 'D500_DU', 'D500_POM', 'D500_SO4', 'D500_SS', 'D550_BC', 'D550_DU', 'D550_POM', 'D550_SO4', 'D550_SS', 'BETOTVIS', 'ASYMMDRY', 'AB550DRY', 'ABS440', 'ABS500', 'ABS550', 'ABSDRYOC', 'ABSDRYSS', 'BS550AER', 'BETOTVIS', 'T', 'PS', 'U', 'V', ]


varl = list_sized_vars_noresm + vars_extra_ns

varl_sec = varl

# %%
for case in cases:
    get_field_fixed(case,varl, startyear, endyear, #raw_data_path=constants.get_input_datapath(),
                pressure_adjust=True, model = 'NorESM', history_fld=history_field, comp='atm', chunks=None)
for case in cases_sec:
    get_field_fixed(case, varl, startyear, endyear, #raw_data_path=constants.get_input_datapath(),
                        pressure_adjust=True, model = 'NorESM', history_fld=history_field, comp='atm', chunks=None)
    # %%
maps_dic = get_averaged_fields.get_maps_cases(cases,varl,startyear, endyear,
                                              avg_over_lev=avg_over_lev,
                                              pmin=pmin,
                                              pressure_adjust=pressure_adjust, p_level=p_level)
maps_dic = get_averaged_fields.get_maps_cases(cases_sec,varl,startyear, endyear,
                                              avg_over_lev=avg_over_lev,
                                              pmin=pmin,
                                              pressure_adjust=pressure_adjust, p_level=p_level)


for period in ['JJA','DJF']:
    maps_dic = get_averaged_fields.get_maps_cases(cases,varl,startyear, endyear,
                                              avg_over_lev=avg_over_lev,
                                              pmin=pmin,
                                              pressure_adjust=pressure_adjust,
                                              p_level=p_level,
                                              time_mask=period)
    maps_dic = get_averaged_fields.get_maps_cases(cases_sec,varl_sec,startyear, endyear,
                                              avg_over_lev=avg_over_lev,
                                              pmin=pmin,
                                              pressure_adjust=pressure_adjust, p_level=p_level,
                                              time_mask=period)
