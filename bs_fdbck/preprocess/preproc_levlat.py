# %%

from useful_scit.imps import *

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

#startyear = '2008-01'
#endyear = '2008-12'
#startyear = '0003-01'
#endyear = '0003-12'

startyear = '0003-01'
endyear = '0004-12'

p_level=1013.
pmin = 850.  # minimum pressure level
avg_over_lev = True  # True # True#False#True
pressure_adjust = True  # Can only be false if avg_over_lev false. Plots particular hybrid sigma lev
p_levels = [1013.,900., 800., 700., 600.]  # used if not avg
# %%
cases_sec = [
    #'NF1850_SECT_ctrl',
    #'NF1850_SECT_gord',
    #'NF1850_aeroxid2014_SECT_ctrl',

    #'NF1850_aero2014_SECT_ctrl',
    #'NF1850_aeroxid2014_SECT_gord',
    #'NF1850_aeroxid2014_fbvoc_SECT_ctrl',
    #'NF1850_fbvoc_SECT_ctrl'
]
#,'SECTv21_decY','SECTv21_incY']
#cases_sec = [
#    'SECTv21_mind3',
#    'SECTv21_nb3',
#    'SECTv21_ctrl_koagD',
#    'SECTv21_decY',
#    'SECTv21_incY',
#    'SECTv21_decSO2',
#    'SECTv21_incSO2',
#     'SECTv21_decNuc',
#     'SECTv21_incNuc',
#]
cases_orig = [
    #'NF1850_noSECT_def',
    #'NF1850_aeroxid2014_noSECT_def',
    #'NF1850_noSECT_ox_ricc',
    #'NF1850_aeroxid2014_noSECT_ox_ricc',
    'NF1850_noSECT_def_smax',
    #'NF1850_aero2014_noSECT_ox_ricc',
    #'NF1850_aeroxid2014_fbvoc_noSECT_ox_ricc'
    #'NF1850_fbvoc_noSECT_ox_ricc'

    #'NF1850_aeroxid2014_fbvoc_noSECT_ox_ricc',
    #'NF1850_fbvoc_noSECT_ox_ricc',
    #'NF1850_noSECT_ox_gord',
    #'NF1850_aeroxid2014_noSECT_ox_ricc_test_gordon',
    #'NF1850_aeroxid2014_LU2000_noSECT_ox_ricc'

    ]
#cases_orig = [
##    'noSECTv21_default_dd',
##    'noSECTv21_ox_ricc_dd',
##    'noSECTv21_ox_ricc_decY',
##    'noSECTv21_ox_ricc_incY',
##    'noSECTv21_def_incY',
##    'noSECTv21_def_decY',
#    'noSECTv21_def_decSO2',
#    'noSECTv21_def_incSO2',
#    'noSECTv21_ox_ricc_decSO2',
#    'noSECTv21_ox_ricc_incSO2',
#    'noSECTv21_ox_ricc_decNuc',
#    'noSECTv21_ox_ricc_incNuc',
#    'noSECTv21_def_decNuc',
#    'noSECTv21_def_incNuc',
#
#    ]

cases = cases_sec + cases_orig
# %%
varl = [
    'Smax','NDROPSRC','NDROPMIX','NDROPCOL','WTKE','Smax_supZero','Smax_w'
    'NACT_FRAC01','NACT_FRAC02','NACT_FRAC03','NACT_FRAC04','NACT_FRAC05','NACT_FRAC06','NACT_FRAC07','NACT_FRAC08','NACT_FRAC09','NACT_FRAC10','NACT_FRAC11','NACT_FRAC12','NACT_FRAC13','NACT_FRAC14',
    'HYGRO01',
    'SO2','DMS','isoprene','monoterp',
    'N_AER','NCONC01', 'NMR01','GR','NUCLRATE','FORMRATE',
    'H2SO4','SOA_LV','SOA_SV','SOA_NA','SO4_NA','SOA_A1',
]
varl_sec = ['nrSOA_SEC_tot', 'nrSO4_SEC_tot','nrSEC_tot']


# %%
for case in cases:
    get_field_fixed(case,varl, startyear, endyear, #raw_data_path=constants.get_input_datapath(),
                pressure_adjust=True, model = 'NorESM', history_fld='.h0.', comp='atm', chunks=None)
for case in cases_sec:
    get_field_fixed(case, varl_sec, startyear, endyear, #raw_data_path=constants.get_input_datapath(),
                        pressure_adjust=True, model = 'NorESM', history_fld='.h0.', comp='atm', chunks=None)
    # %%
levlat_dic = get_averaged_fields.get_levlat_cases(cases,
                                                varl,
                                                startyear,
                                                endyear,
                                                pressure_adjust = pressure_adjust,
                                                )
levlat_dic = get_averaged_fields.get_levlat_cases(cases_sec,
                                                varl_sec,
                                                startyear,
                                                endyear,
                                                pressure_adjust=pressure_adjust,
                                                )


