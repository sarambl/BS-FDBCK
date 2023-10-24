# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from bs_fdbck_clean.preprocess.launch_monthly_station_collocation_from_full_grid import launch_monthly_station_output_noresm
from bs_fdbck_clean.util.Nd.sizedist_class_v2.SizedistributionBins import SizedistributionStationBins
from bs_fdbck_clean.util.collocate.collocateLONLAToutput import CollocateLONLATout
from bs_fdbck_clean.data_info.variable_info import list_sized_vars_nonsec, list_sized_vars_noresm
import useful_scit.util.log as log
import time

log.ger.setLevel(log.log.INFO)








# %% [markdown]
# ## Settings:

# %%
# %%
from_t1   =    '2012-01-01'
to_t1     =    '2015-01-01'

from_t2   =    '2015-01-01'
to_t2     =    '2019-01-01'

history_field = '.h1.'

# %% [markdown]
# ## Cases:
case1 = 'OsloAero_intBVOC_f09_f09_mg17_full'
case2 = 'OsloAero_intBVOC_f09_f09_mg17_ssp245'


case2details_dic = {case1:{'from_t':from_t1, 'to_t':to_t1, 'is_sectional':False},
                    case2:{'from_t':from_t2, 'to_t':to_t2, 'is_sectional':False},
                    }

# %%
vars_extra_ns = ['AWNC', 'AREL', 'FREQL', 'ACTNL',
                 #'ACTNI',
                 'ACTREL', 'ACTREI', 'FCTL', 'FCTI', 'CDNUMC', 'CLDTOT', 'TOT_CLD_VISTAU', 'TOT_ICLD_VISTAU', 'TGCLDIWP', 'TGCLDLWP', 'TGCLDCWP', 'FSNT', 'FLNT', 'FSNT_DRF', 'FLNT_DRF', 'FSNTCDRF', 'FLNTCDRF',
                 #'FLNS',
                 'FSNS', 'FLNSC', 'FSNSC', 'FSDSCDRF', 'FSDS_DRF', 'FSUTADRF',
                 #'FLUTC',
                 'FSUS_DRF', 'FLUS', 'CLDFREE', 'AOD_VIS', 'DOD440', 'DOD500', 'DOD550',
                 #'DOD670',
                 'DOD870', 'ABSVIS', 'AEROD_v', 'OD550DRY', 'RHW', 'MMR_AH2O', 'EC550AER', 'DAERH2O', 'D500_POM', 'D500_SO4', 'D500_BC', 'D500_DU', 'D500_SS', 'DER', 'DERGT05', 'DERLT05', 'BETOTVIS', 'ASYMMDRY', 'BS550AER', 'BETOTVIS', 'PMTOT', 'PM2P5', 'PM25', 'SO4_NA', 'SO4_A1', 'SO4_A2', 'SO4_AC', 'SO4_PR', 'SOA_NA', 'SOA_A1', 'BC_N', 'BC_AX', 'BC_NI', 'BC_A', 'BC_AI', 'BC_AC', 'OM_NI', 'OM_AI', 'OM_AC', 'DST_A2', 'DST_A3', 'SS_A1', 'SS_A2', 'SS_A3', 'SOA_LV', 'SOA_SV', 'H2SO4', 'SO2', 'monoterp', 'isoprene', 'DMS', 'SFisoprene', 'SFmonoterp', 'NUCLRATE', 'FORMRATE', 'GRH2SO4', 'GRSOA', 'GR', 'COAGNUCL', 'CCN1', 'CCN2', 'CCN3', 'CCN4', 'CCN5', 'CCN6', 'CCN7', 'CCN_B', 'NNAT_0', 'NCONC01', 'NCONC02', 'NCONC03', 'NCONC04', 'NCONC05', 'NCONC06', 'NCONC07', 'NCONC08', 'NCONC09', 'NCONC10', 'NCONC11', 'NCONC12', 'NCONC13', 'NCONC14', 'SIGMA01', 'SIGMA02', 'SIGMA03', 'SIGMA04', 'SIGMA05', 'SIGMA06', 'SIGMA07', 'SIGMA08', 'SIGMA09', 'SIGMA10', 'SIGMA11', 'SIGMA12', 'SIGMA13', 'SIGMA14', 'NMR01', 'NMR02', 'NMR03', 'NMR04', 'NMR05', 'NMR06', 'NMR07', 'NMR08', 'NMR09', 'NMR10', 'NMR11', 'NMR12', 'NMR13', 'NMR14', 'T', 'PS', 'U', 'V', 'GRIDAREA']

#'FREQI',

# %%'cosp_reffice','cosp_reffliq','cosp_tau_modis','cosp_tau',

log.ger.info(f'TIMES:****: {from_t} {to_t}')

# %% [markdown]
# ## launches subprocesses that compute monthly

# %%
for case_name in case2details_dic.keys():
    from_t = case2details_dic[case_name]['from_t']
    to_t = case2details_dic[case_name]['to_t']
    is_sectional =case2details_dic[case_name]['is_sectional']
    launch_monthly_station_output_noresm(case_name, is_sectional, from_time=from_t, to_time=to_t)
    # %% [markdown]
    # ## Merge monthly



    print('DONE WITH MONTHLY FIELDS!!!!')

    # %%
    varlist = list_sized_vars_nonsec + vars_extra_ns # list_sized_vars_noresm
    c = CollocateLONLATout(case_name, from_t, to_t,
                           is_sectional,
                           'hour',
                           space_res='locations',
                           history_field=history_field)
    if c.check_if_load_raw_necessary(varlist):
        time1 = time.time()
        print('Running merge monthly: ')
        a = c.make_station_data_merge_monthly(varlist)
        print(a)

        time2 = time.time()
        print('DONE : took {:.3f} s'.format((time2 - time1)))
    else:
        print('UPS')

    # %%

    # %% [markdown]
    #
    # ## Compute binned dataset

    # %% [markdown]
    # ### Make station N50 etc.

    # %%


    s = SizedistributionStationBins(case_name, from_t, to_t, None, False, 'hour',
                                        space_res='full',
                                        nr_bins=None, history_field=history_field)
    s.compute_Nd_vars()

