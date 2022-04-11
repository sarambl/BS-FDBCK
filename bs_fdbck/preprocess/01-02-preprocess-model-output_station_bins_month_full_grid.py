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
from bs_fdbck.preprocess.launch_monthly_station_collocation_from_full_grid import launch_monthly_station_output
from bs_fdbck.util.Nd.sizedist_class_v2.SizedistributionBins import SizedistributionStationBins
from bs_fdbck.util.collocate.collocateLONLAToutput import CollocateLONLATout
from bs_fdbck.data_info.variable_info import list_sized_vars_nonsec, list_sized_vars_noresm
import useful_scit.util.log as log
import time

log.ger.setLevel(log.log.INFO)

# %% [markdown]
# ## Settings:

# %%
nr_of_bins = 5
maxDiameter = 39.6
minDiameter = 5.0  #
history_field = '.h1.'

# %%
from_t = '2012-01-01'
to_t = '2015-01-01'

#from_t = '2015-01-01'
#to_t = '2019-01-01'


# %% [markdown]
# ## Cases:

# %%
cases_sec = []#'OsloAeroSec_intBVOC_f19_f19',]
cases_orig = [
    ##'OsloAero_intBVOC_f19_f19_mg17_ssp245',
    #'OsloAero_intBVOC_pertSizeDist2_f19_f19_mg17_full',
    'OsloAero_intBVOC_pertSizeDist_f19_f19_mg17_full',
    #'OsloAero_intBVOC_f19_f19_mg17_full',
    #'OsloAero_intBVOC_f19_f19_mg17_incY_full',

]
# %%

vars_extra_ns = ['SOA_A1','FLNSC','FSNS','FSNSC','FSNT','FSNT_DRF','FSNT','FLNT_DRF','FLNT','FREQL','FSNTCDRF','TOT_CLD_VISTAU','TOT_ICLD_VISTAU','TGCLDIWP','TGCLDLWP','ACTREI','ACTREL','AREL','ACTNL','DOD440','DOD500','DOD550','DOD670','DOD870',
                 #'MEANTAU_ISCCP',
                 'CABSVIS','ABSVIS','D550_SS','D550_SO4','D550_POM','D550_DU','D550_BC','NNAT_0','PS','COAGNUCL','FORMRATE','NUCLRATE','SOA_LV','H2SO4','SOA_NA','SO4_NA','NCONC01','NCONC02','NCONC03','NCONC04','NCONC05','NCONC06','NCONC07','NCONC08','NCONC09','NCONC10','NCONC11','NCONC12','NCONC13','NCONC14','SIGMA01','SIGMA02','SIGMA03','SIGMA04','SIGMA05','SIGMA06','SIGMA07','SIGMA08','SIGMA09','SIGMA10','SIGMA11','SIGMA12','SIGMA13','SIGMA14','NMR01','NMR02','NMR03','NMR04','NMR05','NMR06','NMR07','NMR08','NMR09','NMR10','NMR11','NMR12','NMR13','NMR14', #'SOA_NAcoagTend', 'SO4_NAcoagTend', 'SOA_NAcondTend', 'SO4_NAcondTend', 'SOA_A1condTend','SO4_A1condTend',
                 'FSNS','FSDS_DRF','T','GR','GRH2SO4','GRSOA','SO4_NAclcoagTend',#'SO4_NAcoagTend',
                 'CCN1','CCN2','CCN3','CCN4','CCN5','CCN6','CCN7','CCN_B', 'TGCLDCWP','U','V','cb_H2SO4','cb_SOA_LV','cb_SOA_NA','cb_SO4_NA','CLDTOT','CDNUMC', 'OH','SO2','isoprene','monoterp','SOA_SV',
                 #'OH_vmr','O3_vmr','NO3_vmr','GS_SO2', 'GS_H2SO4','GS_monoterp','GS_isoprene',
                 'AOD_VIS','CAODVIS','CLDFREE','CDOD550','CDOD440','CDOD870','AEROD_v','CABS550','CABS550A']
vars_extra_ns = ['AWNC', 'FREQL', 'AREL', 'ACTNL', 'FCTL', 'ACTREI', 'FCTI', 'ACTREL', 'CDNUMC', 'CLDTOT', 'TOT_CLD_VISTAU', 'TOT_ICLD_VISTAU', 'TGCLDIWP', 'TGCLDLWP', 'TGCLDCWP',  'FSNTCDRF', 'FSNT_DRF', 'FSNT', 'FLNT_DRF', 'FLNT', 'FSNT_DRF', 'FSNT', 'FLNT_DRF', 'FLNT', 'FSNT_DRF', 'FSNTCDRF', 'FLNT_DRF', 'FLNTCDRF', 'FSNT_DRF', 'FSNTCDRF', 'FLNT_DRF', 'FLNTCDRF', 'FLUS', 'FSDSCDRF', 'FSDS_DRF', 'FSUS_DRF', 'FSUTADRF', 'FSNS', 'FLNSC', 'FSNSC', 'PMTOT', 'PM2P5', 'PM25', 'SO4_NA', 'SO4_A1', 'SO4_A2', 'SO4_AC', 'SO4_PR', 'SOA_NA', 'SOA_A1', 'BC_N', 'BC_AX', 'BC_NI', 'BC_A', 'BC_AI', 'BC_AC', 'OM_NI', 'OM_AI', 'OM_AC', 'DST_A2', 'DST_A3', 'SS_A1', 'SS_A2', 'SS_A3', 'SOA_LV', 'SOA_SV', 'H2SO4', 'SO2', 'monoterp', 'isoprene', 'DMS', 'NUCLRATE', 'FORMRATE', 'GRH2SO4', 'GRSOA', 'GR', 'COAGNUCL', 'SFisoprene', 'SFmonoterp', 'CCN1', 'CCN2', 'CCN3', 'CCN4', 'CCN5', 'CCN6', 'CCN7', 'CCN_B',  'MEANTAU_ISCCP', 'TAUTMODIS', 'REFFCLWMODIS', 'CLHMODIS', 'CLIMODIS', 'CLLMODIS', 'CLMMODIS', 'CLMODIS', 'CLRIMODIS', 'CLRLMODIS', 'CLTMODIS', 'CLWMODIS', 'IWPMODIS', 'LWPMODIS', 'PCTMODIS', 'REFFCLIMODIS', 'REFFCLWMODIS', 'TAUILOGMODIS', 'TAUIMODIS', 'TAUTLOGMODIS', 'TAUTMODIS', 'TAUWLOGMODIS', 'TAUWMODIS', 'NCONC01', 'NCONC02', 'NCONC03', 'NCONC04', 'NCONC05', 'NCONC06', 'NCONC07', 'NCONC08', 'NCONC09', 'NCONC10', 'NCONC11', 'NCONC12', 'NCONC13', 'NCONC14', 'SIGMA01', 'SIGMA02', 'SIGMA03', 'SIGMA04', 'SIGMA05', 'SIGMA06', 'SIGMA07', 'SIGMA08', 'SIGMA09', 'SIGMA10', 'SIGMA11', 'SIGMA12', 'SIGMA13', 'SIGMA14', 'NNAT_0', 'NMR01', 'NMR02', 'NMR03', 'NMR04', 'NMR05', 'NMR06', 'NMR07', 'NMR08', 'NMR09', 'NMR10', 'NMR11', 'NMR12', 'NMR13', 'NMR14', 'AOD_VIS', 'ABSVIS', 'CABSVIS', 'CAODVIS', 'CLDFREE', 'CDOD550', 'CDOD440', 'CDOD870','DOD500',  'DOD550', 'DOD440', 'DOD870', 'AEROD_v', 'CABS550', 'CABS550A', 'OD550DRY', 'RHW', 'MMR_AH2O', 'MMRPM2P5', 'GRIDAREA', 'EC550AER', 'DAERH2O', 'D500_POM', 'DER', 'DERGT05', 'DERLT05', 'D500_BC', 'D500_DU', 'D500_POM', 'D500_SO4', 'D500_SS', 'D550_BC', 'D550_DU', 'D550_POM', 'D550_SO4', 'D550_SS', 'BETOTVIS', 'ASYMMDRY', 'AB550DRY', 'ABS440', 'ABS500', 'ABS550', 'ABSDRYOC', 'ABSDRYSS', 'BS550AER', 'BETOTVIS', 'T', 'PS', 'U', 'V', ]
#                 'nrSO4_SEC01', 'nrSO4_SEC02', 'nrSO4_SEC03', 'nrSO4_SEC04', 'nrSO4_SEC05', 'nrSOA_SEC01', 'nrSOA_SEC02', 'nrSOA_SEC03', 'nrSOA_SEC04', 'nrSOA_SEC05', 'SO4_SEC01', 'SO4_SEC02', 'SO4_SEC03', 'SO4_SEC04', 'SO4_SEC05', 'SOA_SEC01', 'SOA_SEC02', 'SOA_SEC03', 'SOA_SEC04', 'SOA_SEC05'
#                 ]
vars_extra_ns = ['AWNC', 'FREQL', 'AREL', 'ACTNL', 'FCTL', 'ACTREI', 'FCTI', 'ACTREL', 'CDNUMC', 'CLDTOT', 'TOT_CLD_VISTAU', 'TOT_ICLD_VISTAU', 'TGCLDIWP', 'TGCLDLWP', 'TGCLDCWP',  'FSNTCDRF', 'FSNT_DRF', 'FSNT', 'FLNT_DRF', 'FLNT', 'FSNT_DRF', 'FSNT', 'FLNT_DRF', 'FLNT', 'FSNT_DRF', 'FSNTCDRF', 'FLNT_DRF', 'FLNTCDRF', 'FSNT_DRF', 'FSNTCDRF', 'FLNT_DRF', 'FLNTCDRF', 'FLUS', 'FSDSCDRF', 'FSDS_DRF', 'FSUS_DRF', 'FSUTADRF', 'FSNS', 'FLNSC', 'FSNSC', 'PMTOT', 'PM2P5', 'PM25', 'SO4_NA', 'SO4_A1', 'SO4_A2', 'SO4_AC', 'SO4_PR', 'SOA_NA', 'SOA_A1', 'BC_N', 'BC_AX', 'BC_NI', 'BC_A', 'BC_AI', 'BC_AC', 'OM_NI', 'OM_AI', 'OM_AC', 'DST_A2', 'DST_A3', 'SS_A1', 'SS_A2', 'SS_A3', 'SOA_LV', 'SOA_SV', 'H2SO4', 'SO2', 'monoterp', 'isoprene', 'DMS', 'NUCLRATE', 'FORMRATE', 'GRH2SO4', 'GRSOA', 'GR', 'COAGNUCL', 'SFisoprene', 'SFmonoterp', 'CCN1', 'CCN2', 'CCN3', 'CCN4', 'CCN5', 'CCN6', 'CCN7', 'CCN_B',  'MEANTAU_ISCCP', 'TAUILOGMODIS', 'TAUIMODIS', 'TAUTLOGMODIS', 'TAUTMODIS', 'TAUWLOGMODIS', 'TAUWMODIS', 'NCONC01', 'NCONC02', 'NCONC03', 'NCONC04', 'NCONC05', 'NCONC06', 'NCONC07', 'NCONC08', 'NCONC09', 'NCONC10', 'NCONC11', 'NCONC12', 'NCONC13', 'NCONC14', 'SIGMA01', 'SIGMA02', 'SIGMA03', 'SIGMA04', 'SIGMA05', 'SIGMA06', 'SIGMA07', 'SIGMA08', 'SIGMA09', 'SIGMA10', 'SIGMA11', 'SIGMA12', 'SIGMA13', 'SIGMA14', 'NNAT_0', 'NMR01', 'NMR02', 'NMR03', 'NMR04', 'NMR05', 'NMR06', 'NMR07', 'NMR08', 'NMR09', 'NMR10', 'NMR11', 'NMR12', 'NMR13', 'NMR14', 'AOD_VIS', 'ABSVIS', 'CABSVIS', 'CAODVIS', 'CLDFREE', 'CDOD550', 'CDOD440', 'CDOD870','DOD500',  'DOD550', 'DOD440', 'DOD870', 'AEROD_v', 'CABS550', 'CABS550A', 'OD550DRY', 'RHW', 'MMR_AH2O', 'MMRPM2P5', 'GRIDAREA', 'EC550AER', 'DAERH2O', 'D500_POM', 'DER', 'DERGT05', 'DERLT05', 'D500_BC', 'D500_DU', 'D500_POM', 'D500_SO4', 'D500_SS', 'D550_BC', 'D550_DU', 'D550_POM', 'D550_SO4', 'D550_SS', 'BETOTVIS', 'ASYMMDRY', 'AB550DRY', 'ABS440', 'ABS500', 'ABS550', 'ABSDRYOC', 'ABSDRYSS', 'BS550AER', 'BETOTVIS', 'T', 'PS', 'U', 'V', ]

vars_extra_ns = ['AWNC', 'AREL', 'FREQL', 'ACTNL',
                 'ACTNI',
                 'ACTREL', 'ACTREI', 'FCTL', 'FCTI', 'CDNUMC', 'CLDTOT', 'TOT_CLD_VISTAU', 'TOT_ICLD_VISTAU', 'TGCLDIWP', 'TGCLDLWP', 'TGCLDCWP', 'FSNT', 'FLNT', 'FSNT_DRF', 'FLNT_DRF', 'FSNTCDRF', 'FLNTCDRF',
                 'FLNS',
                 'FSNS', 'FLNSC', 'FSNSC', 'FSDSCDRF', 'FSDS_DRF', 'FSUTADRF',
                 'FLUTC',
                 'FSUS_DRF', 'FLUS', 'CLDFREE', 'AOD_VIS', 'DOD440', 'DOD500', 'DOD550',
                 'DOD670',
                 'DOD870', 'ABSVIS', 'AEROD_v', 'OD550DRY', 'RHW', 'MMR_AH2O', 'EC550AER', 'DAERH2O', 'D500_POM', 'D500_SO4', 'D500_BC', 'D500_DU', 'D500_SS', 'DER', 'DERGT05', 'DERLT05', 'BETOTVIS', 'ASYMMDRY', 'BS550AER', 'BETOTVIS', 'PMTOT', 'PM2P5', 'PM25', 'SO4_NA', 'SO4_A1', 'SO4_A2', 'SO4_AC', 'SO4_PR', 'SOA_NA', 'SOA_A1', 'BC_N', 'BC_AX', 'BC_NI', 'BC_A', 'BC_AI', 'BC_AC', 'OM_NI', 'OM_AI', 'OM_AC', 'DST_A2', 'DST_A3', 'SS_A1', 'SS_A2', 'SS_A3', 'SOA_LV', 'SOA_SV', 'H2SO4', 'SO2', 'monoterp', 'isoprene', 'DMS', 'SFisoprene', 'SFmonoterp', 'NUCLRATE', 'FORMRATE', 'GRH2SO4', 'GRSOA', 'GR', 'COAGNUCL', 'CCN1', 'CCN2', 'CCN3', 'CCN4', 'CCN5', 'CCN6', 'CCN7', 'CCN_B', 'NNAT_0', 'NCONC01', 'NCONC02', 'NCONC03', 'NCONC04', 'NCONC05', 'NCONC06', 'NCONC07', 'NCONC08', 'NCONC09', 'NCONC10', 'NCONC11', 'NCONC12', 'NCONC13', 'NCONC14', 'SIGMA01', 'SIGMA02', 'SIGMA03', 'SIGMA04', 'SIGMA05', 'SIGMA06', 'SIGMA07', 'SIGMA08', 'SIGMA09', 'SIGMA10', 'SIGMA11', 'SIGMA12', 'SIGMA13', 'SIGMA14', 'NMR01', 'NMR02', 'NMR03', 'NMR04', 'NMR05', 'NMR06', 'NMR07', 'NMR08', 'NMR09', 'NMR10', 'NMR11', 'NMR12', 'NMR13', 'NMR14', 'T', 'PS', 'U', 'V', 'GRIDAREA']

#'FREQI',

# %%'cosp_reffice','cosp_reffliq','cosp_tau_modis','cosp_tau',

log.ger.info(f'TIMES:****: {from_t} {to_t}')

# %% [markdown]
# ## launches subprocesses that compute monthly

# %%
for case_name in cases_sec:
    #continue
    launch_monthly_station_output(case_name, True, from_time=from_t, to_time=to_t)
for case_name in cases_orig:
    #continue
    launch_monthly_station_output(case_name, False, from_time=from_t, to_time=to_t)
# %% [markdown]
# ## Merge monthly



print('DONE WITH MONTHLY FIELDS!!!!')

# %%
for case_name in cases_sec:
    varlist = list_sized_vars_noresm + vars_extra_ns
    c = CollocateLONLATout(case_name, from_t, to_t,
                           True,
                           'hour',
                           space_res='locations',

                           history_field=history_field)
    if c.check_if_load_raw_necessary(varlist):
        time1 = time.time()
        a = c.make_station_data_merge_monthly(varlist)
        print(a)

        time2 = time.time()
        print('DONE : took {:.3f} s'.format((time2 - time1)))
    else:
        print('UPS')
for case_name in cases_orig:
    varlist = list_sized_vars_nonsec + vars_extra_ns # list_sized_vars_noresm
    c = CollocateLONLATout(case_name, from_t, to_t,
                           False,
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
for case_name in cases_sec:
    s = SizedistributionStationBins(case_name, from_t, to_t, [minDiameter, maxDiameter], True, 'hour',
                                    space_res='full',
                                    nr_bins=nr_of_bins, history_field=history_field)
    s.compute_Nd_vars()

for case_name in cases_orig:
    s = SizedistributionStationBins(case_name, from_t, to_t, [minDiameter, maxDiameter], False, 'hour',
                                    space_res='full',
                                    nr_bins=nr_of_bins, history_field=history_field)
    s.compute_Nd_vars()

