# %%
import sys

import useful_scit.util.log as log

from ukesm_bs_fdbck.data_info.variable_info import list_sized_vars_nonsec, list_sized_vars_noresm
from ukesm_bs_fdbck.util.imports.get_fld_fixed import get_field_fixed

log.ger.setLevel(log.log.INFO)

# %% [markdown]
# ### Settings

vars_extra_ns = ['SOA_A1','FLNSC','FSNS','FSNSC','FSNT','FSNT_DRF','FSNT','FLNT_DRF','FLNT','FREQL','FSNTCDRF','TOT_CLD_VISTAU','TOT_ICLD_VISTAU','TGCLDIWP','TGCLDLWP','ACTREI','ACTREL','AREL','ACTNL','DOD440','DOD500','DOD550','DOD670','DOD870','MEANTAU_ISCCP','CABSVIS','ABSVIS','D550_SS','D550_SO4','D550_POM','D550_DU','D550_BC','NNAT_0','PS','COAGNUCL','FORMRATE','NUCLRATE','SOA_LV','H2SO4','SOA_NA','SO4_NA','NCONC01','NCONC02','NCONC03','NCONC04','NCONC05','NCONC06','NCONC07','NCONC08','NCONC09','NCONC10','NCONC11','NCONC12','NCONC13','NCONC14','SIGMA01','SIGMA02','SIGMA03','SIGMA04','SIGMA05','SIGMA06','SIGMA07','SIGMA08','SIGMA09','SIGMA10','SIGMA11','SIGMA12','SIGMA13','SIGMA14','NMR01','NMR02','NMR03','NMR04','NMR05','NMR06','NMR07','NMR08','NMR09','NMR10','NMR11','NMR12','NMR13','NMR14', #'SOA_NAcoagTend', 'SO4_NAcoagTend', 'SOA_NAcondTend', 'SO4_NAcondTend', 'SOA_A1condTend','SO4_A1condTend',
                 'FSNS','FSDS_DRF','T','GR','GRH2SO4','GRSOA','SO4_NAclcoagTend',#'SO4_NAcoagTend',
                 'CCN1','CCN2','CCN3','CCN4','CCN5','CCN6','CCN7','CCN_B', 'TGCLDCWP','U','V','cb_H2SO4','cb_SOA_LV','cb_SOA_NA','cb_SO4_NA','CLDTOT','CDNUMC', 'OH','SO2','isoprene','monoterp','SOA_SV',
                 #'OH_vmr','O3_vmr','NO3_vmr','GS_SO2', 'GS_H2SO4','GS_monoterp','GS_isoprene',
                 'AOD_VIS','CAODVIS','CLDFREE','CDOD550','CDOD440','CDOD870','AEROD_v','CABS550','CABS550A']
vars_extra_ns = ['AWNC', 'FREQL', 'AREL', 'ACTNL', 'FCTL', 'ACTREI', 'FCTI', 'ACTREL', 'CDNUMC', 'CLDTOT', 'TOT_CLD_VISTAU', 'TOT_ICLD_VISTAU', 'TGCLDIWP', 'TGCLDLWP', 'TGCLDCWP',  'FSNTCDRF', 'FSNT_DRF', 'FSNT', 'FLNT_DRF', 'FLNT', 'FSNT_DRF', 'FSNT', 'FLNT_DRF', 'FLNT', 'FSNT_DRF', 'FSNTCDRF', 'FLNT_DRF', 'FLNTCDRF', 'FSNT_DRF', 'FSNTCDRF', 'FLNT_DRF', 'FLNTCDRF', 'FLUS', 'FSDSCDRF', 'FSDS_DRF', 'FSUS_DRF', 'FSUTADRF', 'FSNS', 'FLNSC', 'FSNSC', 'PMTOT', 'PM2P5', 'PM25', 'SO4_NA', 'SO4_A1', 'SO4_A2', 'SO4_AC', 'SO4_PR', 'SOA_NA', 'SOA_A1', 'BC_N', 'BC_AX', 'BC_NI', 'BC_A', 'BC_AI', 'BC_AC', 'OM_NI', 'OM_AI', 'OM_AC', 'DST_A2', 'DST_A3', 'SS_A1', 'SS_A2', 'SS_A3', 'SOA_LV', 'SOA_SV', 'H2SO4', 'SO2', 'monoterp', 'isoprene', 'DMS', 'NUCLRATE', 'FORMRATE', 'GRH2SO4', 'GRSOA', 'GR', 'COAGNUCL', 'SFisoprene', 'SFmonoterp', 'CCN1', 'CCN2', 'CCN3', 'CCN4', 'CCN5', 'CCN6', 'CCN7', 'CCN_B',  'MEANTAU_ISCCP', 'TAUTMODIS', 'REFFCLWMODIS', 'CLHMODIS', 'CLIMODIS', 'CLLMODIS', 'CLMMODIS', 'CLMODIS', 'CLRIMODIS', 'CLRLMODIS', 'CLTMODIS', 'CLWMODIS', 'IWPMODIS', 'LWPMODIS', 'PCTMODIS', 'REFFCLIMODIS', 'REFFCLWMODIS', 'TAUILOGMODIS', 'TAUIMODIS', 'TAUTLOGMODIS', 'TAUTMODIS', 'TAUWLOGMODIS', 'TAUWMODIS', 'NCONC01', 'NCONC02', 'NCONC03', 'NCONC04', 'NCONC05', 'NCONC06', 'NCONC07', 'NCONC08', 'NCONC09', 'NCONC10', 'NCONC11', 'NCONC12', 'NCONC13', 'NCONC14', 'SIGMA01', 'SIGMA02', 'SIGMA03', 'SIGMA04', 'SIGMA05', 'SIGMA06', 'SIGMA07', 'SIGMA08', 'SIGMA09', 'SIGMA10', 'SIGMA11', 'SIGMA12', 'SIGMA13', 'SIGMA14', 'NNAT_0', 'NMR01', 'NMR02', 'NMR03', 'NMR04', 'NMR05', 'NMR06', 'NMR07', 'NMR08', 'NMR09', 'NMR10', 'NMR11', 'NMR12', 'NMR13', 'NMR14', 'AOD_VIS', 'ABSVIS', 'CABSVIS', 'CAODVIS', 'CLDFREE', 'CDOD550', 'CDOD440', 'CDOD870','DOD500',  'DOD550', 'DOD440', 'DOD870', 'AEROD_v', 'CABS550', 'CABS550A', 'OD550DRY', 'RHW', 'MMR_AH2O', 'MMRPM2P5', 'GRIDAREA', 'EC550AER', 'DAERH2O', 'D500_POM', 'DER', 'DERGT05', 'DERLT05', 'D500_BC', 'D500_DU', 'D500_POM', 'D500_SO4', 'D500_SS', 'D550_BC', 'D550_DU', 'D550_POM', 'D550_SO4', 'D550_SS', 'BETOTVIS', 'ASYMMDRY', 'AB550DRY', 'ABS440', 'ABS500', 'ABS550', 'ABSDRYOC', 'ABSDRYSS', 'BS550AER', 'BETOTVIS', 'T', 'PS', 'U', 'V', ]




# %%

nr_of_bins = 5
maxDiameter = 39.6
minDiameter = 5.0
chunks = {'lon':10, 'lat':10}
history_field = '.h1.'
from_t = sys.argv[1]
to_t = sys.argv[2]
case = sys.argv[3]
sectional = sys.argv[4].strip()
if sectional == 'True':
    sectional = True
elif sectional == 'False':
    sectional = False
else:
    sys.exit('Last arguemnt must be True or False')
if sectional:
    cases_sec = [case]
    cases_orig = []
else:
    cases_orig = [case]
    cases_sec = []
# %% [markdown]
# ## Compute collocated datasets from latlon specified output

# %% jupyter={"outputs_hidden": true}


time_res = 'hour'

for case_name in cases_sec:
    isSectional= True
    varlist = list_sized_vars_noresm + vars_extra_ns
    get_field_fixed(case,varlist, from_t, to_t, #raw_data_path=constants.get_input_datapath(),
                    pressure_adjust=False, model = 'NorESM', history_fld=history_field, comp='atm', chunks=chunks)

for case_name in cases_orig:
    isSectional= False
    varlist = list_sized_vars_nonsec+ vars_extra_ns
    get_field_fixed(case,varlist, from_t, to_t, #raw_data_path=constants.get_input_datapath(),
                    pressure_adjust=False, model = 'NorESM', history_fld=history_field, comp='atm', chunks=chunks)
