# %%
import sys
import time

import useful_scit.util.log as log

from ukesm_bs_fdbck.data_info.variable_info import list_sized_vars_nonsec, list_sized_vars_noresm
from ukesm_bs_fdbck.util.collocate.collocate import CollocateModel

log.ger.setLevel(log.log.INFO)

# %% [markdown]
# ### Settings

vars_extra_ns = ['AWNC', 'AREL', 'FREQL', 'ACTNL',
                 'ACTNI',
                 'CCN1', 'CCN2', 'CCN3', 'CCN4', 'CCN5', 'CCN6', 'CCN7', 'CCN_B', 'NNAT_0', 'NCONC01', 'NCONC02', 'NCONC03', 'NCONC04', 'NCONC05', 'NCONC06', 'NCONC07',
                 'NCONC08', 'NCONC09', 'NCONC10', 'NCONC11', 'NCONC12', 'NCONC13', 'NCONC14', 'SIGMA01', 'SIGMA02', 'SIGMA03', 'SIGMA04', 'SIGMA05', 'SIGMA06', 'SIGMA07', 'SIGMA08', 'SIGMA09', 'SIGMA10', 'SIGMA11', 'SIGMA12', 'SIGMA13', 'SIGMA14', 'NMR01', 'NMR02', 'NMR03', 'NMR04', 'NMR05', 'NMR06', 'NMR07', 'NMR08', 'NMR09', 'NMR10', 'NMR11', 'NMR12', 'NMR13', 'NMR14', 'T', 'PS', 'U', 'V',
                 'ACTREL', 'ACTREI', 'FCTL', 'FCTI', 'CDNUMC', 'CLDTOT',
                 'TOT_CLD_VISTAU', 'TOT_ICLD_VISTAU', 'TGCLDIWP', 'TGCLDLWP',
                 'TGCLDCWP', 'FSNT', 'FLNT', 'FSNT_DRF', 'FLNT_DRF', 'FSNTCDRF', 'FLNTCDRF',
                 'FLNS',
                 'FSNS', 'FLNSC', 'FSNSC', 'FSDSCDRF', 'FSDS_DRF', 'FSUTADRF',
                 'FLUTC',
                 'FSUS_DRF', 'FLUS', 'CLDFREE', 'AOD_VIS', 'DOD440', 'DOD500', 'DOD550',
                 'DOD670',
                 'TOT_CLD_VISTAU',
                 'DOD870', 'ABSVIS', 'AEROD_v', 'OD550DRY', 'RHW', 'MMR_AH2O', 'EC550AER',
                 'DAERH2O', 'D500_POM', 'D500_SO4', 'D500_BC', 'D500_DU', 'D500_SS', 'DER', 'DERGT05', 'DERLT05', 'BETOTVIS', 'ASYMMDRY',
                 'BS550AER', 'BETOTVIS', 'PMTOT', 'PM2P5', 'PM25', 'SO4_NA', 'SO4_A1', 'SO4_A2', 'SO4_AC', 'SO4_PR', 'SOA_NA', 'SOA_A1', 'BC_N',
                 'BC_AX', 'BC_NI', 'BC_A', 'BC_AI', 'BC_AC', 'OM_NI', 'OM_AI', 'OM_AC', 'DST_A2', 'DST_A3', 'SS_A1', 'SS_A2', 'SS_A3', 'SOA_LV', 'SOA_SV',
                 'H2SO4', 'SO2', 'monoterp', 'isoprene', 'DMS', 'SFisoprene', 'SFmonoterp', 'NUCLRATE', 'FORMRATE', 'GRH2SO4', 'GRSOA', 'GR', 'COAGNUCL',

                 #'GRIDAREA'
                 ]


# %%

nr_of_bins = 5
maxDiameter = 39.6
minDiameter = 5.0
history_field = '.h1.'
#from_t = '2012-07-01'
from_t =sys.argv[1]


to_t = sys.argv[2]
#to_t = '2012-08-01'
case = sys.argv[3]
#case = 'OsloAero_intBVOC_f09_f09_mg17_full'
sectional = sys.argv[4].strip()

#sectional = False#sys.argv[4].strip()
# %%

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
    c = CollocateModel(case_name,
                       from_t,
                       to_t,
                       isSectional,
                       time_res,
                       history_field=history_field
                       )
    if c.check_if_load_raw_necessary(varlist):
        time1 = time.time()
        a = c.make_station_data_all()
        time2 = time.time()
        print('****************DONE: took {:.3f} s'.format((time2 - time1)))
    else:
        print(f'Already computed for {case_name} ')

# %%
for case_name in cases_orig:
    # %%
    isSectional = False

    varlist = list_sized_vars_nonsec+ vars_extra_ns


    c = CollocateModel(case_name,
                   from_t,
                   to_t,
                   isSectional,
                   time_res,
                   history_field=history_field
                   )
   # %%
    if c.check_if_load_raw_necessary(varlist):
        a = c.make_station_data_all()
    else:
        print(f'Already computed for {case_name} ')
