# %%
import sys
import time

import useful_scit.util.log as log

from bs_fdbck_clean.util.collocate.collocate_ec_earth import CollocateModelECEarth

log.ger.setLevel(log.log.INFO)

# %% [markdown]
# ### Settings

varlist = ['tempair']
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
#    'GAS_O3',
#    'GAS_SO2',
#    'GAS_TERP',
#    'GAS_OH',
#    'GAS_ISOP',
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
#    'loadoa',
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
#    'var68',
#    'var69',
#    'var70',
#    'var71',
#    'var72',
#    'var73',
#    'var74',
#    'var75',
#    'var176',
#    'var177',
#    'var178',
#    'var179',
#    'var208',
#    'var209',
#    'var210',
#    'var211',
#    'var136',
#    'var137',
#    'var78',
#    'var79',
#    'var164',
#    'var20',
#    'var130',
#    'var131',
#    'var132',
#    'var167',
#    'var248',
#    'var54',
]
varlist_ifs_t =[
    'var130',
]

varlist_dic ={
    'TM5':varlist_tm5,
    'IFS_T':varlist_ifs_t,
    'IFS_GG':varlist_ifs_gg
}
# %%

from_t = sys.argv[1]
to_t = sys.argv[2]
case = sys.argv[3]
cases_orig = [case]
# %% [markdown]
# ## Compute collocated datasets from latlon specified output

# %% jupyter={"outputs_hidden": true}


time_res = 'hour'
print('hey2222  !!')
for mod_ver in ['TM5', 'IFS_T', 'IFS_GG']:
    print(mod_ver)
    varlist = varlist_dic[mod_ver]
    for case_name in cases_orig:
        c = CollocateModelECEarth(
            case_name,
            from_t,
            to_t,
            time_res=time_res,
            which= mod_ver,
        )
        if c.check_if_load_raw_necessary(varlist):
            for v in varlist:
                if c.check_if_load_raw_necessary([v]):
                    a = c.make_station_da(v)
        else:
            print(f'Already computed for {case_name} ')
