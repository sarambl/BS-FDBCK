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

import time

import useful_scit.util.log as log

from bs_fdbck.preprocess.launch_monthly_station_collocation_from_full_grid import launch_monthly_station_output_ec_earth
from bs_fdbck.util.collocate.collocate_ec_earth import CollocateModelECEarth


from IPython import get_ipython

# noinspection PyBroadException
try:
    _ipython = get_ipython()
    _magic = _ipython.magic
    _magic('load_ext autoreload')
    _magic('autoreload 2')
except:
    pass
# %load_ext autoreload
# %autoreload 2
# %%
log.ger.setLevel(log.log.INFO)

case_name = 'ECE3_output_Sara'
time_res = 'some_hour'
space_res = 'locations'
model_name = 'EC-Earth'

# %% [markdown]
# ## Settings:

# %%
from_t = '2012-01-01'
to_t = '2019-01-01'


# %% [markdown]
# ## Cases:

# %%
cases_orig = [
    case_name,
]
# %%



log.ger.info(f'TIMES:****: {from_t} {to_t}')

# %% [markdown]
# ## launches subprocesses that compute monthly


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

time_res = 'hour'

for case_name in cases_orig:
    launch_monthly_station_output_ec_earth(case_name, from_time=from_t, to_time=to_t)
# %% [markdown]
# ## Merge monthly
# %%
for mod_ver in ['TM5', 'IFS_T', 'IFS_GG']:
    varlist = varlist_dic[mod_ver]

    for case_name in cases_orig:
        c = CollocateModelECEarth(case_name, from_t, to_t,
                                  False,
                                  'hour',
                                  space_res='locations',
                                  which=mod_ver
                                  )
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

for mod_ver in ['TM5', 'IFS_T', 'IFS_GG']:
    varlist = varlist_dic[mod_ver]

    for case_name in cases_orig:
        c = CollocateModelECEarth(
            case_name,
            from_t,
            to_t,
            which=mod_ver
        )
        if c.check_if_load_raw_necessary(varlist):
            time1 = time.time()
            a = c.make_station_data_all(varlist)
            time2 = time.time()
            print('****************DONE: took {:.3f} s'.format((time2 - time1)))
        else:
            print(f'Already computed for {case_name} ')

    print('DONE WITH MONTHLY FIELDS!!!!')

# %%

# %% [markdown]
#
# ## Compute binned dataset

# %% [markdown]
# ### Make station N50 etc.

# %%
