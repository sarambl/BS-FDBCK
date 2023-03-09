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

from bs_fdbck.preprocess.launch_monthly_station_collocation_from_full_grid import \
    launch_monthly_station_output_ec_earth, launch_monthly_station_output_ukesm

from IPython import get_ipython

from bs_fdbck.util.collocate.collocate_ukesm import CollocateModelUkesm

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

case_name = 'AEROCOMTRAJ'
from_time = '2012-01'
to_time = '2015-01'
time_res = 'hour'
space_res = 'locations'
model_name = 'UKESM'

# %% [markdown]
# ## Settings:

# %%
from_t = '2013-01-01'
to_t = '2019-01-01'

# %% [markdown]
# ## Cases:

# %%
cases_orig = [
    case_name,
]
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
    'SFisoprene',
    'SFterpene',
]
# ,'vw','uv','ccn10','ccn2','ceff','apm','cod','lcdnc']
# "'mmrtrN50','mmrtrN100','mmrtrN200','mmrtrN250','mmrtrN5','ccn02',']# 'SO2_gas']
# 'FREQI',

# %%'cosp_reffice','cosp_reffliq','cosp_tau_modis','cosp_tau',

log.ger.info(f'TIMES:****: {from_t} {to_t}')

# %% [markdown]
# ## launches subprocesses that compute monthly


# %%


for case_name in cases_orig:
    launch_monthly_station_output_ukesm(case_name, from_time=from_t, to_time=to_t)
# %% [markdown]
# ## Merge monthly

for case_name in cases_orig:
    c = CollocateModelUkesm(case_name, from_t, to_t,
                            False,
                            'hour',
                            space_res='locations',
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


for case_name in cases_orig:
    c = CollocateModelUkesm(case_name, from_t, to_t, )
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
