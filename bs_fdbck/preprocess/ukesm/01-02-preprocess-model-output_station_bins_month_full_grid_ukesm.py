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
from bs_fdbck.preprocess.launch_monthly_station_collocation_from_full_grid import launch_monthly_station_output_echam

#%load_ext autoreload
#%autoreload 2



# %%
import useful_scit.util.log as log
import time

from bs_fdbck.util.collocate.collocateLONLAToutput import CollocateLONLATout
from bs_fdbck.util.collocate.collocate_echam_salsa import CollocateModelEcham
from bs_fdbck.util.collocate.collocate_ukesm import CollocateModelUkesm

log.ger.setLevel(log.log.INFO)






case_name = 'CRES'
from_time = '2012-01'
to_time = '2015-01'
time_res = 'hour'
space_res='locations'
model_name='UKESM'



# %% [markdown]
# ## Settings:

# %%
# %%
from_t = '2012-01-01'
to_t = '2015-01-01'


#from_t = '2015-01-01'
#to_t = '2019-01-01'


# %% [markdown]
# ## Cases:

# %%
cases_orig = [
    'CRES',
]

# %%

varlist = [#'tempair'
           'tas',
           'sfmmroa',

           ]
#,'vw','uv','ccn10','ccn2','ceff','apm','cod','lcdnc']
           #"'mmrtrN50','mmrtrN100','mmrtrN200','mmrtrN250','mmrtrN5','ccn02',']# 'SO2_gas']
#'FREQI',

# %%'cosp_reffice','cosp_reffliq','cosp_tau_modis','cosp_tau',

log.ger.info(f'TIMES:****: {from_t} {to_t}')

# %% [markdown]
# ## launches subprocesses that compute monthly


# %%



import xarray as xr



# %%
#for case_name in cases_orig:
#    launch_monthly_station_output_echam(case_name, from_time=from_t, to_time=to_t)
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
dic_ds = dict()
for ca in cases_orig:
    c = CollocateLONLATout(ca, from_t, to_t,
                           True,
                           'hour',
                           model_name=model_name
                           )
    # history_field=history_field)
    ds = c.get_collocated_dataset(varlist)
# %%

# %% [markdown]
#
# ## Compute binned dataset

# %% [markdown]
# ### Make station N50 etc.

# %%
