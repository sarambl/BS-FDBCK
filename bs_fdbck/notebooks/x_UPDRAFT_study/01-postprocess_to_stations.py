# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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

# %%
# %% [markdown]
# ## Settings:

# %%
nr_of_bins = 5
maxDiameter = 39.6
minDiameter = 5.0  #
history_field = '.h1.'

# %%
from_t = '2009-01-01'
to_t = '2010-01-01'

# %%
# %%
cases_sec = []
cases_orig = [

    'OsloAero_f19_f19_mg17_act',
]

# %%
log.ger.info(f'TIMES:****: {from_t} {to_t}')

# %% [markdown]
# ## launches subprocesses that compute monthly

# %%
for case_name in cases_sec:
    #continue
    launch_monthly_station_output(case_name, True, from_time=from_t, to_time=to_t, history_field=history_field)
for case_name in cases_orig:
    #continue
    launch_monthly_station_output(case_name, False, from_time=from_t, to_time=to_t, history_field=history_field)
# %% [markdown]
# ## Merge monthly

# %%
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
