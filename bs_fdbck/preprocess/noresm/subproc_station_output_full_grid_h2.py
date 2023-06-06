# %%
import sys
import time

import useful_scit.util.log as log

from bs_fdbck.data_info.variable_info import list_sized_vars_nonsec, list_sized_vars_noresm
from bs_fdbck.util.collocate.collocate import CollocateModel

log.ger.setLevel(log.log.INFO)

# %% [markdown]
# ### Settings

vars_extra_ns = ['SFmonoterp','SFisoprene']

# %%

nr_of_bins = 5
maxDiameter = 39.6
minDiameter = 5.0
history_field = '.h2.'
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

for case_name in cases_orig:
    isSectional= False
    varlist = list_sized_vars_nonsec+ vars_extra_ns


    c = CollocateModel(case_name,
                   from_t,
                   to_t,
                   isSectional,
                   time_res,
                   history_field=history_field
                   )
   
    if c.check_if_load_raw_necessary(varlist):
        a = c.make_station_data_all()
    else:
        print(f'Already computed for {case_name} ')
