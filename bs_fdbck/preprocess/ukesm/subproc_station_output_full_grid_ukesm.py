# %%
import sys
import time

import useful_scit.util.log as log

from bs_fdbck.util.collocate.collocate_echam_salsa import CollocateModelEcham
from bs_fdbck.util.collocate.collocate_ukesm import CollocateModelUkesm

log.ger.setLevel(log.log.INFO)

# %% [markdown]
# ### Settings

varlist = ['tempair']






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


# %%

history_field = '.h1.'

from_t = sys.argv[1]
to_t = sys.argv[2]
case = sys.argv[3]
cases_orig = [case]


# %% [markdown]
# ## Compute collocated datasets from latlon specified output

# %% jupyter={"outputs_hidden": true}


time_res = 'hour'

for case_name in cases_orig:
    isSectional= False

    c = CollocateModelUkesm(
        case_name,
        from_t,
        to_t,
        time_res=time_res,
        model_name='ECHAM-SALSA',
    )

    if c.check_if_load_raw_necessary(varlist):
        for v in varlist:
            if c.check_if_load_raw_necessary([v]):
                a = c.make_station_data_all(varlist=[v])
    else:
        print(f'Already computed for {case_name} ')

# %%