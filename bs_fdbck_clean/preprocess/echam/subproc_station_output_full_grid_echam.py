# %%
import sys
import time

import useful_scit.util.log as log

from bs_fdbck_clean.util.collocate.collocate_echam_salsa import CollocateModelEcham

log.ger.setLevel(log.log.INFO)

# %% [markdown]
# ### Settings

varlist = ['tempair']






varlist = [#'tempair'
    'apm',
    'geom',
    'airdens',
    'tempair',
    'uw',
    'vw',
    'ccn02',
    'ccn10',
    'cod',
    'cwp',
    'ceff',
    'ceff_ct',
    'lcdnc',
    'lcdnc_ct',
    'clfr',
    'cl_time',
    'aot550nm',
    'aot865nm',
    'ang550865',
    'up_sw',
    'up_sw_cs',
    'up_sw_noa',
    'up_sw_cs_noa',
    'up_lw',
    'up_lw_cs',
    'up_lw_noa',
    'up_lw_cs_noa',
    'mmrtrN500',
    'mmrtrN250',
    'mmrtrN200',
    'mmrtrN100',
    'mmrtrN50',
    'mmrtrN3',
    'oh_con',
    'emi_monot_bio',
    'emi_isop_bio',
    'SO2_gas',
    'APIN_gas',
    'TBETAOCI_gas',
    'BPIN_gas',
    'LIMON_gas',
    'SABIN_gas',
    'MYRC_gas',
    'CARENE3_gas',
    'ISOP_gas',
    'VBS0_gas',
    'VBS1_gas',
    'VBS10_gas',
    'ORG_mass',
]


# %%

nr_of_bins = 5
maxDiameter = 39.6
minDiameter = 5.0
history_field = '.h1.'
#from_t = sys.argv[1]
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

    c = CollocateModelEcham(
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
