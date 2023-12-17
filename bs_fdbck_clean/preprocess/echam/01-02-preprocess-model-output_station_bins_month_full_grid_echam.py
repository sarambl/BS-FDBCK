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
from bs_fdbck_clean.preprocess.launch_monthly_station_collocation_from_full_grid import launch_monthly_station_output_echam

#%load_ext autoreload
#%autoreload 2

from bs_fdbck_clean.util.Nd.sizedist_class_v2.SizedistributionBins import SizedistributionStationBins


# %%
# from bs_fdbck_clean.util.collocate.collocateLONLAToutput import CollocateLONLATout
# from bs_fdbck_clean.data_info.variable_info import list_sized_vars_nonsec, list_sized_vars_noresm
import useful_scit.util.log as log
import time

from bs_fdbck_clean.util.collocate.collocateLONLAToutput import CollocateLONLATout
from bs_fdbck_clean.util.collocate.collocate_echam_salsa import CollocateModelEcham

log.ger.setLevel(log.log.INFO)






case_name = 'SALSA_BSOA_feedback'
from_time = '2012-01'
to_time = '2012-02'
time_res = 'hour'
space_res='locations'
model_name='ECHAM-SALSA'



# %% [markdown]
# ## Settings:

# %%
# %%
from_t = '2012-01-01'
to_t = '2019-01-01'
# %%
#to_t = '2012-03-01'



#from_t = '2015-01-01'
#to_t = '2019-01-01'


# %% [markdown]
# ## Cases:

# %%
cases_orig = [
    'SALSA_BSOA_feedback',
]
# %%

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
#,'vw','uv','ccn10','ccn2','ceff','apm','cod','lcdnc']
           #"'mmrtrN50','mmrtrN100','mmrtrN200','mmrtrN250','mmrtrN5','ccn02',']# 'SO2_gas']
#'FREQI',

# %%'cosp_reffice','cosp_reffliq','cosp_tau_modis','cosp_tau',

log.ger.info(f'TIMES:****: {from_t} {to_t}')

# %% [markdown]
# ## launches subprocesses that compute monthly


# %%


fl = ['/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_APIN_gas.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_BPIN_gas.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_CARENE3_gas.nc', '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_ISOP_gas.nc', '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_LIMON_gas.nc', '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_MYRC_gas.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_ORG_mass.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_SABIN_gas.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_SO2_gas.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_TBETAOCI_gas.nc', '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_VBS0_gas.nc', '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_VBS10_gas.nc', '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_VBS1_gas.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_airdens.nc',
      # 

      # #'/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_ang550865.nc',
      # #'/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_aot550nm.nc',
      # #'/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_aot865nm.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_apm.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_ccn02.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_ccn10.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_ceff.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_ceff_ct.nc',
      #
      # #'/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_cl_time.nc',
      # #'/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_clfr.nc',
      # #'/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_cod.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_cwp.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_emi_isop_bio.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_emi_monot_bio.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_geom.nc',

      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_lcdnc.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_lcdnc_ct.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_mmrtrN100.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_mmrtrN200.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_mmrtrN250.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_mmrtrN3.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_mmrtrN50.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_mmrtrN500.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_oh_con.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_tempair.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_up_lw.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_up_lw_cs.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_up_lw_cs_noa.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_up_lw_noa.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_up_sw.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_up_sw_cs.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_up_sw_cs_noa.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_up_sw_noa.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_uw.nc',
      '/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_vw.nc'
      ]

import xarray as xr

#ds = xr.open_mfdataset(fl)
# %%
#ds2 = xr.open_dataset('/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_ORG_mass.nc',decode_cf=False)
# %%
#ds2['time'].attrs

#xr.decode_cf(ds2)
# %%
#ds['cod'].attrs['units'] = '1'#.attrs = ds2['time'].attrs

# %%
#ds = xr.open_mfdataset(['/proj/bolinc/users/x_sarbl/other_data/BS-FDBCK/ECHAM-SALSA/SALSA_BSOA_feedback/SALSA_BSOA_feedback_201201_cod.nc'], decode_cf=False)
#ds

# %%
#ds['time'].attrs['units'] = 'days since 2010-01-01 00:00:00'
# %%
#xr.decode_cf(ds)

# %%


time_res = 'hour'

for case_name in cases_orig:
    launch_monthly_station_output_echam(case_name, from_time=from_t, to_time=to_t)
# %% [markdown]
# ## Merge monthly

for case_name in cases_orig:
    c = CollocateModelEcham(case_name, from_t, to_t,
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
    c = CollocateModelEcham(case_name, from_t, to_t, )
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
