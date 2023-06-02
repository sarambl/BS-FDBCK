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
from bs_fdbck.notebooks.x_forEscienceCourse.launch_monthly_station_collocation_from_full_grid import launch_monthly_station_output
from bs_fdbck.util.Nd.sizedist_class_v2.SizedistributionBins import SizedistributionStationBins
from bs_fdbck.util.collocate.collocateLONLAToutput import CollocateLONLATout
from bs_fdbck.data_info.variable_info import list_sized_vars_nonsec, list_sized_vars_noresm,list_sized_vars_nonsec
import useful_scit.util.log as log
import time
from bs_fdbck.util.imports.import_fields_xr_v2 import xr_import_NorESM
log.ger.setLevel(log.log.INFO)

# %%
import pandas as pd

# %%

collocate_locations = pd.read_csv('locations.csv', index_col=0)

collocate_locations


# %%
from bs_fdbck.constants import get_input_datapath
# %% [markdown] tags=[]
# ## Settings:

# %%
nr_of_bins = 5
maxDiameter = 39.6  #    23.6 #e-9
minDiameter = 5.0  # e-9
history_field='.h1.'

# %%
from_t ='2012-01-01'
to_t = '2015-01-01'
#from_t = '2014-02-01'#'2012-01-01'
#to_t = '2014-03-01'#2015-01-01'

# %%
skip_subproc = True

# %%
from_t2 = '2015-01-01'
to_t2 = '2018-01-01'
#from_t2 = '2017-02-01'
#to_t2 = '2017-03-01'

# %%
cases_noresm1 = ['OsloAero_intBVOC_f09_f09_mg17_full']
cases_noresm2 = ['OsloAero_intBVOC_f09_f09_mg17_ssp245']
# %%
case_mod = 'OsloAero_intBVOC_f09_f09_mg17_fssp'
case_noresm = 'OsloAero_intBVOC_f09_f09_mg17_fssp'

# %%
model_name = 'NorESM'



# %%
from bs_fdbck.constants import package_base_path


package_base_path.parent


# %% [markdown]
# ### Variables

# %%
varl =['N100','SOA_NA','SOA_A1','SO4_NA','DOD500','DOD440','ACTREL',#'TGCLDLWP',
       'H2SO4','SOA_LV','COAGNUCL','FORMRATE','FSNSC',
       'NUCLRATE','NCONC01','NCONC02','NCONC03','NCONC04','NCONC05','NCONC06','NCONC07',
       'NCONC08','NCONC09','NCONC10','NCONC11','NCONC12','NCONC13','NCONC14','SIGMA01',
       'SIGMA02','SIGMA03','SIGMA04','SIGMA05','SIGMA06','SIGMA07','SIGMA08','SIGMA09',
       'SIGMA10','SIGMA11','SIGMA12','SIGMA13','SIGMA14','NMR01','NMR02','NMR03','NMR04',
       'NMR05','NMR06','NMR07','NMR08','NMR09','NMR10','NMR11','NMR12','NMR13','NMR14', 
      'FSNS','FSDS_DRF','T','GR','GRH2SO4','GRSOA','TGCLDCWP','U','V', 'SO2','isoprene',
       'monoterp','GS_SO2', 'GS_H2SO4','GS_monoterp','GS_isoprene',
      ]


varl =['N100','DOD500','DOD440','ACTREL',#,'SOA_A1',
       'H2SO4','SOA_LV','COAGNUCL','FORMRATE','T',
       'N500',
       'NCONC01',#'N50','N150','N200',#'DOD500',
       #'NCONC01',
       #'SFisoprene',
       #'SFmonoterp',
       #'DOD500',
      'SFmonoterp','SFisoprene',
       'PS',
      
      'SOA_NA','SOA_A1','OM_NI','OM_AI','OM_AC','SO4_NA','SO4_A1','SO4_A2','SO4_AC','SO4_PR',
      'BC_N','BC_AX','BC_NI','BC_A','BC_AI','BC_AC','SS_A1','SS_A2','SS_A3','DST_A2','DST_A3', 
      ] 


# %%
vars_extra_ns = varl

# %% [markdown]
# vars_extra_ns = []
#     
#            'AWNC',
#        #'AWNC_incld',
#        'AREL', 
#        'FREQL', 
#        'FREQI', 
#        #'ACTNL_incld',
#        'ACTNL',
#        'ACTREL', 
#        'ACTREI', 
#        'FCTL', 'FCTI',
#        'Z3',
#        'Smax_cldv',
#        'Smax_cldv_supZero',
#        'Smax_incld',
#        'Smax_incld_supZero',
#        'WSUB',
#        'WTKE',
#        'WSUBI',
#        'T',
#        'LCLOUD', # liquid cloud fraction used in stratus activation
#        'CLDTOT',
#        'CLOUD',
#        'CLOUDCOVER_CLUBB',
#        'CLOUDFRAC_CLUBB',
#
# ]
# %%
cases_sec = []
cases_orig = [

    'OsloAero_f09_f09_mg17_full',
    'OsloAero_f09_f09_mg17_ssp2',
]
case_name = cases_orig[0]
print(case_name)

# %% tags=[]
log.ger.info(f'TIMES:****: {from_t} {to_t}')

# %% [markdown]
# ## launches subprocesses that compute monthly

# %%
time_set = [[from_t, to_t],[from_t2, to_t2]]

# %%
varl_merge  = [
#varl_dumb=[
#'gw',
 'hyam',
 'hybm',
 'P0',
 'hyai',
 'hybi',
 'date',
 'datesec',
 'time_bnds',
 'date_written',
 'time_written',
 'ndbase',
 'nsbase',
 'nbdate',
 'nbsec',
 'mdt',
 'ndcur',
 'nscur',
 'co2vmr',
 'ch4vmr',
 'n2ovmr',
 'f11vmr',
 'f12vmr',
 'sol_tsi',
 'nsteph',
 'ABSVIS',
 'ACTNI',
 'ACTNL',
 'ACTREI',
 'ACTREL',
 'AEROD_v',
 'AOD_VIS',
 'AREL',
 'ASYMMDRY',
 'AWNC',
 'BC_A',
 'BC_AC',
 'BC_AI',
 'BC_AX',
 'BC_N',
 'BC_NI',
 'BETOTVIS',
 'BS550AER',
 'CCN1',
 'CCN2',
 'CCN3',
 'CCN4',
 'CCN5',
 'CCN6',
 'CCN7',
 'CCN_B',
 'CDNUMC',
 'CLDFREE',
 'CLDTOT',
 'COAGNUCL',
 'D500_BC',
 'D500_DU',
 'D500_POM',
 'D500_SO4',
 'D500_SS',
 'DAERH2O',
 'DER',
 'DERGT05',
 'DERLT05',
 'DMS',
 'DOD440',
 'DOD500',
 'DOD550',
 'DOD670',
 'DOD870',
 'DST_A2',
 'DST_A3',
 'EC550AER',
 'FCTI',
 'FCTL',
 'FLNS',
 'FLNSC',
 'FLNT',
 'FLNTCDRF',
 'FLNT_DRF',
 'FLUS',
 'FLUTC',
 'FORMRATE',
 'FREQI',
 'FREQL',
 'FSDSCDRF',
 'FSDS_DRF',
 'FSNS',
 'FSNSC',
 'FSNT',
 'FSNTCDRF',
 'FSNT_DRF',
 'FSUS_DRF',
 'FSUTADRF',
 'GR',
 'GRH2SO4',
 'GRIDAREA',
 'GRSOA',
 'H2SO4',
 'MMR_AH2O',
 'NCONC01',
 'NCONC02',
 'NCONC03',
 'NCONC04',
 'NCONC05',
 'NCONC06',
 'NCONC07',
 'NCONC08',
 'NCONC09',
 'NCONC10',
 'NCONC11',
 'NCONC12',
 'NCONC13',
 'NCONC14',
 'NMR01',
 'NMR02',
 'NMR03',
 'NMR04',
 'NMR05',
 'NMR06',
 'NMR07',
 'NMR08',
 'NMR09',
 'NMR10',
 'NMR11',
 'NMR12',
 'NMR13',
 'NMR14',
 'NNAT_0',
 'NUCLRATE',
 'OD550DRY',
 'OM_AC',
 'OM_AI',
 'OM_NI',
 'PM25',
 'PM2P5',
 'PMTOT',
 'PS',
 'RHW',
 'SFisoprene',
 'SFmonoterp',
 'SIGMA01',
 'SIGMA02',
 'SIGMA03',
 'SIGMA04',
 'SIGMA05',
 'SIGMA06',
 'SIGMA07',
 'SIGMA08',
 'SIGMA09',
 'SIGMA10',
 'SIGMA11',
 'SIGMA12',
 'SIGMA13',
 'SIGMA14',
 'SO2',
 'SO4_A1',
 'SO4_A2',
 'SO4_AC',
 'SO4_NA',
 'SO4_PR',
 'SOA_A1',
 'SOA_LV',
 'SOA_NA',
 'SOA_SV',
 'SS_A1',
 'SS_A2',
 'SS_A3',
 'T',
 'TGCLDCWP',
 'TGCLDIWP',
 'TGCLDLWP',
 'TOT_CLD_VISTAU',
 'TOT_ICLD_VISTAU',
 'U',
 'V',
 'isoprene',
 'monoterp']

# %% tags=[]
for case_name in cases_noresm1:#,cases_noresm2], time_set):
    if skip_subproc:
        continue
    continue
    f_t = from_t
    t_t = to_t
    launch_monthly_station_output(case_name, False, from_time=f_t, to_time=t_t, history_field=history_field)
# %% tags=[]
for case_name in cases_noresm2:#,cases_noresm2], time_set):
    if skip_subproc:
        continue

    f_t = from_t2
    t_t = to_t2
    launch_monthly_station_output(case_name, False, from_time=f_t, to_time=t_t, history_field=history_field)
# %% [markdown]
# ## Merge monthly

# %%
print('DONE WITH MONTHLY FIELDS!!!!')

# %%
list_sized_vars_noresm

# %% tags=[]
for case_name in cases_noresm1:
    #continue
    varlist = list_sized_vars_nonsec + varl_merge
    c = CollocateLONLATout(case_name, from_t, to_t,
                           is_sectional=False,
                           time_res='hour',
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
for case_name in cases_noresm2:
    varlist = list_sized_vars_nonsec + varl_merge#vars_extra_ns # list_sized_vars_noresm
    c = CollocateLONLATout(case_name, from_t2, to_t2,
                           False,
                           time_res='hour',
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

# %% [markdown]
#
# ## Compute binned dataset

# %% [markdown] tags=[]
# ### Make station N50 etc.

# %%
for case_name in cases_noresm1:
    s = SizedistributionStationBins(case_name, from_t, to_t, [minDiameter, maxDiameter], False, 'hour',
                                    space_res='full',
                                    nr_bins=nr_of_bins, history_field=history_field)
    s.compute_Nd_vars()

for case_name in cases_noresm2:
    s = SizedistributionStationBins(case_name, from_t2, to_t2, [minDiameter, maxDiameter], False, 'hour',
                                    space_res='full',
                                    nr_bins=nr_of_bins, history_field=history_field)
    s.compute_Nd_vars()

# %%

# %% [markdown] tags=[]
# # MERGE:

# %% [markdown] tags=[]
# ### Load data

# %% [markdown]
# for case_name in cases_noresm1:
#     varlist = varl
#     c = CollocateLONLATout(case_name, from_t, to_t,
#                            True,
#                            'hour',
#                            history_field=history_field)
#     if c.check_if_load_raw_necessary(varlist ):
#         time1 = time.time()
#         a = c.make_station_data_merge_monthly(varlist)
#         print(a)
#
#         time2 = time.time()
#         print('DONE : took {:.3f} s'.format( (time2-time1)))
#     else:
#         print('UUUPS')
# %% [markdown]
# for case_name in cases_noresm2:
#     varlist = varl# list_sized_vars_noresm
#     c = CollocateLONLATout(case_name, from_t2, to_t2,
#                            False,
#                            'hour',
#                            history_field=history_field)
#     if c.check_if_load_raw_necessary(varlist ):
#         time1 = time.time()
#         a = c.make_station_data_merge_monthly(varlist)
#         print(a)
#
#         time2 = time.time()
#         print('DONE : took {:.3f} s'.format( (time2-time1)))
#     else:
#         print('UUUPS')

# %% [markdown] tags=[]
# dic_ds = dict()
# for ca in cases_noresm1:
#     c = CollocateLONLATout(ca, from_t, to_t,
#                            False,
#                            'hour',
#                            history_field=history_field)
#     ds = c.get_collocated_dataset(varl)
#     #ds2 = c.get_collocated_dataset(['DOD500'])
#     if 'location' in ds.coords:
#         ds = ds.rename({'location':'station'})
#     dic_ds[ca]=ds

# %% [markdown] tags=[]
# #dic_ds = dict()
# for ca in cases_noresm2:
#     c = CollocateLONLATout(ca, from_t2, to_t2,
#                            False,
#                            'hour',
#                            history_field=history_field)
#     ds = c.get_collocated_dataset(varl)
#     if 'location' in ds.coords:
#         ds = ds.rename({'location':'station'})
#     dic_ds[ca]=ds

# %% [markdown]
# case1 = cases_noresm1[0]
# case2 = cases_noresm2[0]
#
# ds1 = dic_ds[case1]
# ds2 = dic_ds[case2]
#
# st_y = from_t.split('-')[0]
# mid_y_t = str(int(to_t.split('-')[0])-1)
# mid_y_f = to_t.split('-')[0]
# end_y = to_t2.split('-')[0]
#
# print(st_y, mid_y_t, mid_y_f, end_y)

# %% [markdown]
# _ds1 = ds1.sel(time=slice(st_y, mid_y_t))
# _ds2 = ds2.sel(time=slice(mid_y_f, end_y))
# ds_comb = xr.concat([_ds1, _ds2], dim='time')#.sortby('time')
