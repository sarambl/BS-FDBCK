sized_varListNorESM = {'NCONC': ['NCONC01', 'NCONC02', 'NCONC04', 'NCONC05', 'NCONC06', 'NCONC07', 'NCONC08',
                           'NCONC09', 'NCONC10', 'NCONC12', 'NCONC14'],
                 'SIGMA': ['SIGMA01', 'SIGMA02', 'SIGMA04', 'SIGMA05', 'SIGMA06', 'SIGMA07', 'SIGMA08',
                           'SIGMA09', 'SIGMA10', 'SIGMA12', 'SIGMA14'],
                 'NMR': ['NMR01', 'NMR02', 'NMR04', 'NMR05', 'NMR06', 'NMR07', 'NMR08',
                         'NMR09', 'NMR10', 'NMR12', 'NMR14']}
sized_varlist_SOA_SEC = ['nrSOA_SEC01', 'nrSOA_SEC02', 'nrSOA_SEC03', 'nrSOA_SEC04', 'nrSOA_SEC05']
sized_varlist_SO4_SEC = ['nrSO4_SEC01', 'nrSO4_SEC02', 'nrSO4_SEC03', 'nrSO4_SEC04', 'nrSO4_SEC05']
list_sized_vars_noresm = sized_varListNorESM['NCONC'] + \
                         sized_varListNorESM['SIGMA'] + \
                         sized_varListNorESM['NMR'] + \
                         sized_varlist_SOA_SEC + \
                         sized_varlist_SO4_SEC
list_sized_vars_nonsec = sized_varListNorESM['NCONC'] + sized_varListNorESM['SIGMA'] + sized_varListNorESM['NMR']
import_always_include = ['P0', 'area', 'LANDFRAC', 'hyam', 'hybm', 'PS', 'gw', 'LOGR',
                  'hyai', 'hybi', 'ilev','slon','slat'] # , 'date',  'LANDFRAC','Press_surf',
import_constants = ['P0', 'GRIDAREA', 'landfrac', 'hyam', 'hybm', 'gw', 'LOGR',
                    'hyai', 'hybi', 'ilev', 'LANDFRAC','slon','slat']
not_pressure_coords = ['P0','hyam', 'hybm', 'PS', 'gw', 'LOGR', 'aps',
                  'hyai', 'hybi', 'ilev', 'date']
vars_time_not_dim = ['P0', 'area',]
default_units = dict(
    numberconc={
        'units':'#/cm3',
        'factor':1.e-6,
        'exceptions':['N_AER']
    },
    NMR={
        'units':'nm',
        'factor':1.e9,
        'exceptions':[]
    },
    mixingratio= {
        'units':'$\mu$g/kg',
        'factor':1e9,
        'exceptions':[]
    },
    percent = {
    'units':'%',
    'factor':1e2,
    'exceptions':[]
    }

)