from ukesm_bs_fdbck.constants import path_data_info
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr

def get_casen_by_type_mod(case_type, model_type):
    if case_type=='PD':
        ct = 'PIaerPD'
    else:
        ct = case_type
    sims = pd.read_csv(Path(path_data_info) / 'simulation_types.csv', index_col=0)
    case = sims.loc[ct, model_type]
    return case

def get_diff_by_type(cases_dic,
                     varl,
                     case_types=None,
                     mod_types=None,
                     ctrl='ctrl',
                     relative=False):
    """
    Calculates difference between cases case relevant ctrl.
    Example: If
    ctrl='PI'
    case_types=['PI', 'PD'],
    mod_types=['OsloAeroSec', 'OsloAero$_{def}$']
    then the output will be the difference between the PD and PI runs for OsloAeroSec and
    OsloAero_def respectively, outputting dic[model_type] dic.
    :param cases_dic: dictionary with case names as keys and xr.Datasets as elements
    :param varl:
    :param case_types:
    :param mod_types:
    :param ctrl:
    :param relative:
    :return:
    """
    sims = pd.read_csv(Path(path_data_info) / 'simulation_types.csv', index_col=0)
    if case_types is None:
        case_types = ['incYield', 'decYield']
    if mod_types is None:
        mod_types = ['OsloAeroSec', 'OsloAero$_{imp}$', 'OsloAero$_{def}$']
    di = {}
    print(case_types, mod_types)
    for case_type in case_types:
        ctlab = f'{case_type}-{ctrl}'
        if relative:
            ctlab = f'({case_type}-{ctrl})/{ctrl}'
        di[ctlab] = {}
        for mod_type in mod_types:
            case = sims.loc[case_type, mod_type]
            case_ctrl = sims.loc[ctrl, mod_type]
            # _df = df2[var]
            if relative:
                di[ctlab][mod_type] = 100. * (cases_dic[case][varl] - cases_dic[case_ctrl][varl]) / np.abs(
                    cases_dic[case_ctrl][varl])
                if isinstance(di[ctlab][mod_type], xr.Dataset):
                    for var in varl:
                        di[ctlab][mod_type][var].attrs['units']='%'
            else:
                print(f'subtracting {case}-{case_ctrl}')
                di[ctlab][mod_type] = cases_dic[case][varl] - cases_dic[case_ctrl][varl]

    return di



def get_abs_by_type(cases_dic,
                    case_types=None,
                    mod_types=None):
    """
    From dict(case_name:xr.Dataset,...) to dict(case_type:{mod_type:xr.Dataset})
    :param cases_dic: dict
    :param case_types: list
    :param mod_types: list
    :return: dict
    """

    sims = pd.read_csv(Path(path_data_info) / 'simulation_types.csv', index_col=0)
    if case_types is None:
        case_types = ['incYield', 'decYield']
    if mod_types is None:
        mod_types = ['OsloAeroSec', 'OsloAero$_{imp}$', 'OsloAero$_{def}$']
    di = {}
    print(case_types, mod_types)
    for case_type in case_types:
        #ctlab = f'{case_type}-{ctrl}'
        di[case_type] = {}
        for mod_type in mod_types:
            case = sims.loc[case_type, mod_type]
            di[case_type][mod_type] = cases_dic[case]
    return di


def transpose_2lev_dic(dic_abs, ctrl='ctrl'):
    dic_absT = {}
    _case_t = list(dic_abs.keys())
    _mod_t = list(dic_abs[_case_t[0]].keys())
    if ctrl is None:
        l = _case_t
    else:
        l = _case_t +[ctrl]
    for mo in _mod_t:
        dic_absT[mo]={}
        for type in l:
            dic_absT[mo][type]=dic_abs[type][mo]
    return dic_absT