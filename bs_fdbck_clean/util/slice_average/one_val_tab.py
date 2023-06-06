import sys
from pathlib import Path

import pandas as pd

from bs_fdbck_clean.constants import path_data_info
from bs_fdbck_clean.data_info import get_nice_name_case, simulation_types
from bs_fdbck_clean.util.naming_conventions.var_info import get_fancy_var_name, get_fancy_unit_xr
from bs_fdbck_clean.util.slice_average.avg_pkg import yearly_mean_dic


# %%
def get_tab_yearly_mean(varl,
                        cases,
                        startyear,
                        endyear,
                        pmin=850.,
                        pressure_adjust=True,
                        average_over_lev=True,
                        groupby='time.year',
                        dims=None,
                        area='Global',
                        invert_dic=False,
                        use_nn=False
                        ):
    if groupby is None:
        avg_dim = 'time'
    else:
        avg_dim = groupby.split('.')[-1]
    # %%
    dummy_dic = yearly_mean_dic(varl,
                                cases,
                                startyear,
                                endyear,
                                pmin,
                                pressure_adjust,
                                model='NorESM',
                                avg_over_lev=average_over_lev,
                                groupby=groupby,
                                dims=dims,
                                area=area)
    # %%
    # varl, cases, avg_over_lev=average_over_lev, groupby=groupby, dims=dims, area=area)
    di = {}
    di_vals = {}
    for var in varl:
        _di_tmp = {}

        val_tab = pd.DataFrame()
        for case in cases:
            if use_nn:
                ncase = get_nice_name_case(case)
            else:
                ncase=case
            _ds = dummy_dic[case]
            val_tab[ncase] = _ds[var].to_pandas()  # .mean(avg_dim).values)
            # _di_tmp[ncase]['$\sigma$'] = float(_ds[var].std(avg_dim).values)
        di_vals[var] = val_tab.copy()
        nvar = get_fancy_var_name(var)
        un = get_fancy_unit_xr(_ds[var], var)
        _di_tmp['nice_name'] = nvar
        _di_tmp['unit'] = un
        di[var] = _di_tmp.copy()
    df = pd.DataFrame(di)
    # %%
    if invert_dic:
        di_vals2 = {}
        for case in cases:
            if use_nn:
                ncase = get_nice_name_case(case)
            else:
                ncase=case
            di_vals2[case] = {}
            for var in varl:
                di_vals2[case][var] = di_vals[var][case]
        for case in cases:
            di_vals2[case] = pd.DataFrame(di_vals2[case])
        return df, di_vals2
    return df, di_vals


def get_pd_yearly_mean(varl,
                       cases,
                       startyear,
                       endyear,
                       pmin=850.,
                       pressure_adjust=True,
                       average_over_lev=True,
                       groupby='time.year',
                       dims=None,
                       area='Global'
                       ):
    if groupby is None:
        avg_dim = 'time'
    else:
        avg_dim = groupby.split('.')[-1]
    dummy_dic = yearly_mean_dic(varl,
                                cases,
                                startyear,
                                endyear,
                                pmin,
                                pressure_adjust,
                                model='NorESM',
                                avg_over_lev=average_over_lev,
                                groupby=groupby,
                                dims=dims,
                                area=area)
    # varl, cases, avg_over_lev=average_over_lev, groupby=groupby, dims=dims, area=area)
    dummy_dic
    di = {}
    for var in varl:
        _di_tmp = {}
        for case in cases:
            ncase = get_nice_name_case(case)
            _ds = dummy_dic[case]
            _di_tmp[ncase] = {}
            _di_tmp[ncase]['$\mu$'] = float(_ds[var].mean(avg_dim).values)
            _di_tmp[ncase]['$\sigma$'] = float(_ds[var].std(avg_dim).values)
        nvar = get_fancy_var_name(var)
        un = get_fancy_unit_xr(_ds[var], var)
        di[f'{nvar} [{un}]'] = _di_tmp.copy()
    d = {(i, j): di[i][j]
         for i in di.keys()
         for j in di[i].keys()}
    df = pd.DataFrame(d)
    return df
    # %%


def get_pd_yearly_stat(varl,
                       cases,
                       startyear,
                       endyear,
                       pmin=850.,
                       stat='mean',
                       pressure_adjust=True,
                       average_over_lev=True,
                       groupby='time.year',
                       dims=None,
                       area='Global',
                       fancy_table=True
                       ):
    if groupby is None:
        avg_dim = 'time'
    else:
        avg_dim = groupby.split('.')[-1]
    # %%
    dummy_dic = yearly_mean_dic(varl,
                                cases,
                                startyear,
                                endyear,
                                pmin,
                                pressure_adjust,
                                model='NorESM',
                                avg_over_lev=average_over_lev,
                                groupby=groupby,
                                dims=dims,
                                area=area)
    # %%
    # varl, cases, avg_over_lev=average_over_lev, groupby=groupby, dims=dims, area=area)
    dummy_dic
    if fancy_table:
        df = fancy_pd(avg_dim, cases, dummy_dic, stat, varl)
    else:
        df = practical_pd(avg_dim, cases, dummy_dic, varl, stat)
    return df


def practical_pd(avg_dim, cases, dummy_dic, varl, stat=None):
    if stat is None:
        stat='mean'
    # %%
    di = {}
    for var in varl:
        di[var] = {}
        for case in cases:
            _di_tmp = {}
            # _di_tmp['case']=case
            ncase = get_nice_name_case(case)
            _di_tmp['case_nn'] = ncase
            _ds = dummy_dic[case]
            # _di_tmp[ncase] = {}
            if stat == 'mean':
                _di_tmp['$\mu$'] = float(_ds[var].mean(avg_dim).values)
            elif stat == 'std':
                _di_tmp['$\sigma$'] = float(_ds[var].std(avg_dim).values)
            else:
                sys.exit(f'Cannot recognize statistic {stat}')

            nvar = get_fancy_var_name(var)
            un = get_fancy_unit_xr(_ds[var], var)
            _di_tmp['unit'] = un
            _di_tmp['var_nn'] = nvar
            _di_tmp['var'] = var
            di[var][case] = _di_tmp
    # %%
    # di[f'{nvar} [{un}]'] = _di_tmp.copy()
    d = {(i, j): di[i][j]
         for i in di.keys()
         for j in di[i].keys()}
    df = pd.DataFrame(d)
    # %%
    return df


def fancy_pd(avg_dim, cases, dummy_dic, stat, varl):
    di = {}
    for var in varl:
        _di_tmp = {}
        for case in cases:
            print(case)
            ncase = get_nice_name_case(case)
            _ds = dummy_dic[case]
            _di_tmp[ncase] = {}
            if stat == 'mean':
                _di_tmp[ncase]['$\mu$'] = float(_ds[var].mean(avg_dim).values)
            elif stat == 'std':
                _di_tmp[ncase]['$\sigma$'] = float(_ds[var].std(avg_dim).values)
            else:
                sys.exit(f'Cannot recognize statistic {stat}')
        nvar = get_fancy_var_name(var)
        un = get_fancy_unit_xr(_ds[var], var)
        di[f'{nvar} [{un}]'] = _di_tmp.copy()
    d = {(i, j): di[i][j]
         for i in di.keys()
         for j in di[i].keys()}
    df = pd.DataFrame(d)
    return df


# %%
def test():
    # %%
    startyear = '2008-01'
    endyear = '2010-12'
    pmin = 850.
    cases = ['SECTv21_ctrl_koagD', 'noSECTv21_ox_ricc_dd', 'noSECTv21_default_dd']
    varl = ['N_AER', 'NCONC01']  # ,'ACTREL']
    groupby = 'time.year'

    pressure_adjust = True

    dims = None
    area = 'Global'
    average_over_lev = True
    # %%
    df_mean = get_pd_yearly_stat(varl,
                                 cases,
                                 startyear,
                                 endyear,
                                 pmin=pmin,
                                 stat='mean',
                                 pressure_adjust=pressure_adjust,
                                 average_over_lev=average_over_lev,
                                 groupby=groupby,
                                 dims=dims,
                                 area=area
                                 )
    # %%
    df_std = get_pd_yearly_stat(varl,
                                cases,
                                startyear,
                                endyear,
                                pmin=pmin,
                                stat='std',
                                pressure_adjust=pressure_adjust,
                                average_over_lev=average_over_lev,
                                groupby=groupby,
                                dims=dims,
                                area=area
                                )
    # %%
    df = get_pd_yearly_mean(varl,
                            cases,
                            startyear,
                            endyear,
                            pmin=pmin,
                            pressure_adjust=pressure_adjust,
                            average_over_lev=average_over_lev,
                            groupby=groupby,
                            dims=dims,
                            area=area
                            )

    # %%


def get_diff_by_type(_df, case_types=None, mod_types=None, ctrl='ctrl', col='$\mu$',
                     relative=False):
    sims = pd.read_csv(Path(path_data_info) / 'simulation_types.csv', index_col=0)

    if case_types is None:
        case_types = ['decYield', 'incYield']
    if mod_types is None:
        mod_types = ['OsloAeroSec', 'OsloAero$_{imp}$', 'OsloAero$_{def}$']
    di = {}
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
                di[ctlab][mod_type] = 100 * (_df.loc[case, col] - _df.loc[case_ctrl, col]) / np.abs(
                    _df.loc[case_ctrl, col])
            else:
                #   print(f'subtractiving {case}-{case_ctrl}')
                di[ctlab][mod_type] = _df.loc[case, col] - _df.loc[case_ctrl, col]

    df_diff = pd.DataFrame(di)
    return df_diff


def get_abs_by_type(_df,
                    case_types=None,
                    mod_types=None,
                    col='$\mu$'
                    ):
    sims = pd.read_csv(Path(path_data_info) / 'simulation_types.csv', index_col=0)

    if case_types is None:
        case_types = ['decYield', 'ctrl', 'incYield']
    if mod_types is None:
        mod_types = ['OsloAeroSec', 'OsloAero$_{imp}$', 'OsloAero$_{def}$']
    di = {}
    for case_type in case_types:
        di[case_type] = {}
        for mod_type in mod_types:
            case = sims.loc[case_type, mod_type]
            # case_ctrl = sims.loc['ctrl',mod_type]
            # _df = df2[var]
            di[case_type][mod_type] = _df.loc[case, col]  # - _df.loc[case_ctrl,col]

    df_abs = pd.DataFrame(di)
    return df_abs


from useful_scit.imps import (plt, np)


def plt_var(var, _df, axs=None, figsize=None, relative=False, case_types=None,
            model_types=None,
            ctrl='ctrl', df_sig=None):
    df_diff = get_diff_by_type(_df, relative=relative,
                               case_types=case_types,
                               mod_types=model_types,
                               ctrl=ctrl)
    df_abs = get_abs_by_type(_df, mod_types=model_types,
                             case_types=case_types)
    if df_sig is not None:
        df_abs_s = get_abs_by_type(df_sig, mod_types=model_types,
                                   case_types=case_types, col='$\sigma$')
        df_abs_s = df_abs_s.transpose()
    else:
        df_abs_s = None

    if relative:
        _l = f'({ctrl}-{ctrl})/{ctrl}'
    else:
        _l = f'{ctrl}-{ctrl}'
    linepl = True
    if _l in df_diff.columns:
        df_diff = df_diff.drop(_l, axis=1)
        linepl = False

    if axs is None:
        if linepl:
            sn = 3
        else:
            sn = 2
        if figsize is None:
            figsize = [3 * sn, 4]
        fig, axs = plt.subplots(1, sn, figsize=figsize)

    df_abs.transpose().plot.bar(ax=axs[0], yerr=df_abs_s)  # .transpose())
    i = 1
    if linepl:
        df_diff['Zero'] = 0
        cols = df_diff.columns
        df_diff = df_diff[[cols[0], 'Zero', *cols[1:-1]]]
        df_diff.transpose().plot(marker='*', ax=axs[i])
        axs[i].grid(True)
        axs[i].legend(frameon=False)
        df_diff = df_diff.drop('Zero', axis=1)
        i += 1

    df_diff.transpose().plot.bar(ax=axs[i])

    for ax in axs:
        if 'unit' in _df.columns:
            ax.set_ylabel(_df['unit'][0])
        if 'var_nn' in _df.columns:
            ax.set_title(_df['var_nn'][0])
    if relative:
        for ax in axs[1:]:
            ax.set_ylabel('%')
    return axs, df_abs, df_diff


def get_mean_std_by_type(dic_vals, varl, case_types=None, model_types=None, ctrl=None,
                         col_out=None,
                         relative=False):
    if case_types is None:
        case_types = ['PI', 'PD']
    if ctrl is None:
        ctrl=case_types[0]
    if model_types is None:
        model_types = ['OsloAeroSec', 'OsloAero$_{imp}$', 'OsloAero$_{def}$']
    if col_out is None:
        col_out = f'{case_types[1]}-{ctrl}'
        if relative:
            col_out =f'({case_types[1]}-{ctrl})/{ctrl}'# '(PD-PI)/PI'
    d = simulation_types.get_diff_by_type(dic_vals, varl, case_types=case_types,
                                          relative=relative, mod_types=model_types,
                                          ctrl=ctrl)
    d2 = d[col_out]
    std = dict()
    mean = dict()
    for model in model_types:
        mea = d2[model].mean()
        sd = d2[model].std()
        std[model] = sd
        mean[model] = mea
    mean = pd.DataFrame(mean)
    mean_nn = mean.rename({key: get_fancy_var_name(key) for key in mean.index}, axis=0)
    std = pd.DataFrame(std)
    std_nn = std.rename({key: get_fancy_var_name(key) for key in std.index}, axis=0)
    return mean, std, mean_nn, std_nn
