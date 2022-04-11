import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bs_fdbck.util.naming_conventions.var_info
from bs_fdbck.util import practical_functions


def plot_bars_ctr_and_diff(nested_cases, var,
                           versions_dic,
                           ctrl, sens_cases, versions,
                           avg_over_lev, pmin, p_level, area,
                           relative=False, pressure_coord=True,
                           figsize=None, FONT_SIZE = 12):
    """
    Plots two or more versions or models with corresponding sensitivity tests. E.g. NorESM and EC-Earth with
    sensitivity tests 2xCO2 and 4xCO2 would look like:
    EXAMPLE:
    nested_cases={'NorESM_CTRL': xr_ds1, 'EC_Earth_CTRL': xr_ds2, 'NorESM_2xCO2':xr_ds3,'EC_Earth_2xCO2': xr_ds4 etc}
    versions_dic = {'NorESM':{'CTRL':'NorESM_CTRL', '2xCO2':'NorESM_2xCO2', '4xCO2':NorESM_4xCO2'},
                    'EC-Earth':{'CTRL':'EC_Earth_CTRL','2xCO2':'EC_Earth_2xCO2,..etc.}}
    var = 'N_AER' # number concentration aerosols
    ctrl = 'CTRL'
    sens_cases = ['2xCO2','4xCO2']
    versions = ['NorESM', 'EC-Earth']
    avg_over_lev=True # if plotted variable avg over lev
    pmin= 850. # avg up to 850.
    p_level = 1013. # if not avg_over_lev
    relative = False
    plot_bars_ctr_and_diff(nested_cases, var,
                           versions_dic,
                           ctrl, sens_cases, versions,
                           avg_over_lev, pmin, p_level,
                           relative=relative)
    :param area:
    :param pressure_coord:
    :param nested_cases:
    :param var:
    :param versions_dic:
    :param ctrl:
    :param sens_cases:
    :param versions:
    :param avg_over_lev:
    :param pmin:
    :param p_level:
    :param relative:
    :param figsize:
    :param FONT_SIZE:
    :return:
    """
    if figsize is None:
        figsize = [6, 4]
    fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, len(sens_cases)]})
    plot_bars_2gr_diff(axs, ctrl, nested_cases, relative, sens_cases, var, versions, versions_dic)
    plt.tight_layout()
    plt_path = 'plots/bar_plots/'
    figname = plt_path + area + '/'#
    for version in versions:
        figname = figname + '_' + version.replace(' ','_')
    figname = figname + '/' + var
    if avg_over_lev:
        figname = figname + '_2lev%.0f' %pmin
    else:
        figname = figname + '_atlev%.0f' %p_level
    if not pressure_coord:
        figname = figname + '_not_pres_coord'
    for version in versions:
        figname = figname + '_' + version.replace(' ','_')
    for case in sens_cases:
        figname = figname + '_' + case.replace(' ','_')
    if relative:
        figname = figname + 'relative'
    figname = figname + '.png'
    axs[1].set_title('Difference to CTRL')#, fontsize=FONT_SIZE)
    axs[0].set_title('CTRL')#, fontsize=FONT_SIZE)
    practical_functions.make_folders(figname)
    print(figname)
    plt.savefig(figname, dpi=300)
    return axs


def plot_bars_2gr_diff(axs, ctrl, nested_cases, relative, sens_cases, var, versions, versions_dic):
    case_types = [ctrl] + sens_cases  # case_types.union(versions_dic[version])
    abs_df = pd.DataFrame(columns=versions, index=case_types)  # [ctrl]+sens_cases)
    for version in versions:
        for case_type in set(versions_dic[version]).intersection(case_types):
            abs_df.loc[case_type, version] = float(nested_cases[versions_dic[version][case_type]][var].values)
    diff_df = pd.DataFrame(columns=versions, index=list(set(case_types) - {ctrl}))
    # for case_type in list(case_types-{ctrl}):
    if relative:
        diff_df = 100 * (abs_df.loc[list(set(case_types) - {ctrl}), :] - abs_df.loc[ctrl, :]) / np.abs(
            abs_df.loc[ctrl, :])
    else:
        diff_df = abs_df.loc[list(set(case_types) - {ctrl}), :] - abs_df.loc[ctrl, :]
    abs_df.loc[ctrl, :].plot.bar(ax=axs[0], legend=False)#, fontsize=FONT_SIZE)
    diff_df.plot.bar(ax=axs[1])#, fontsize=FONT_SIZE)
    d_xr = nested_cases[versions_dic[versions[0]][ctrl]][var]
    axs[0].set_ylabel(bs_fdbck.util.naming_conventions.var_info.get_fancy_var_name_xr(d_xr, var) +
                      ' [%s]' % bs_fdbck.util.naming_conventions.var_info.get_fancy_unit_xr(d_xr, var))#, fontsize=FONT_SIZE)
    if relative:
        axs[1].set_ylabel(bs_fdbck.util.naming_conventions.var_info.get_fancy_var_name_xr(d_xr, var) + ' [%]')#,
                          #fontsize=FONT_SIZE)
    else:
        axs[1].set_ylabel(bs_fdbck.util.naming_conventions.var_info.get_fancy_var_name_xr(d_xr, var) +
                          ' [%s]' % bs_fdbck.util.naming_conventions.var_info.get_fancy_unit_xr(d_xr, var))#, fontsize=FONT_SIZE)
    plt.legend()#fontsize=FONT_SIZE)