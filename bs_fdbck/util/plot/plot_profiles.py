import numpy as np
import matplotlib.pyplot as plt

# import analysis_tools.practical_functions
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
from matplotlib.ticker import ScalarFormatter

from bs_fdbck.data_info import get_area_defs_pd
from bs_fdbck.util.naming_conventions.var_info import get_fancy_var_name

import bs_fdbck.util.naming_conventions.var_info
from bs_fdbck.util.plot.colors import get_area_col


# %%

def plot_profile_multivar(ds, case, var, ax, xscale='log', yscale='log', label='',
                          ylim=None, pressure_coords=True, kwargs=None, title=''):
    if ylim is None:
        ylim = [1e3, 100]
    if kwargs is None:
        kwargs = {}
    ds[var].where(np.logical_and(ds['lev'] < ylim[0], ds['lev'] > ylim[1])).plot(y='lev', label=var, xscale=xscale,
                                                                                 yscale=yscale, **kwargs, ax=ax,
                                                                                 ylim=ylim)
    ax.set_title(case)
    if pressure_coords:
        ax.set_ylabel('Pressure [hPa]')
    ax.grid(True, which='both')
    ax.set_ylim(ylim)


def plot_profile(da, ax, xscale='log', yscale='log',
                 label='',
                 ylim=None,
                 pressure_coords=True,
                 kwargs=None,
                 title='',**more_kwargs):
    if kwargs is None:
        kwargs = {}
    if ylim is None:
        ylim = [1e3, 100]
    if 'ylim' in kwargs:
        ylim = kwargs['ylim']
    for key, val in zip(['xscale', 'yscale', 'ylim', 'label'], [xscale, yscale, ylim, label]):
        if key not in kwargs:
            kwargs[key] = val
    plt_da = da.where(np.logical_and(da['lev'] <= ylim[0], da['lev'] >= ylim[1]))
    plt_da.plot(y='lev', **kwargs, ax=ax)

    ylim = [da.lev.where(plt_da.notnull()).max(),
            da.lev.where(plt_da.notnull()).min()]
    ax.set_ylim(ylim)
    if len(title) > 0:
        ax.set_title(title)
    if pressure_coords:
        ax.set_ylabel('Pressure [hPa]')
    var = da.name
    xlabel = get_fancy_var_name(var) + ' [%s]' % bs_fdbck.util.naming_conventions.var_info.get_fancy_unit_xr(da,
                                                                                                                 var)
    ax.set_xlabel(xlabel)
    ax.grid(True, which='both')


def plot_var_multicase(ax, var, nested_cases, cases, pltkwargs=None, area='', area_in_label=False):
    if pltkwargs is None:
        pltkwargs = {}
    linecolors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
    for case, linecolor in zip(cases, linecolors):
        if area_in_label:
            label = area + ': ' + case
        else:
            label = case
        if 'color' in pltkwargs:
            plot_profile(nested_cases[case][var], ax=ax, label=label, kwargs=pltkwargs)
        else:
            plot_profile(nested_cases[case][var], ax=ax, label=label, kwargs={**pltkwargs, 'color': linecolor})
    xlab = '%s [%s]' % (
    bs_fdbck.util.naming_conventions.var_info.get_fancy_var_name_xr(nested_cases[case][var], var),
    bs_fdbck.util.naming_conventions.var_info.get_fancy_unit_xr(nested_cases[case][var], var))
    ax.set_xlabel(xlab)


def plot_var_multicase_sets_cases(caselistlist, list_nested_cases, plot_vars_list, nr_rows=2,
                                  figsize=None, xscale='log', title='', ylim=None, axs=None,
                                  color=None, area_in_label=False, area='', legend=True):
    if ylim is None:
        ylim = [1e3, 100]
    if figsize is None:
        figsize = [12, 8]
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    nr_cols = int(np.ceil(max([len(plot_vars) for plot_vars in plot_vars_list]) / nr_rows))
    if axs is None:
        fig, axs = plt.subplots(nr_rows, nr_cols, figsize=figsize, sharey=True)
    all_vars = []
    for plot_vars in plot_vars_list:
        for var in plot_vars:
            if var not in all_vars:
                all_vars.append(var)
    # if not isinstance(axs, np.ndarray):
    #    axs = axs
    # else: axs=axs.flatten()
    for var, ax in zip(all_vars, axs.flatten()):
        ii = 0
        ax.set_title(title)
        for cases, nested_cases, plot_vars in zip(caselistlist, list_nested_cases, plot_vars_list):
            pltkwargs = {'linestyle': linestyles[ii], 'xscale': xscale, 'ylim': ylim}
            if color is not None:
                pltkwargs['color'] = color
            if var in plot_vars:
                plot_var_multicase(ax, var, nested_cases, cases, pltkwargs=pltkwargs, area=area,
                                   area_in_label=area_in_label)
            ii += 1
        ax.set_xscale(xscale)
        if len(title) > 0:
            ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    # plt.show()
    return axs


def plot_profiles_multivar_cases(axs, nested_cases, cases, plot_vars, xlim=None, xscale='log', kwargs=None,
                                 title=''):
    if kwargs is None:
        kwargs = {}
    for case, ax in zip(cases, axs.flatten()):
        for var in plot_vars:
            plot_profile_multivar(nested_cases[case], case, var, ax=ax, label=case, kwargs=kwargs, title=title,
                                  xscale=xscale)
        ax.legend()
        if xlim is not None:
            ax.set_xlim(xlim)


def plot_profiles_multivar_sets_cases(caselistlist, list_nested_cases, plot_vars_list, figsize=None,
                                      xscale='log', title=''):
    """
    Ex:
    cases_gr1 =[case_11, case12]
    cases_gr2 = [case_21, case22]
    nested_cases_gr1 ={case_11: xr_ds11, case_12:xr_ds12}
    nested_cases_gr2 ={case_21: xr_ds21, case_22:xr_ds22}
    plot_vars_gr1 = ['NCONC01', 'NCONC02']
    plot_vars_gr2 = ['NCONC01', 'NCONC02', 'ACTREL']
    axs= plot_profiles_multivar_sets_cases([cases_gr1, cases_gr2],
                [nested_cases_gr1, nested_gases_gr2],
                [plot_vars_gr1, plot_vars_gr2], figsize = figsize, xscale=xscale)

    :param caselistlist: list of lists of cases
    :param list_nested_cases: list of dic containing xr Datasets for each case
    :param plot_vars_list: list of variables for groups of cases
    :param figsize: figsize
    :param xscale: xscale
    :param title: plot title
    :return:
    """
    if figsize is None:
        figsize = [12, 8]
    nr_cols = int(max([len(cases) for cases in caselistlist]))
    nr_rows = len(caselistlist)
    fig, axs = plt.subplots(nr_rows, nr_cols, figsize=figsize, sharex=True, sharey=True)
    tot_cases = []
    tot_nested = {}
    for cases in caselistlist:
        tot_cases = tot_cases + cases
    for nested_cases in list_nested_cases:
        tot_nested = {**tot_nested, **nested_cases}
    for cases, nested_cases, plot_vars, axind in zip(caselistlist, list_nested_cases, plot_vars_list,
                                                     range(0, nr_rows)):
        plot_profiles_multivar_cases(axs[axind, :], nested_cases, cases, plot_vars, xscale=xscale, title=title)
    plt.tight_layout()
    plt.show()
    return axs


def get_minmax_list_xr(list_xr, varList):
    minv = 9e99
    maxv = -9e99
    for xr_ds in list_xr:
        for var in varList:
            minv = min(minv, xr_ds[var].min().values)
            maxv = max(maxv, xr_ds[var].max().values)
    return [minv, maxv]


def set_legend_area_profs(ax, areas, cases, linestd,
                          textsize=10,
                          loc_area='lower right',
                          loc_case='upper right',
                          bbox_to_anchor_area=(.68, .1, .4, 1.),
                          bbox_to_anchor_case=(.68, .8, .4, .2),

                          ):
    custom_lines = []

    for area in areas:
        custom_lines.append(Line2D([0], [0], color=get_area_col(area), lw=1))
    df_area = get_area_defs_pd().transpose().to_dict()
    for area in areas:
        if area not in df_area:
            df_area[area] = dict(nice_name=area)
    areas_nice = [df_area[area]['nice_name'] for area in areas]
    lgn1 = ax.legend(custom_lines, areas_nice, title_fontsize=textsize, prop=dict(size=textsize), title="Area:",
                     bbox_to_anchor=bbox_to_anchor_area,
                     loc=loc_area,
                     frameon=False)

    ax.add_artist(lgn1)
    custom_lines = []

    for case in cases:
        custom_lines.append(Line2D([0], [0], color='k', linestyle=linestd[case], lw=1))
    lgn2 = ax.legend(custom_lines, cases, title_fontsize=textsize, prop=dict(size=textsize), title="Model:",
                     bbox_to_anchor=bbox_to_anchor_case, loc=loc_case,
                     frameon=False)
    ax.add_artist(lgn2)


def set_scalar_formatter(ax):
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_minor_formatter(formatter)

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
