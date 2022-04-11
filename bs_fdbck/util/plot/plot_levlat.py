# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import useful_scit.plot.plot

from bs_fdbck.data_info import get_nice_name_case
from bs_fdbck.util.naming_conventions.var_info import get_fancy_var_name, get_fancy_unit_xr


# %%

#def plot_levlat_abs(var, case, cases_dic, title=None, ax=None, ylim=None, cbar_orientation='vertical', figsize=None,
#                    cmap=None, **plt_kwargs):
#    plt_xa = cases_dic[case][var]
#    nn_case = get_nice_name_case(case)
#    if ax is None:
#        fig, ax = plt.subplots(1, figsize=figsize)
#    if title is None:
#        title = nn_case
#    label = get_fancy_var_name(var) + ' [%s]' % get_fancy_unit_xr(plt_xa, var)
#    plt_kwargs = make_cbar_kwargs(label, plt_kwargs, cbar_orientation)
#    ax = plot_levlat(ax, cmap, plt_xa, title, ylim, **plt_kwargs)
#    return ax
    # %%


def make_cbar_kwargs(label, plt_kwargs, cbar_orientation='vertical'):
    cba_kwargs = dict(aspect=12, label=label)
    if cbar_orientation == 'horizontal':
        cba_kwargs['orientation'] = cbar_orientation
        cba_kwargs['pad'] = 0.05
        cba_kwargs['shrink'] = 0.75
        cba_kwargs['aspect'] = 16
    plt_kwargs['cbar_kwargs'] = cba_kwargs
    return plt_kwargs


def plot_levlat(ax, cmap, plt_xa, title, ylim, **plt_kwargs):
    if ax is None:
        fig, ax = plt.subplots(1)  # , figsize=figsize)

    if ylim is None:
        ylim = [1e3, plt_xa['lev'].min()]
    if 'rasterized' not in plt_kwargs:
        plt_kwargs['rasterized'] = True
    plt_kwargs['cmap'] = cmap
    plt_kwargs['ylim'] = ylim
    plt_kwargs['yscale'] = 'log'
    #plt_kwargs['norm'] = colors.LogNorm(vmin=1, vmax=1000)
    plt_kwargs['robust'] = True
    plt_kwargs['ax'] = ax
    im = plt_xa.plot(y='lev', **plt_kwargs)
    ax.set_title(title)
    ax.set_ylim(ylim)

    #plt.show()
    return ax, im

def plot_levlat_abs(var, case, cases_dic,
                    cbar_orientation='vertical',
                    title=None,
                    ax=None,
                    ylim=None,
                    figsize=None,
                    cmap='Reds',
                    add_colorbar=True,
                    use_ds_units=False,
                    **plt_kwargs):
    plt_da = cases_dic[case][var]
    label = get_cbar_label(plt_da, var, use_ds_units)
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    if ylim is None:
        ylim = [1e3, plt_da['lev'].min()]
    if title is None:
        title = set_title_abs(case)
    if add_colorbar:
        plt_kwargs = make_cbar_kwargs(label, plt_kwargs, cbar_orientation)
    plt_kwargs['robust'] = True
    plt_kwargs['add_colorbar']=add_colorbar
    ax, im = plot_levlat(ax, cmap, plt_da, title, ylim, **plt_kwargs)
    #ax.set_yticks([1000,800])
    yticks = ax.get_yticks()#[0:-1:2])
    yticks = yticks[yticks>=ylim[1]]

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.yaxis.set_minor_formatter(mtick.FormatStrFormatter('%.0f'))

    return ax, im


def get_cbar_label(plt_da, var, use_ds_units=False, diff=False, relative=False):

    unit = ' [%s]' % get_fancy_unit_xr(plt_da, var)
    if use_ds_units:
        unit = ' [%s]' % plt_da.units
    label = get_fancy_var_name(var) + unit
    if diff and relative:
        label = f'rel.$\Delta$ {get_fancy_var_name(var)} [%]'
    if diff and not relative:
        label = f'$\Delta$ {get_fancy_var_name(var)} {unit}'

    return label


def plot_levlat_diff(var, case_ctrl, case_oth, cases_dic,
                     cbar_orientation='vertical',
                     relative=False,
                     title=None,
                     ax=None,
                     ylim=None,
                     figsize=None,
                     cmap='RdBu_r',
                     add_colorbar=True,
                     **plt_kwargs):
    if 'rasterized' not in plt_kwargs:
        plt_kwargs['rasterized']= True

    ctrl_da = cases_dic[case_ctrl][var]
    oth_da = cases_dic[case_oth][var]

    if relative:
        func = frelative
        unit_diff = ' [%]'
        label = f'rel.$\Delta${get_fancy_var_name(var)}{unit_diff}'
        # + get_fancy_var_name(var) + unit_diff
    else:
        func = fdifference
        unit_diff = ' [%s]' % get_fancy_unit_xr(ctrl_da, var)
        label = '$\Delta$' + get_fancy_var_name(var) + unit_diff
    plt_xa = func(oth_da, ctrl_da)
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    if ylim is None:
        ylim = [1e3, plt_xa['lev'].min()]
    if title is None:
        title = set_title_diff(case_ctrl, case_oth, relative)
    if add_colorbar:
        plt_kwargs = make_cbar_kwargs(label, plt_kwargs, cbar_orientation)

    plt_kwargs['robust'] = True
    plt_kwargs['add_colorbar']=add_colorbar
    ax, im = plot_levlat(ax, cmap, plt_xa, title, ylim, **plt_kwargs)
    yticks = ax.get_yticks()#[0:-1:2])
    yticks = yticks[yticks<=ylim[1]]
    ax.set_yticks(yticks[0:-1:2])

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.yaxis.set_minor_formatter(mtick.FormatStrFormatter('%.0f'))
    return ax, im


def frelative(xr1, xr2):
    out_xr = (xr1 - xr2) / np.abs(xr2) * 100
    out_xr.attrs['units'] = '%'
    return out_xr


def fdifference(xr1, xr2):
    return xr1 - xr2

def set_title_abs(case_ctrl, glob_diff=''):
    nn_ctrl = get_nice_name_case(case_ctrl)
    return f'{nn_ctrl}{glob_diff}'


def set_title_diff(case_ctrl, case_oth, relative, glob_diff=''):
    nn_ctrl = get_nice_name_case(case_ctrl)
    nn_oth = get_nice_name_case(case_oth)
    if relative:
        return f'{nn_oth} to {nn_ctrl}{glob_diff}'
    else:
        return f'{nn_oth}-{nn_ctrl}{glob_diff}'

def get_cbar_eq_kwargs(cases,  case_oth, relative, cases_dic, var):
    kwargs_diff = dict()
    if relative:
        func = frelative
    else:
        func = fdifference
    set_vmin_vmax_diff(cases,True, case_oth,func,kwargs_diff,cases_dic,var)
    return  kwargs_diff




def set_vmin_vmax_diff(cases, cbar_equal, case_oth, func, kwargs_diff, cases_dic, var):
    plt_not_ctrl = [func(cases_dic[case][var], cases_dic[case_oth][var]) for case in
                    (set(cases) - {case_oth})]
    if 'vmin' in kwargs_diff and 'vmax' in kwargs_diff:
        if kwargs_diff['vmin'] is not None:
            return
    set_vmin_vmax(cbar_equal, kwargs_diff, plt_not_ctrl)
    return

def set_vmin_vmax_abs(cases, cbar_equal, kwargs, cases_dic, var):
    if kwargs is None:
        kwargs=dict()
    plt_not_ctrl = [cases_dic[case][var] for case in
                    cases]
    if 'vmin' in kwargs and 'vmax' in kwargs:
        if kwargs['vmin'] is not None:
            return
    set_vmin_vmax(cbar_equal, kwargs, plt_not_ctrl)
    return kwargs


def set_vmin_vmax(cbar_equal, kwargs, plt_not_ctrl):
    """
    Set vmin and vmax in kwargs for plotting with same colorbar.
    :param cbar_equal:
    :param kwargs:
    :param plt_not_ctrl:
    :return:
    """
    if 'vmin' in kwargs and 'vmax' in kwargs:
        if kwargs['vmin'] is not None:
            return
    vmax, vmin = get_vmin_vmax(plt_not_ctrl)
    kwargs['robust'] = True
    if cbar_equal and 'vmin' not in kwargs:
        kwargs['vmin'] = vmin
        kwargs['vmax'] = vmax
        if vmin < 0 and vmax <= 0:
            kwargs['cmap'] = 'Blues_r'
        elif vmin > 0 and vmax > 0:
            kwargs['cmap'] = 'Reds'
    return

def get_vmin_vmax(plt_not_ctrl):
    vmin, vmax = useful_scit.plot.plot.calc_vmin_vmax(plt_not_ctrl, quant=0.01)
    vmin = vmin.values
    vmax = vmax.values
    if vmin < 0 < vmax:
        minmax = max(abs(vmin), vmax)
        vmin = -minmax
        vmax = minmax
    return vmax, vmin
