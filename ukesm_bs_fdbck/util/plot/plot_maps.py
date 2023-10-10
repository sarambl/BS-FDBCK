# import analysis_tools.import_fields_xr as import_fields_xr
import ukesm_bs_fdbck.util.naming_conventions.var_info
# import analysis_tools.fix_xa_dataset as fix_xa_dataset
# from ukesm_bs_fdbck.util.plot import  plot_settings
from ukesm_bs_fdbck.data_info import get_nice_name_case, get_area_specs
from ukesm_bs_fdbck.util.slice_average.area_mod import lon180_2lon360
from ukesm_bs_fdbck.util.slice_average.avg_pkg import average_model_var  # , get_lat_wgts_matrix, masked_average
# import analysis_tools.area_pkg_sara as area_pkg_sara
import numpy as np
import matplotlib.ticker as mticker
from ukesm_bs_fdbck.util.naming_conventions.var_info import get_fancy_var_name
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
# %%

import useful_scit.plot

map_projection = ccrs.Miller()
default_save_path = 'plots/maps'
default_save_path_levlat = 'plots/levlat'
# %%

# %%

def fix_axis4map_plot(ax):
    ax.set_global()
    #ax.set_extent((-190,190,-90,90,),crs=ccrs.PlateCarree())

    ax.coastlines(linewidth=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    #gl:plt.grid
    #gl.left_labels(top=False)

    gl.ylabels_left = False
    # gl.xlines = False
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
    return


def frelative(xr1, xr2):
    out_xr = (xr1 - xr2) / np.abs(xr2) * 100
    out_xr.attrs['units'] = '%'
    return out_xr


def fdifference(xr1, xr2): return xr1 - xr2


def subplots_map(*vars, **kwargs):
    """
    Subplots with map stuff
    :param vars:
    :param kwargs:
    :return:
    """
    #if 'rasterized' not in kwargs:
    #    kwargs['rasterized'] = True
    if 'subplot_kw' not in kwargs:
        kwargs['subplot_kw'] = {'projection': ccrs.Robinson()}
    elif 'projection' not in kwargs['subplot_kw']:
        kwargs['subplot_kw']['projection'] = ccrs.Robinson()
    return plt.subplots(*vars, **kwargs)



def plot_map(var, case, cases_dic,
             figsize=None,
             kwargs_abs=None, ax=None,
             cmap_abs='Reds',
             cbar_orientation='vertical',
             **kwargs):
    """
    Plot absolute for case
    :param cbar_orientation:
    :param ax:
    :param var: variable to plot
    :param case: cases to plot
    :param cases_dic: dictionary of datasets, keys are case names.
    :param figsize: figure size
    :param kwargs_abs:
    :param cmap_abs:
    :return:
    """

    plt_var = cases_dic[case][var]

    if kwargs_abs is None:
        kwargs_abs = {}
    if figsize is None:
        figsize = [6, 3]
    if 'cmap' not in kwargs_abs:
        kwargs_abs['cmap'] = cmap_abs


    label = get_fancy_var_name(var) + ' [%s]' % ukesm_bs_fdbck.util.naming_conventions.var_info.get_fancy_unit_xr(plt_var,
                                                                                                                var)
    cba_kwargs = dict(aspect=12, label=label)
    if cbar_orientation=='horizontal':
        cba_kwargs['orientation']=cbar_orientation
        cba_kwargs['pad']=0.05
        cba_kwargs['shrink']=0.75
        cba_kwargs['aspect']=16
    if 'add_colorbar' in kwargs:
        if kwargs['add_colorbar']:
            kwargs_abs['cbar_kwargs'] = cba_kwargs
    else:
        kwargs_abs['cbar_kwargs'] = cba_kwargs
    ax, im = plt_map(plt_var, ax, figsize, **kwargs_abs,**kwargs)

    unit = ukesm_bs_fdbck.util.naming_conventions.var_info.get_fancy_unit_xr(plt_var, var)
    glob_avg_ctrl = get_global_avg_map(case, cases_dic, var)  # .values
    glob_diff = ', $\mu$=%.1f %s' % (glob_avg_ctrl, unit)

    tit = set_title_abs(case, glob_diff=glob_diff)
    ax.set_title(tit)
    ax.set_aspect('auto', adjustable=None)

    plt.tight_layout()
    return ax, im


def plt_map(plt_var, ax=None, figsize=None, **kwargs_abs):
    if 'rasterized' not in kwargs_abs:
        kwargs_abs['rasterized'] = True
    if ax is None:
        fig, ax = subplots_map(1, figsize=figsize, subplot_kw={'projection': ccrs.Robinson()})  # Orthographic(10, 0))
    kwargs_abs['robust'] = True
    im = plt_var.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs_abs)  # vmin=vmin, vmax=vmax)
    fix_axis4map_plot(ax)
    return ax, im


def plot_map_diff(var,
                  case_ctrl,
                  case_oth,
                  cases_dic,
                  figsize=None,
                  relative=False,
                  tit_ext='',
                  contourf=False,
                  ax=None,
                  cmap_diff='RdBu_r',
                  cbar_orientation='vertical',
                  rasterized=True,
                  **kwargs_diff):
    """
    Plot absolute for case
    :param case_oth:
    :param relative:
    :param tit_ext:
    :param contourf:
    :param cbar_orientation:
    :param rasterized:
    :param ax:
    :param var: variable to plot
    :param case_ctrl: cases to plot
    :param cases_dic: dictionary of datasets, keys are case names.
    :param figsize: figure size
    :param kwargs_diff:
    :param cmap_diff:
    :return:
    """
    ctrl_da = cases_dic[case_oth][var]
    if relative:
        func = frelative
        unit_diff = ' [%]'
        label = 'rel.$\Delta$'+ get_fancy_var_name(var) + unit_diff

    else:
        func = fdifference
        unit_diff = ' [%s]' % ukesm_bs_fdbck.util.naming_conventions.var_info.get_fancy_unit_xr(ctrl_da, var)
        label = '$\Delta$'+ get_fancy_var_name(var) + unit_diff

    #get_vmin_vmax(plt_not_ctrl)
    if 'rasterized' not in kwargs_diff:
        kwargs_diff['rasterized']=rasterized
    if kwargs_diff is None:
        kwargs_diff = {}
    if figsize is None:
        figsize = [6, 3]
    if 'cmap' not in kwargs_diff:
        kwargs_diff['cmap'] = cmap_diff
    if ax is None:
        fig, ax = subplots_map(1, figsize=figsize, subplot_kw={'projection': ccrs.Robinson()})  # Orthographic(10, 0))
    kwargs_diff['robust'] = True
    #plt_var = cases_dic[case][var]
    plt_var = func(cases_dic[case_oth][var], cases_dic[case_ctrl][var])
    if 'vmax' not in kwargs_diff:
        set_vmin_vmax_diff([case_oth],False, case_ctrl, func, kwargs_diff, cases_dic,var)

    cba_kwargs = dict(aspect=12, label=label, shrink=0.95)
    if cbar_orientation=='horizontal':
        cba_kwargs['orientation']=cbar_orientation
        cba_kwargs['pad']=0.05
    #
    if 'add_colorbar' in kwargs_diff.keys():
        if kwargs_diff['add_colorbar']:
            kwargs_diff['cbar_kwargs'] = cba_kwargs
    else:
        kwargs_diff['cbar_kwargs'] = cba_kwargs

    if contourf:
        im = plt_var.plot.contourf(ax=ax,
                                   transform=ccrs.PlateCarree(),
                                   **kwargs_diff)
    else:
        if 'locator' in kwargs_diff:
            kwargs_diff.pop('locator')
        im = plt_var.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs_diff)
    glob_diff = get_avg_diff(case_oth, case_ctrl, cases_dic, relative, var)
    tit = set_title_diff(case_ctrl, case_oth, relative, glob_diff=glob_diff)
    ax.set_title(tit+tit_ext)
    ax.set_aspect('auto', adjustable=None)

    fix_axis4map_plot(ax)
    return ax,im, kwargs_diff



def make_box(area, ax, facecolor=None, alpha=0., edgecolor='#66a61e', kwargs=None):
    # %%
    if kwargs is None:
        kwargs = dict()
    area_specs = get_area_specs(area)
    if area_specs is None:
        print('ups, didnt find area!')
        #return
    min_lat = area_specs['min_lat']
    max_lat = area_specs['max_lat']
    min_lon = area_specs['min_lon']
    max_lon = area_specs['max_lon']
    min_lon = lon180_2lon360(min_lon)
    max_lon = lon180_2lon360(max_lon)
    box = sgeom.box(minx=min_lon, maxx=max_lon, miny=min_lat, maxy=max_lat)
    ax.add_geometries([box], ccrs.PlateCarree(), facecolor=facecolor,
                      edgecolor=edgecolor, alpha=alpha, **kwargs)
    # %%






# %%

def plot_map_abs_abs_diff(var, cases, cases_dic, relative=False, figsize=None, cbar_equal=True,
                          kwargs_diff=None,
                          kwargs_abs=None, axs=None, cmap_abs='Reds', cmap_diff='RdBu_r'):
    """
    Plots absolute absolute difference for cases in cases
    :param var: variable to plot
    :param cases: cases to plot
    :param cases_dic: dictionary of datasets, keys are case names.
    :param relative: if difference relative
    :param figsize: figure size
    :param cbar_equal:
    :param kwargs_diff:
    :param kwargs_abs:
    :param axs:
    :param cmap_abs:
    :param cmap_diff:
    :return:
    """
    # Default values:
    if figsize is None:
        figsize = [15, 3]
    if kwargs_diff is None:
        kwargs_diff = {}
    if kwargs_abs is None:
        kwargs_abs = {}
    if 'cmap' not in kwargs_abs:
        kwargs_abs['cmap'] = cmap_abs
    if 'cmap' not in kwargs_diff:
        kwargs_diff['cmap'] = cmap_diff
    # set ctrl case:
    ctrl_case = cases[0]
    n_rows = int(np.ceil(len(cases) / 2))
    if axs is None:
        if len(cases) == 1:
            fig, axs = subplots_map(1, figsize=figsize,
                                    subplot_kw={'projection': ccrs.Robinson()})  # Orthographic(10, 0))
            axs = [axs]
        else:
            fig, axs = subplots_map(n_rows, 3, figsize=figsize,
                                    subplot_kw={'projection': ccrs.Robinson()})  # Orthographic(10, 0))

    if relative:
        func = frelative
    else:
        func = fdifference
    # Difference to ctr
    set_vmin_vmax_diff(cases, cbar_equal, ctrl_case, func, kwargs_diff, cases_dic, var)

    set_vminmax_abs(cases, cases_dic, cbar_equal, kwargs_abs, var)
    # %%
    cases_not_ctrl = cases[1:]
    for i, case_oth in enumerate(cases_not_ctrl):
        if len(cases_not_ctrl)==1:
            saxs= axs
        else:
            saxs = axs[i,:]
        for ax, case in zip(saxs,[case_oth,ctrl_case]):
            plot_map(var,case, cases_dic, kwargs_abs=kwargs_abs, ax=ax,cmap_abs=cmap_abs)
        ax = saxs[-1]
        plot_map_diff(var, ctrl_case, case_oth,cases_dic,relative=relative,
                    ax=ax,**kwargs_diff)
    plt.tight_layout()
    return axs, kwargs_diff.copy()


def set_vminmax_abs(cases, cases_dic, cbar_equal, kwargs_abs, var):
    plt_abs = [cases_dic[case][var] for case in cases]
    vmax_abs, vmin_abs = get_vmin_vmax(plt_abs)
    if cbar_equal and 'vmin' not in kwargs_abs:
        kwargs_abs['vmin'] = vmin_abs
        kwargs_abs['vmax'] = vmax_abs
    if (vmin_abs<0) and (vmax_abs>0):
        kwargs_abs['cmap']='RdBu_r'
    if (vmin_abs<0) and (vmax_abs<0):
        kwargs_abs['cmap']='Blues'



# %%
def plot_map_diff_2case(var, case1, case2, cases_dic, relative=False, ax=None, vmax=None, vmin=None,
                        kwargs_diff=None, cmap_diff='RdBu_r', figsize=None, cbar_eq = False):
    if kwargs_diff is None:
        kwargs_diff = {}
    if vmin is not None:
        kwargs_diff['vmin'] = vmin
    if vmax is not None:
        kwargs_diff['vmax'] = vmax

    plot_map_diff(var,case1,case2,cases_dic,figsize=figsize, relative=relative,ax=ax,
                  cmap_diff=cmap_diff, **kwargs_diff)
    return
def iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True

def plot_map_diff_only(var, cases, cases_dic, relative=False, figsize=None, cbar_equal=True,
                       kwargs_diff=None, axs=None, cmap_diff='RdBu_r',
                       cbar_loc='side', tight_layout=True, inverse_diff=False):
    """
    Plots absolute absolute difference for cases in cases
    :param cbar_loc:
    :param tight_layout:
    :param inverse_diff:
    :param var: variable to plot
    :param cases: cases to plot
    :param cases_dic: dictionary of datasets, keys are case names.
    :param relative: if difference relative
    :param figsize: figure size
    :param cbar_equal:
    :param kwargs_diff:
    :param axs:
    :param cmap_diff:
    :return:
    """
    if figsize is None:
        figsize = [4, 3]
    if kwargs_diff is None:
        kwargs_diff = {}
    if not iterable(axs):
        axs=[axs]
    if 'cmap' not in kwargs_diff:
        kwargs_diff['cmap'] = cmap_diff

    ctrl_case = cases[0]
    n_rows = int(np.ceil((len(cases) - 1) / 2))
    if axs is None:
        if len(cases) == 2:
            fig, axs = subplots_map(1, figsize=figsize,
                                    subplot_kw={'projection': ccrs.Robinson()})  # Orthographic(10, 0))
            axs = [axs]

        else:
            fig, axs = subplots_map(n_rows, 2, figsize=figsize,
                                    subplot_kw={'projection': ccrs.Robinson()})  # Orthographic(10, 0))
            axs = axs.flatten()
    if relative:
        func = frelative
    else:
        func = fdifference
    # Difference to ctr
    set_vmin_vmax_diff(cases, cbar_equal, ctrl_case, func, kwargs_diff, cases_dic, var, inv_diff=inverse_diff)
    print(cases[1:])
    print(axs)
    for to_case, ax in zip(cases[1:], axs):
        if inverse_diff:
            _to_case = ctrl_case
            _ctrl_case = to_case
        else:
            _to_case = to_case
            _ctrl_case = ctrl_case
        if cbar_loc=='under':
            cbar_orientation='horizontal'
        else:
            cbar_orientation='vertical'

        ax,im, kwargs_diff = plot_map_diff(var,
                                        _ctrl_case,
                                        _to_case,
                                        cases_dic,
                                        relative=relative,
                                        ax=ax,
                                        cbar_orientation=cbar_orientation,
                                        **kwargs_diff)
    if tight_layout:
        plt.tight_layout()
    return axs, kwargs_diff.copy()


def plot_map_abs_only(var, cases, cases_dic, relative=False, figsize=None, cbar_equal=True,
                      kwargs_abs=None, axs=None, cmap_abs='Reds', cbar_loc='side', nr_rows=1):
    """
    Plots absolute absolute difference for cases in cases
    :param cbar_loc:
    :param nr_rows:
    :param var: variable to plot
    :param cases: cases to plot
    :param cases_dic: dictionary of datasets, keys are case names.
    :param relative: if difference relative
    :param figsize: figure size
    :param cbar_equal:
    :param kwargs_abs:
    :param axs:
    :param cmap_abs:
    :return:
    """
    if figsize is None:
        figsize = [4, 3]
    if kwargs_abs is None:
        kwargs_abs = {}
    if 'cmap' not in kwargs_abs:
        kwargs_abs['cmap'] = cmap_abs

    ctrl_case = cases[0]
    nr_cols = int(np.ceil((len(cases) - 1) / nr_rows))
    if axs is None:
        if len(cases) == 2:
            fig, axs = subplots_map(1, figsize=figsize,
                                    subplot_kw={'projection': ccrs.Robinson()})  # Orthographic(10, 0))
            axs = [axs]

        else:
            fig, axs = subplots_map(nr_rows, 2, figsize=figsize,
                                    subplot_kw={'projection': ccrs.Robinson()})  # Orthographic(10, 0))
            axs = axs.flatten()

    set_vminmax_abs(cases, cases_dic, cbar_equal, kwargs_abs, var)

    for case, ax in zip(cases, axs):
        if cbar_loc=='under':
            cbar_orientation='horizontal'
        else:
            cbar_orientation='vertical'
            plot_map(var,case, cases_dic, kwargs_abs=kwargs_abs, ax=ax,cmap_abs=cmap_abs,
                     cbar_orientation=cbar_orientation)

        #ax, kwargs_abs = plot_map_diff(var, ctrl_case, case, cases_dic, relative=relative,
        #                               kwargs_diff=kwargs_abs, ax=ax,
        #                               cbar_orientation=cbar_orientation)
    plt.tight_layout()
    return axs, kwargs_abs.copy()


def plot_map_abs_only_onecase(varl, case, cases_dic, relative=False, sfigsize=None, cbar_equal=True,
                              kwargs_abs=None, axs=None, cmap_abs='Reds', cbar_loc='side', nr_rows=1):
    """
    Plots absolute absolute difference for cases in cases
    :param varl:
    :param case:
    :param sfigsize:
    :param cbar_loc:
    :param nr_rows:
    :param cases_dic: dictionary of datasets, keys are case names.
    :param relative: if difference relative
    :param cbar_equal:
    :param kwargs_abs:
    :param axs:
    :param cmap_abs:
    :return:
    """
    if kwargs_abs is None:
        kwargs_abs = {}
    if sfigsize is None:
        sfigsize=[3,4]

    if 'cmap' not in kwargs_abs:
        kwargs_abs['cmap'] = cmap_abs
    ctrl_case = case
    nr_cols = int(np.ceil((len(varl) ) / nr_rows))
    figsize=[sfigsize[0]*nr_cols, sfigsize[1]*nr_rows]

    if axs is None:
        fig, axs = subplots_map(nr_rows,nr_cols, figsize=figsize,
                                    subplot_kw={'projection': ccrs.Robinson()})  # Orthographic(10, 0))
        if len(varl) == 1:
            axs = [axs]
        else:
            axs = axs.flatten()


    for var, ax in zip(varl, axs):
        if cbar_loc=='under':
            cbar_orientation='horizontal'
        else:
            cbar_orientation='vertical'
            plot_map(var,case, cases_dic, kwargs_abs=kwargs_abs, ax=ax,cmap_abs=cmap_abs,
                     cbar_orientation=cbar_orientation)

        #ax, kwargs_abs = plot_map_diff(var, ctrl_case, case, cases_dic, relative=relative,
        #                               kwargs_diff=kwargs_abs, ax=ax,
        #                               cbar_orientation=cbar_orientation)
    plt.tight_layout()
    return axs, kwargs_abs.copy()











def set_title_diff(case_ctrl, case_oth, relative, glob_diff=''):
    nn_ctrl = get_nice_name_case(case_ctrl)
    nn_oth = get_nice_name_case(case_oth)
    if relative:
        return f'{nn_oth} to {nn_ctrl}{glob_diff}'
    else:
        return f'{nn_oth}-{nn_ctrl}{glob_diff}'

def set_title_abs(case_ctrl, glob_diff=''):
    nn_ctrl = get_nice_name_case(case_ctrl)
    return f'{nn_ctrl}{glob_diff}'

# %%


def set_vmin_vmax_diff(cases, cbar_equal, ctrl_case, func, kwargs_diff, nested_cases, var, inv_diff=False):
    if inv_diff:
        plt_not_ctrl = [func(nested_cases[ctrl_case][var], nested_cases[case][var]) for case in
                        (set(cases) - {ctrl_case})]
    else:
        plt_not_ctrl = [func(nested_cases[case][var], nested_cases[ctrl_case][var]) for case in
                    (set(cases) - {ctrl_case})]
    if 'vmin' in kwargs_diff and 'vmax' in kwargs_diff:
        if kwargs_diff['vmin'] is not None:
            return
    vmax, vmin = get_vmin_vmax(plt_not_ctrl)
    kwargs_diff['robust'] = True
    if cbar_equal and 'vmin' not in kwargs_diff:
        kwargs_diff['vmin'] = vmin
        kwargs_diff['vmax'] = vmax
        if vmin < 0 and vmax <= 0:
            print(vmin, vmax)
            print('blues')
            kwargs_diff['cmap'] = 'Blues_r'
        elif vmin > 0 and vmax > 0:
            kwargs_diff['cmap'] = 'Reds'
            print('reds')
    return


def get_vmin_vmax(plt_not_ctrl, quant=0.01):
    vmin, vmax = useful_scit.plot.plot.calc_vmin_vmax(plt_not_ctrl, quant=quant)
    vmin= vmin#.values
    vmax = vmax#.values
    if vmin < 0 < vmax:
        minmax = max(abs(vmin), vmax)
        vmin = -minmax
        vmax = minmax
    return vmax, vmin


def get_global_avg_map(ctrl_case, nested_cases, var):
    glob_avg_ctrl = average_model_var(nested_cases[ctrl_case], var)[var]
    return glob_avg_ctrl.values


def get_avg_diff(case_oth, case_ctrl, case_dic, relative, var):
    """

    :param case_oth:
    :param case_ctrl:
    :param case_dic:
    :param relative:
    :param var:
    :return:
    """
    # glob_avg_ctr = masked_average(xr1, weights=lat_wgt)
    glob_avg_ctrl = average_model_var(case_dic[case_ctrl], var)[var].values
    glob_avg_oth = average_model_var(case_dic[case_oth], var)[var].values
    if relative:
        _a = (glob_avg_oth - glob_avg_ctrl) / abs(glob_avg_ctrl) * 100
        return ', $\mu$=%.2f %%' % _a
    else:
        _a = (glob_avg_oth - glob_avg_ctrl)
        unit = ukesm_bs_fdbck.util.naming_conventions.var_info.get_fancy_unit_xr(case_dic[case_ctrl][var], var)
        return ', $\mu$=%.2f %s' % (_a, unit)


def save_map_name(var, cases, pressure, pressure_coord, to_pressure, avg_lev, pmin, path=default_save_path,
                  logscale=False, relative=False, addstring=''):
    plotname = path + '/' + var + '_' + addstring
    if relative: plotname = plotname + '_rel'
    for case in cases:
        plotname = plotname + '_' + case.replace(' ', '-')
    if avg_lev:
        plotname = plotname + '_to_lev%.0f' % to_pressure
    else:
        plotname = plotname + '_lev%.0f' % pressure
    if logscale: plotname = plotname + '_logscale'
    if pressure_coord:
        plotname = plotname + '_pres_coords'
    plotname = plotname + '.png'
    return plotname



