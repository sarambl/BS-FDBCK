import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs as ccrs
from matplotlib import colors
from matplotlib import gridspec
from useful_scit.plot.fig_manip import subp_insert_abc

from bs_fdbck.util.naming_conventions.var_info import get_fancy_var_name, get_fancy_unit_xr
from bs_fdbck.util.plot.plot_maps import plot_map_diff, plot_map, frelative, fdifference, \
    get_vmin_vmax


# noinspection DuplicatedCode
def abs_diffs(di_dict,
              vl,
              ctrl=None,
              cases_oth=None,
              sfg_size=2.9,
              asp_rat=.8,
              norm_abs=None,
              relative=False,
              norm_dic=None,
              locator_dic_diff = None,
              locator_dic_abs = None,
              invert_diff = False,
              add_abc=True
              ):
    if norm_abs is None:
        norm_abs = {}
        for v in vl:
            norm_abs[v] = None
    if norm_dic is None:
        norm_dic = {}
        for v in vl:
            norm_dic[v] = None
    if locator_dic_diff is None:
        locator_dic_diff = {}
        for v in vl:
            locator_dic_diff[v] = None

    if cases_oth is None:
        cases_oth = ['OsloAero$_{imp}$', 'OsloAero$_{def}$']
    if ctrl is None:
        ctrl = 'OsloAeroSec'

    height_ratios = [3, .3, 2, 3, .1, 3, .3]
    # plt.rcParams['figure.constrained_layout.use'] = False

    fg = plt.figure(figsize=[sfg_size * len(vl), sfg_size * 3 * asp_rat])
    nrows = len(height_ratios)
    ncols = len(vl)

    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols,
                           height_ratios=height_ratios,
                           hspace=0.2,
                           )  # width_ratios=width_ratios)
    axs_dict = {}
    forc_dic = {'NCFT_Ghan':'ERF$_{aci}$',
                'SWCF_Ghan':'ERF$_{aci,SW}$',
                'LWCF_Ghan':'ERF$_{aci,LW}$',
                }
    for i, var in enumerate(vl):
        # gs_s = gs[:,i]
        axs_dict[var] = {}
        axs_dict[var][ctrl] = fg.add_subplot(gs[0, i], projection=ccrs.Robinson(),
                                             )
        for j, ca in enumerate(cases_oth):
            axs_dict[var][ca] = fg.add_subplot(gs[3 + j * 2, i], projection=ccrs.Robinson())
            # axs_dic[var][cases_oth[1]] =fig.add_subplot(gs[5,i], projection=ccrs.Robinson())

        axs_dict[var]['cbar'] = fg.add_subplot(gs[-1, i])
        axs_dict[var]['cbar_abs'] = fg.add_subplot(gs[1, i])

    for i, var in enumerate(vl):
        saxs = axs_dict[var]
        for case_oth in cases_oth:
            ax = saxs[case_oth]
            if invert_diff:
                from_ca = case_oth
                to_ca = ctrl
            else:
                from_ca = ctrl
                to_ca = case_oth
            ax, im, kw = plot_map_diff(var,
                                       from_ca,
                                       to_ca,
                                       di_dict,
                                       ax=ax,
                                       norm=norm_dic[var],
                                       relative=relative,
                                       #contourf=cont_,
                                       #locator=ticker.SymmetricalLogLocator(linthresh=10,base=10),
                                       add_colorbar=False,
                                           )
            ax.set_title(f'{to_ca}-{from_ca}')
        # cbar:

        if var in forc_dic.keys():#'NCFT_Ghan':
            nnvar = forc_dic[var]#  'ERF$_{aci}$'
        else:
            nnvar = '$\Delta_{\mathrm{PD-PI}}$%s' %get_fancy_var_name(var)
        la = nnvar + ' [%s]' % get_fancy_unit_xr(di_dict[ctrl][var], var)
        la = f'$\Delta${la}'
        # noinspection PyUnboundLocalVariable
        plt.colorbar(im, cax=axs_dict[var]['cbar'],
                     label=la,
                     orientation='horizontal',
                     extend='both')
    for i, var in enumerate(vl):
        ax = axs_dict[var][ctrl]
        # plt_map(di_dic[ctrl][var], ax=ax,
        ax, im = plot_map(var, ctrl, di_dict,
                          ax=ax,
                          kwargs_abs=dict(
                              cmap='PuOr_r'
                          ),
                          add_colorbar=False,
                          norm=norm_abs[var],
                          )
        # cbar:
        if var in forc_dic.keys():#'NCFT_Ghan':
            nnvar = forc_dic[var]#  'ERF$_{aci}$'
        else:
            nnvar = f'$\Delta${get_fancy_var_name(var)}'
            nnvar = '$\Delta_{\mathrm{PD-PI}}$%s' %get_fancy_var_name(var)

        _la =nnvar    + ' [%s]' % get_fancy_unit_xr(di_dict[ctrl][var], var)
        la = f'{_la}'

        plt.colorbar(im, cax=axs_dict[var]['cbar_abs'],
                     label=la,
                     orientation='horizontal',
                     extend='both'
                     )
    if add_abc:
        md_ls = [ctrl] + cases_oth
        axs_abc = []

        for ct in md_ls:
            _axs = [axs_dict[v][ct] for v in vl]
            axs_abc = axs_abc + _axs
        subp_insert_abc(np.array(axs_abc), pos_x=0.0,pos_y=0.0)

    return fg, axs_dict


def abs_diffs_PI_PD_sep(dic_type_mod,
                        var,
                        case_types=None,
                        ctrl=None,
                        cases_oth=None,
                        sfg_size=2.9,
                        asp_rat=.9,
                        relative=False,
                        norm_abs=None,
                        norm_diff=None,
                        type_nndic=None,
                        height_ratios=None,
                        switch_diff=False,
                        add_abc=True,
                        cmap_abs=None
                        ):
    type_nndic={'PI':'Pre-industrial','PIaerPD':'Present day'}
    if case_types is None:
        case_types = ['PI', 'PIaerPD']
    if norm_abs is None:
        norm_abs = None
    if cases_oth is None:
        cases_oth = ['OsloAero$_{imp}$', 'OsloAero$_{def}$']
    if ctrl is None:
        ctrl = 'OsloAeroSec'

    if norm_diff is None:
        # noinspection DuplicatedCode
        norm_diff = get_lin_norm(case_types, cases_oth, ctrl, dic_type_mod,
                                 norm_diff,
                                 relative,
                                 var,
                                 switch_diff=switch_diff
                                 )

    if height_ratios is None:
        height_ratios = [.3, 3, .3, 1.9]+[.1, 3]*len(cases_oth)+ [.3]

    fg = plt.figure(figsize=[sfg_size * len(case_types),
                             sfg_size * (len(cases_oth)+1) * asp_rat],
                    dpi=150,
                    )

    nrows = len(height_ratios)
    ncols = len(case_types)

    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols,
                           height_ratios=height_ratios,
                           hspace=0.2,
                           )  # width_ratios=width_ratios)
    # Add subplots:
    axs_dict = setup_subplots(case_types, cases_oth, ctrl, fg, gs)
    # plot diffs:
    for i, ct in enumerate(case_types):
        saxs = axs_dict[ct]
        for case_oth in cases_oth:
            ax = saxs[case_oth]
            if switch_diff:
                from_c = case_oth
                to_c = ctrl
            else:
                from_c =ctrl
                to_c = case_oth
            ax, im, kw = plot_map_diff(var,
                                       from_c,
                                       to_c,
                                       dic_type_mod[ct],
                                       ax=ax,
                                       norm=norm_diff,
                                       relative=relative,
                                       contourf=False,
                                       add_colorbar=False,
                                       #cmap=cmap_abs
                                       )
            ax.set_title(f'{to_c}-{from_c}')
        # cbar:
        la = get_fancy_var_name(var) + ' [%s]' % get_fancy_unit_xr(dic_type_mod[ct][ctrl][var], var)
        if relative:
            la = get_fancy_var_name(var) + ' [%]'
            la = f'rel.$\Delta${la}'
        else:
            la = f'$\Delta${la}'
        # noinspection PyUnboundLocalVariable
        plt.colorbar(im, cax=axs_dict[ct]['cbar'],
                     label=la,
                     orientation='horizontal',
                     extend='both')

        axs_dict[ct]['type_tit'].axis('off')
        if type_nndic is not None:
            if ct in type_nndic:
                ctnn = type_nndic[ct]
            else:
                ctnn = ct
        else:
            ctnn = ct
        ax = axs_dict[ct]['type_tit']
        # noinspection PyUnboundLocalVariable
        ax.set_title(ctnn, fontsize=14)

    for i, ct in enumerate(case_types):
        ax = axs_dict[ct][ctrl]
        # plt_map(di_dic[ctrl][var], ax=ax,

        ax, im = plot_map(var, ctrl, dic_type_mod[ct],
                          ax=ax,
                          kwargs_abs=dict(
                              # cmap='Reds'
                          ),
                          add_colorbar=False,
                          norm=norm_abs,
                          cmap_abs = cmap_abs
                          )
        # cbar:
        _la = get_fancy_var_name(var) + ' [%s]' % get_fancy_unit_xr(dic_type_mod[ct][ctrl][var], var)
        la = f'{_la}'

        plt.colorbar(im, cax=axs_dict[ct]['cbar_abs'],
                     label=la,
                     orientation='horizontal',
                     extend='both')

    md_ls = [ctrl] + cases_oth

    axs_abc = []
    for ct in case_types:
        _axs = [axs_dict[ct][m] for m in md_ls]
        axs_abc = axs_abc + _axs
    if add_abc:
        subp_insert_abc(np.array(axs_abc), pos_x=0.0,pos_y=0.01)

    return fg, axs_dict


def get_lin_norm(case_types, cases_oth, ctrl, dic_type_mod, norm_diff, relative, var,
                 switch_diff=False):
    plt_not_ctrl = []
    if relative:
        func = frelative
    else:
        func = fdifference
    for ct in case_types:
        _di = dic_type_mod[ct]
        if switch_diff:
            plt_not_ctrl = plt_not_ctrl + [func(_di[ctrl][var], _di[cs_o][var]) for cs_o in cases_oth]
        else:
            plt_not_ctrl = plt_not_ctrl + [func(_di[cs_o][var], _di[ctrl][var]) for cs_o in cases_oth]
    vmax, vmin = get_vmin_vmax(plt_not_ctrl, quant=0.05)
    norm_diff = colors.Normalize(vmin=vmin, vmax=vmax)
    return norm_diff


def setup_subplots(case_types, cases_oth, ctrl, fg, gs):
    axs_dict = {}
    for i, ct in enumerate(case_types):
        axs_dict[ct] = {}
        axs_dict[ct][ctrl] = fg.add_subplot(gs[1, i], projection=ccrs.Robinson(),
                                            )
        for j, ca in enumerate(cases_oth):
            axs_dict[ct][ca] = fg.add_subplot(gs[5 + j * 2, i], projection=ccrs.Robinson())
            # axs_dic[var][cases_oth[1]] =fig.add_subplot(gs[5,i], projection=ccrs.Robinson())

        axs_dict[ct]['cbar'] = fg.add_subplot(gs[-1, i])
        axs_dict[ct]['cbar_abs'] = fg.add_subplot(gs[2, i])
        axs_dict[ct]['type_tit'] = fg.add_subplot(gs[0, i])
    return axs_dict




def diffs_PI_PD_sep(dic_type_mod,
                    varl,
                    case_types=None,
                    ctrl=None,
                    case_oth=None,
                    sfg_size=3.5,
                    asp_rat=.5,
                    relative=False,
                    norm_diff_dic=None,
                    locator_diff_dic=None,
                    type_nndic =None,
                    height_ratios=None
                    ):


    if type_nndic is None:
        type_nndic=dict(PI='Pre-industrial',PIaerPD='Present day',PD='Present day')
    if locator_diff_dic is None:
        locator_diff_dic={}
    if case_types is None:
        case_types = ['PI', 'PIaerPD']
    if ctrl is None:
        ctrl = 'OsloAero$_{imp}$'
    if case_oth is None:
        case_oth = 'OsloAeroSec'
    if norm_diff_dic is None:
        norm_diff_dic={}
        for var in varl:
            norm_diff = get_lin_norm(case_types, [case_oth], ctrl, dic_type_mod, None, relative, var)
            norm_diff_dic[var]=norm_diff

    nrows = len(varl)
    ncols = len(case_types)
    if height_ratios is None:
        height_ratios = [.1]+[8]*nrows

    gs = gridspec.GridSpec(nrows=(nrows+1), ncols=ncols,
                           height_ratios=height_ratios,
                           hspace=0.2,
                           )  # width_ratios=width_ratios)
    fg = plt.figure(figsize=[sfg_size*ncols, sfg_size*nrows*asp_rat], dpi=150)#
    # Add subplots:
    axs = []
    for r in range(nrows):
        rl = []
        for c in range(ncols):
            sp = fg.add_subplot(gs[r+1, c], projection=ccrs.Robinson(),)
            rl.append(sp)
        axs.append(rl)
    axs_tit = [fg.add_subplot(gs[c]) for c in range(ncols) ]
    # plot diffs:
    var = varl[0]
    for i, ct in enumerate(case_types):
        for j,var in enumerate(varl):
            ax = axs[j][i]
            if var in norm_diff_dic.keys():
                norm_diff = norm_diff_dic[var]
            else:
                norm_diff = None
            if var in locator_diff_dic:
                loc = locator_diff_dic[var]
                ax, im, kw = plot_map_diff(var,
                                       ctrl,
                                       case_oth,
                                       dic_type_mod[ct],
                                       ax=ax,
                                       norm=norm_diff,
                                       relative=relative,
                                       contourf=False,
                                       add_colorbar=True,
                                       locator=loc

                                       #cbar_orientation='horizontal'
                                       )

            else:
                loc=None
                ax, im, kw = plot_map_diff(var,
                                           ctrl,
                                           case_oth,
                                           dic_type_mod[ct],
                                           ax=ax,
                                           norm=norm_diff,
                                           relative=relative,
                                           contourf=False,
                                           add_colorbar=True,

                                           #cbar_orientation='horizontal'
                                           )

            ax.set_title('')#f'{case_oth}-{ctrl}')
        # cbar:
        la = get_fancy_var_name(var) + ' [%s]' % get_fancy_unit_xr(dic_type_mod[ct][ctrl][var], var)
        if relative:
            la = get_fancy_var_name(var) + ' [%]'
            la = f'rel.$\Delta${la}'
        else:
            la = f'$\Delta${la}'

        axs_tit[i].axis('off')
        if type_nndic is not None:
            if ct in type_nndic:
                ctnn = type_nndic[ct]
            else:
                ctnn = ct
        else:
            ctnn = ct
        tit = f'{ctnn}: \n {case_oth}-{ctrl}'
        axs_tit[i].set_title(tit)#, fontsize=14)
    subp_insert_abc(np.array(axs), pos_x=0.01,pos_y=1.0)

    return fg, axs



# %%
