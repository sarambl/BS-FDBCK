import matplotlib.pyplot as plt
from matplotlib import colors, gridspec
import matplotlib.ticker as mtick

from bs_fdbck.util.naming_conventions.var_info import get_fancy_var_name, get_fancy_unit_xr
from bs_fdbck.util.plot.plot_levlat import plot_levlat_diff, get_cbar_label, plot_levlat_abs, frelative, fdifference, \
    get_vmin_vmax
from useful_scit.plot.fig_manip import subp_insert_abc
import numpy as np

# noinspection DuplicatedCode
def abs_diffs(di_dic, ctrl, cases_oth, varl,
              subfig_size=2.9,
              asp_ratio=.9,
              norm_dic=None,
              switch_diff=False,
              ylim=None,
              yticks=None
              ):
    if yticks is None:
        yticks = [900, 700, 500, 300]

    if ylim is None:
        ylim=[1e3,200]
    if norm_dic is None:
        print('must set norm_dic')
        return
    fig, axs = plt.subplots(4, len(varl),
                            gridspec_kw={'height_ratios': [4, 3, 3, .3]},
                            figsize=[subfig_size * len(varl), subfig_size * 3 * asp_ratio])
    axs_diff = axs[1:-1, :]
    axs_diff_cb = axs[-1, :]
    axs_plts = axs[0:-1, :]
    #axs_plts = np.delete(axs, 1, 0)
    # ctrl = 'OsloAeroSec'
    # cases_oth = ['OsloAero$_{imp}$','OsloAero$_{def}$']
    for i, var in enumerate(varl):
        saxs = axs_diff[:, i]
        # noinspection DuplicatedCode
        for case_oth, ax in zip(cases_oth, saxs.flatten()):
            if switch_diff:
                from_c = case_oth
                to_c=ctrl
            else:
                from_c = ctrl
                to_c=case_oth
            _, im = plot_levlat_diff(var, from_c, to_c,
                                     di_dic,
                                     cbar_orientation='horizontal',
                                     # title=None,
                                     ax=ax,
                                     ylim=ylim,
                                     # figsize=None,
                                     cmap='RdBu_r',
                                     # use_ds_units=True,
                                     # add_colorbar=True,
                                     norm=norm_dic[var],
                                     add_colorbar=False
                                     )

        # ax.set_title(f'{key}: PIaerPD-PI')
        lab = f'$\Delta${get_cbar_label(di_dic[ctrl][var], var, diff=True)}'
        # noinspection PyUnboundLocalVariable
        plt.colorbar(im,
                     cax=axs_diff_cb[i],
                     label=lab,
                     orientation='horizontal',
                     extend='both'
                     )

    for i, var in enumerate(varl):
        ax = axs[0, i]
        _, im = plot_levlat_abs(var, ctrl,
                                di_dic,
                                cbar_orientation='horizontal',
                                # title=None,
                                ax=ax,
                                ylim=ylim,
                                # figsize=None,
                                cmap='PuOr_r',
                                # use_ds_units=True,
                                # add_colorbar=True,
                                norm=norm_dic[var],
                                add_colorbar=False
                                )

        # ax.set_title(f'{key}: PIaerPD-PI')
        lab = get_cbar_label(di_dic[ctrl][var], var, diff=True)
        plt.colorbar(im,
                     ax=ax,
                     label=lab,
                     orientation='horizontal',
                     extend='both'
                     )

    for i in range(len(axs_plts[:, 0])):
        for j in range(len(axs_plts[0, :])):
            ax = axs_plts[i, j]
            ax.set_yticks([],minor=True)
            ax.set_yticks(yticks,minor=False)

            if i < (len(axs_plts[:, 0]) - 1):
                ax.set_xlabel('')
                plt.setp(ax.get_xticklabels(), visible=False)
            if i == (len(axs_plts[:, 0]) - 1):
                ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f$^\circ$N'))
                ax.xaxis.set_minor_formatter(mtick.FormatStrFormatter('%.0f$^\circ$N'))
                ax.set_xlabel('Latitude [$^\circ$N]')

            if j==0:
                ax.set_yticklabels(yticks, minor=False)
            else:
                ax.yaxis.set_major_formatter(mtick.NullFormatter())#('%.0f'))



                #if j > 0:
            #    plt.setp(ax.get_yticklabels(), visible=False)
                ax.set_ylabel('')
    for ax in axs[0, :]:
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in axs[0, 1:]:
        ax.set_ylabel('')
        plt.setp(ax.get_yticklabels(), visible=False)
    subp_insert_abc(axs_plts, pos_x=-.05,pos_y=1.04 )
    fig.tight_layout()
    return axs, fig


def abs_diffs_PI_PD_sep(dic_type_mod,
                        var,
                        case_types=None,
                        ctrl=None,
                        cases_oth=None,
                        sfg_size=2.9,
                        asp_rat=.9,
                        ylim=None,
                        relative=False,
                        norm_abs=None,
                        norm_diff=None,
                        type_nndic=None,
                        height_ratios=None,
                        switch_diff=False,
                        add_abc=True,
                        yticks=None
                        ):
    if ylim is None:
        ylim = [1e3, 200]
    if yticks is None:
        yticks = [900, 700, 500, 300]
    if type_nndic is None:
        type_nndic={'PI':'Pre-industrial','PIaerPD':'Present day'}
    if case_types is None:
        case_types = ['PI', 'PD']
    #for c in case_types:
    #    if c not in sup_titles_dic.keys():
    #        sup_titles_dic[c]=c

    if norm_abs is None:
        norm_abs = None
    if cases_oth is None:
        cases_oth = ['OsloAero$_{imp}$', 'OsloAero$_{def}$']
    if ctrl is None:
        ctrl = 'OsloAeroSec'

    if norm_diff is None:
        # noinspection DuplicatedCode
        plt_not_ctrl = []
        if relative:
            func = frelative
            print('relative!!')
        else:
            func = fdifference
        for ct in case_types:
            _di = dic_type_mod[ct]
            if switch_diff:
                plt_not_ctrl = plt_not_ctrl + [func(_di[ctrl][var], _di[cs_o][var]) for cs_o in cases_oth]
            else:
                plt_not_ctrl = plt_not_ctrl + [func(_di[cs_o][var], _di[ctrl][var]) for cs_o in cases_oth]

        vmax, vmin = get_vmin_vmax(plt_not_ctrl)
        norm_diff = colors.Normalize(vmin=vmin, vmax=vmax)

    #height_ratios = [.3, 3, .3, 2, 3, .1, 3,.4, .3]
    # plt.rcParams['figure.constrained_layout.use'] = False
    if height_ratios is None:
        height_ratios = [.3, 3, .3, 1.]+[.6, 3]*len(cases_oth)+ [.4,.3]

    fg = plt.figure(figsize=[sfg_size * len(case_types), sfg_size * (len(cases_oth)+1) * asp_rat])

    nrows = len(height_ratios)
    ncols = len(case_types)

    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols,
                           height_ratios=height_ratios,
                           hspace=0.2,
                           )  # width_ratios=width_ratios)
    axs_dict = {}
    for i, ct in enumerate(case_types):
        # gs_s = gs[:,i]
        axs_dict[ct] = {}
        axs_dict[ct][ctrl] = fg.add_subplot(gs[1, i],
                                            )
        for j, ca in enumerate(cases_oth):
            axs_dict[ct][ca] = fg.add_subplot(gs[5 + j * 2, i], )
            # axs_dic[var][cases_oth[1]] =fig.add_subplot(gs[5,i], projection=ccrs.Robinson())

        axs_dict[ct]['cbar'] = fg.add_subplot(gs[-1, i])
        axs_dict[ct]['cbar_abs'] = fg.add_subplot(gs[2, i])
        axs_dict[ct]['type_tit'] = fg.add_subplot(gs[0, i])

    for i, ct in enumerate(case_types):
        saxs = axs_dict[ct]
        for case_oth in cases_oth:
            ax = saxs[case_oth]
            if switch_diff:
                from_c = case_oth
                to_c = ctrl
            else:
                from_c = ctrl
                to_c=case_oth
            ax, im = plot_levlat_diff(var, from_c, to_c, dic_type_mod[ct],
                                          cbar_orientation='horizontal',
                                          # title=None,
                                      relative=relative,
                                      ylim=ylim,
                                          ax=ax,
                                          # ylim=None,
                                          # figsize=None,
                                          cmap='RdBu_r',
                                          # use_ds_units=True,
                                          # add_colorbar=True,
                                          norm=norm_diff,
                                          add_colorbar=False,
                                      yscale='log',
                                      #type_nndic=None
                                      )
            ax.set_title(f'{to_c}-{from_c}')

        # cbar:
        # noinspection DuplicatedCode
        la = get_fancy_var_name(var) + ' [%s]' % get_fancy_unit_xr(dic_type_mod[ct][ctrl][var], var)
        if relative:
            la = get_fancy_var_name(var) + ' [%]'
            la = f'rel.$\Delta${la}'
        else:
            la = f'$\Delta${la}'
        # noinspection PyUnboundLocalVariable
        plt.colorbar(im,
                     cax=axs_dict[ct]['cbar'],
                     label=la,
                     orientation='horizontal',
                     extend='both'
                     )
        # axs_dict[ct]['cbar'].set_title(f'{ct}$\Updownarrow$')

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
        # axs_dict[ct]['type_tit'].set_xlabel(f'{ct} $\\Updownarrow$')


    for i, ct in enumerate(case_types):
        ax = axs_dict[ct][ctrl]
        # plt_map(di_dic[ctrl][var], ax=ax,
        ax, im = plot_levlat_abs(var, ctrl,
                                 dic_type_mod[ct],
                                 cbar_orientation='horizontal',
                                 # title=None,
                                 ax=ax,
                                 # ylim=None,
                                 # figsize=None,
                                 cmap='Reds',
                                 ylim=ylim,

                                 # use_ds_units=True,
                                 # add_colorbar=True,
                                 norm=norm_abs,
                                 add_colorbar=False,
                                 yscale='log'
                                 )

        # cbar:

        _la = get_fancy_var_name(var) + ' [%s]' % get_fancy_unit_xr(dic_type_mod[ct][ctrl][var], var)
        la = f'{_la}'

        plt.colorbar(im,
                     cax=axs_dict[ct]['cbar_abs'],
                     label=la,
                     orientation='horizontal',
                     extend='both')
    saxs = axs_dict[ct]
    #for ax in saxs[1:]:
    #       ax.set_yticklabels([])
    for ct in case_types:
        saxs = axs_dict[ct]
        for case in cases_oth+ [ctrl]:
            ax = saxs[case]
            #ax.tick_params(labelbottom=False)
            ax.set_yticks([],minor=True)
            ax.set_yticks(yticks,minor=False)

            if ct !=case_types[0]:
                ax.yaxis.set_major_formatter(mtick.NullFormatter())#('%.0f'))
                ax.yaxis.set_minor_formatter(mtick.NullFormatter())#('%.0f'))
                ax.set_ylabel('')
            else:
                ax.set_yticklabels(yticks, minor=False)
                #ax.set_yticklabels([1000,500,200],minor=False)

            if case !=cases_oth[-1]:
                ax.set_xlabel('')
                #ax.set_yticklabels([])
                ax.tick_params(labelbottom=False)

            else:
                ax.set_xlabel('')#Latitude [$^\circ$N]')
                ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f$^\circ$N'))
                ax.xaxis.set_minor_formatter(mtick.FormatStrFormatter('%.0f$^\circ$N'))
                
        ax = axs_dict[ct][ctrl]
        if ct !=case_types[0]:
            ax.yaxis.set_major_formatter(mtick.NullFormatter())#('%.0f'))
            ax.yaxis.set_minor_formatter(mtick.NullFormatter())#('%.0f'))
            ax.set_ylabel('')
        #if case !=cases_oth[-1]:
        #    ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_xticklabels([])


    md_ls = [ctrl] + cases_oth

    axs_abc = []
    for ct in case_types:
        _axs = [axs_dict[ct][m] for m in md_ls]
        axs_abc = axs_abc + _axs
    if add_abc:
        subp_insert_abc(np.array(axs_abc), pos_x=-0.12,pos_y=.95)


    return fg, axs_dict
