from bs_fdbck_clean.util.plot.plot_maps import subplots_map, plot_map_diff


def plt_var_types_mods_diff(dic_absT, var, case_types=None,
                            mod_types=None,
                            ctrl='ctrl',
                            figsize_x =3.4,
                            figsize_y =3.,
                            kwargs_diff=None,
                            relative=True,
                            kwargs_diff_types=None,
                            **kwargs
                            ):
    if mod_types is None:
        mod_types = ['OsloAeroSec', 'OsloAero$_{imp}$', 'OsloAero$_{def}$']
    if case_types is None:
        case_types = ['decYield', 'incYield']
    #dic_abs = get_abs_by_type(maps_dic, case_types=case_types+[ctrl],
    #                      mod_types=mod_types)
    #dic_absT = transpose_2lev_dic(dic_abs, ctrl=ctrl)
    lx = len(case_types)
    ly = len(mod_types)
    fig, axs = subplots_map(ly,lx, figsize=[lx*figsize_x,ly*figsize_y])

    for type, iax in zip(case_types,range(len(case_types))):
        if ly==1:
            saxs = [axs[iax]]
        elif lx==1:
            saxs=axs
        else:
            saxs = axs[:,iax]
        if kwargs_diff_types is not None:
            kwargs_diff = kwargs_diff_types[type]
        for key, ax in zip(mod_types,saxs):
            plot_map_diff(var,
                          'ctrl',
                          type,
                          dic_absT[key],
                          figsize=None,
                          relative=relative,
                          kwargs_diff=kwargs_diff,
                          ax=ax,
                          tit_ext=f' {key}',
                          cmap_diff='RdBu_r',
                          cbar_orientation='vertical',
                          **kwargs
                          )
    return axs