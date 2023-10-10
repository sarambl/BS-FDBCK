import ukesm_bs_fdbck.util.plot.make_break_axis as mba
from matplotlib import colors

from ukesm_bs_fdbck.util.plot import plot_settings
from ukesm_bs_fdbck.util import practical_functions
import seaborn as sns
from ukesm_bs_fdbck.util.plot.plot_settings import  set_equal_axis
import matplotlib.pyplot as plt
import numpy as np

sns.reset_orig()





def plot_Nd_bars_in_ax_CTRL(ax, model, pl_pd, cmap, header=None):
    """
    Plots Absolute of CTRL column
    :param header:
    :param ax:
    :param model:
    :param pl_pd:
    :param cmap:
    :return:
    """
    if header is None:
        header=model
    kwargs = {'fontsize': 14, 'color': cmap, 'edgecolor': 'k', 'grid': {'b': True, 'axis': 'y'}}
    pl_pd['CTRL'].transpose().plot.bar(ax=ax, title='%s: CTRL' % header, width=1, **kwargs)  # , color=cmap,
    ax.set_title('%s: CTRL' % model, fontsize=14)
    ax.xaxis.grid(False)

def plot_Nd_bars_in_ax_DIFF(ax, model, pl_pd, relative, cmap, header=None):
    """
    Plots difference to CTRL column either relative or absolute
    :param header:
    :param ax:
    :param model:
    :param pl_pd:
    :param relative: if relative, plots relative difference to CTRL
    :param cmap:
    :return:
    """
    if header is None:
        header=model
    kwargs = {'fontsize': 14, 'color': cmap, 'edgecolor': 'k', 'grid': {'b': True, 'axis': 'y'}}
    #ax.set_title('%s: CTRL' % model, fontsize=14)
    if relative:
        plt_diff = pl_pd.drop('CTRL', axis=1).sub(pl_pd['CTRL'], axis=0).div(np.abs(pl_pd['CTRL']), axis=0) * 100.
    else:
        plt_diff = pl_pd.drop('CTRL', axis=1).sub(pl_pd['CTRL'], axis=0)  # .div(pl_pd['CTRL'])
    if relative:
        kwargs['title'] = '%s: relative difference' % header
        kwargs['width'] = 0.85
        kwargs['legend'] = False
        kwargs['ax'] = ax
    else:
        kwargs['title'] = '%s: difference' % header
        kwargs['width'] = 0.9
        kwargs['legend'] = False
        kwargs['ax'] = ax
        kwargs['ax'] = ax
    plt_diff.transpose().plot.bar(**kwargs)
    ax.set_title(kwargs['title'], fontsize=14)
    ax.xaxis.grid(False)






def plot_Nd_bars_all_models_break_ECHAM(nested_pd_model, models, area, N_vars,
                                        relative=True, sharey_for_diffrel=False,
                                        sharey_for_ctrl=False, fn_bs='',
                                        plt_path='plots/bars_non_stacked/',
                                        cmap = 'colorblind',
                                        format = 'png'):


    """
    Break axis for ECHAM
    :param fn_bs:
    :param format:
    :param nested_pd_model:
    :param models:
    :param area:
    :param N_vars:
    :param relative:
    :param sharey_for_diffrel:
    :param sharey_for_ctrl:
    :param plt_path:
    :param cmap:
    :return:
    """
    plt_path = plt_path + area + '/'
    practical_functions.make_folders(plt_path)
    filen_base = plt_path + fn_bs
    cmap = sns.color_palette(cmap, len(N_vars))
    # If relative, break only ctrl:
    if relative:
        fig, gs, axs = mba.make_my_grid_subplots([1, 5], [4, 4, 1, 3])
    else:
        fig, gs, axs = mba.make_my_grid_subplots([3, 8], [4, 4, 1, 3])

    for ax in axs.flatten():
        ax.grid(True, axis='x')
    ii = 0
    filen = filen_base  # plt_path+'bars_Nd_ctr_diff'
    for model in models + ['ECHAM']:
        pl_pd = nested_pd_model[model]  # .drop(['N$_{d<20}$'])#, axis=1)#, axis=0)#.transpose()
        pl_pd.index.name = None

        plot_Nd_bars_in_ax_CTRL(axs[ii,0], model, pl_pd, cmap)
        plot_Nd_bars_in_ax_DIFF(axs[ii,1], model, pl_pd, relative, cmap)
        if relative:
            axs[ii, 0].set_ylabel('#/cm$^3$', fontsize=14)
            axs[ii, 1].set_ylabel('%', fontsize=14)
        else:
            axs[ii, 0].set_ylabel('#/cm$^3$', fontsize=14)
            axs[ii, 1].set_ylabel('#/cm$^3$', fontsize=14)
        filen = filen + '_' + model
        if ii < len(models): # remove tick labels
            axs[ii, 1].get_xaxis().set_ticklabels([])
            axs[ii, 0].get_xaxis().set_ticklabels([])
        ii += 1
    if relative:
        model = 'ECHAM'
        ax = fig.add_subplot(gs[8:, 1:])
        pl_pd = nested_pd_model[model]
        pl_pd.index.name = None
        plot_Nd_bars_in_ax_DIFF(ax, model, pl_pd, relative, cmap)
        ax.set_ylabel('%', fontsize=14)
        fig.delaxes(axs[2, 1])
        fig.delaxes(axs[3, 1])
        plot_settings.insert_abc_where(ax, 14, 2 * 2 + 1, ratioxy=1.2)

    axs[2, 0].set_ylabel('', visible=False)
    axs[3, 0].set_ylabel('             #/cm$^3$', fontsize=14)  # , visible=False)
    if not relative:
        axs[2, 1].set_ylabel('', visible=False)
        axs[3, 1].set_ylabel('             #/cm$^3$', fontsize=14)  # , visible=False)

    for ax in axs[3, :]:
        ax.title.set_visible(False)
    if sharey_for_diffrel:
        set_equal_axis(axs[:, 1], which='y')
    if sharey_for_ctrl:
        set_equal_axis(axs[:, 0], which='y')
    for ii in np.arange(len(models)-1):
        plot_settings.insert_abc_where(axs[ii, 0], 14, ii * 2,ratioxy=.8 )
        plot_settings.insert_abc_where( axs[ii, 1], 14, ii * 2 + 1, ratioxy=1.2)

    plot_settings.insert_abc_where(axs[2, 0], 14, 2 * 2, ratioxy=.8)
    if not relative:
        plot_settings.insert_abc_where( axs[2, 1], 14, 2 * 2 + 1, ratioxy=3.)


    if relative:
        filen = filen + '_rel.%s'%format
    else:
        filen = filen + '.%s'%format
    print(filen)
    if relative:
        mba.broken_axis(axs[2:, 0], [1000, 5500])
    else:
        mba.broken_axis(axs[2:, 0], [1000, 5500])
        mba.broken_axis(axs[2:, 1], [21, 25])
    gs.tight_layout(fig, pad=0.3)

    print('Saving file to: %s' % filen)
    plt.savefig(filen, dpi=300)
    plt.show()








def plot_Nd_bars_all_models(nested_pd_model, models, area, N_vars, relative=True, sharey_for_diffrel=False,
                            sharey_for_ctrl=False, without_be20=True, plt_path='plots/bars_non_stacked/',
                            cmap='colorblind', format='png'):
    """

    :param format:
    :param nested_pd_model:
    :param models:
    :param area:
    :param N_vars:
    :param relative:
    :param sharey_for_diffrel:
    :param sharey_for_ctrl:
    :param without_be20:
    :param plt_path:
    :param cmap:
    :return:
    """
    plt_path=plt_path +area+'/'
    practical_functions.make_folders(plt_path)
    filen_base=plt_path+'bars_Nd_ctr_diff'
    cmap = sns.color_palette(cmap, len(N_vars))

    fig, axs = plt.subplots(len(models),2, figsize=[12,11],gridspec_kw = {'width_ratios':[1, 5]})

    ii=0
    filen=filen_base #plt_path+'bars_Nd_ctr_diff'
    if without_be20:
        filen=filen+'no_sub20'
    for model in models:
        if without_be20 and ('N$_{d<20}$' in nested_pd_model[model].index):
            pl_pd= nested_pd_model[model].drop(['N$_{d<20}$'])#, axis=1)#, axis=0)#.transpose()
        else:
            pl_pd = nested_pd_model[model]#.drop(['N$_{d<20}$'])#, axis=1)#, axis=0)#.transpose()
        pl_pd.index.name = None

        plot_Nd_bars_in_ax_CTRL(axs[ii,0], model, pl_pd, cmap)
        plot_Nd_bars_in_ax_DIFF(axs[ii,1], model, pl_pd, relative, cmap)
        if relative:
            axs[ii,0].set_ylabel('#/cm$^3$', fontsize=14)
            axs[ii,1].set_ylabel('%', fontsize=14)
        else:
            axs[ii,0].set_ylabel('#/cm$^3$', fontsize=14)
        filen = filen+'_'+model
        if ii<len(models)-1:
            axs[ii,1].get_xaxis().set_ticklabels([])
            axs[ii,0].get_xaxis().set_ticklabels([])
        ii+=1
    if sharey_for_diffrel:
        set_equal_axis(axs[:, 1], which='y')
    if sharey_for_ctrl:
        set_equal_axis(axs[:, 0], which='y')
    for ii in np.arange(len(models)):
        plot_settings.insert_abc_where(axs[ii, 0], 14, ii * 2,ratioxy=.8 )
        plot_settings.insert_abc_where( axs[ii, 1], 14, ii * 2 + 1, ratioxy=1.2)
        #sensitivity_scripts.plot_settings.insert_abc(axs[ii,0],14,ii*2)
        #sensitivity_scripts.plot_settings.insert_abc(axs[ii,1],14,ii*2+1)
    if relative:
        filen = filen+'_rel.%s'%format
    else:
        filen = filen+'.%s'%format
    print(filen)
    plt.tight_layout(pad=2.)
    plt.savefig(filen, dpi=300)
    plt.show()


def plot_Nd_bars_n_areas(nested_pd_areas, model, areas, N_vars, cases, areas_label, relative=True, sharey_for_diffrel=False,
                            sharey_for_ctrl=False, without_be20=True, plt_path='plots/bars_non_stacked/',
                            cmap='colorblind', fn_base='bars_Nd_areas', format='png'):
    """

    :param nested_pd_areas:
    :param model:
    :param areas:
    :param cases:
    :param areas_label:
    :param fn_base:
    :param format:
    :param N_vars:
    :param relative:
    :param sharey_for_diffrel:
    :param sharey_for_ctrl:
    :param without_be20:
    :param plt_path:
    :param cmap:
    :return:
    """
    areas_str= '_'.join(areas)
    plt_path = plt_path+'m_areas/'
    practical_functions.make_folders(plt_path)
    filen_base = plt_path + fn_base + areas_str
    cmap = sns.color_palette(cmap, len(N_vars))
    if len(cases)==3:#Yields_only:

        fig, axs = plt.subplots(2,2, figsize=[8,4*len(areas)],gridspec_kw = {'width_ratios':[1, 2]})


    else:
        fig, axs = plt.subplots(2,2, figsize=[11,5*len(areas)],gridspec_kw = {'width_ratios':[1, 4]})
    #fig, axs = plt.subplots(len(areas), 2, figsize=[12, 11], gridspec_kw={'width_ratios': [1, 5]})

    ii = 0
    filen = filen_base  # plt_path+'bars_Nd_ctr_diff'
    if without_be20:
        filen = filen + 'no_sub20'
    for area, ii in zip(areas,range(0, len(areas))):
        nested_pd_model = nested_pd_areas[area][model][cases]

        if without_be20 and ('N$_{d<20}$' in nested_pd_model.index):
            pl_pd = nested_pd_model.drop(['N$_{d<20}$'])  # , axis=1)#, axis=0)#.transpose()
        else:
            pl_pd = nested_pd_model  # .drop(['N$_{d<20}$'])#, axis=1)#, axis=0)#.transpose()
        pl_pd.index.name = None

        plot_Nd_bars_in_ax_CTRL(axs[ii, 0], model, pl_pd, cmap, header=areas_label[area])
        plot_Nd_bars_in_ax_DIFF(axs[ii, 1], model, pl_pd, relative, cmap, header=areas_label[area])
        if relative:
            axs[ii, 0].set_ylabel('#/cm$^3$', fontsize=14)
            axs[ii, 1].set_ylabel('%', fontsize=14)
        else:
            axs[ii, 0].set_ylabel('#/cm$^3$', fontsize=14)
            axs[ii, 1].set_ylabel('#/cm$^3$', fontsize=14)
        filen = filen + '_' + model
        if ii < len(areas) - 1:
            axs[ii, 1].get_xaxis().set_ticklabels([])
            axs[ii, 0].get_xaxis().set_ticklabels([])
        ii += 1
    if sharey_for_diffrel:
        set_equal_axis(axs[:, 1], which='y')
    if sharey_for_ctrl:
        set_equal_axis(axs[:, 0], which='y')
    for ii in np.arange(len(areas)):
        plot_settings.insert_abc_where(axs[ii, 0], 14, ii * 2, ratioxy=.8)
        plot_settings.insert_abc_where(axs[ii, 1], 14, ii * 2 + 1, ratioxy=1.1)
        # sensitivity_scripts.plot_settings.insert_abc(axs[ii,0],14,ii*2)
        # sensitivity_scripts.plot_settings.insert_abc(axs[ii,1],14,ii*2+1)
    if relative:
        filen = filen + '_rel.%s'%format
    else:
        filen = filen + '.%s'%format
    print(filen)
    plt.tight_layout(pad=2.)
    plt.savefig(filen, dpi=300)
    plt.show()


def plot_sizedist_time(ds, ss_start_t, ss_end_t,
                       location=None,
                       var=None,
                       ax=None,
                       figsize=None,
                       vmin=None, vmax=None,
                       norm_fun=colors.LogNorm,
                       **plt_kwargs):
    if figsize is None:
        figsize = [5, 10]
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)

    def_kwargs={'ylim':[3,1e3], 'yscale':'log', 'ax':ax}
    for key in def_kwargs.keys():
        if key not in plt_kwargs.keys():
            plt_kwargs[key]=def_kwargs[key]

    if location is not None:
        ds = ds.sel(location=location)
    if 'dNdlogD_sec' in ds:
        ds['dNdlogD'] = ds['dNdlogD_sec'] + ds['dNdlogD_mod']

    else:
        ds['dNdlogD']= ds['dNdlogD_mod']
    ds['dNdlogD'].attrs = ds['dNdlogD_mod'].attrs
    ds['dNdlogD'].attrs['long_name'] = 'dNdlogD'
    if var is None:
        var = 'dNdlogD'

    if 'norm' not in plt_kwargs:
        plt_kwargs['norm']=norm_fun(vmin=vmin, vmax=vmax)

    da =ds[var].mean('lev', keep_attrs=True)# ds[var]#.sel(time=slice(ss_start_t,ss_end_t))
    #return da
    da.plot(x='time',**plt_kwargs)
    if 'case_name' in da.attrs:
        tit = da.attrs['case_name']+', '+ location
    else:
        tit = location
    ax.set_title(tit)