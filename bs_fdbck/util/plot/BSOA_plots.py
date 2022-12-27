import matplotlib
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt, gridspec as gridspec
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression

from matplotlib.colors import ListedColormap

import seaborn as sns
years_used = list(np.arange(2012, 2019))


cmap_list = ['#441FE0','#BBE01F'][::-1]
palette_OA = cmap_list[0:2]


cols = [
    # '#ffff33',
    '#0074c3',
    '#eb4600',
    '#f8ae00',
    '#892893',
    '#66ae00',
    '#00c1f3',
    '#b00029',
]


cdic_model = {
    'NorESM': '#d8651e', ##d26f1a',##9d134b',
    'ECHAM-SALSA':'#0476b2', #6e75c0',##f6a90d',# '#36AE7C',
    'EC-Earth':'#e6a01c', ##6e75c0',#f6a90d',
    'UKESM':'#2a9e76', #f24a4a',##f24a5f',#2a717e',
    'Observations':'k',

}


my_cmap = ListedColormap(cols)

col_dic = {}
for y, c in zip(range(2012, 2019), cols):
    col_dic[y] = c


def add_cbar(cax):
    _cols = [col_dic[int(y)] for y in years_used]
    _my_cmap = ListedColormap(_cols)

    bounds = [y - 0.5 for y in years_used] + [years_used[-1] + .5]
    return mpl.colorbar.ColorbarBase(cmap=_my_cmap, ax=cax, boundaries=bounds, ticks=years_used)


def make_cool_grid(figsize=None,
                   width_ratios=None,
                   ncols=2,
                   nrows=1,
                   add_gs_kw=None,
                   sharex='col',
                   sharey='row',
                   w_plot = 4,
                   w_cbar = 0.3


                   ):
    if figsize is None:
        figsize = [ncols * w_plot + w_cbar, w_plot*nrows]
    # if figsize is None:
    #    figsize= [18,4]
    # c = 0.1/2.1
    # siz_bar = c*ncols/(1-c)
    if width_ratios is None:
        width_ratios = [1] * ncols + [w_cbar / w_plot]
    if add_gs_kw is None:
        add_gs_kw = dict()
    if 'hspace' not in add_gs_kw.keys():
        add_gs_kw['hspace'] = 0
    if 'wspace' not in add_gs_kw.keys():
        add_gs_kw['wspace'] = 0
    # add_gs_kw['width_ratios'] = width_ratios
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols + 1, **add_gs_kw)
    w_r1 = [sum(width_ratios[:-1]), width_ratios[-1]]
    gs0 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=w_r1, **add_gs_kw)

    gs00 = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs0[0], **add_gs_kw)

    # gs_s = gs[:,:(ncols+1)].subgridspec(nrows=nrows, ncols=ncols, wspace=add_gs_kw['wspace'], hspace=add_gs_kw['hspace'])
    axs = gs00.subplots(sharex=sharex, sharey=sharey, )
    cax = fig.add_subplot(gs0[:, -1])
    add_cbar(cax)
    #axs = np.array(axs)
    return fig, axs, cax


def plot_scatter(v_x, v_y, df_s, df_sy, ca,
                 xlims=None,
                 ylims=None,
                 xlab=None,
                 ylab=None,
                 figsize=[6, 5],
                 ax=None,
                 fig = None,
                 add_cbar=False,
                 # fig = None
                 legend_loc='upper left',
                 ):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    # for y,co in zip(df_s['year'].unique(), cols):

    #    _df = df_s[df_s['year']==y]
    df_s.plot.scatter(x=v_x, y=v_y, ax=ax,
                      c='year',
                      cmap=my_cmap,
                      colorbar=add_cbar,
                      vmax=2018.5, vmin=2011.5)  # , label=y )#, c='year', cmap='Paired')
    # df_sy = df_s.resample('Y').median()
    if df_sy is not None:
        for y, co in zip(df_sy['year'].unique(), cols):
            # df_sy = dic_df_sm[ca]
            co = col_dic[int(y)]
            _dfm = df_sy[df_sy['year'] == y]

            # _dfm = _df.median()
            scat_obj = ax.scatter(_dfm[v_x], _dfm[v_y], c=co, label='__nolegend__', marker='s', s=200, edgecolor='k')

    _df_s = df_s[df_s[v_x].notnull() & df_s[v_y].notnull()]
    x = np.array(_df_s[v_x].values).reshape(-1, 1)
    y = np.array(_df_s[v_y].values).reshape(-1, 1)

    model = LinearRegression().fit(x, y)

    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)

    print('intercept:', model.intercept_)

    print('slope:', model.coef_)
    x_s = np.linspace(x.min(), x.max(), 10)
    a = model.coef_[0]
    b = model.intercept_[0]
    if b < 0:
        sig = ''
    else:
        sig = '+'
    if a < 1:

        lab = r'fit: $y= %.3fx%s%.3f$, r$^2$=%.02f' % (a, sig, b, r_sq)
    elif a > 10:
        lab = r'fit: $y= %.1fx%s%.1f$, r$^2$=%.02f' % (a, sig, b, r_sq)
    elif a > 100:
        lab = r'fit: $y= %.0fx%s%.0f$, r$^2$=%.02f' % (a, sig, b, r_sq)
    else:
        lab = r'fit: $y= %.2fx%s%.2f$, r$^2$=%.02f' % (a, sig, b, r_sq)

    ax.plot(x_s, (a * x_s + b), c='k')
    plt.legend(frameon=False, bbox_to_anchor=(1, 1,))
    # ax.hlines(2000, 5,30, color='k', linewidth=1)

    ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color='#0074c3', marker='s', markeredgecolor='k', markersize=10, linewidth=0),
        Line2D([0], [0], color='#0074c3', marker='o', linewidth=0),
        Line2D([0], [0], color='k', lw=1),
        # Patch( color='b', lw=4),
        # Line2D([0], [0], color=cmap(1.), lw=4)
    ]

    ax.legend(custom_lines, ['Daily median', 'Summer median', lab, ], frameon=False,
              loc=legend_loc)
    return fig, ax



def make_cool_grid3(figsize=None,
                    width_ratios=None,
                    ncols=1,
                    nrows=1,
                    nrows_extra = 3,
                    add_gs_kw=None,
                    ncols_extra=1,
                    sharex='col',
                    sharey='row',
                    w_plot = 5.5,
                    w_cbar = 0.7,

                    w_ratio_sideplot = 0.3
                    ):


    if figsize is None:
        figsize = [ncols * w_plot + w_cbar + ncols_extra*w_plot*w_ratio_sideplot, w_plot*nrows]
    width_ratios = None
    add_gs_kw = None
    if width_ratios is None:
        width_ratios = [1] * ncols + [w_cbar / w_plot] #+ [1]* ncols_extra
    if add_gs_kw is None:
        add_gs_kw = dict()


    if 'hspace' not in add_gs_kw.keys():
        add_gs_kw['hspace'] = 0
    if 'wspace' not in add_gs_kw.keys():
        add_gs_kw['wspace'] = 0


    # add_gs_kw['width_ratios'] = width_ratios
    fig = plt.figure(figsize=figsize,dpi=150)

    gs = fig.add_gridspec(nrows, ncols + 1, **add_gs_kw)

    w_r1 = [sum(width_ratios[:-1]), width_ratios[-1]]
    h_r1 = [width_ratios[-1],sum(width_ratios[:-1]), ]
    gs0 = gridspec.GridSpec(1, 2, figure=fig,width_ratios = [1,w_ratio_sideplot*ncols_extra])# width_ratios=w_r1, height_ratios=h_r1, **add_gs_kw)

    gs00 = gridspec.GridSpecFromSubplotSpec(nrows+1, ncols+1, width_ratios=w_r1, height_ratios=h_r1, subplot_spec=gs0[0], **add_gs_kw)

    gs01 = gridspec.GridSpecFromSubplotSpec(nrows_extra, ncols_extra, subplot_spec=gs0[1])#, **add_gs_kw)

    # gs_s = gs[:,:(ncols+1)].subgridspec(nrows=nrows, ncols=ncols, wspace=add_gs_kw['wspace'], hspace=add_gs_kw['hspace'])
    axs = gs00.subplots(sharex=sharex, sharey=sharey, )
    axs_extra = gs01.subplots(sharex=sharex, sharey=sharey, )
    #dax1 = fig.add_subplot(gs0[1, -1])
    #dax1 = fig.add_subplot(gs0[1, -1])
    #dax2 = fig.add_subplot(gs0[0,0])
    #add_cbar(cax)
    axs[0,1].clear()
    axs[0,1].axis("off")
    daxs = dict(x=axs[0,0],y=axs[1,1])
    for a in daxs:
        _ax = daxs[a]
        sns.despine(bottom=False, left=False, ax=_ax)
        _ax.axis("off")
    #daxs = [dax1,dax2]
    #axs = np.array(axs)

    ax = axs[1,0]


    return fig, ax, daxs, axs_extra
#make_cool_grid3()
def make_cool_grid2(figsize=None,
                    width_ratios=None,
                    ncols=1,
                    nrows=1,
                    add_gs_kw=None,
                    sharex='col',
                    sharey='row',
                    w_plot = 5,
                    w_cbar = 0.7


                    ):


    #figsize = None
    #add_gs_kw = None
    #w_cbar = 0.3
    #width_ratios = None
    #w_plot = 5; ncols=1; nrows=1;
    if figsize is None:
        figsize = [ncols * w_plot + w_cbar, w_plot*nrows]
    # if figsize is None:
    #    figsize= [18,4]
    # c = 0.1/2.1
    # siz_bar = c*ncols/(1-c)
    #sharex=False
    #sharey = False

    if width_ratios is None:
        width_ratios = [1] * ncols + [w_cbar / w_plot]
    if add_gs_kw is None:
        add_gs_kw = dict()


    if 'hspace' not in add_gs_kw.keys():
        add_gs_kw['hspace'] = 0
    if 'wspace' not in add_gs_kw.keys():
        add_gs_kw['wspace'] = 0


    # add_gs_kw['width_ratios'] = width_ratios
    fig = plt.figure(figsize=figsize,dpi=150)

    gs = fig.add_gridspec(nrows, ncols + 1, **add_gs_kw)
    w_r1 = [sum(width_ratios[:-1]), width_ratios[-1]]
    h_r1 = [width_ratios[-1],sum(width_ratios[:-1]), ]
    gs0 = gridspec.GridSpec(1, 1, figure=fig,)# width_ratios=w_r1, height_ratios=h_r1, **add_gs_kw)

    gs00 = gridspec.GridSpecFromSubplotSpec(nrows+1, ncols+1, width_ratios=w_r1, height_ratios=h_r1, subplot_spec=gs0[0], **add_gs_kw)

    # gs_s = gs[:,:(ncols+1)].subgridspec(nrows=nrows, ncols=ncols, wspace=add_gs_kw['wspace'], hspace=add_gs_kw['hspace'])
    axs = gs00.subplots(sharex=sharex, sharey=sharey, )
    #dax1 = fig.add_subplot(gs0[1, -1])
    #dax2 = fig.add_subplot(gs0[0,0])
    #add_cbar(cax)
    axs[0,1].clear()
    axs[0,1].axis("off")
    daxs = dict(x=axs[0,0],y=axs[1,1])
    for a in daxs:
        _ax = daxs[a]
        sns.despine(bottom=False, left=False, ax=_ax)
        _ax.axis("off")
    #daxs = [dax1,dax2]
    #axs = np.array(axs)

    ax = axs[1,0]

    return fig, ax, daxs
#make_cool_grid2()


def fix_ax_labs(axs, x=True, y=True):
    if len(axs.shape)==1:
        _axs = np.expand_dims(axs, axis=0)
    else:
        _axs = axs


    #print(axs)
    if x:
        for ax in _axs[-1,:]:

            print(ax)
            x_ticks_new = ax.get_xticklabels()#[1:-1]
            t = x_ticks_new[-1]
            nt = matplotlib.text.Text(t.get_position()[0],t.get_position()[1])
            x_ticks_new[-1] = nt
            ax.set_xticklabels(x_ticks_new)
    if y:
        for ax in _axs[:,0]:
            #t = ax.get_xticklabels()[0]
            #nt = matplotlib.text.Text(t.get_position()[0],t.get_position()[1])
            y_ticks_new = ax.get_yticklabels()#[1:-1]
            #x_ticks_new[0] = nt
            t = y_ticks_new[-1]
            nt = matplotlib.text.Text(t.get_position()[0],t.get_position()[1])
            y_ticks_new[-1] = nt
            ax.set_yticklabels(y_ticks_new)
    return




def make_cool_grid5(figsize=None,
                    width_ratios=None,
                    ncols=1,
                    nrows=1,
                    num_subplots_per_big_plot=2,
                    size_big_plot=5,
                    add_gs_kw=None,
                    sharex='col',
                    sharey='row',

                    w_plot = 5.,
                    w_cbar = 1,
                    w_ratio_sideplot = 0.6,
                    frac_dist_axis_from_big = .15
                    ):
    width_small_plot = size_big_plot/num_subplots_per_big_plot
    width_dist_ax = size_big_plot*frac_dist_axis_from_big

    if figsize is None:

        figsize = [size_big_plot + width_small_plot+ width_dist_ax,
                   size_big_plot + width_small_plot+ width_dist_ax,
                   ]
    width_ratios = None
    add_gs_kw = None

    if width_ratios is None:
        width_ratios = [1] * ncols + [w_cbar / w_plot] #+ [1]* ncols_extra
    if add_gs_kw is None:
        add_gs_kw = dict()


    if 'hspace' not in add_gs_kw.keys():
        add_gs_kw['hspace'] = 0
    if 'wspace' not in add_gs_kw.keys():
        add_gs_kw['wspace'] = 0

    fig = plt.figure(figsize=figsize,
                     dpi=100)

    w_r1 = [size_big_plot,size_big_plot*frac_dist_axis_from_big]
    h_r1 = [frac_dist_axis_from_big,1, ]

    gs0 = gridspec.GridSpec(2, 2, figure=fig, height_ratios= [size_big_plot+width_dist_ax,width_small_plot],
                            width_ratios = [size_big_plot+width_dist_ax,width_small_plot])

    gs00 = gridspec.GridSpecFromSubplotSpec(nrows+1, ncols+1, width_ratios=w_r1, height_ratios=h_r1, subplot_spec=gs0[0,0], **add_gs_kw)
    # for the small plots:
    gs01 = gridspec.GridSpecFromSubplotSpec(num_subplots_per_big_plot+1,1, subplot_spec=gs0[:,1])#, **add_gs_kw)
    gs03 = gridspec.GridSpecFromSubplotSpec(1,num_subplots_per_big_plot, subplot_spec=gs0[1,:1])#, **add_gs_kw)

    axs = gs00.subplots(sharex=sharex, sharey=sharey, )
    axs_extra = gs01.subplots(sharex=sharex, sharey=sharey, )
    axs_extra2 = gs03.subplots(sharex=sharex, sharey=sharey, )
    axs_extra = np.concatenate((axs_extra, axs_extra2,))
    axs[0,1].clear()
    axs[0,1].axis("off")
    daxs = dict(x=axs[0,0],y=axs[1,1])
    # distribution axis
    for a in daxs:
        _ax = daxs[a]
        sns.despine(bottom=False, left=False, ax=_ax)
        _ax.axis("off")

    ax = axs[1,0]

    for ax_e in axs_extra:
        ax_e.set_xlabel('')
        ax_e.set_ylabel('')
        ax_e.set_ylim(ax.get_ylim())
        ax_e.set_xlim(ax.get_xlim())
        ax_e.axes.xaxis.set_ticklabels([])
        ax_e.axes.yaxis.set_ticklabels([])

        sns.despine(ax = ax_e)


    return fig, ax, daxs, axs_extra


