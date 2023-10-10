import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def make_my_grid_subplots(ratios_c, ratios_r, gridspec_kwargs=None):
    if gridspec_kwargs is None:
        gridspec_kwargs = {'wspace': 0.2, 'hspace': 0.2, 'left': .2, 'top': 0.9}
    gridspec_kwargs['hspace']=.5
    gridspec_kwargs['left']=.2
    gridspec_kwargs['bottom']=.2

    fig,gs, axs = _make_grid_subplots(ratios_c, ratios_r, gridspec_kwargs=gridspec_kwargs)
    for ax in axs[0:-1,:].flatten():
        plt.setp(ax.get_xticklabels(), visible=False)
    return fig, gs, axs


def _make_grid_subplots(ratios_c, ratios_r, gridspec_kwargs=None,
                        fig_kwargs=None):
    """
    :param ratios_c: list of whole numbers
    :param ratios_r: list of whole numbers  sd
    :return:
    """
    if gridspec_kwargs is None:
        gridspec_kwargs = {}
    if fig_kwargs is None:
        fig_kwargs = {'figsize': [12, 11]}
    fig2 = plt.figure(constrained_layout=True, **fig_kwargs)
    spec2 = gridspec.GridSpec(ncols=sum(ratios_c), nrows=sum(ratios_r), figure=fig2, **gridspec_kwargs)
    ax_rows = []
    a=0
    for row_ii in ratios_r:
        axs_r = []
        b=0
        for col_ii in ratios_c:
            axs_r.append(fig2.add_subplot(spec2[a:(a+row_ii), b:(b+col_ii)]))#ii, co_id]))
            b+=col_ii
        ax_rows.append(axs_r)
        a+=row_ii
    return fig2, spec2, np.array(ax_rows)




def broken_axis(axs, break_between, ax_break='y'):
    """
    Plot the same plot in axis and break the axis between limits
    given in break_between
    :param axs: list of axis
    :param break_between: list [low_lim, high_lim]
    :param ax_break: 'x' or 'y'
    :return:
    Example:
    >>> fig, axs = plt.subplots(3,3)
    >>> broken_axis(axs[0,1:], [3,10], ax_break='x')
    >>> plt.show()
    """
    spines = {'y': ['bottom', 'top'], 'x': ['right', 'left']}
    # remove spines between plots
    for s, ax in zip(spines[ax_break], axs):
        ax.spines[s].set_visible(False)

    if ax_break == 'y':
        # remove xaxis in between
        axs[0].xaxis.set_visible(False)
        # add dash on axis:
        _add_break_line(axs[0], where='bottom')
        _add_break_line(axs[1], where='top')
        # set limits on the plot:
        lims = axs[1].get_ylim()
        axs[1].set_ylim([lims[0], break_between[0]])
        lims = axs[0].get_ylim()
        axs[0].set_ylim([break_between[1], lims[1]])
    elif ax_break == 'x':
        # Set ticks on left.
        axs[0].yaxis.tick_left()
        # remove ticks to right
        axs[0].tick_params(labelright=False)  # 'off')
        axs[1].yaxis.tick_right()
        # add dash on axis:
        _add_break_line(axs[0], where='right', axis='x')
        _add_break_line(axs[1], where='left', axis= 'x')

        lims = axs[0].get_xlim()
        axs[0].set_xlim([lims[0], break_between[0]])
        lims = axs[1].get_xlim()
        axs[1].set_xlim([break_between[1], lims[1]])

def _add_break_line(ax,where='top', fs= 16, wgt='bold', axis='y'):
    """
    Adds break line to plot
    :param ax: the axis to add to
    :param where: 'top','bottom','left', 'right' (where to add)
    :param fs: size of the line (font size)
    :param wgt: weight of the font
    :param axis: which axis
    :return:
    """
    anno_opts = dict( xycoords='axes fraction',
                      va='center', ha='center', rotation=-45, fontsize=fs, fontweight=wgt)
    if where in ['top', 'right']:k=1
    else: k=0
    if axis=='y':
        ax.annotate('|',xy=(0,k ), **anno_opts)
        ax.annotate('|',xy=(1,k ), **anno_opts)
    else:
        ax.annotate('|',xy=(k,0), **anno_opts)
        ax.annotate('|',xy=(k,1 ), **anno_opts)

