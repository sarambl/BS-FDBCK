import numpy as np
from matplotlib.ticker import Locator

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
EVEN_BIGGER=24
colors_models={'NorESM':'r','EC-Earth':'b','ECHAM':'m'}
linestyle_models={'NorESM':'-','EC-Earth':'-.', 'ECHAM':':'}
linewidth=2.
figsize_3by3 = [20,10]
figsize_3by3_pres = [20,10]
normal_settings=[SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, EVEN_BIGGER, colors_models, linestyle_models, linewidth, figsize_3by3]
present_settings= [SMALL_SIZE+2, MEDIUM_SIZE+2, BIGGER_SIZE+2, EVEN_BIGGER+2, colors_models, linestyle_models, linewidth+1, figsize_3by3_pres]
plot_settings={'normal': normal_settings, 'presentation': present_settings}

def set_presentation_mode():
    global MEDIUM_SIZE, BIGGER_SIZE, EVEN_BIGGER, linewidth
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18
    EVEN_BIGGER = 24
    linewidth = 2.5
    return

def set_plot_vars(plot_mode):
    return plot_settings[plot_mode]


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        """Return the locations of the ticks"""
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        #for i in xrange(1, len(majorlocs)):
        for i in np.arange(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))


def insert_abc(ax1, Font_size, subfig_count):
    #print('inserting abc')
    ax1.annotate('(%s)' %(chr(ord('a') + subfig_count)), xy=get_axis_limits(ax1), size = Font_size)


def get_axis_limits(ax, scale_1=1., scale_2=1.):
    return ax.get_xlim()[1]*scale_1, ax.get_ylim()[1]*scale_2


def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)


def set_equal_axis(axs,which='y'):
    if which=='y':
        ylim=[1e99999, -1e9999]
        for ax in axs:
            ylim_d= ax.get_ylim()
            ylim = [min(ylim[0], ylim_d[0]), max(ylim[1],ylim_d[1])]
        for ax in axs:
            ax.set_ylim(ylim)
    elif which == 'x':
        xlim=[1e99999, -1e9999]
        for ax in axs:
            xlim_d= ax.get_xlim()
            xlim = [min(xlim[0], xlim_d[0]), max(xlim[1],xlim_d[1])]
        for ax in axs:
            ax.set_xlim(xlim)
    return