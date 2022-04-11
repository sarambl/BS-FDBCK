import matplotlib.pyplot as plt
from matplotlib import colors
from useful_scit.plot import get_cmap_dic

from bs_fdbck import constants as constants
from bs_fdbck.constants import paths_plotsave
from bs_fdbck.util.Nd.sizedist_class_v2.SizedistributionSurface import SizedistributionSurface
from bs_fdbck.util.plot.Nd_plot import plot_sizedist_time
from bs_fdbck.util.practical_functions import make_folders


def plot_sizedist_time_cases(cases, ss_start_t, ss_end_t, location, figsize=None,
                             resolution='month', history_field='.h0.', minDiameter=5., maxDiameter=39.6,
                             vmin=10., vmax=5.e3):
    if figsize is None:
        figsize = [14, 15]
    fig, axs = plt.subplots(len(cases), 1, figsize=figsize, sharex=True, sharey=True)
    for case, ax in zip(cases, axs):
        isSectional = 'noSECT' not in case
        s = SizedistributionSurface(case, ss_start_t, ss_end_t,
                                    [minDiameter, maxDiameter], isSectional, resolution,
                                    history_field=history_field)
        ds = s.get_collocated_dataset()
        a = plot_sizedist_time(ds, ss_start_t, ss_end_t,
                               location=location,
                               var=None,
                               ax=ax,
                               figsize=figsize,
                               vmin=vmin, vmax=vmax,
                               norm_fun=colors.LogNorm,
                               )
    plt.tight_layout()
    path_save = str(paths_plotsave['sizedist_time']) + '/month/' + '_'.join(cases) + 'surf_%s_%s.png' % (ss_start_t, ss_end_t)
    make_folders(path_save)
    plt.savefig(path_save, dpi=200)
    plt.show()


def plot_seasonal_surface_loc_sizedistributions(cases_sec, cases_orig, from_t, to_t,
                                                variables=None,
                                                minDiameter=5., maxDiameter=39.6, resolution='month',
                                                history_field='.h0.',
                                                figsize=None, locations=constants.collocate_locations.keys()):
    if variables is None:
        variables = ['dNdlogD']
    if figsize is None:
        figsize = [12, 6]
    cl = SizedistributionSurface
    s_list = []
    for case in cases_sec:
        s1 = cl(case, from_t, to_t,
                [minDiameter, maxDiameter], True, resolution,
                history_field=history_field)
        s_list.append(s1)
    for case in cases_orig:
        s1 = cl(case, from_t, to_t,
                [minDiameter, maxDiameter], False, resolution,
                history_field=history_field)
        s_list.append(s1)
    cmap_dic = get_cmap_dic(cases_sec + cases_orig)

    for loc in locations:
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs = axs.flatten()
        for s in s_list:
            ls = variables
            # if s.isSectional:
            #    ls = ls + ['dNdlogD_sec']
            s.plot_location(variables=ls,
                            c=cmap_dic[s.case_name],
                            axs=axs,
                            loc=loc,
                            ylim=[10, 1e4])
        fig.tight_layout()
        savepath = constants.paths_plotsave['sizedist'] + '/season/'
        _cas = '_'.join(cases_sec) + '_' + '_'.join(cases_orig)
        savepath = savepath + loc + _cas + 'mean_%s-%s_%s.png' % (from_t, to_t, resolution)
        make_folders(savepath)
        plt.savefig(savepath, dpi=200)
        plt.show()
