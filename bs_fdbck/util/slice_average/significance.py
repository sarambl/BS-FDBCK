import matplotlib as mpl
from cartopy import crs as ccrs
from scipy import stats
import numpy as np

from bs_fdbck.util.slice_average.avg_pkg import yearly_mean_dic
# %%
# %%
def get_significance_map_paired_monthly(var,
                                        case1,
                                        case2,
                                        startyear,
                                        endyear,
                                        pmin=850.,
                                        pressure_adjust=True,
                                        avg_over_lev=True,
                                        ci=.95,
                                        groupby=None,
                                        dims=None,
                                        area='Global',
                                        avg_dim='time'
                                        ):
    """
    Two-tailed paired t-test with level ci.
    Source: Modern Mathematical Statistics with Applications by Devore & Berk
    Assumptions:
    - data consists of independently selected pairs (X1,Y1)
    - Di=Xi-Yi are assumed to be normally distributed.

    :param avg_dim:
    :param pressure_adjust:
    :param pmin:
    :param endyear:
    :param startyear:
    :param var:
    :param case1:
    :param case2:
    :param avg_over_lev:
    :param ci:
    :param groupby:
    :param dims:
    :param area:
    :return:
    """
    dic_means_yr = yearly_mean_dic([var], [case1, case2], startyear, endyear, pmin, pressure_adjust,
                                   avg_over_lev=avg_over_lev, groupby=groupby, dims=dims, area=area)
    da1 = dic_means_yr[case1][var]
    da2 = dic_means_yr[case2][var]
    T, data4comp, sig_map, t = calc_significance_field(case1, case2, da1, da2, avg_dim, ci)
    return T, t, sig_map, data4comp

    # %%


def calc_significance_field(case1, case2, da1, da2, avg_dim, ci):
    data4comp = {case1: da1, case2: da2}
    # difference:
    # da1 = da1.coarsen({'lat':8,'lon':8}).mean()
    # da2 = da2.coarsen({'lat':8,'lon':8}).mean()
    D = da1 - da2
    data4comp[f'{case1}-{case2}'] = D.copy()
    T, sig_map, t = calc_sign_over_fields(da1, da2, avg_dim, ci)
    return T, data4comp, sig_map, t


def calc_sign_over_fields(da1, da2, avg_dim='time', ci=.90):
    """
    Source: Modern Mathematical Statistics with Applications by Devore & Berk
    Assumptions:
    - data consists of independently selected pairs (X1,Y1)
    - Di=Xi-Yi are assumed to be normally distributed.

    :param da1:
    :param da2:
    :param avg_dim:
    :param ci:
    :return:
    """
    D = da1 - da2
    n = len(D.coords[avg_dim])
    mean_D = D.mean(avg_dim)
    # sample std:
    # population std= sqrt(sum((xi-mu)^2)/n)
    # sample std =  sqrt(sum((xi-mu)^2)/(n-1))
    # sample std = (population std)*(sqrt(n/(n-1))
    s_std = D.std(avg_dim) * np.sqrt(n / (n - 1))
    t = mean_D / (s_std / np.sqrt(n))
    # for small samples (<50) we use t-statistics
    # n = 9, degree of freedom = 9-1 = 8
    # for 99% confidence interval, alpha = 1% = 0.01 and alpha/2 = 0.005
    T = stats.t.ppf(1 - ((1 - ci) / 2), n - 1)  # 99% CI, t8,0.005
    sig_map = (t > T) | (t < -T)
    return T, sig_map, t


def hatch_area_sign(t,T,ax, hatches=None, hatch_lw = .5, reverse=False, transform=None):
    """
    hatch significant area
    :param transform:
    :param t:
    :param T:
    :param ax:
    :param hatches:
    :param hatch_lw:
    :param reverse:
    :return:
    """
    if hatches is None:
        hatches=['....','']
    TF = ((t>T)|(t<-T))
    hatch_area(TF,ax,hatches=hatches, hatch_lw=hatch_lw,reverse=reverse,transform=transform)


def hatch_area(TF, ax, hatches=None, hatch_lw = .5, reverse=False, transform=None):
    mpl.rcParams['hatch.linewidth'] = hatch_lw
    if reverse:
        TF=~TF
    if hatches is None:
        hatches=['....','']
    if set(TF.dims)=={'lat','lev'}:
        x_coord = TF.lat
        y_coord = TF.lev
    elif set(TF.dims)=={'lat','lon'}:
        x_coord = TF.lon
        y_coord = TF.lat
    else:
        x_coord = TF[TF.dims[0]]
        y_coord = TF[TF.dims[1]]

    #ax.contourf( x_coord,y_coord, TF.where(TF), hatches=hatches, transform=transform, alpha=0, extend='both')
    plt_kwgs = dict(hatch=hatches[0], rasterized=True, alpha=0)
    if transform is not None:
        plt_kwgs['transform']=transform
    ax.pcolor( x_coord,y_coord, TF.where(TF), **plt_kwgs)#hatch=hatches[0],rasterized=True, transform=transform, alpha=0)


def load_and_plot_sign(to_case,from_cases, saxs, var, startyear, endyear,
                       pressure_adjust=True,
                       avg_over_lev=True,
                       ci=.95,
                       groupby=None,
                       dims=('lev',),
                       area='Global',
                       avg_dim='time',
                       hatches=None, hatch_lw = 1,
                       transform=ccrs.PlateCarree(),
                       reverse=False):
    if hatches is None:
        hatches=['....', '']
        #hatches=['/////', '']

    for case_f,ax in zip(from_cases, saxs):
        T, t, sig_map, data4comp = get_significance_map_paired_monthly(var,
                                                                      to_case,
                                                                      case_f,
                                                                      startyear,
                                                                      endyear,
                                                                      pressure_adjust=pressure_adjust,
                                                                      avg_over_lev=avg_over_lev,
                                                                      ci=ci,
                                                                      groupby=groupby,
                                                                      dims=dims,
                                                                      area=area,
                                                                      avg_dim=avg_dim
                                                                      )
        hatch_area_sign(t, T, ax, hatches=hatches, hatch_lw = hatch_lw, transform=transform, reverse=reverse)#, reverse=True)
    return t, T