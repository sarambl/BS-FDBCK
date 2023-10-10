import pprint
# %%
from glob import glob

import numpy as np
from useful_scit.imps import (plt, sns, pd)

from ukesm_bs_fdbck.constants import path_eusaar_data, path_eusaar_outdata, package_base_path  # path_eusaar_data
from ukesm_bs_fdbck.data_info import get_nice_name_case
from ukesm_bs_fdbck.util.plot.colors import get_case_col

p_histc = path_eusaar_data / 'HISTC/'
p_gen = path_eusaar_data / 'GEN/'
p_distc = path_eusaar_data / 'DISTC/'
station_table_path = package_base_path / 'ukesm_bs_fdbck' / 'misc' / 'eusaar_table2_asmi2011.csv'

fl = glob(str(p_histc) + '*')
fl = [f.split('/')[-1] for f in fl]
station_codes = pd.Series([f[:3] for f in fl]).unique()
# subs list
# %%
fl = glob(str(p_distc) + '*')
fl = [f.split('/')[-1] for f in fl]
subs_codes = pd.Series([f[4:7] for f in fl]).unique()
years_codes = ['BOTH', '2008', '2009']
# %%
# The time array is exactly equal this
time_h = pd.date_range(start='2008-01-01', end='2010-01-01', freq='h', name='time')
standard_varlist_histc = ['N30', 'N50', 'N100', 'N250']
long_name_var_dic = dict(
    N30='N$_{30-50}$',
    N50='N$_{50-500}$',
    N100='N$_{100-500}$',
    N250='N$_{250-500}$'
)

savepath_histc_vars = path_eusaar_outdata / 'HISTC_Nvars.nc'

savepath_histc_flags = path_eusaar_outdata / 'HISTC_flags.nc'


# %%

def get_station_locations():
    df = pd.read_csv(station_table_path).set_index('Station code')
    return df.transpose()


# %%
def dms_to_dd(d, m, s):
    dd = d + float(m) / 60 + float(s) / 3600
    return dd


# %%
def get_min_deg(lat):
    #
    deg, min = lat.split('◦')
    dir = min.strip()[-1]
    if min.strip()[0].isdigit():
        min = int(min.strip().split(' ')[0])  # .strip()
    else:
        min = 0
    deg = int(deg)
    return deg, min, dir  # , min


def rewrite_coord(lat, spec='◦'):
    deg, min, dir = get_min_deg(lat)
    if min >= 100:
        min = int(min / 10)
    return '%s %s %s %s' % (deg, spec, min, dir)


def min_to_dec(st):
    lat, lon, alt = st.split(',')
    deg, min, dir = get_min_deg(lat)
    lat_dd = dms_to_dd(deg, min, 0)
    deg, min, dir = get_min_deg(lon)
    lon_dd = dms_to_dd(deg, min, 0)

    return lat_dd, lon_dd, alt


def clean_df(df, sc='eusaar'):
    """
    cleanes one source
    author: aliaga daliaga_at_chacaltaya.edu.bo
    :param df:
    :param sc:
    :return:
    """
    df_eu = df[df['source'] == sc].copy()
    df_eu = df_eu.set_index(['time', 'station'])
    df_eu = df_eu.drop(
        # ['Unnamed: 0', 'source', 'index'],
        ['source'],
        axis=1)
    return df_eu


def clean_quantiles(dfm, qM, qm, xv, yv):
    """
    removes above quantile values.
    author: aliaga daliaga_at_chacaltaya.edu.bo

    :param dfm:
    :param qM:
    :param qm:
    :param xv:
    :param yv:
    :return:
    """
    qs = dfm.quantile([qm, qM])
    _b = (dfm[xv] > qs[xv][qm]) & \
         (dfm[xv] < qs[xv][qM]) & \
         (dfm[yv] > qs[yv][qm]) & \
         (dfm[yv] < qs[yv][qM])
    dfq = dfm[_b]
    return dfq, qs


def get_merge(df_dic, var, xl, yl):
    """
    Merge variable from two dataframes and removes nans
    Add suffixes
    author: aliaga daliaga_at_chacaltaya.edu.bo

    :param df_dic:
    :param var:
    :param xl: suffix
    :param yl: suffix
    :return:
    """

    dfm = pd.merge(
        df_dic[xl][[var]], df_dic[yl][[var]],
        left_index=True, right_index=True,
        suffixes=[f'_{xl}', f'_{yl}']
    )
    dfm: pd.DataFrame
    dfm = dfm.dropna(axis=0)
    return dfm


def get_dfs(df_dic, qM, qm, var, xl, yl):
    """
    returns dic with normal, quantile cleaned, log
    author: aliaga daliaga_at_chacaltaya.edu.bo
    :param df_dic:
    :param qM:
    :param qm:
    :param var:
    :param xl:
    :param yl:
    :return:
    """
    xv = f'{var}_{xl}'
    yv = f'{var}_{yl}'
    dfm = get_merge(df_dic, var, xl, yl)
    dfq, quan_df = clean_quantiles(dfm, qM, qm, xv, yv)
    dfml = np.log10(dfq)

    dfs = {'norm': dfm, 'quan': dfq, 'log': dfml,
           'xv': xv, 'yv': yv, 'quan_df': quan_df}

    return dfs


def plot_res_season(ddic, sec, typ='quan', axs=None, figsize=None):
    """
    author: aliaga daliaga_at_chacaltaya.edu.bo
    :param axs:
    :param figsize:
    :param ddic:
    :param sec:
    :param typ:
    :return:
    """
    if figsize is None:
        figsize = [7, 7]
    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=figsize)
    # %%
    color = get_case_col(sec)
    dif_ = get_dif(ddic, sec, typ)
    # %%
    dif_2 = get_time_season(dif_)
    # dif_
    # %%
    for seas, ax in zip(['DJF', 'MAM', 'JJA', 'SON'], axs.flatten()):
        pdi = dif_2[dif_2['season'] == seas]
        sns.distplot(pdi['dif'], label=get_nice_name_case(sec), ax=ax, color=color)
        ax.set_title(seas)
    # sns.distplot(dif_, label=sec)#, ax=ax)
    # plt.show()
    # %%
    return dif_


def plot_res(ddic, sec, typ='quan', ax=None):
    """
    author: aliaga daliaga_at_chacaltaya.edu.bo
    :param ax:
    :param ddic:
    :param sec:
    :param typ:
    :return:
    """
    # %%
    dif_ = get_dif(ddic, sec, typ)
    dif_
    color = get_case_col(sec)

    # %%
    sns.distplot(dif_, label=get_nice_name_case(sec), ax=ax, color=color)
    # %%
    return dif_


def get_dif(ddic, sec, typ):
    """
    author: Aliaga Daliaga_at_chacaltaya.edu.bo

    :param ddic:
    :param sec:
    :param typ:
    :return:
    """
    dic = ddic[sec]
    dfq = dic[typ]
    yv_ = dic['yv']
    xv_ = dic['xv']
    dif_ = dfq[yv_] - dfq[xv_]
    return dif_


def plot_den_dist_comp(ddic, nsec, sec, var, ax=None, xlabel=None):
    """
    author: aliaga daliaga_at_chacaltaya.edu.bo

    :param ax:
    :param xlabel:
    :param ddic:
    :param nsec:
    :param sec:
    :param var:
    :return:
    """
    typ = 'quan'
    plot_res(ddic, sec, typ=typ, ax=ax)
    plot_res(ddic, nsec, typ=typ, ax=ax)

    if ax is None:
        ax: plt.Axes = plt.gca()
    ax.legend()
    ax.set_ylabel('density dist.')
    if xlabel is None:
        xlabel = fr'''$m_{{c_s}} - o_{{c_s}}$ 
        where o=observed, m=modeled, and $c_s$={var}'''
    ax.set_xlabel(xlabel)
    ax.axvline(x=0, color='w', linestyle='--')
    # ax.set_xlim(-.8e4,.8e4)
    # plt.tight_layout()
    return ax


def plot_den_dist_comp_cases(ddic, cases, var, ax=None, xlabel=None):
    """
    author: aliaga daliaga_at_chacaltaya.edu.bo

    :param cases:
    :param ax:
    :param xlabel:
    :param ddic:
    :param var:
    :return:
    """
    typ = 'quan'
    for case in cases:
        plot_res(ddic, case, typ=typ, ax=ax)

    if ax is None:
        ax: plt.Axes = plt.gca()
    ax.legend()
    ax.set_ylabel('density dist.')
    if xlabel is None:
        xlabel = fr'''$m_{{c_s}} - o_{{c_s}}$ 
        where o=observed, m=modeled, and $c_s$={var}'''
    ax.set_xlabel(xlabel)
    ax.axvline(x=0, color='w', linestyle='--')
    # ax.set_xlim(-.8e4,.8e4)
    # plt.tight_layout()
    return ax


def plot_den_dist_comp_seas(ddic, nsec, sec, var, axs=None, xlabel=None, unit='[cm$^{-3}$]'):
    """
    author: aliaga daliaga_at_chacaltaya.edu.bo

    :param axs:
    :param xlabel:
    :param unit:
    :param ddic:
    :param nsec:
    :param sec:
    :param var:
    :return:
    """
    if axs is None:
        fig, axs = plt.subplots(2, 2)
    typ = 'quan'
    plot_res_season(ddic, sec, typ=typ, axs=axs)
    plot_res_season(ddic, nsec, typ=typ, axs=axs)
    for ax in axs.flatten()[0::2]:
        ax.set_ylabel('density dist.')
    for ax in axs.flatten():
        ax.set_xlabel('')
    for ax in axs.flatten()[2:]:
        # if xlabel is None:
        xlabel = 'm(%s)-o(%s) %s' % (var, var, unit)
        # fr'''$m_{{c_s}} - o_{{c_s}}$
        # where o=observed, m=modeled, and $c_s$={var}'''

        ax.set_xlabel(xlabel)
    for ax in axs.flatten():
        ax.legend()
        ax.axvline(x=0, color='w', linestyle='--')
    # ax.set_xlim(-.8e4,.8e4)
    # plt.tight_layout()
    return ax


def plot_den_dist_comp_seas_cases(ddic, cases, var, axs=None, xlabel=None, unit='[cm$^{-3}$]'):
    """
    author: aliaga daliaga_at_chacaltaya.edu.bo

    :param cases:
    :param axs:
    :param xlabel:
    :param unit:
    :param ddic:
    :param var:
    :return:
    """
    if axs is None:
        fig, axs = plt.subplots(2, 2)
    typ = 'quan'
    for case in cases:
        plot_res_season(ddic, case, typ=typ, axs=axs)
    # plot_res_season(ddic, nsec, typ=typ, axs=axs)
    for ax in axs.flatten()[0::2]:
        ax.set_ylabel('density dist.')
    for ax in axs.flatten():
        ax.set_xlabel('')
    for ax in axs.flatten()[2:]:
        # if xlabel is None:
        xlabel = 'm(%s)-o(%s) %s' % (var, var, unit)
        # fr'''$m_{{c_s}} - o_{{c_s}}$
        # where o=observed, m=modeled, and $c_s$={var}'''

        ax.set_xlabel(xlabel)
    for ax in axs.flatten():
        ax.legend()
        ax.axvline(x=0, color='w', linestyle='--')
    # ax.set_xlim(-.8e4,.8e4)
    # plt.tight_layout()
    return ax


def filter_sta_sea(_sd, sta, sea):
    """
        author: aliaga daliaga_at_chacaltaya.edu.bo

    :param _sd:
    :param sta:
    :param sea:
    :return:
    """

    _sd1 = _sd
    _sd1 = _sd1[_sd1['station'] == sta]

    _sd1 = _sd1[_sd1['season'] == sea]
    return _sd1


def get_time_season(sec_d):
    """
    Adds season flag to dataframe
        author: aliaga daliaga_at_chacaltaya.edu.bo

    :param sec_d:
    :return:
    """
    _in: pd.MultiIndex = sec_d.index
    tim = pd.to_datetime(_in.get_level_values('time'))
    _sd = sec_d
    _sd.name = 'dif'
    _sd = _sd.reset_index()
    _sd['time'] = tim
    _sd = _sd.set_index('time')
    _sd['season'] = _sd.to_xarray()['time'].dt.season.to_dataframe()
    return _sd


def plot_hist_grid(nsec_d, sec_d, var, unit='cm$^{-3}$', nsec_name='nsec', sec_name='sec'):
    """
        author: aliaga daliaga_at_chacaltaya.edu.bo

    :param var:
    :param unit:
    :param nsec_name:
    :param sec_name:
    :param nsec_d:
    :param sec_d:
    :return:
    """
    stations = ['VHL', 'SMR', 'CMN', 'KPO', 'OBK', 'JFJ', 'PDD', 'BIR', 'MPZ',
                'HWL', 'ASP', 'ZSF', 'MHD', 'PAL', 'ZEP', 'WAL', 'FKL', 'CBW',
                'HPB']
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    l_sta = len(stations)
    l_sea = len(seasons)
    _sd = get_time_season(sec_d)
    _nd = get_time_season(nsec_d)
    bins = np.arange(-6000, 6000, 250)
    f, axs = plt.subplots(nrows=l_sta, ncols=l_sea, sharey=True, sharex=True,
                          figsize=(10, 13))
    csec = get_case_col(sec_name)
    cnsec = get_case_col(nsec_name)
    for r in range(l_sta):
        for c in range(l_sea):

            sea = seasons[c]
            sta = stations[r]

            ax: plt.Axes = axs[r, c]
            ax.axvline(x=0, color='k', alpha=.5, zorder=0, linestyle='--')
            _sd1 = filter_sta_sea(_sd, sta, sea)
            sns.distplot(_sd1['dif'], kde=False, bins=bins, ax=ax, color=csec)
            _nd1 = filter_sta_sea(_nd, sta, sea)
            sns.distplot(_nd1['dif'], kde=False, bins=bins, ax=ax, color=cnsec)
            ax.set_xlabel('')
            ax.set_yticks(np.arange(0, 600, 200))
            ax.set_xlim(-3999, 3999)
            ax.set_ylim(0.1, 500)
            if r is not l_sta - 1:
                sns.despine(ax=ax, left=True, bottom=True)
            else:
                sns.despine(ax=ax, left=True, bottom=False)

            if r == 0:
                ax.set_title(sea)
            if c == 0:
                ax.set_ylabel(sta)
            else:
                ax.yaxis.set_ticks_position('none')
            if r == l_sta - 1:
                ax.set_xlabel('m(%s)-o(%s) %s' % (var, var, unit))
    f: plt.Figure
    f.subplots_adjust(hspace=0, wspace=0)
    return f, axs


# %%
from ukesm_bs_fdbck.constants import collocate_locations
from useful_scit.util.pd_fix import pd_custom_sort_values

dall_c = "Dall'Osto 2018 categories"
sorter_def = ['North', 'Center', 'South (spring)', 'South (winter)', 'Overlap']


def get_ordered_stations(cat=None, sorter=None):
    if sorter is None:
        sorter = sorter_def
    coll_ltr = collocate_locations.transpose()
    coll_ltr = pd_custom_sort_values(coll_ltr, sorter, cat)
    return coll_ltr.index


# %%


# %%

def plot_hist_grid_cases(case_dic, var, unit='cm$^{-3}$'):
    """
        author: aliaga daliaga_at_chacaltaya.edu.bo

    :param case_dic:
    :param var:
    :param unit:
    :return:
    """
    stations = list(get_ordered_stations())  # ['VHL', 'SMR', 'CMN', 'KPO', 'OBK', 'JFJ', 'PDD', 'BIR', 'MPZ',
    # 'HWL', 'ASP', 'ZSF', 'MHD', 'PAL', 'ZEP', 'WAL', 'FKL', 'CBW',
    # 'HPB']
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    l_sta = len(stations)
    l_sea = len(seasons)
    _cd_dic = {}
    for case in case_dic.keys():
        _cd_dic[case] = get_time_season(case_dic[case])
    # _sd = get_time_season(sec_d)
    # _nd = get_time_season(nsec_d)
    bins = np.arange(-6000, 6000, 250)
    f, axs = plt.subplots(nrows=l_sta, ncols=l_sea, sharey=True, sharex=True,
                          figsize=(10, 15))
    # csec=get_case_col(sec_name)
    # cnsec=get_case_col(nsec_name)
    for r in range(l_sta):
        for c in range(l_sea):

            sea = seasons[c]
            sta = stations[r]

            ax: plt.Axes = axs[r, c]
            ax.axvline(x=0, color='k', alpha=.5, zorder=0, linestyle='--')
            for case in case_dic.keys():
                _d1 = filter_sta_sea(_cd_dic[case], sta, sea)
                sns.distplot(_d1['dif'], kde=False, bins=bins, ax=ax, color=get_case_col(case))
            # _sd1 = filter_sta_sea(_sd, sta, sea)
            # sns.distplot(_sd1['dif'], kde=False, bins=bins, ax=ax, color=csec)
            # _nd1 = filter_sta_sea(_nd, sta, sea)
            # sns.distplot(_nd1['dif'], kde=False, bins=bins, ax=ax, color=cnsec)
            ax.set_xlabel('')
            ax.set_yticks(np.arange(0, 600, 200))
            ax.set_xlim(-3999, 3999)
            ax.set_ylim(0.1, 500)
            if r is not l_sta - 1:
                sns.despine(ax=ax, left=True, bottom=True)
            else:
                sns.despine(ax=ax, left=True, bottom=False)

            if r == 0:
                ax.set_title(sea)
            if c == 0:
                ax.set_ylabel(sta)
            else:
                ax.yaxis.set_ticks_position('none')
            if r == l_sta - 1:
                ax.set_xlabel('m(%s)-o(%s) %s' % (var, var, unit))
    f: plt.Figure
    f.subplots_adjust(hspace=0, wspace=0)
    return f, axs


def print_improvements(nsec_d, sec_d):
    """
        author: aliaga daliaga_at_chacaltaya.edu.bo

    :param nsec_d:
    :param sec_d:
    :return:
    """
    sec_range = sec_d.quantile([.05, .95]).diff().iloc[-1]
    nsec_range = nsec_d.quantile([.05, .95]).diff().iloc[-1]
    ratio = sec_range / nsec_range * 100
    pars = {
        'sec range 5-95%': sec_range,
        'nsec range 5-95%': nsec_range,
        'ratio sec/nsec [%]': round(ratio)
    }
    pprint.pprint(pars)


def get_ddic(df_dic, nsec, qM, qm, sec, var, xl):
    """
    Get dictionary of dictionary of cases  sec and nosec, for one variable.
    Returns dic[case_name][normal|quantile
        author: aliaga daliaga_at_chacaltaya.edu.bo

    :param df_dic: dic of dataframes
    :param nsec: no sectional model key
    :param qM: quantile max threshold
    :param qm: quantile min threshold
    :param sec: sectional model key
    :return: dictionary with the dataframes for nsec,sec, and xl
    :param var: variable to be extracted e.g. 30-100
    :param xl: observational data key
    :returns A dictionary of the dataframe dic for case sec and nsec.
    """
    yls = [sec, nsec]
    ddic = {}
    for yl in yls:
        ddic[yl] = get_dfs(df_dic, qM, qm, var, xl, yl)
    return ddic


def get_ddic_cases(df_dic, cases, qM, qm, var, xl):
    """
    Get dictionary of dictionary of cases  sec and nosec, for one variable.
    Returns dic[case_name][normal|quantile
        author: aliaga daliaga_at_chacaltaya.edu.bo

    :param cases:
    :param df_dic: dic of dataframes
    :param qM: quantile max threshold
    :param qm: quantile min threshold
    :return: dictionary with the dataframes for nsec,sec, and xl
    :param var: variable to be extracted e.g. 30-100
    :param xl: observational data key
    :returns A dictionary of the dataframe dic for case sec and nsec.
    """
    yls = cases  # [sec, nsec]
    ddic = {}
    for yl in yls:
        ddic[yl] = get_dfs(df_dic, qM, qm, var, xl, yl)
    return ddic
