# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import xarray as xr

# %% [markdown]
# ## OA and sizedistribution plots: ATTO station

# %%
# %load_ext autoreload
 
# %autoreload 2

from pathlib import Path
from bs_fdbck_clean.util.BSOA_datamanip import ds2df_inc_preprocessing, ds2df_echam
from bs_fdbck_clean.util.collocate.collocateLONLAToutput import CollocateLONLATout
from bs_fdbck_clean.util.collocate.collocate_echam_salsa import CollocateModelEcham
import useful_scit.util.log as log
from bs_fdbck_clean.util.plot.BSOA_plots import make_cool_grid, plot_scatter
log.ger.setLevel(log.log.INFO)
import time
import xarray as xr
import matplotlib.pyplot as plt

# %%
import pandas as pd
from bs_fdbck_clean.constants import path_measurement_data
import numpy as np



# %%
import scienceplots
import scienceplots
plt.style.use([
    'default',
    #'science',
    #'acp',
    'nature',
    # 'sp-grid',
    'no-black',
    'no-latex',
    'illustrator-safe'
])


# %%

select_station = 'ATTO'
model_lev_i = -2

# %%
plot_path = Path(f'Plots/{select_station}')


# %% pycharm={"name": "#%% \n"}
def make_fn_scat(case, v_x, v_y):
    _x = v_x.split('(')[0]
    _y = v_y.split('(')[0]
    f = f'scat_all_years_echam_noresm_{case}_{_x}_{_y}-ATTO_ukesm_lev{model_lev_i}.png'
    return plot_path /f


# %%
plot_path.mkdir(exist_ok=True, parents=True)

# %%
plot_path

# %%
from bs_fdbck_clean.constants import path_measurement_data
postproc_data = path_measurement_data /'model_station'/select_station
postproc_data_obs = path_measurement_data /select_station/'processed'


# %%
fn_obs_comb_data_full_time =postproc_data_obs /'ATTO_data_comb_hourly.nc'

# %% [markdown]
# # Load observational data: 

# %%
postproc_data_obs = path_measurement_data /'ATTO'/'processed'

# %%


ds_ATTO = xr.open_dataset(fn_obs_comb_data_full_time)

# %%
fn_obs_comb_data_full_time

# %% [markdown] tags=[]
# # Read in model data:

# %%
models = ['ECHAM-SALSA','NorESM', 'EC-Earth', 'UKESM']
mod2cases = {'ECHAM-SALSA' : ['SALSA_BSOA_feedback'],
             'NorESM' : ['OsloAero_intBVOC_f09_f09_mg17_fssp'],
             'EC-Earth' : ['ECE3_output_Sara'],
             'UKESM' : ['AEROCOMTRAJ'],
             'Observations':['Obs'],
            }
di_mod2cases = mod2cases.copy()


# %%
select_station='ATTO'

# %%
dic_df_pre=dict()
for mod in models:
    print(mod)
    dic_df_pre[mod] = dict()
    for ca in mod2cases[mod]:
        print(mod, ca)
        #if model_lev_i !=-2:
        fn_out = postproc_data/f'{select_station}_station_{mod}_{ca}_ilev{model_lev_i}.csv'
        #else:
        #    fn_out = postproc_data/f'{select_station}_station_{mod}_{ca}.csv'
        #fn_out = postproc_data/f'{select_station}_station_{mod}_{ca}.csv'
        print(fn_out)
        dic_df_pre[mod][ca] = pd.read_csv(fn_out, index_col=0)
        dic_df_pre[mod][ca].index = pd.to_datetime(dic_df_pre[mod][ca].index)
        #dic_df_mod_case[mod][ca].to_csv(fn_out)

# %% [markdown]
# ## Read in observations

# %%
ds_ATTO = xr.open_dataset(fn_obs_comb_data_full_time, engine='netcdf4')

# %%
ds_ATTO = ds_ATTO.sel(time_traj=0)

# %%
varl = ['Pressure_reanalysis', 'Potential_Temperature_reanalysis', 'Temperature_reanalysis', 'Rainfall', 'Mixing_Depth', 'Relative_Humidity', 'Specific_Humidity_reanalysis',
 'Mixing_Ratio','Solar_Radiation', 'condensation_sink', 'N50-500', 'N100-500', 'N200-500', 'timeUTC-3', 'pressure', 'temperature', 'humidity','precip',#'Solar_inc_Wm2',
        #'Solar_out_Wm2',
        #'wind_speed',
        #'wind_speed_v',
        'OA'
]
ds_ATTO[varl].squeeze().to_dataframe()

# %%
dic_df_pre['Observations'] = dict()
dic_df_pre['Observations']['Observations'] = ds_ATTO[varl].squeeze().to_dataframe()
dic_df_pre['Observations']['Observations'].index = pd.to_datetime(dic_df_pre['Observations']['Observations'].index)

# %%
mod2cases['Observations'] = ['Observations']


# %%
dic_mod_ca = dic_df_pre.copy()

# %%
_ds =dic_mod_ca['ECHAM-SALSA']['SALSA_BSOA_feedback']

_ds =dic_mod_ca['NorESM'][mod2cases['NorESM'][0]]

#_ds['SFisoprene'].plot()


# %% [markdown]
# ### Save result in dictionary

# %%
dic_df_mod_case = di_mod2cases
for mo in models:
    cs = mod2cases[mo]
    for c in cs: 
        if len(cs)>1:
            use_name = f'{mo}_{c}'
        else:
            use_name =mo


# %%
dic_df_mod_case = dic_mod_ca.copy()

# %%
from bs_fdbck_clean.util.BSOA_datamanip import calculate_daily_median_summer,calculate_summer_median

# %% [markdown]
# ## Rename STP values

# %%
for mod in models:
    for ca in mod2cases[mod]:
        _df = dic_df_mod_case[mod][ca]
        for v in ['OA','N50','N100','N200','N500']:
            if f'{v}_STP' in _df.columns:
                if v in _df.columns:
                    _df = _df.rename({v:f'{v}_orig'}, axis=1)
                _df = _df.rename({f'{v}_STP':v}, axis=1)
        dic_df_mod_case[mod][ca] = _df

# %% [markdown]
# ### Calculate Nx-500:
#

# %% tags=[]
for mod in models:
    print(mod)
    for ca in dic_df_mod_case[mod].keys():
        print(ca)
        _df = dic_df_mod_case[mod][ca]
        for v in ['N50','N100','N200']:
            _df[f'{v}-500'] = _df[v] -_df['N500'] 
        dic_df_mod_case[mod][ca] = _df

# %% [markdown] tags=[]
# # Merge with observations:

# %%
dic_df_pre = dict()#dic_df_mod_case.copy()#deep=True)
for mod in dic_df_mod_case.keys():
    dic_df_pre[mod] = dic_df_mod_case[mod].copy()

# %%
vars_obs = ['OA', 'N100-500','N50-500','N200-500','temperature']

# %%
ds_ATTO

# %%
df_ATTO = ds_ATTO[vars_obs].drop('time_traj').to_dataframe()

df_ATTO['some_obs_missing'] = df_ATTO.isnull().any(axis=1)

# %%
df_for_merge = df_ATTO[['OA','N100-500', 'some_obs_missing']].rename({'OA':'obs_OA','N100-500':'obs_N100-500',},axis=1)

# %%
for mod in dic_df_mod_case.keys():
    print(mod)
    for ca in dic_df_mod_case[mod].keys():
        dic_df_mod_case[mod][ca] = pd.merge(dic_df_pre[mod][ca], df_for_merge ,right_on='time', left_on='time', how='outer')
        dic_df_mod_case[mod][ca]['year'] = dic_df_mod_case[mod][ca].index.year

# %%
df_ATTO_obs_rename = df_ATTO.rename({'Org':'OA','temperature':'T_C'}, axis=1)

# %%
df_ATTO_obs_rename

# %% [markdown]
# ## Add observations to dictionary

# %%
dic_df_mod_case['Observations'] = dict()
dic_df_mod_case['Observations']['Observations'] = df_ATTO_obs_rename

# %%
dic_df_mod_case['Observations'].keys()

# %%
dic_mod2case = mod2cases


# %%
def add_log(df, varl=None):
    if varl is None:
        varl = ['OA','N100', 'Org','N100 (cm^-3)','N50','N150','N200']
    var_exist = df.columns
    
    varl_f = set(varl).intersection(var_exist)
    print(varl_f)
    for v in varl_f:
        df[f'log10({v})'] = np.log10(df[v])
    return df


for mod in dic_df_mod_case.keys():
    for c in dic_df_mod_case[mod].keys():
    
        dic_df_mod_case[mod][c] = add_log(dic_df_mod_case[mod][c])
        #dic_dfsm_mod_case[mod][c] = add_log(dic_dfsm_mod_case[mod][c])
        
df_ATTO = add_log(df_ATTO)


# %%
mod='NorESM'

# %%
ca = mod2cases[mod][0]

# %%
mask_obs_N = dic_df_mod_case[mod][ca]['obs_N100-500'].notnull()
mask_obs_OA = dic_df_mod_case[mod][ca]['obs_OA'].notnull()

# %% [markdown]
# ## Compute daily medians:

# %%
path_save_daily_medians = Path(f'Temp_data/{select_station}_daily_medians')
path_save_daily_medians.parent.mkdir(exist_ok=True)

# %%
path_save_daily_medians

# %% [markdown]
# ## For emission analysis we don't remove missing data because it interfers.

# %% [markdown]
# Minimum number of obs in a day:

# %%
min_obs = 5

# %%
dic_df_med = dict()
for mo in dic_df_mod_case.keys():
    for ca in dic_df_mod_case[mo].keys():
        if len(dic_df_mod_case[mo].keys())>1:
            use_name = f'{mo}_{ca}'
        else:
            use_name = mo
            
        _df = dic_df_mod_case[mo][ca]
        
        #_df = _df[_df['some_obs_missing']==False]
        _df_med = _df.resample('D').median()
        _df_count =_df.resample('D').count()
        
        
        _df_med = _df_med[_df_count['OA']>min_obs]
        dic_df_med[use_name] = _df_med
        
        fp = path_save_daily_medians.parent / f'{path_save_daily_medians.name}_{use_name}.csv'
        #dic_df_med[use_name].to_csv(fp)


# %%
for k in dic_df_med.keys():
    dic_df_med[k]['N100-500'].plot(label=k)
plt.legend()

# %%
month_mask = dic_df_med['Observations'].index.month.isin([2,3,4])


# %%
from bs_fdbck_clean.util.plot.BSOA_plots import cdic_model
import seaborn as sns
from matplotlib import pyplot as plt, gridspec as gridspec
from bs_fdbck_clean.util.plot.BSOA_plots import make_cool_grid2, make_cool_grid3
import scipy

# %% [markdown]
# ### Fit funcs

# %%
from bs_fdbck_clean.util.BSOA_datamanip.fits import *
from bs_fdbck_clean.util.BSOA_datamanip.atto import season2month


# %% [markdown] tags=[]
# ### season to monthseason2month

# %%
def select_months(df, season = None, month_list=None):
    if season is not None: 
        month_list = season2month[season]
    

    df['month'] = df.index.month
    return df['month'].isin(month_list)

# %%
from bs_fdbck_clean.util.plot.BSOA_plots import cdic_model

# %%
season = 'FMA'
mo ='ECHAM-SALSA'
df_s2 =  dic_df_med[mo]
print(mo)
mask_months = select_months(df_s2, season=season)
df_s2 = df_s2[mask_months].copy()
print(len(df_s2.dropna()))

df_s2.plot.scatter(x='OA',y='N50-500')
plt.show()
fi, ax = plt.subplots()
df_s2['OA'].plot(marker='.',linewidth=0)

season = 'FMA'
mo ='NorESM'
df_s1 =  dic_df_med[mo]
print(mo)
mask_months = select_months(df_s1, season=season)
df_s1 = df_s1[mask_months].copy()
print(len(df_s1.dropna()))
df_s1['OA'].plot(marker='.', linewidth=0, ax=ax.twinx(), c='r')


# %%
season = 'FMA'
mo ='ECHAM-SALSA'
df_s2 =  dic_df_med[mo]
print(mo)
mask_months = select_months(df_s2, season=season)
df_s2 = df_s2[mask_months].copy()
print(len(df_s2.dropna()))

df_s2.plot.scatter(x='OA',y='N50-500')
plt.show()
fi, ax = plt.subplots()
df_s2['N50-500'].plot(marker='.',linewidth=0)

season = 'FMA'
mo ='NorESM'
df_s1 =  dic_df_med[mo]
print(mo)
mask_months = select_months(df_s1, season=season)
df_s1 = df_s1[mask_months].copy()
print(len(df_s1.dropna()))
df_s1['N50-500'].plot(marker='.', linewidth=0, ax=ax.twinx(), c='r')


# %%
models

# %%
models_and_obs =  models + ['Observations'] 

# %% [markdown]
# ## T to OA

# %%
label_dic =dict(
    T_C=r'T  [$^\circ$C]',
    OA =r'OA [$\mu g m^{-3}$]',
)


# %% [markdown]
# ## Define grid

# %%

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
    #figsize=[10,10]
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
    fig = plt.figure(figsize=figsize,
                     dpi=100)

    #gs = fig.add_gridspec(nrows, ncols, **add_gs_kw)

    
    w_r1 = [size_big_plot,size_big_plot*frac_dist_axis_from_big]
    h_r1 = [frac_dist_axis_from_big,1, ]
    
    gs0 = gridspec.GridSpec(2, 2, figure=fig, height_ratios= [size_big_plot+width_dist_ax,width_small_plot],
                            width_ratios = [size_big_plot+width_dist_ax,width_small_plot])
    #fig.show()
    
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows+1, ncols+1, width_ratios=w_r1, height_ratios=h_r1, subplot_spec=gs0[0,0], **add_gs_kw)
    # for the small plots:
    gs01 = gridspec.GridSpecFromSubplotSpec(num_subplots_per_big_plot+1,1, subplot_spec=gs0[:,1])#, **add_gs_kw)
    gs03 = gridspec.GridSpecFromSubplotSpec(1,num_subplots_per_big_plot, subplot_spec=gs0[1,:1])#, **add_gs_kw)

    # gs_s = gs[:,:(ncols+1)].subgridspec(nrows=nrows, ncols=ncols, wspace=add_gs_kw['wspace'], hspace=add_gs_kw['hspace'])
    axs = gs00.subplots(sharex=sharex, sharey=sharey, )
    axs_extra = gs01.subplots(sharex=sharex, sharey=sharey, )
    axs_extra2 = gs03.subplots(sharex=sharex, sharey=sharey, )
    axs_extra = np.concatenate((axs_extra2, axs_extra[::-1],))
    #for i, ax  in enumerate(axs_extra):
    #    ax.text(1,1,f'{i}')
    axs[0,1].clear()
    axs[0,1].axis("off")
    daxs = dict(x=axs[0,0],y=axs[1,1])
    # distribution axis
    for a in daxs:
        _ax = daxs[a]
        sns.despine(bottom=False, left=False, ax=_ax)
        _ax.axis("off")
    #daxs = [dax1,dax2]
    #axs = np.array(axs)

    ax = axs[1,0]


    return fig, ax, daxs, axs_extra


fig, ax, daxs, axs_extra = make_cool_grid5(#ncols_extra=1, nrows_extra=1
                                           )# w_ratio_sideplot=.5)
#for ax_e in axs_extra:
#    ax_e.set_xlabel('')
#    ax_e.set_ylabel('')
#    ax_e.set_ylim(ax.get_ylim())
#    ax_e.set_xlim(ax.get_xlim())
#    ax_e.axes.xaxis.set_ticklabels([])
#    ax_e.axes.yaxis.set_ticklabels([])

#    sns.despine(ax = ax_e)



# %%
#parameters, cov= curve_fit(f, x, y)

#model = scipy.odr.odrpack.Model(f_wrapper_for_odr)
#data = scipy.odr.odrpack.Data(x,y)
#myodr = scipy.odr.odrpack.ODR(data, model, beta0=parameters,  maxit=0)
#myodr.set_job(fit_type=2)
def compute_p_value(df_s, out, popt):
    parameters = popt
    parameterStatistics = out#myodr.run()    
    x = df_s.dropna()
    df_e = len(x) - len(popt) # degrees of freedom, error
    cov_beta = parameterStatistics.cov_beta # parameter covariance matrix from ODR
    sd_beta = parameterStatistics.sd_beta * parameterStatistics.sd_beta
    ci = []
    t_df = scipy.stats.t.ppf(0.975, df_e)
    ci = []
    for i in range(len(parameters)):
        ci.append([parameters[i] - t_df * parameterStatistics.sd_beta[i], parameters[i] + t_df * parameterStatistics.sd_beta[i]])

    tstat_beta = parameters / parameterStatistics.sd_beta # coeff t-statistics
    pstat_beta = (1.0 - scipy.stats.t.cdf(np.abs(tstat_beta), df_e)) * 2.0    # coef. p-values

    for i in range(len(parameters)):
        print('parameter:', parameters[i])
        print('   conf interval:', ci[i][0], ci[i][1])
        print('   tstat:', tstat_beta[i])
        print('   pstat:', pstat_beta[i])
        print()


# %% [markdown]
# ## Make plot

# %%
def make_plot(v_x, v_y, xlims, ylims, season, 
              xlab=None, ylab=None, alpha_scat=.2,
             source_list = models_and_obs, fig=None, ax=None, daxs=None, axs_extra=None,
              xscale='linear', yscale='linear',
              dic_df_med = dic_df_med,
             ):
    if fig is None: 
        fig, ax, daxs, axs_extra = make_cool_grid3(ncols_extra=2, nrows_extra=3,)# w_ratio_sideplot=.5)

    if xlab is None: 
        if xlab in label_dic:
            xlab = label_dic[v_x]
    if ylab is None: 
        if ylab in label_dic:
            ylab = label_dic[v_y]

    for mo, ax_ex in zip(source_list, axs_extra[:]):
        print(mo)
        df_s =  dic_df_med[mo]

        mask_months = select_months(df_s, season=season)
        df_s = df_s[mask_months].copy()


        sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt*.7, 
                    label='__nolegend__',
                    ax = ax,
                    #facecolor='none',
                    edgecolor=cdic_model[mo],
                        marker='.',
                    
                   )
        sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    #facecolor='none',
                    edgecolor=cdic_model[mo],
                        marker='.',
                    
                    
                   )
        ax_ex.set_title(mo, y=.95)
        
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    fig.suptitle(f'{select_station}, {season} season, 2012-2018', y=.95)
    xlim_dist = list(daxs['y'].get_xlim())
    for mo in source_list:
        print(mo)

        df_s =  dic_df_med[mo]

        mask_months = select_months(df_s, season=season)
        df_s = df_s[mask_months].copy()
        if xscale=='log':
            xbins = np.logspace(np.log10(xlims[0]),np.log10(xlims[1]),20)
        else:
            xbins = np.linspace(xlims[0],xlims[1],20)
            
        if yscale=='log':
            ybins = np.logspace(np.log10(ylims[0]),np.log10(ylims[1]),20)
        else:
            ybins = np.linspace(ylims[0],ylims[1],20)
            

        sns.histplot(#x=v_x,
                    x= df_s[v_x], 
            edgecolor=cdic_model[mo],
            #log_scale=(xscale=='log'),
            color=cdic_model[mo], 
            element="step",
            label=mo,
            linewidth=1,
            #log_scale=(xscale=='log',False,),

            alpha=.1,
            bins=xbins,
            ax = daxs['x'],
                    
                   )
        print(daxs['x'].get_ylim())
        _fi, ax_test = plt.subplots();
        ax_test = sns.histplot(#x=v_x,
            y=df_s[v_y],
            color=cdic_model[mo], 
            element="step",
            label=mo,
            ax = ax_test,
            #ax = daxs['y'],
            linewidth=2,
            
            #edgecolor=None,
            #log_scale=(False,yscale=='log'),
            alpha=.1,
            bins=ybins,
        );
        
        sns.histplot(#x=v_x,
            y=df_s[v_y],
            color=cdic_model[mo], 
            element="step",
            label=mo,
            ax = daxs['y'],
            linewidth=1,
            
            #edgecolor=None,
            #log_scale=(xscale=='log',yscale=='log'),
            #log_scale=(False,yscale=='log'),
            
            alpha=.1,
            bins=ybins,
            )
        xlim_dist_n = list(ax_test.get_xlim())
        _fi.clf()
        #if xlim_dist_n[1]>xlim_dist_n[1]:
        xlim_dist[1] = max(xlim_dist_n[1],xlim_dist[1])
        #daxs['y'].set_xlim([0,xlim_dist[1]])
        
        #plt.show()

    ax.set_ylim(ylims)
    ax.set_xlim(xlims)


    for ax_e in axs_extra:
        ax_e.set_xlabel('')
        ax_e.set_ylabel('')
        ax_e.set_ylim(ax.get_ylim())
        ax_e.set_xlim(ax.get_xlim())
        ax_e.axes.xaxis.set_ticklabels([])
        ax_e.axes.yaxis.set_ticklabels([])

        sns.despine(ax = ax_e)

    return

#### WET_mid

# %% [markdown]
# ## T to OA, exp

# %% [markdown]
# def get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12]):
#     v_log_y = f'ln({v_y})'
#     df_s[v_log_y] = np.log(df_s[v_y])
#     popt, pov, label, func = get_odr_fit_and_labs(df_s, v_x, v_log_y, fit_func = 'linear', return_func=True, beta0=beta0)
#     print('****ignore****')
#     _, _, _, func = get_odr_fit_and_labs(df_s, v_x, v_log_y, fit_func = 'exp', return_func=True, beta0=beta0, pprint=False)
#     print('****stop ignore****')
#     
#     a = np.exp(popt[-1])
#     b = popt[0]
#     if np.abs(a)< 0.009:
#         #a_lab = ((str("%.2e" % a)).replace("e", ' \\cdot 10^{ ')).replace("+0", ") + ' } ')
#         label = '($%.1E) \cdot e^{%5.2fx}$' %(a,b,)
#     else:
#         label = '$%5.2f e^{%5.2fx}$' %(a,b,)
#     popt = [a,b]
#
#     return popt, pov, label, func

# %%
def get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12]):
    v_log_y = f'ln({v_y})'
    df_s[v_log_y] = np.log(df_s[v_y])
    popt, pov, label, func_lin = get_odr_fit_and_labs(df_s, v_x, v_log_y, fit_func = 'linear', return_func=True, beta0=beta0)
    print('****ignore****')
    _, _, _, func = get_odr_fit_and_labs(df_s, v_x, v_log_y, fit_func = 'exp', return_func=True, beta0=beta0, pprint=False)
    print('****stop ignore****')
    popt_lin = popt
    a = np.exp(popt[-1])
    b = popt[0]
    
    if np.abs(a)< 0.009:
        #a_lab = ((str("%.2e" % a)).replace("e", ' \\cdot 10^{ ')).replace("+0", ") + ' } ')
        label = '($%.1E) \cdot e^{%5.2fx}$' %(a,b,)
    else:
        label = '$%5.2f e^{%5.2fx}$' %(a,b,)
    popt = [a,b]

    return popt, pov, label, func, func_lin, popt_lin


# %% [markdown]
# # Emissions

# %%
dict_emissions= {
    'emiisop':'emiisop',
    'emiterp':'emiterp',
    'SFisoprene':'emiisop',
    'SFmonoterp':'emiterp',
    'emi_isop_bio':'emiisop',
    'emi_monot_bio':'emiterp',
    'SFisoprene':'emiisop',
    'SFterpene':'emiterp',
}

# %% [markdown]
# ### UKESM has units: kgC m-2 s-1
#
# for isoprene that should be a correction of M_isop = 68.11702 versus M_isop_c_only = 60.05

# %%
M_isop = 68.11702
M_isop_c_only = 12.01*5
ukesm_isop_unit_correction = M_isop/M_isop_c_only
print('correction isop',ukesm_isop_unit_correction)

M_terp = 136.24
M_terp_c_only = 12.01*10
ukesm_terp_unit_correction = M_terp/M_terp_c_only
print('correction terp',ukesm_terp_unit_correction)

# %%
for m in models:
    print(m)
    _df = dic_df_med[m]
    if m=='UKESM':
        if 'SFisoprene' in _df.columns:
            _df['SFisoprene'] = _df['SFisoprene']*ukesm_isop_unit_correction
            print('corrected UKESM isop emi by factor',ukesm_isop_unit_correction)
        if 'SFterpene' in _df.columns:
            _df['SFterpene'] = _df['SFterpene']*ukesm_terp_unit_correction
            print('corrected UKESM terp emi by factor',ukesm_terp_unit_correction)
        
    _df = _df.rename(dict_emissions, axis=1)
    
    dic_df_med[m] = _df


# %% [markdown]
# ### NorESM has units: kg m-2 s-1

# %% [markdown]
# ### EC-Earth has units kg m-2 s-1

# %% [markdown] tags=[]
# ### ECHAM-SALSA has units kg m-2 s-1

# %% [markdown]
# ### Change units to $\mu g$

# %%
vars_emi = ['emiisop','emiterp']


# %%
for m in models:
    _df = dic_df_med[m]
    for v in vars_emi:
        print(_df[v].mean())
        if _df[v].mean()<1e-9:
            _df[v] = _df[v]*1e9
            print('changes units for {v} from kg/m2/s to ug/m2/s')


# %% [markdown] tags=[]
# ### FMA

# %%
models = ['ECHAM-SALSA', 'NorESM', 'EC-Earth', 'UKESM']

# %% [markdown]
# ### UPS MASK THE INCOMING 

# %%
fig, ax, daxs, axs_extra = make_cool_grid5()##ncols_extra=2, nrows_extra=2,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.flatten()

## Settings
alpha_scatt = 0.6

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu$gm$^{-3}$]'#[$\mu g m^{-3}$]'


linewidth=2
xlims =[5,30]
ylims = [.1,25]
xlims = [22,37]


season='FMA'
v_x = 'T_C'
v_y = 'OA'

dic_df_med_adj = dict()

for k in dic_df_med.keys():
    _df  = dic_df_med[k].copy()
    _df = _df[~_df.index.year.isin([2015,2016])]
    
    dic_df_med_adj[k] = _df

make_plot(v_x, v_y, xlims, ylims, season, 
              xlab, ylab, .3, models, fig, ax, daxs, axs_extra,
          yscale='log',
          dic_df_med = dic_df_med_adj,
          #source_list = models
         
         )


for mo, ax_ex in zip(models[:-1], axs_extra[:]):
    if mo is None:
        continue
    print(mo)
    df_s =  dic_df_med_adj[mo]
    
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    df_s = df_s[df_s[[v_x,v_y]].notna().all(axis=1)]
    #popt, pov, label, func = get_odr_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    popt, pov, label, func,func_lin, popt_lin = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    ax.plot(x, func(x, *popt), c='w', linewidth=linewidth+2,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=linewidth+1, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=linewidth+1,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
               linewidth=linewidth,
              )
    ax_ex.set_yscale('log')
ax.set_yscale('log')


    
fn = make_fn_scat(f'exp1_no2015-2016_{season}', v_x, v_y)
ax.legend(frameon=False)
print(fn)
fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %%
fig, ax, daxs, axs_extra = make_cool_grid5()##ncols_extra=2, nrows_extra=2,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.flatten()

## Settings
alpha_scatt = 0.6

xlab = r'T  [$^\circ$C]'
ylab = r'Isoprene emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'


linewidth=2
xlims =[5,30]
ylims = [1e-3,8e-1]
xlims = [22,37]


season='FMA'
v_x = 'T_C'
v_y = 'emiisop'


make_plot(v_x, v_y, xlims, ylims, season, 
              xlab, ylab, .3, models, fig, ax, daxs, axs_extra,
          yscale='log',
          dic_df_med = dic_df_med,
          #source_list = models
         
         )


for mo, ax_ex in zip(models[:-1], axs_extra[:]):
    if mo is None:
        continue
    print(mo)
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    if 'emi' in v_y:
        df_s = df_s[df_s[v_y]>0]
    df_s = df_s[df_s[[v_x,v_y]].notna().all(axis=1)]
    #popt, pov, label, func = get_odr_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    popt, pov, label, func,func_lin, popt_lin = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    ax.plot(x, func(x, *popt), c='w', linewidth=linewidth+2,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=linewidth+1, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=linewidth+1,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
               linewidth=linewidth,
              )
    ax_ex.set_yscale('log')
ax.set_yscale('log')


    
fn = make_fn_scat(f'exp1_{season}', v_x, v_y)
ax.legend(frameon=False)
print(fn)
fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %%

fig, ax, daxs, axs_extra = make_cool_grid5()##ncols_extra=2, nrows_extra=2,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.flatten()

## Settings
alpha_scatt = 0.6

xlab = r'T  [$^\circ$C]'
ylab = r'Monoterp. emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'


linewidth=2
xlims =[5,30]
ylims = [4e-3,4e-1]
xlims = [22,37]


season='FMA'
v_x = 'T_C'
v_y = 'emiterp'


make_plot(v_x, v_y, xlims, ylims, season, 
              xlab, ylab, .3, models, fig, ax, daxs, axs_extra,
          yscale='log',
          dic_df_med = dic_df_med,
          #source_list = models
         
         )


for mo, ax_ex in zip(models, axs_extra[:]):
    print(mo)
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    if 'emi' in v_y:
        df_s = df_s[df_s[v_y]>0]
    df_s = df_s[df_s[[v_x,v_y]].notna().all(axis=1)]
    #popt, pov, label, func = get_odr_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    popt, pov, label, func,func_lin, popt_lin = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])

    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    
    ax.plot(x, func(x, *popt), c='w', linewidth=linewidth+2,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=linewidth+1, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=linewidth+1,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
               linewidth=linewidth,
              )
    ax_ex.set_yscale('log')
ax.set_yscale('log')


    
fn = make_fn_scat(f'exp1_{season}', v_x, v_y)
ax.legend(frameon=False)
fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %%
fig, ax, daxs, axs_extra = make_cool_grid5()##ncols_extra=2, nrows_extra=2,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.flatten()

## Settings
alpha_scatt = 0.6

xlab = r'Monoterp. emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'

ylab = r'OA[$\mu$g cm$^{-3}$]'#[$\mu g m^{-3}$]'


linewidth=2
lims =[5,30]
xlims = [4e-3,2e-1]
ylims = [1e-1,20]


season='FMA'
v_x = 'emiterp'
v_y = 'OA'


make_plot(v_x, v_y, xlims, ylims, season, 
              xlab, ylab, .3, models, fig, ax, daxs, axs_extra,
          yscale='linear',
          dic_df_med = dic_df_med,
          #source_list = models
         
         )


for mo, ax_ex in zip(models, axs_extra[:]):
    print(mo)
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    if 'emi' in v_y:
        df_s = df_s[df_s[v_y]>0]
    df_s = df_s[df_s[[v_x,v_y]].notna().all(axis=1)]
    #popt, pov, label, func = get_odr_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    #popt, pov, label, func,func_lin, popt_lin = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    popt, pov, label, func= get_linear_fit(df_s, v_x, v_y,  return_func=True,)# beta0=[0.01,.12])

    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    
    ax.plot(x, func(x, *popt), c='w', linewidth=linewidth+2,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=linewidth+1, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=linewidth+1,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
               linewidth=linewidth,
              )
    ax_ex.set_yscale('linear')
ax.set_yscale('linear')
ax.set_xscale('linear')


    
fn = make_fn_scat(f'exp1_{season}', v_x, v_y)
ax.legend(frameon=False)
fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)



plt.show()

# %%
fig, ax, daxs, axs_extra = make_cool_grid5()##ncols_extra=2, nrows_extra=2,)# w_ratio_sideplot=.5)
axs_extra = axs_extra.flatten()

## Settings
alpha_scatt = 0.6

xlab = r'Isoprene emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'

ylab = r'OA[$\mu$g cm$^{-3}$]'#[$\mu g m^{-3}$]'


linewidth=2
lims =[5,30]
xlims = [1e-9,2.5e-1]
ylims = [1e-8,20]


season='FMA'
v_x = 'emiisop'
v_y = 'OA'


make_plot(v_x, v_y, xlims, ylims, season, 
              xlab, ylab, .3, models, fig, ax, daxs, axs_extra,
          yscale='linear',
          dic_df_med = dic_df_med,
          #source_list = models
         
         )


for mo, ax_ex in zip(models, axs_extra[:]):
    print(mo)
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    if 'emi' in v_y:
        df_s = df_s[df_s[v_y]>0]
    df_s = df_s[df_s[[v_x,v_y]].notna().all(axis=1)]
    #popt, pov, label, func = get_odr_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    popt, pov, label, func= get_linear_fit(df_s, v_x, v_y,  return_func=True,)# beta0=[0.01,.12])

    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    
    ax.plot(x, func(x, *popt), c='w', linewidth=linewidth+2,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=linewidth+1, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=linewidth+1,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
               linewidth=linewidth,
              )
    ax_ex.set_yscale('linear')
ax.set_yscale('linear')
ax.set_xscale('linear')


    
fn = make_fn_scat(f'exp1_{season}', v_x, v_y)
ax.legend(frameon=False)
fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)

print(fn)

plt.show()

# %%
axs_extra

# %%
from matplotlib import colors

# %%
y_lab = 'OA [$\mu$gm$^{-3}$]'
v_x1 = 'emiisop'
v_x2 = 'emiterp'

x_lab1 = 'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
x_lab2 = 'Em. monoterp. [$\mu$gm$^{-2}$s$^{-1}$]'

v_y = 'OA'
v_z = 'year'
season = 'FMA'
f, axs_extra = plt.subplots(2,4, figsize=[10,5])

def do_it(v_x,v_y, v_z, axs, xlab, ylab):

    for mo, ax in zip(models, axs):
        print(mo)
        df_mo =  dic_df_med[mo]
        print(mo)
        mask_months = select_months(df_mo, season=season)
        df_mo = df_mo[mask_months].copy()

        s = ax.scatter(df_mo[v_x],df_mo[v_y], 
               edgecolor=None, 
               alpha=.8, 
               c=df_mo[v_z], 
                   s=4,
               cmap=plt.cm.get_cmap('Spectral', lut=7),
                   norm = colors.Normalize(
                   vmin=2011.5,
                   vmax=2018.5,
                   )
              )
        _co = df_mo[v_x].corr( df_mo[v_y])
        print(_co)
        #ax.text(0.1,8, f'corr:{_co:.2f}')
        #ax.set_title(mo)
        ax.set_title(f'{mo}, corr:{_co:.2f}')
    
        ax.set_ylabel(v_y)
        if xlab is None:
            ax.set_xlabel(v_x)
        else:
            ax.set_xlabel(xlab)
        if ylab is None:
            ax.set_ylabel(v_y)
        else:
            ax.set_ylabel(ylab)
            
            
    return s 

################################
axs = axs_extra[0,:]
v_x = v_x1
x_lab = x_lab1
s = do_it(v_x,v_y, v_z, axs, x_lab, ylab)
    
################################
    
################################

axs = axs_extra[1,:]
v_x = v_x2
x_lab = x_lab2
#s = do_it(v_x2,v_y, v_z, axs)
s = do_it(v_x,v_y, v_z, axs, x_lab, ylab)
################################

cbar_ax = f.add_axes([1, 0.33, 0.03, 0.33])
f.colorbar(s, cax=cbar_ax)
f.tight_layout()
for ax in axs_extra.flatten():
    ax.set_xlabel('')
    ax.set_ylabel('')
for ax in axs_extra[0,:]:
    ax.set_xlabel(x_lab1)

for ax in axs_extra[1,:]:
    ax.set_xlabel(x_lab2)

axs_extra[0,0].set_ylabel(y_lab)
axs_extra[1,0].set_ylabel(y_lab)


ax.legend(frameon=False)
fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

f.savefig(fn, dpi=150, bbox_inches="tight")
f.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

print(fn) 

plt.show()

# %%
_df = dic_df_med['Observations']
index_observed_values_exist = _df[~_df[['OA','T_C']].isna().any(axis=1)].index
index_observed_values_exist

# %%
y_lab = 'OA [$\mu$gm$^{-3}$]'
v_x = 'T_C'

x_lab = 'Temperature [$^\circ$C]'
v_y = 'OA'
v_z = 'year'
season = 'FMA'
f, axs_extra = plt.subplots(1,5, figsize=[10,3], sharey=True, sharex=True)

def do_it(v_x,v_y, v_z, axs, xlab, ylab):

    for mo, ax in zip(models_and_obs, axs):
        print(mo)
        df_mo =  dic_df_med[mo]
        print(mo)
        mask_months = select_months(df_mo, season=season)
        df_mo = df_mo[mask_months].copy()
        df_mo['year'] = df_mo.index.year
        #df_mo = df_mo.loc[index_observed_values_exist[index_observed_values_exist.isin(df_mo.index)]]
        s = ax.scatter(df_mo[v_x],df_mo[v_y], 
               edgecolor=None, 
               alpha=.8, 
               c=df_mo[v_z], 
                   s=4,
               cmap=plt.cm.get_cmap('Spectral', lut=7),
                   norm = colors.Normalize(
                   vmin=2011.5,
                   vmax=2018.5,
                   )
              )
        _co = df_mo[v_x].corr( df_mo[v_y])
        print(_co)
        #ax.text(0.1,8, f'corr:{_co:.2f}')
        #ax.set_title(mo)
        ax.set_title(f'{mo}, corr:{_co:.2f}')
    
        ax.set_ylabel(v_y)
        if xlab is None:
            ax.set_xlabel(v_x)
        else:
            ax.set_xlabel(xlab)
        if ylab is None:
            ax.set_ylabel(v_y)
        else:
            ax.set_ylabel(ylab)
            
            
    return s 

################################
axs = axs_extra#[0,:]
s = do_it(v_x,v_y, v_z, axs, x_lab, ylab)
    
################################
    
################################
################################

cbar_ax = f.add_axes([1, 0.1, 0.01, 0.8])
f.colorbar(s, cax=cbar_ax)
f.tight_layout()
for ax in axs_extra.flatten():
    ax.set_xlabel('')
    ax.set_ylabel('')
for ax in axs_extra:
    ax.set_xlabel(x_lab)
    ax.set_yscale('log')

axs_extra[0].set_ylabel(y_lab)



ax.legend(frameon=False)
fn = make_fn_scat(f'scatter_{season}_z{v_z}', v_x, v_y)



f.savefig(fn, dpi=150, bbox_inches="tight")
f.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

print(fn) 

plt.show()

# %%
y_lab = 'OA [$\mu$gm$^{-3}$]'
v_x = 'T_C'

x_lab = 'Temperature [$^\circ$C]'
v_y = 'OA'
v_z = 'year'
season = 'FMA'
f, axs_extra = plt.subplots(1,4, figsize=[10,3], sharey=True)

def do_it(v_x,v_y, v_z, axs, xlab, ylab):

    for mo, ax in zip(models, axs):
        print(mo)
        df_mo =  dic_df_med[mo].copy()
        print(mo)
        df_mo = df_mo[~df_mo.index.year.isin([2015,2016])]
        mask_months = select_months(df_mo, season=season)
        df_mo = df_mo[mask_months].copy()

        s = ax.scatter(df_mo[v_x],df_mo[v_y], 
               edgecolor=None, 
               alpha=.8, 
               c=df_mo[v_z], 
                   s=4,
               cmap=plt.cm.get_cmap('Spectral', lut=7),
                   norm = colors.Normalize(
                   vmin=2011.5,
                   vmax=2018.5,
                   )
              )
        _co = df_mo[v_x].corr( df_mo[v_y])
        print(_co)
        #ax.text(0.1,8, f'corr:{_co:.2f}')
        #ax.set_title(mo)
        ax.set_title(f'{mo}, corr:{_co:.2f}')
    
        ax.set_ylabel(v_y)
        if xlab is None:
            ax.set_xlabel(v_x)
        else:
            ax.set_xlabel(xlab)
        if ylab is None:
            ax.set_ylabel(v_y)
        else:
            ax.set_ylabel(ylab)
            
            
    return s 

################################
axs = axs_extra#[0,:]
s = do_it(v_x,v_y, v_z, axs, x_lab, ylab)
    
################################
    
################################
################################

cbar_ax = f.add_axes([1, 0.1, 0.01, 0.8])
f.colorbar(s, cax=cbar_ax)
f.tight_layout()
for ax in axs_extra.flatten():
    ax.set_xlabel('')
    ax.set_ylabel('')
for ax in axs_extra:
    ax.set_xlabel(x_lab)
    ax.set_yscale('log')

axs_extra[0].set_ylabel(y_lab)



ax.legend(frameon=False)
fn = make_fn_scat(f'scatter_no2015_2016_{season}_z{v_z}', v_x, v_y)



f.savefig(fn, dpi=150, bbox_inches="tight")
f.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

print(fn) 

plt.show()

# %%
v_x = 'T_C'
v_y1 = 'emiisop'
v_y2 = 'emiterp'
y_lab1 = 'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
y_lab2 = 'Em. monoterp. [$\mu$gm$^{-2}$s$^{-1}$]'
x_lab = 'Temperature [$^\circ$C]'
v_z = 'year'

season = 'FMA'
f, axs_extra = plt.subplots(2,4, figsize=[10,5])

def do_it(v_x,v_y, v_z, axs, xlab, ylab):

    for mo, ax in zip(models, axs):
        print(mo)
        df_mo =  dic_df_med[mo]
        print(mo)
        mask_months = select_months(df_mo, season=season)
        df_mo = df_mo[mask_months].copy()

        s = ax.scatter(df_mo[v_x],df_mo[v_y], 
               edgecolor=None, 
               alpha=.8, 
               c=df_mo[v_z], 
                   s=4,
               cmap=plt.cm.get_cmap('Spectral', lut=7),
                   norm = colors.Normalize(
                   vmin=2011.5,
                   vmax=2018.5,
                   )
              )
        _co = df_mo[v_x].corr( df_mo[v_y])
        print(_co)
        #ax.text(0.1,8, f'corr:{_co:.2f}')
        #ax.set_title(mo)
        ax.set_title(f'{mo}, corr:{_co:.2f}')
    
        ax.set_ylabel(v_y)
        if xlab is None:
            ax.set_xlabel(v_x)
        else:
            ax.set_xlabel(xlab)
        if ylab is None:
            ax.set_ylabel(v_y)
        else:
            ax.set_ylabel(ylab)
            
            
    return s 

####################################
axs = axs_extra[0,:]
v_y = v_y1
y_lab = y_lab1
s = do_it(v_x,v_y, v_z, axs, x_lab, y_lab)
    
####################################
v_y = v_y2
y_lab = y_lab2
axs = axs_extra[1,:]
#s = do_it(v_x2,v_y, v_z, axs)
s = do_it(v_x,v_y, v_z, axs, x_lab, y_lab)
####################################
cbar_ax = f.add_axes([1, 0.33, 0.03, 0.33])
f.colorbar(s, cax=cbar_ax)
f.tight_layout()


ax.legend(frameon=False)
fn = make_fn_scat(f'scatter_{season}_{v_x}_z{v_z}', v_y1, v_y2)
for ax in axs_extra.flatten():
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yscale('log')
for ax in axs_extra[1,:]:
    ax.set_xlabel(x_lab)

axs_extra[0,0].set_ylabel(y_lab1)
axs_extra[1,0].set_ylabel(y_lab2)



f.savefig(fn, dpi=150, bbox_inches="tight")
f.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

print(fn) 

plt.show()

# %%
from bs_fdbck_clean.util.plot.plot_settings import insert_abc_axs

# %%
model_name = 'NorESM'
fig, axs = plt.subplots(2,2, figsize = [6,5], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral', lut=7),
               norm = colors.Normalize(
                   vmin=2011.5,
                   vmax=2018.5,
                   )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'T_C'    
v_x2 = 'FSDS_DRF'
v_y1 = 'emiisop'    
v_y2 = 'emiterp'

xlab1 = 'Temperature [$^\circ$C]'
xlab2 = 'SW downelling flux at surface'
ylab1 = 'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = 'Em. monoterp. [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'year'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('log')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('log')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('log')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('log')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

fig.savefig(fn, dpi=150, bbox_inches="tight")
fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(2,2, figsize = [6,5], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral', lut=7),
               norm = colors.Normalize(
                   vmin=2011.5,
                   vmax=2018.5,
                   )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'ISOP_gas'    

v_x2 = 'APIN_gas'
v_y1 = 'oh_con'    
v_y2 = 'VBS1_gas_conc'

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. monoterp. [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'year'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

#fig.savefig(fn, dpi=150, bbox_inches="tight")
#fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(2,2, figsize = [6,5], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral', lut=7),
               norm = colors.Normalize(
                   vmin=2011.5,
                   vmax=2018.5,
                   )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'emiisop'    
v_x2 = 'emiterp'
v_y1 = 'ISOP_gas'    
v_y2 = 'APIN_gas'

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. monoterp. [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'year'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

#fig.savefig(fn, dpi=150, bbox_inches="tight")
#fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(2,2, figsize = [6,5], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral', lut=7),
               norm = colors.Normalize(
                   vmin=2011.5,
                   vmax=2018.5,
                   )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'ISOP_gas'    
v_x2 = 'APIN_gas'
v_y1 = 'VBS0_gas'    
v_y2 = 'VBS1_gas'

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. terpene [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'year'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

#fig.savefig(fn, dpi=150, bbox_inches="tight")
#fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(2,2, figsize = [6,5], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral', lut=7),
               norm = colors.Normalize(
                   vmin=2011.5,
                   vmax=2018.5,
                   )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'ISOP_gas'    
v_x2 = 'APIN_gas'
v_y1 = 'VBS1_gas'    
v_y2 = 'VBS10_gas'

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. terpene [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'year'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

#fig.savefig(fn, dpi=150, bbox_inches="tight")
#fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(2,2, figsize = [6,5], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral', lut=7),
               norm = colors.Normalize(
                   vmin=2011.5,
                   vmax=2018.5,
                   )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'T_C'    
v_x2 = 'APIN_gas'
v_y1 = 'VBS1_gas'    
v_y2 = 'VBS10_gas'

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. terpene [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'year'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

#fig.savefig(fn, dpi=150, bbox_inches="tight")
#fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(2,2, figsize = [6,5], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral', lut=7),
               norm = colors.Normalize(
                   vmin=2011.5,
                   vmax=2018.5,
                   )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'T_C'    
v_x2 = 'APIN_gas'
v_y1 = 'VBS0_gas'    
v_y2 = 'VBS1_gas'

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. terpene [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'year'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

#fig.savefig(fn, dpi=150, bbox_inches="tight")
#fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(2,2, figsize = [6,5], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral_r', lut=7),
               #norm = colors.Normalize(
               #    vmin=2011.5,
               #    vmax=2018.5,
               #    )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'emiterp'    
v_x2 = 'VBS0_gas'
v_y1 = 'OA'    
v_y2 = 'VBS1_gas'

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. terpene [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'T_C'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
print(v_z, lab_z)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
s = do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

#fig.savefig(fn, dpi=150, bbox_inches="tight")
#fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(2,2, figsize = [6,5], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral_r', lut=7),
               #norm = colors.Normalize(
               #    vmin=2011.5,
               #    vmax=2018.5,
               #    )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'emiterp'    
v_x2 = 'VBS1_gas'
v_y1 = 'OA'    
v_y2 = 'VBS1_gas'

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. terpene [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'T_C'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
print(v_z, lab_z)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
s = do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

#fig.savefig(fn, dpi=150, bbox_inches="tight")
#fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(3,2, figsize = [6,8], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral_r', lut=7),
               #norm = colors.Normalize(
               #    vmin=2011.5,
               #    vmax=2018.5,
               #    )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'emiterp'    
v_x2 = 'emiisop'
v_y1 = 'VBS0_gas'
v_y2 = 'VBS1_gas'    
v_y3 = 'VBS10_gas'    

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. terpene [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'year'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
print(v_z, lab_z)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
s = do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################

########################
v_x = v_x1
v_y = v_y3
lab_x = xlab1
lab_y = ylab2
ax = axs[2,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y3
lab_x = xlab2
lab_y = ylab2
ax = axs[2,1]
print(v_x,v_y)
s = do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

#fig.savefig(fn, dpi=150, bbox_inches="tight")
#fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(3,2, figsize = [6,8], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral_r', lut=7),
               #norm = colors.Normalize(
               #    vmin=2011.5,
               #    vmax=2018.5,
               #    )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'emiterp'    
v_x2 = 'emiisop'
v_y1 = 'VBS0_gas'
v_y2 = 'VBS1_gas'    
v_y3 = 'VBS10_gas'    

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. terpene [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'T_C'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
print(v_z, lab_z)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
s = do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################

########################
v_x = v_x1
v_y = v_y3
lab_x = xlab1
lab_y = ylab2
ax = axs[2,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y3
lab_x = xlab2
lab_y = ylab2
ax = axs[2,1]
print(v_x,v_y)
s = do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

#fig.savefig(fn, dpi=150, bbox_inches="tight")
#fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(2,2, figsize = [6,5], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral_r', lut=7),
               #norm = colors.Normalize(
               #    vmin=2011.5,
               #    vmax=2018.5,
               #    )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'emiterp'    
v_x2 = 'emiisop'
v_y1 = 'OA'    
v_y2 = 'VBS0_gas'

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. terpene [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'T_C'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
print(v_z, lab_z)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
s = do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

#fig.savefig(fn, dpi=150, bbox_inches="tight")
#fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(2,2, figsize = [6,5], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral_r', lut=7),
               #norm = colors.Normalize(
               #    vmin=2011.5,
               #    vmax=2018.5,
               #    )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'emiterp'    
v_x2 = 'emiisop'
v_y1 = 'OA'    
v_y2 = 'VBS10_gas'

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. terpene [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'T_C'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
print(v_z, lab_z)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
s = do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

fig.savefig(fn, dpi=150, bbox_inches="tight")
fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(2,2, figsize = [6,5], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral_r', lut=7),
               #norm = colors.Normalize(
               #    vmin=2011.5,
               #    vmax=2018.5,
               #    )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'T_C'    
v_x2 = 'emiisop'
v_y1 = 'OA'    
v_y2 = 'VBS10_gas'

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. terpene [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'emiterp'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
print(v_z, lab_z)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
s = do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

fig.savefig(fn, dpi=150, bbox_inches="tight")
fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(2,2, figsize = [6,5], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral_r', lut=7),
               #norm = colors.Normalize(
               #    vmin=2011.5,
               #    vmax=2018.5,
               #    )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'emiterp'    
v_x2 = 'APIN_gas'
v_y1 = 'OA'    
v_y2 = 'VBS1_gas'

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. terpene [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'T_C'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
print(v_z, lab_z)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
s = do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

#fig.savefig(fn, dpi=150, bbox_inches="tight")
#fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(2,2, figsize = [6,5], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral_r', lut=7),
               #norm = colors.Normalize(
               #    vmin=2011.5,
               #    vmax=2018.5,
               #    )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'emiterp'    
v_x2 = 'oh_con'
v_y1 = 'OA'    
v_y2 = 'VBS1_gas'

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. terpene [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'T_C'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
print(v_z, lab_z)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
s = do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

#fig.savefig(fn, dpi=150, bbox_inches="tight")
#fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(2,2, figsize = [6,5], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral_r', lut=7),
               #norm = colors.Normalize(
               #    vmin=2011.5,
               #    vmax=2018.5,
               #    )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'emiterp'    
v_x2 = 'APIN_gas'
v_y1 = 'OA'    
v_y2 = 'VBS10_gas'

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. terpene [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'T_C'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
print(v_z, lab_z)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
s = do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

#fig.savefig(fn, dpi=150, bbox_inches="tight")
#fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %% tags=[]
model_name = 'ECHAM-SALSA'
fig, axs = plt.subplots(2,2, figsize = [6,5], sharey='row', sharex='col')
axs = axs

## Settings
alpha_scatt = 0.6
linewidth=2
season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()

def do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax):
    s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               s=10,
               
               cmap=plt.cm.get_cmap('Spectral', lut=7),
               norm = colors.Normalize(
                   vmin=2011.5,
                   vmax=2018.5,
                   )

              )

    return s





#df_mo = df_mo.loc['2015':None,:]
v_x1 = 'emiisop'    
v_x2 = 'emiterp'
v_y1 = 'SO2_gas'    
v_y2 = 'VBS10_gas_conc'

xlab1 = v_x1#'Temperature [$^\circ$C]'
xlab2 = v_x2#'SW downelling flux at surface'
ylab1 = v_y1#'Em. isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
ylab2 = v_y2#'Em. terpene [$\mu$gm$^{-2}$s$^{-1}$]'
v_z = 'year'

lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


########################
v_x = v_x1
v_y = v_y1
lab_x = xlab1
lab_y = ylab1
ax = axs[0,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
#ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y1
lab_x = xlab2
lab_y = ylab1
ax = axs[0,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
#ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')
########################

########################
v_x = v_x1
v_y = v_y2
lab_x = xlab1
lab_y = ylab2
ax = axs[1,0]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)
ax.set_yscale('linear')
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

########################
########################
v_x = v_x2
v_y = v_y2
lab_x = xlab2
lab_y = ylab2
ax = axs[1,1]
print(v_x,v_y)
do_it(v_x, v_y, v_z, lab_x, lab_y,lab_z, ax)

ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

########################


cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
fig.colorbar(s,label=lab_z, cax=cbar_ax)


fn = make_fn_scat(f'scatter_{season}_{v_x1}_{v_x2}_z{v_z}', v_x1, v_y)

print(fn)
insert_abc_axs(axs,)# scale_1=0, scale_2=.5)
plt.tight_layout()

#fig.savefig(fn, dpi=150, bbox_inches="tight")
#fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

plt.show()


# %%
li= list(df_mo.columns)
li.sort()
li

# %%
from matplotlib.lines import Line2D
fig, ax = plt.subplots()

v_x ='OA'
v_y='emiisop'
v_y2='emiterp'


for mo in models:
    print(mo)
    df_s =  dic_df_med[mo]
    print(mo)
    _df_cov = df_s[v_y].rolling(30).corr(df_s[v_x])#.groupby('m').mean()#.plot()
    _df_cov.groupby(_df_cov.index.month).mean().plot(label='__nolabel__', c= cdic_model[mo], linestyle=':')
plt.legend()

lines = []
for mo in models:
    print(mo)
    df_s =  dic_df_med[mo]
    print(mo)
    _df_cov = df_s[v_y2].rolling(30).corr(df_s[v_x])#.groupby('m').mean()#.plot()
    l = _df_cov.groupby(_df_cov.index.month).mean().plot(label=mo, c= cdic_model[mo])
    lines.append(l)

    
    
leg1 = ax.legend(bbox_to_anchor=(1,1,))
custom_lines = [Line2D([0], [0], color='k',linestyle='solid', lw=1),
                Line2D([0], [0], color='k',linestyle=':', lw=1)]

ax.legend(custom_lines, ['corr(OA, Em. Terp)','corr(OA, Em. Isop)'], bbox_to_anchor=(1,.3,))

ax.add_artist(leg1)
plt.title(f'Average monthly correlation between OA and BVOC emissions')
plt.ylabel('Pearson corr. coeff.')
plt.xlabel('Month')
ax.set_xticks(np.arange(1,13))

fn = make_fn_scat(f'correlation_through_year_{v_y2}', v_x, v_y)
fig.savefig(fn, dpi=150, bbox_inches="tight")
fig.savefig(fn.with_suffix('.pdf'), dpi=150,bbox_inches="tight")

print(fn)

# %%
mo

# %%
_df_cov = df_s['emiisop'].rolling(30).cov(df_s['OA'])#.groupby('m').mean()#.plot()
_df_cov.groupby(_df_cov.index.month).mean().plot()

# %%

# %%
model_name = 'NorESM'
fig, axs = plt.subplots(1,2, figsize = [8,4])
axs = axs.flatten()

## Settings
alpha_scatt = 0.6

#xlab = r'T  [$^\circ$C]'
#ylab = r'Terpene emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'
linewidth=2
#xlims =[5,30]
#ylims = [4e-3,4e-1]
#xlims = [22,37]


season='FMA'
v_x = 'T_C'
v_y = 'emiisop'
    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()
ax = axs[0]
sns.scatterplot(v_x,v_y,data = df_mo, ax= ax, edgecolor=None, alpha=.3)

ax.set_yscale('log')



season='FMA'
v_x = 'FSDS_DRF'
v_y = 'emiisop'
    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()
ax = axs[1]
print(v_x,v_y)
sns.scatterplot(v_x,v_y,data = df_mo, ax= ax, edgecolor=None, alpha=.3)
ax.set_yscale('log')

plt.show()

# %%
from matplotlib.colors import Normalize

# %%
import matplotlib.colors as colors


# %%
class PiecewiseNorm(colors.Normalize):
    def __init__(self, levels, clip=False):
        # input levels
        self._levels = np.sort(levels)
        # corresponding normalized values between 0 and 1
        self._normed = np.linspace(0, 1, len(levels))
        Normalize.__init__(self, None, None, clip)

    def __call__(self, value, clip=None):
        # linearly interpolate to get the normalized value
        return np.ma.masked_array(np.interp(value, self._levels, self._normed))

    def inverse(self, value):
        return 1.0 - self.__call__(value)


# %%
model_name = 'NorESM'
fig, axs = plt.subplots(1,2, figsize = [7,3], sharey=True)
axs = axs.flatten()

## Settings
alpha_scatt = 0.6

#xlab = r'T  [$^\circ$C]'
#ylab = r'Terpene emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'
linewidth=2
#xlims =[5,30]
#ylims = [4e-3,4e-1]
#xlims = [22,37]


season='FMA'
    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()
v_x = 'T_C'    
v_y = 'emiterp'
v_z = 'FSDS_DRF'

lab_x = 'Temperature [$^\circ$C]'
lab_y = 'Em terp [$\mu$gm$^{-2}$s$^{-1}$]'
lab_z = 'SW radiation surf [Wm$^{-2}$]'


ax = axs[0]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=10),
               norm=colors.LogNorm(1,200, )#,extend='both'
              
              )
plt.colorbar(s, label=lab_z, ax=ax, )
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

ax.set_yscale('log')






v_x = 'FSDS_DRF'
v_z = 'T_C'    

lab_x = 'SW radiation surf [Wm$^{-2}$]'
lab_z = 'Temperature [$^\circ$C]'


ax = axs[1]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], 
               edgecolor=None, 
               alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=10),
              )
plt.colorbar(s, label=lab_z, ax=ax)

ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('log')
#ax.set_xscale('log')

plt.tight_layout()
plt.show()


# %%
model_name = 'NorESM'
fig, axs = plt.subplots(1,2, figsize = [7,3], sharey=True)
axs = axs.flatten()

## Settings
alpha_scatt = 0.6

#xlab = r'T  [$^\circ$C]'
#ylab = r'Terpene emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'
linewidth=2
#xlims =[5,30]
#ylims = [4e-3,4e-1]
#xlims = [22,37]


season='FMA'
v_x = 'T_C'
v_y = 'emiisop'
    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()
ax = axs[0]
v_x = 'T_C'    
v_y = 'emiisop'
v_z = 'FSDS_DRF'

lab_x = 'Temperature [$^\circ$C]'
lab_y = 'Em isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
lab_z = 'SW radiation surf [Wm$^{-2}$]'


ax = axs[0]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=10),
               norm=colors.LogNorm(1,200, )#,extend='both'
              
              )
plt.colorbar(s, label=lab_z, ax=ax, )
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

ax.set_yscale('linear')






v_x = 'FSDS_DRF'
v_y = 'emiisop'
v_z = 'T_C'    

lab_x = 'SW radiation surf [Wm$^{-2}$]'
lab_y = 'Em isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
lab_z = 'Temperature [$^\circ$C]'


ax = axs[1]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], 
               edgecolor=None, 
               alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=10),
              )
plt.colorbar(s, label=lab_z, ax=ax)

ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
#ax.set_xscale('log')

plt.tight_layout()
plt.show()


# %%
model_name = 'NorESM'
fig, axs = plt.subplots(1,2, figsize = [7,3], sharey=True)
axs = axs.flatten()

## Settings
alpha_scatt = 0.6

#xlab = r'T  [$^\circ$C]'
#ylab = r'Terpene emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'
linewidth=2
#xlims =[5,30]
#ylims = [4e-3,4e-1]
#xlims = [22,37]


season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()
#df_mo = df_mo.loc['2015':None,:]
ax = axs[0]
v_x = 'T_C'    
v_y = 'T_C'
v_z = 'year'

lab_x = v_x#'Temperature [$^\circ$C]'
lab_y = v_y#'Em terpene [$\mu$gm$^{-2}$s$^{-1}$]'
lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


ax = axs[0]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               cmap=plt.cm.get_cmap('Spectral', lut=7),
               norm = colors.Normalize(
                   vmin=2011.5,
                   vmax=2018.5,
                   )

              )
plt.colorbar(s, label=lab_z, ax=ax, )
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

ax.set_yscale('log')






v_x = 'FSDS_DRF'
v_y = 'T_C'
v_z = 'year'#'T_C'    

lab_x = v_x#'SW radiation surf [Wm$^{-2}$]'
lab_y = v_y#'Em terpene [$\mu$gm$^{-2}$s$^{-1}$]'
lab_z = v_z#'month'#'Temperature [$^\circ$C]'


ax = axs[1]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], 
               edgecolor=None, 
               alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=10),
               cmap=plt.cm.get_cmap('Spectral', lut=7),
                   norm = colors.Normalize(
                   vmin=2011.5,
                   vmax=2018.5,
                   )

              )
plt.colorbar(s, label=lab_z, ax=ax)

ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')

plt.tight_layout()
plt.show()


# %%
model_name = 'NorESM'
fig, axs = plt.subplots(1,2, figsize = [7,3], sharey=True)
axs = axs.flatten()

## Settings
alpha_scatt = 0.6

#xlab = r'T  [$^\circ$C]'
#ylab = r'Terpene emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'
linewidth=2
#xlims =[5,30]
#ylims = [4e-3,4e-1]
#xlims = [22,37]


season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()
#df_mo = df_mo.loc['2015':None,:]
ax = axs[0]
v_x = 'T_C'    
v_y = 'OA'
v_z = 'year'

lab_x = v_x#'Temperature [$^\circ$C]'
lab_y = v_y#'Em terpene [$\mu$gm$^{-2}$s$^{-1}$]'
lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


ax = axs[0]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
               cmap=plt.cm.get_cmap('Spectral', lut=7),
               norm = colors.Normalize(
                   vmin=2011.5,
                   vmax=2018.5,
                   )

              )
plt.colorbar(s, label=lab_z, ax=ax, )
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

ax.set_yscale('log')






v_x = 'FSDS_DRF'
#v_y = 'emiterp'
v_z = 'year'#'T_C'    

lab_x = v_x#'SW radiation surf [Wm$^{-2}$]'
lab_y = v_y#'Em terpene [$\mu$gm$^{-2}$s$^{-1}$]'
lab_z = v_z#'month'#'Temperature [$^\circ$C]'


ax = axs[1]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], 
               edgecolor=None, 
               alpha=.8, 
               c=df_mo[v_z], 
               #cmap=plt.cm.get_cmap('Reds', lut=10),
               cmap=plt.cm.get_cmap('Spectral', lut=7),
                   norm = colors.Normalize(
                   vmin=2011.5,
                   vmax=2018.5,
                   )

              )
plt.colorbar(s, label=lab_z, ax=ax)

ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('log')
ax.set_xscale('log')

plt.tight_layout()
plt.show()


# %%
model_name = 'NorESM'
fig, axs = plt.subplots(1,2, figsize = [7,3], sharey=True)
axs = axs.flatten()

## Settings
alpha_scatt = 0.6

#xlab = r'T  [$^\circ$C]'
#ylab = r'Monoterp. emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'
linewidth=2
#xlims =[5,30]
#ylims = [4e-3,4e-1]
#xlims = [22,37]


season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()
df_mo = df_mo.loc['2015':None,:]
ax = axs[0]
v_x = 'T_C'    
v_y = 'emiterp'
v_z = 'year'

lab_x = 'Temperature [$^\circ$C]'
lab_y = 'Em monoterp. [$\mu$gm$^{-2}$s$^{-1}$]'
lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


ax = axs[0]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
              
              )
plt.colorbar(s, label=lab_z, ax=ax, )
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

ax.set_yscale('log')






v_x = 'FSDS_DRF'
v_y = 'emiterp'
v_z = 'month'#'T_C'    

lab_x = 'SW radiation surf [Wm$^{-2}$]'
lab_y = 'Em monoterp. [$\mu$gm$^{-2}$s$^{-1}$]'
lab_z = 'month'#'Temperature [$^\circ$C]'


ax = axs[1]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], 
               edgecolor=None, 
               alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=10),
              )
plt.colorbar(s, label=lab_z, ax=ax)

ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('log')
ax.set_xscale('log')

plt.tight_layout()
plt.show()


# %%
model_name = 'NorESM'
fig, axs = plt.subplots(1,2, figsize = [7,3], sharey=True)
axs = axs.flatten()

## Settings
alpha_scatt = 0.6

#xlab = r'T  [$^\circ$C]'
#ylab = r'Terpene emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'
linewidth=2
#xlims =[5,30]
#ylims = [4e-3,4e-1]
#xlims = [22,37]


season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()
df_mo = df_mo.loc['2015':None,:]
ax = axs[0]
v_x = 'T_C'    
v_y = 'OA'
v_z = 'year'

lab_x = v_x#'Temperature [$^\circ$C]'
lab_y = v_y#'Em terpene [$\mu$gm$^{-2}$s$^{-1}$]'
lab_z = v_z#'SW radiation surf [Wm$^{-2}$]'


ax = axs[0]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=8),
               #norm=colors.LogNorm(1,200, )#,extend='both'
              
              )
plt.colorbar(s, label=lab_z, ax=ax, )
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

ax.set_yscale('log')






v_x = 'FSDS_DRF'
v_y = 'OA'
v_z = 'month'#'T_C'    

lab_x = v_x#'SW radiation surf [Wm$^{-2}$]'
lab_y = v_y #'Em terpene [$\mu$gm$^{-2}$s$^{-1}$]'
lab_z = v_z#'month'#'Temperature [$^\circ$C]'


ax = axs[1]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], 
               edgecolor=None, 
               alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=10),
              )
plt.colorbar(s, label=lab_z, ax=ax)

ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('log')
ax.set_xscale('log')

plt.tight_layout()
plt.show()


# %%
model_name = 'NorESM'
fig, axs = plt.subplots(1,2, figsize = [7,3], sharey=True)
axs = axs.flatten()

## Settings
alpha_scatt = 0.6

#xlab = r'T  [$^\circ$C]'
#ylab = r'Terpene emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'
linewidth=2
#xlims =[5,30]
#ylims = [4e-3,4e-1]
#xlims = [22,37]


season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()
df_mo = df_mo.loc['2015':None,:]
ax = axs[0]
v_x = 'emiterp'    
v_y = 'OA'
v_z = 'T_C'

lab_x = v_x
lab_y = v_y
lab_z = v_z


ax = axs[0]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=10),
               #norm=colors.LogNorm(1,200, )#,extend='both'
              
              )
plt.colorbar(s, label=lab_z, ax=ax, )
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

ax.set_yscale('log')


v_z = 'FSDS_DRF'
lab_x = v_x
lab_y = v_y
lab_z = v_z


ax = axs[1]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], 
               edgecolor=None, 
               alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=10),
              )
plt.colorbar(s, label=lab_z, ax=ax)

ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('log')
ax.set_xscale('log')

plt.tight_layout()
plt.show()


# %%
model_name = 'NorESM'
fig, axs = plt.subplots(1,2, figsize = [7,3], sharey=True)
axs = axs.flatten()

## Settings
alpha_scatt = 0.6

#xlab = r'T  [$^\circ$C]'
#ylab = r'Terpene emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'
linewidth=2
#xlims =[5,30]
#ylims = [4e-3,4e-1]
#xlims = [22,37]


season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()
ax = axs[0]
v_x = 'emiterp'    
v_y = 'OA'
v_z = 'month'

lab_x = v_x
lab_y = v_y
lab_z = v_z


ax = axs[0]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('cividis', lut=10),
               #norm=colors.LogNorm(1,200, )#,extend='both'
              
              )
plt.colorbar(s, label=lab_z, ax=ax, )
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

ax.set_yscale('log')


v_z = 'year'
lab_x = v_x
lab_y = v_y
lab_z = v_z


ax = axs[1]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], 
               edgecolor=None, 
               alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('cividis', lut=10),
              )
plt.colorbar(s, label=lab_z, ax=ax)

ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')

plt.tight_layout()
plt.show()


# %%
model_name = 'NorESM'
fig, axs = plt.subplots(1,2, figsize = [7,3], sharey=True)
axs = axs.flatten()

## Settings
alpha_scatt = 0.6

#xlab = r'T  [$^\circ$C]'
#ylab = r'Terpene emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'
linewidth=2
#xlims =[5,30]
#ylims = [4e-3,4e-1]
#xlims = [22,37]


season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()
ax = axs[0]
v_x = 'emiisop'    
v_y = 'OA'
v_z = 'month'

lab_x = v_x
lab_y = v_y
lab_z = v_z


ax = axs[0]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('cividis', lut=10),
               #norm=colors.LogNorm(1,200, )#,extend='both'
              
              )
plt.colorbar(s, label=lab_z, ax=ax, )
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

ax.set_yscale('log')


v_z = 'year'
lab_x = v_x
lab_y = v_y
lab_z = v_z


ax = axs[1]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], 
               edgecolor=None, 
               alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('cividis', lut=10),
              )
plt.colorbar(s, label=lab_z, ax=ax)

ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('linear')
ax.set_xscale('linear')

plt.tight_layout()
plt.show()


# %%
model_name = 'NorESM'
fig, axs = plt.subplots(1,2, figsize = [7,3], sharey=True)
axs = axs.flatten()

## Settings
alpha_scatt = 0.6

#xlab = r'T  [$^\circ$C]'
#ylab = r'Terpene emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'
linewidth=2
#xlims =[5,30]
#ylims = [4e-3,4e-1]
#xlims = [22,37]


season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()
ax = axs[0]
v_x = 'emiisop'    
v_y = 'OA'
v_z = 'T_C'

lab_x = v_x
lab_y = v_y
lab_z = v_z


ax = axs[0]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=10),
               #norm=colors.LogNorm(1,200, )#,extend='both'
              
              )
plt.colorbar(s, label=lab_z, ax=ax, )
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

ax.set_yscale('log')
ax.set_yscale('log')
ax.set_xscale('log')

v_z = 'FSDS_DRF'
lab_x = v_x
lab_y = v_y
lab_z = v_z


ax = axs[1]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], 
               edgecolor=None, 
               alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=10),
              )
plt.colorbar(s, label=lab_z, ax=ax)

ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('log')
ax.set_xscale('log')

plt.tight_layout()
plt.show()


# %%
model_name = 'NorESM'
fig, axs = plt.subplots(1,2, figsize = [7,3], sharey=True)
axs = axs.flatten()

## Settings
alpha_scatt = 0.6

#xlab = r'T  [$^\circ$C]'
#ylab = r'Terpene emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'
linewidth=2
#xlims =[5,30]
#ylims = [4e-3,4e-1]
#xlims = [22,37]


season='FMA'

    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()
ax = axs[0]
v_x = 'emiisop'    
v_y = 'OA'
v_z = 'FSDS_DRF'

lab_x = 'Temperature [$^\circ$C]'
lab_y = 'Em isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
lab_z = 'SW radiation surf [Wm$^{-2}$]'


ax = axs[0]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=10),
               norm=colors.LogNorm(1,200, )#,extend='both'
              
              )
plt.colorbar(s, label=lab_z, ax=ax, )
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

ax.set_yscale('log')




v_x = 'emiisop'    
v_y = 'OA'
v_z = 'T_C'    

lab_x = 'SW radiation surf [Wm$^{-2}$]'
lab_y = 'Em isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
lab_z = 'Temperature [$^\circ$C]'


ax = axs[1]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], 
               edgecolor=None, 
               alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=10),
              )
plt.colorbar(s, label=lab_z, ax=ax)

ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('log')
ax.set_xscale('log')

plt.tight_layout()
plt.show()


# %%

# %%
model_name = 'NorESM'
fig, axs = plt.subplots(1,3, figsize = [9,3], sharey=True)
axs = axs.flatten()

## Settings
alpha_scatt = 0.6

#xlab = r'T  [$^\circ$C]'
#ylab = r'Terpene emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'
linewidth=2
#xlims =[5,30]
#ylims = [4e-3,4e-1]
#xlims = [22,37]


season='FMA'
v_x = 'T_C'
v_y = 'emiterp'
    

df_mo = dic_df_med[model_name]
mask_months = select_months(df_mo, season=season)
df_mo = df_mo[mask_months].copy()
ax = axs[0]
v_x = 'T_C'    
v_y = 'emiisop'
v_z = 'FSDS_DRF'

lab_x = 'Temperature [$^\circ$C]'
lab_y = 'Em isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
lab_z = 'SW radiation surf [Wm$^{-2}$]'


ax = axs[0]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=10),
               norm=colors.LogNorm(1,200, )#,extend='both'
              
              )
plt.colorbar(s, label=lab_z, ax=ax)
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

ax.set_yscale('log')


ax = axs[1]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], edgecolor=None, alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=10),
               norm=colors.Normalize(10,200, )#,extend='both'
              
              )
plt.colorbar(s, label=lab_z, ax=ax)
ax.set_xlabel(lab_x)
ax.set_ylabel(lab_y)

ax.set_yscale('log')



v_x = 'FSDS_DRF'
v_y = 'emiterp'
v_z = 'T_C'    

lab_x = 'SW radiation surf [Wm$^{-2}$]'
lab_y = 'Em isoprene [$\mu$gm$^{-2}$s$^{-1}$]'
lab_z = 'Temperature [$^\circ$C]'


ax = axs[2]
print(v_x,v_y)
s = ax.scatter(df_mo[v_x],df_mo[v_y], 
               edgecolor=None, 
               alpha=.8, 
               c=df_mo[v_z], 
               cmap=plt.cm.get_cmap('Reds', lut=10),
              )
plt.colorbar(s, label=lab_z, ax=ax)

ax.set_xlabel(lab_x)
#ax.set_ylabel(lab_y)

ax.set_yscale('log')
#ax.set_xscale('log')

plt.tight_layout()
plt.show()


# %%

# %%
plt.cm.get_cmap('Reds', lut=10)

# %%
sns.scatterplot(v_x,v_y,data = df_mo, ax= ax, edgecolor=None, alpha=.3)


# %%
model_name

# %%
cn = list(dic_df_mod_case[model_name].keys())[0]
df_mo_full = dic_df_mod_case[model_name][cn]

# %% tags=[]
df_mo['FSDS_DRF'].plot()

# %%
df_mo_full['FSDS_DRF'].resample('D').median().plot()

# %%
df_mo_full['FSDS_DRF'].resample('d').median().plot()

# %%
model_name = 'NorESM'
fig, axs = plt.subplots(1, figsize = [8,4])

## Settings
alpha_scatt = 0.6

#xlab = r'T  [$^\circ$C]'
#ylab = r'Terpene emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'
linewidth=2
#xlims =[5,30]
#ylims = [4e-3,4e-1]
#xlims = [22,37]


season='FMA'
v_x = 'T_C'
v_y = 'emiisop'
    

df_mo = dic_df_med[model_name]
#mask_months = select_months(df_mo, season=season)
#df_mo = df_mo[mask_months].copy()


v_y = 'FSDS_DRF'
#v_y = 'emiisop'
    

ax = axs
#sns.lineplot(y=v_y,data = df_mo, ax= ax, alpha=.3, marker='.')
df_mo[v_y].plot(ax=ax, linewidth=0, marker='.', alpha=.5)
ax.set_yscale('log')

plt.show()

# %%
model_name = 'NorESM'
fig, axs = plt.subplots(1, figsize = [8,4])

## Settings
alpha_scatt = 0.6

#xlab = r'T  [$^\circ$C]'
#ylab = r'Terpene emissions [$\mu$gm$^{-2}$s$^{-1}$]'#[$\mu g m^{-3}$]'
linewidth=2
#xlims =[5,30]
#ylims = [4e-3,4e-1]
#xlims = [22,37]


season='FMA'
v_x = 'T_C'
v_y = 'emiisop'
    

df_mo = dic_df_med[model_name]
#mask_months = select_months(df_mo, season=season)
#df_mo = df_mo[mask_months].copy()


v_x = 'FSDS_DRF'
v_y = 'emiisop'
    

ax = axs
#sns.lineplot(y=v_y,data = df_mo, ax= ax, alpha=.3, marker='.')
df_mo[v_y].plot(ax=ax, linewidth=0, marker='.', alpha=.5)
ax.set_yscale('log')

plt.show()

# %%
df_mo

# %% [markdown]
# ### OA flux'ish:
#

# %%

# %% [markdown]
# ### Residuals

# %%

## Settings
alpha_scatt = 0.5

figsize=[7,10]
xlab = r'T  [$^\circ$C]'
ylab = r'$\Delta$OA [$\mu m^{-3}$]'

season = 'FMA'
xlims = [5,30]

#ylims = [1,700]

# OBS: 
v_y = 'OA'
v_x = 'T_C'


xscale='linear'
yscale='linear'

fig, axs = plt.subplots(len(models_and_obs), sharex=True, sharey= True, figsize=figsize)

## Settings
alpha_scatt = 0.6


for mo, ax in zip(models_and_obs, axs):
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    
    #popt, pov, label, func, out = get_odr_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True, return_out_obj=True)
    #popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    popt, pov, label, func = get_odr_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])

    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    ax.scatter(df_s[v_x],df_s[v_y]-func(df_s[v_x],*popt),
                                    color=cdic_model[mo], 
                #alpha=alpha_scatt, 
                #facecolor='none',
               alpha=alpha_scatt,
               
                edgecolor=cdic_model[mo],

                label=label
                   )
    _xlim = [_mi*.95, _ma*1.05]
    ax_ex.set_yscale(yscale)
    ax_ex.set_xscale(xscale)


    ax.hlines(0, xmin=xlims[0],xmax=xlims[1], color='k', linewidth=1)
    ax.legend(frameon=False)
    ax.set_ylabel(ylab)
    ax.set_title(mo, y=.93)
    ax.set_xlim(xlims)

        
#fig.suptitle('Observations')
axs[-1].set_xlabel(xlab)
fig.suptitle(r'Residuals fits')

sns.despine(fig)    
    
fn = make_fn_scat(f'residual_ln_{season}', v_x, v_y)
ax.legend(frameon=False)
fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)
print(fn)

# %% [markdown]
# # Nx new version

    # %%
    df_s =  dic_df_med[mo]

# %%
_df = dic_df_med['ECHAM-SALSA']
_df['emiisop']

# %%


# %%
dic_df_med[mod]

# %%
dic_df_med[mod]

# %%

df_emiterp_all = pd.DataFrame()

df_emiisop_all = pd.DataFrame()
for mod in dic_df_med.keys():
    if mod=='Obs' or mod=='Observations':
        continue
    print(mod)
    df_emiterp_all[mod] = dic_df_med[mod]['emiterp']
    df_emiisop_all[mod] = dic_df_med[mod]['emiisop']





# %%
df_emiterp_all

# %%
seasons2months = {'DJF':[12,1,2],
        'MAM': [3,4,5],
        'JJA':[6,7,8],
        'SON':[9,10,11],
       }
seasons2months = {'DJF':[12,1,2],
        'MAM': [3,4,5],
        'JJA':[6,7,8],
        'SON':[9,10,11],
       }

# %%
seasons2months2 = {'JFM': [ 1, 2,3], 'AMJ': [ 4, 5,6], 'JAS': [ 7, 8,9], 'OND': [ 10, 11,12]}

# %%
fig, axs = plt.subplots(4,figsize = [4,8,],sharex=True, sharey=True)
_df_use = df_emiterp_all


mi = np.min(np.min(_df_use[_df_use>0]))
ma = np.max(np.max(_df_use[_df_use>0])*.999)
bins_ = 10 ** np.linspace(np.log10(mi), np.log10(ma), 50)
for seas,ax in zip(seasons2months2, axs.flatten()):
    df_OA_all_sub = _df_use.copy()
    
    if 'UKESM' in _df_use.columns:
        mo = 'UKESM'
        df_OA_all_sub[mo] =df_OA_all_sub.loc[:,mo].ffill(limit=3).copy()
        
    df_OA_all_sub = df_OA_all_sub[df_OA_all_sub.index.month.isin(seasons2months2[seas])].copy()


    for mo in models:
        print(mo)
        df_OA_all_sub[mo].plot.hist(bins=bins_, alpha=1, 
                                     color=cdic_model[mo],
                                     label=mo,
                                    ax = ax,
                                    histtype='step'
                                    )
        df_OA_all_sub[mo].plot.hist(bins=bins_, alpha=0.2, 
                                     color=cdic_model[mo],
                                     label='__nolabel__',
                                    ax = ax
                                    )

        ax.set_xscale('log')
    #_mod_an.plot.hist(bins=bins_, alpha=0.5,label='OsloAero, SOA')

    ax.set_xlabel('Em. monoterpene [$\mu$gm$^{-2}$s$^{-1}$]')

    ax.set_title(f'{seas}', y=.95)

ax.legend(frameon=False)
plt.suptitle('Distribution at ATTO')
fn = make_fn_scat('noresm_echam_seasons2_emiterp', 'emiterp','hist')
fig.tight_layout()
sns.despine(fig)
print(fn)
#plt#.savefig(fn, dpi=300)

ax.set_xlim([1e-2,1e0])
plt.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
fig, axs = plt.subplots(4,figsize = [4,8,],sharex=True, sharey=True)
_df_use = df_emiterp_all


mi = np.min(np.min(_df_use[_df_use>0]))
ma = np.max(np.max(_df_use[_df_use>0])*.999)
bins_ = 10 ** np.linspace(np.log10(mi), np.log10(ma), 50)
for seas,ax in zip(seasons2months, axs.flatten()):
    df_OA_all_sub = _df_use.copy()
    
    if 'UKESM' in _df_use.columns:
        mo = 'UKESM'
        df_OA_all_sub[mo] =df_OA_all_sub.loc[:,mo].ffill(limit=3).copy()
        
    df_OA_all_sub = df_OA_all_sub[df_OA_all_sub.index.month.isin(seasons2months[seas])].copy()


    for mo in models:
        print(mo)
        df_OA_all_sub[mo].plot.hist(bins=bins_, alpha=1, 
                                     color=cdic_model[mo],
                                     label=mo,
                                    ax = ax,
                                    histtype='step'
                                    )
        df_OA_all_sub[mo].plot.hist(bins=bins_, alpha=0.2, 
                                     color=cdic_model[mo],
                                     label='__nolabel__',
                                    ax = ax
                                    )

        ax.set_xscale('log')
    #_mod_an.plot.hist(bins=bins_, alpha=0.5,label='OsloAero, SOA')

    ax.set_xlabel('Em. monoterpene [$\mu$gm$^{-2}$s$^{-1}$]')

    ax.set_title(f'{seas}', y=.95)

ax.legend(frameon=False)
plt.suptitle('Distribution at ATTO')
fn = make_fn_scat('noresm_echam_seasons2_emiterp', 'emiterp','hist')
fig.tight_layout()
sns.despine(fig)
print(fn)
#plt#.savefig(fn, dpi=300)

ax.set_xlim([1e-2,5e-1])
plt.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
fig, axs = plt.subplots(4,figsize = [4,8,],sharex=True, sharey=True)
_df_use = df_emiisop_all


mi = np.min(np.min(_df_use[_df_use>0]))*10
ma = np.max(np.max(_df_use[_df_use>0]))
bins_ = 10 ** np.linspace(np.log10(mi), np.log10(ma), 200)
for seas,ax in zip(seasons2months, axs.flatten()):
    df_OA_all_sub = _df_use.copy()
    
    if 'UKESM' in _df_use.columns:
        mo = 'UKESM'
        df_OA_all_sub[mo] =df_OA_all_sub.loc[:,mo].ffill(limit=3).copy()
        
    df_OA_all_sub = df_OA_all_sub[df_OA_all_sub.index.month.isin(seasons2months[seas])].copy()


    for mo in models:
        print(mo)
        df_OA_all_sub[mo].plot.hist(bins=bins_, alpha=1, 
                                     color=cdic_model[mo],
                                     label=mo,
                                    ax = ax,
                                    histtype='step'
                                    )
        df_OA_all_sub[mo].plot.hist(bins=bins_, alpha=0.2, 
                                     color=cdic_model[mo],
                                     label='__nolabel__',
                                    ax = ax
                                    )

        ax.set_xscale('log')
    #_mod_an.plot.hist(bins=bins_, alpha=0.5,label='OsloAero, SOA')

    ax.set_xlabel('Em. monoterpene [$\mu$gm$^{-2}$s$^{-1}$]')

    ax.set_title(f'{seas}', y=.95)

ax.legend(frameon=False)
plt.suptitle('Distribution at ATTO')
fn = make_fn_scat('noresm_echam_seasons2_emiterp', 'emiterp','hist')
fig.tight_layout()
sns.despine(fig)
print(fn)
#plt#.savefig(fn, dpi=300)

ax.set_xlim([1e-4,1e0])

plt.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# # Compare seasons

# %%
