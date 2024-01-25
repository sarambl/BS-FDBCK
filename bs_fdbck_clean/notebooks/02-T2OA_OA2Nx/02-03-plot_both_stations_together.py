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
import matplotlib.pyplot as plt

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
# %load_ext autoreload

# %autoreload 2

from pathlib import Path
from bs_fdbck.util.BSOA_datamanip import ds2df_inc_preprocessing, ds2df_echam
from bs_fdbck.util.collocate.collocateLONLAToutput import CollocateLONLATout
from bs_fdbck.util.collocate.collocate_echam_salsa import CollocateModelEcham
import useful_scit.util.log as log
from bs_fdbck.util.plot.BSOA_plots import make_cool_grid, plot_scatter
log.ger.setLevel(log.log.INFO)
import time
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import scienceplots
import scienceplots
plt.style.use([
    'default',
    # 'science',
    'acp',
    # 'sp-grid',
    'no-black',
    'no-latex',
    'illustrator-safe'
])


# %%
select_station = 'SMR'

# %%
plot_path = Path(f'Plots/Both_stations')


# %%
def make_fn_scat(case, v_x, v_y):
    _x = v_x.split('(')[0]
    _y = v_y.split('(')[0]
    f = f'scat_all_years_2stations_{case}_{_x}_{_y}.png'
    return plot_path /f


# %% [raw]
# fig = plt.figure()

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
models_and_obs =  models + ['Observations'] 
models_and_obs

# %%
dic_season_nicename = {
    'JFM':'Jan-Mar',
    'FMA':'Feb-Apr',
    'FMAM':'Feb-May',
    'JFMAM':'Jan-May',
    'MAM':'Mar-May',
    'JA':'July-Aug',
}


# %% [markdown] tags=[]
# ## Read in data

# %% [markdown]
# ### Data is produced in [02-01-relations_plots_TOANx_SMR.ipynb](02-01-relations_plots_TOANx_SMR.ipynb) and [02-02-relations_plots_TOANx_ATTO.ipynb](02-02-02-relations_plots_TOANx_ATTO.ipynb)

# %%
path_save_daily_medians_SMR = Path(f'Temp_data/SMR_daily_medians')
path_save_daily_medians_ATTO = Path(f'Temp_data/ATTO_daily_medians')

dic_df_med_SMR = dict()
dic_df_med_ATTO = dict()

for mo in di_mod2cases.keys():
    for ca in di_mod2cases[mo]:
        if len(di_mod2cases[mo])>1:
            use_name = f'{mo}_{ca}'
        else:
            use_name = mo
            
        
        fp_smr = path_save_daily_medians_SMR.parent / f'{path_save_daily_medians_SMR.name}_{use_name}.csv'
        fp_atto = path_save_daily_medians_ATTO.parent / f'{path_save_daily_medians_ATTO.name}_{use_name}.csv'
        
        dic_df_med_SMR[use_name] = pd.read_csv(fp_smr, index_col=0,)
        dic_df_med_SMR[use_name].index = pd.to_datetime(dic_df_med_SMR[use_name].index)
        dic_df_med_SMR[use_name]['month'] = dic_df_med_SMR[use_name].index.month
        dic_df_med_ATTO[use_name] = pd.read_csv(fp_atto, index_col=0)
        dic_df_med_ATTO[use_name].index = pd.to_datetime(dic_df_med_ATTO[use_name].index)
        dic_df_med_ATTO[use_name]['month'] = dic_df_med_ATTO[use_name].index.month
        
        


# %%
def make_plot2(v_x, v_y, xlims, ylims, season, 
              xlab=None, ylab=None, alpha_scat=.4,
             source_list = models_and_obs, fig=None, 
               axs=None,
              xscale='linear', yscale='linear',
               select_station= '',
              dic_df_med = None,
               divide_NorESM_by_factor=None,
               divide_UKESM_by_factor=None,
               markersize = 1,
             ):
    if xlab is None: 
        if xlab in label_dic:
            xlab = label_dic[v_x]
    if ylab is None: 
        if ylab in label_dic:
            ylab = label_dic[v_y]

    for mo, ax in zip(source_list, axs[:]):
        print(mo)
        df_s =  dic_df_med[mo]
    
        mask_months = select_months(df_s, season=season)
        df_s = df_s[mask_months].copy()
        
        if (mo =='NorESM') &  (divide_NorESM_by_factor is not None):
            df_s = df_s/divide_NorESM_by_factor
            title = f'{mo}/{divide_NorESM_by_factor}'
            print(title)
            ax.spines['bottom'].set_color('r')
            ax.spines['top'].set_color('r') 
            ax.spines['right'].set_color('red')
            ax.spines['left'].set_color('red')
            ax.set_title(title, c='r')
        elif (mo =='UKESM') &  (divide_UKESM_by_factor is not None):
            df_s = df_s/divide_UKESM_by_factor
            title = f'{mo}/{divide_UKESM_by_factor}'
            print(title)
            ax.spines['bottom'].set_color('m')
            ax.spines['top'].set_color('m') 
            ax.spines['right'].set_color('m')
            ax.spines['left'].set_color('m')
            ax.set_title(title, c='m')

        else:
            title = mo
            ax.set_title(title,)# y=.95)
            
        sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax,
                    #facecolor='none',
                        edgecolor=cdic_model[mo],
                        marker='.',
                        s=markersize,
                        #rasterized = True,
                    
                   )
        #ax.set_title(mo, y=.95)
        
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    fig.suptitle(f'{select_station}, {season} season, 2012-2018', y=.95)
    #xlim_dist = list(daxs['y'].get_xlim())
    for mo,ax in zip(source_list, axs):

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
            

        ax.set_ylim(ylims)
        ax.set_xlim(xlims)

        sns.despine(ax = ax)

    return

#### WET_mid

# %% [markdown]
# ### Fit funcs

# %%
from bs_fdbck.util.BSOA_datamanip.fits import *
from bs_fdbck.util.BSOA_datamanip.atto import season2month

# %% [markdown] tags=[]
# ### season to monthseason2month

# %%
from sklearn.metrics import r2_score


# %%
def select_months(df, season = None, month_list=None):
    if season is not None: 
        month_list = season2month[season]
    

    df['month'] = df.index.month
    return df['month'].isin(month_list)

# %%
from bs_fdbck.util.BSOA_datamanip.fits import *

# %%
from bs_fdbck.util.plot.BSOA_plots import cdic_model, make_cool_grid5

# %%

# %%
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# %%
def calc_table_se(di):
    di['a'] = di['popt'][0]
    di['b'] = di['popt'][1]
    if len(di['popt'])>2:
        di['c'] = di['popt'][2]
    di['SE_a'] = di['standard_error'][0]
    di['SE_b'] = di['standard_error'][1]
    if len(di['popt'])>2:
        di['SE_c'] = di['standard_error'][2]
    else:
        di['SE_c'] = np.nan
        
    for v in ['a','b','c']:
        if len(di['popt'])<=2 and v=='c':
            di[f'{v}_str'] = ''
            continue
        v_SE = f'SE_{v}'
        if np.abs(di[v])>10 and di[v_SE]>1:
            di[f'{v}_str'] = '%.1f $\pm$ %.1f' %(di[v], di[v_SE])
        elif np.abs(di[v])>.1 and di[v_SE]>.01:
            di[f'{v}_str'] = '%.2f $\pm$ %.2f' %(di[v], di[v_SE])
        else:
            di[f'{v}_str'] = '%.2E $\pm$ %.2E' %(di[v], di[v_SE])
            
            

    return di


# %%
def lin_lab_fix(popt,divided_by_factor=None):
    label = '$%5.2fx+ %5.2f$' % tuple(popt)
    if divided_by_factor is not None:
        label = '$%5.2fx+ %5.2f$' % (popt[0],popt[1]*divided_by_factor)
        return label,[popt[0], popt[1]*divided_by_factor] 
    return label,popt
    


# %%
def func_atto(df_s,v_x,v_y,):
    if v_y=='N50-500':
        return get_odr_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)
        #return get_least_square_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)

    elif v_y =='N100-500':
        return get_odr_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)
        #return get_least_square_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)

    elif v_y =='N200-500':
        return get_odr_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)
        #return get_least_square_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)
    
    else:
        print(f'did not recognice v_y:{v_y}')



# %%
def func_smr(df_s,v_x,v_y,):
    if v_y=='N50':
        return get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    elif v_y =='N100':
        try:
            return get_log_fit_abc(df_s,v_x,v_y, return_func=True)
        except RuntimeError:
            return get_odr_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)
            #return get_least_square_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)
            
    elif v_y =='N200':
        return get_odr_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)
        #return get_least_square_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)
    
    else:
        print(f'did not recognice v_y:{v_y}')
def func_smr_lin(df_s,v_x,v_y,):
    if v_y=='N50':
        return get_odr_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)
        #return get_least_square_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)

    elif v_y =='N100':
        return get_odr_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)
        #return get_least_square_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)

    elif v_y =='N200':
        return get_odr_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)
        #return get_least_square_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)
    
    else:
        print(f'did not recognice v_y:{v_y}')



# %%
def get_r2(df_s,v_x,x_y, popt, func):
    
    _df = df_s[[v_x, v_y]].dropna()
    y_pred = func(_df[v_x].values, *popt)
    r2 =  r2_score(_df[v_y].values, y_pred)
    return r2



def get_corr(df_s,v_x,x_y):
    
    _df = df_s[[v_x, v_y]].dropna()
    return _df.corr().loc[v_x,v_y]



# %% tags=[]
def make_pd_of_dic(dic_fits):
    columns = ['station','variable','data source','Fit','a','b','c','R2', 'corr']
    df_test = pd.DataFrame(columns=columns )
    df_list = []
    dic_labels = dict()
    for st in dic_fits.keys():
        dic_labels[st] = dict()
        for v_y in dic_fits[st].keys():
            dic_labels[st][v_y] = dict()
            for mo in dic_fits[st][v_y].keys():
                dic_labels[st][v_y][mo] = dict()
                dic_labels[st][v_y][mo] = dic_fits[st][v_y][mo]['label'] 
                _d = dict()
                _d['station'] = st
                _d['variable'] =v_y
                _d['data source'] = mo
                _d['Fit'] = dic_fits[st][v_y][mo]['label']
                _d['a'] = dic_fits[st][v_y][mo]['a_str']
                _d['b'] = dic_fits[st][v_y][mo]['b_str']
                _d['c'] = dic_fits[st][v_y][mo]['c_str']
                _d['R$^2$'] = np.round(dic_fits[st][v_y][mo]['R2'] , decimals=2)
                _d['r$^2$'] = np.round(dic_fits[st][v_y][mo]['corr'] , decimals=2)
                #print(_d)
                d2 = {st:_d}
                df_list.append(pd.DataFrame(d2).T)
                

    df = pd.DataFrame(dic_labels['SMR']).T
    df['station'] = 'SMR'
    df2 = pd.DataFrame(dic_labels['ATTO']).T
    df2['station'] = 'ATTO'
    df2 = df2.reset_index().set_index(['station','index'])
    df1 = df.reset_index().set_index(['station','index'])
    df_out = pd.concat([df1,df2],)
    df_out = pd.concat(df_list, axis=0).reset_index()    
    df_out = df_out.drop('index', axis=1).set_index(['station','variable','data source'])
    return df_out

get_corr

# %%
plot_path.mkdir(exist_ok=True)

# %%
left, width = .1, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

# %%

# %%

# %% [markdown]
# ## Writing out data to source data file for Paper Fig 2 and 3

# %%
rename_dictionary_for_source_data = {
    'T_C':'T [deg C]',
    'OA':'Organic aerosol mass [ugm-3]',
    'N50-500': 'N50 [#cm-3]',
    'N100-500': 'N100 [#cm-3]',
    'N200-500': 'N200 [#cm-3]',
    'N50': 'N50 [#cm-3]',
    'N100': 'N100 [#cm-3]',
    'N200': 'N200 [#cm-3]',
}

dic_list_of_vars_for_source_data = {'SMR': ['station','data_source','T_C','OA','N50','N100','N200',],
                    'ATTO':['station','data_source','T_C','OA','N50-500','N100-500','N200-500'],
                   }


# %%

list_of_datasets_for_source_data = []

season_atto = 'FMA'
season_smr = 'JA'


## SMEAR:
dic_df_med = dic_df_med_SMR
select_station = 'SMR'
season=season_smr

for mo in models_and_obs:
    print(f'******{mo}*********')
    df_s =  dic_df_med[mo]
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    df_s_out = df_s.copy()
    df_s_out['station'] = select_station
    df_s_out['data_source'] = mo
    df_s_out =df_s_out[dic_list_of_vars_for_source_data[select_station]]
    df_s_out = df_s_out.rename(rename_dictionary_for_source_data,axis=1)
    list_of_datasets_for_source_data.append(df_s_out.copy())
    

## ATTO:
dic_df_med = dic_df_med_ATTO
select_station = 'ATTO'
season = season_atto

for mo in models_and_obs:
    print(f'******{mo}********')
    df_s =  dic_df_med[mo]
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    df_s_out = df_s.copy()
    df_s_out['station'] = select_station
    df_s_out['data_source'] = mo
    df_s_out =df_s_out[dic_list_of_vars_for_source_data[select_station]]
    df_s_out = df_s_out.rename(rename_dictionary_for_source_data,axis=1)
    list_of_datasets_for_source_data.append(df_s_out.copy())
    
    

    
    
fn = make_fn_scat(f'together_{season_smr}_{season_atto}', 'Fig2', 'Fig3')
save_data_df = pd.concat(list_of_datasets_for_source_data,axis=0 )
save_data_df.to_csv(fn.with_suffix('.csv'))
print(fn.with_suffix('.csv'))

# %% [markdown]
# ## Paper figure 3:

# %%
fx = .92
fig_main = plt.figure(#constrained_layout=True,
                  figsize=[15*fx,5.5*fx],
                      
                 )


spec2 = gridspec.GridSpec( nrows=2,ncols=3, 
                          width_ratios=[4,.01,4],
                          height_ratios=[1,20], 
                          figure=fig_main,
                         )


markersize = 4

subfig1 =  fig_main.add_subfigure(spec2[1, 0],frameon=True )
subfig2 =  fig_main.add_subfigure(spec2[1, 1])
subfig3 =  fig_main.add_subfigure(spec2[1, 2],frameon=True,)


axs_smr = subfig1.subplots(3,5, sharex='col', sharey='row')
#ax_fits = subfig2.subplots(3,1, sharex='col', sharey='row')
axs_atto =subfig3.subplots(3,5, sharex='col', sharey='row')
#subfig2.set_facecolor('#e9f2f9')##e5f8f8')
# subfig3.set_facecolor('#fff4ea')

dic_fits = {}
dic_fits['SMR'] =dict()
dic_fits['ATTO'] =dict()

#subfig1.suptitle('SMEARII, Jul & Aug')
#subfig3.suptitle('ATTO, JFM')
#subfig2.suptitle('Fits')
subfig_up =  fig_main.add_subfigure(spec2[0, 0])
subfig_up2 =  fig_main.add_subfigure(spec2[0, -1])

ax_dum = subfig_up.subplots(1)
ax_dum.axis('off')
ax_dum2 = subfig_up2.subplots(1)
ax_dum2.axis('off')


divide_NorESM_by_factor = 8 
divide_UKESM_by_factor = None


varlistplot = ['N50','N100','N200']
xlab = r'OA [$\mu$gm$^{-3}$]'
alpha_scatt = 0.4
source_list = models_and_obs[::-1]
v_x = 'OA'

## Settings
dic_lims = {
    'N50': {'xlims':[.01,12], 'ylims':[1,4000]},
    'N100': {'xlims':[.01,12], 'ylims':[1,2500]},
    'N200': {'xlims':[.01,12], 'ylims':[1,1500]},
    'N50-500': {'xlims':[.01,7], 'ylims':[1,1200]},
    'N100-500': {'xlims':[.01,7], 'ylims':[1,900]},
    'N200-500': {'xlims':[.01,7], 'ylims':[1,500]},

}

dic_ylabels = {
    'N50' : r'N$_{50}$  [cm$^{-3}$]',
    'N100' : r'N$_{100}$  [cm$^{-3}$]',
    'N200' : r'N$_{200}$  [cm$^{-3}$]',
    'N50-500' : r'N$_{50}$  [cm$^{-3}$]',
    'N100-500' : r'N$_{100}$  [cm$^{-3}$]',
    'N200-500' : r'N$_{200}$  [cm$^{-3}$]',

}




select_station = 'SMR'
season = 'JA'


xscale='linear'
yscale='linear'


# OBS: 

dic_df_med = dic_df_med_SMR
axs_all = axs_smr
fig = subfig1


#fig, axs_all = plt.subplots(3,6,figsize=figsize, sharey='row', sharex='col')
## Settings
legends_smr = []
legends_atto = []
legs =[]

for i,v_y in enumerate(varlistplot):
    dic_fits[select_station][v_y] = dict()
    # Make plot
    ylab = dic_ylabels[v_y]
    ylims = dic_lims[v_y]['ylims']
    xlims = dic_lims[v_y]['xlims']
    
    axs_sub = axs_all[i,:]
    axs_sub[0].set_ylabel(ylab)
    make_plot2(v_x, v_y, xlims, ylims, season, 
              xlab=xlab, ylab=ylab, alpha_scat=alpha_scatt,
             source_list = source_list, fig=fig, 
               axs=axs_sub,
              xscale='linear', yscale='linear',
              dic_df_med = dic_df_med,
           select_station= select_station,
               markersize=markersize,
         )

    
    for mo, ax in zip(source_list, axs_sub):
        dic_fits[select_station][v_y][mo] = dict()

        df_s =  dic_df_med[mo]
        
        print(mo)
        mask_months = select_months(df_s, season=season)
        df_s = df_s[mask_months].copy()
        popt, pov, label, func = func_smr(df_s,v_x,v_y)
            
            
        legends_smr.append(label)

        plot_fit(func, popt, mo, xlims, yscale, xscale, ax, label,)
        #plot_fit(func, popt, mo, xlims, yscale, xscale,  ax_fits[i],label,extra_plot=True)

        dic_fits[select_station][v_y][mo]['label'] = label
        dic_fits[select_station][v_y][mo]['popt'] = popt
        dic_fits[select_station][v_y][mo]['pcov'] = pov
        dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))
        
        dic_fits[select_station][v_y][mo]['func'] = func
        dic_fits[select_station][v_y][mo]['R2'] = get_r2(df_s,v_x,v_y, popt, func)
        dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
        print( get_r2(df_s,v_x,v_y, popt, func))
        dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])   
        
        
        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
    #leg = axs_sub[-1].legend(bbox_to_anchor=(1,1,), frameon=False)

    #legs.append(leg)


    
for ax in axs_sub:
    ax.set_xlabel(xlab)
sns.despine(fig) 








varlistplot = ['N50-500','N100-500','N200-500']

select_station = 'ATTO'
season = 'FMA'

xscale='linear'
yscale='linear'


# OBS: 

dic_df_med = dic_df_med_ATTO
axs_all = axs_atto
fig = subfig3
select_station = 'ATTO'


#fig, axs_all = plt.subplots(3,6,figsize=figsize, sharey='row', sharex='col')
## Settings

legs =[]

for i,v_y in enumerate(varlistplot):
    dic_fits[select_station][v_y] = dict()
    
    # Make plot
    ylab = dic_ylabels[v_y]
    ylims = dic_lims[v_y]['ylims']
    xlims = dic_lims[v_y]['xlims']
    axs_sub = axs_all[i,:]
    axs_sub[0].set_ylabel(ylab)

    make_plot2(v_x, v_y, xlims, ylims, season, 
              xlab=xlab, ylab=ylab, alpha_scat=alpha_scatt,
             source_list = source_list, fig=fig, 
               axs=axs_sub,
              xscale='linear', yscale='linear',
              dic_df_med = dic_df_med,
           select_station= select_station,
               divide_NorESM_by_factor=divide_NorESM_by_factor,
               divide_UKESM_by_factor = divide_UKESM_by_factor,
               markersize=markersize,     
         )


    for mo, ax in zip(source_list, axs_sub):
        dic_fits[select_station][v_y][mo] = dict()
        
        df_s =  dic_df_med[mo]
        print(mo)
        mask_months = select_months(df_s, season=season)
        df_s = df_s[mask_months].copy()
        if (mo =='NorESM') &  (divide_NorESM_by_factor is not None):
            df_s = df_s/divide_NorESM_by_factor
            ax.set_facecolor('#fff6f6')
        if (mo =='UKESM') &  (divide_UKESM_by_factor is not None):
            df_s = df_s/divide_UKESM_by_factor
            ax.set_facecolor('#f0e3f2')
        popt, pov, label, func = func_atto(df_s,v_x,v_y)
        if (mo =='NorESM') &  (divide_NorESM_by_factor is not None):
            label,popt_out =  lin_lab_fix(popt, divide_NorESM_by_factor)
        elif (mo =='UKESM') &  (divide_UKESM_by_factor is not None):
             label,popt_out =  lin_lab_fix(popt, divide_UKESM_by_factor)
        else:
            popt_out = popt
        legends_atto.append(label)
        
        plot_fit(func, popt, mo, xlims, yscale, xscale, ax,label)#, linestyle='dashed')
        #plot_fit(func, popt, mo, xlims, yscale, xscale,  ax_fits[i],label,  extra_plot=True,linestyle='dashed',)
        ax.set_xlim(xlims)
        
        dic_fits[select_station][v_y][mo]['label'] = label
        dic_fits[select_station][v_y][mo]['popt'] = popt_out
        dic_fits[select_station][v_y][mo]['pcov'] = pov
        dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))
        
        #dic_fits[select_station][v_y][mo]['func'] = func
        dic_fits[select_station][v_y][mo]['R2'] = get_r2(df_s,v_x,v_y, popt, func)
        dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
        dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])        

        _a = label.split('x')[0][1:]
        _b = label.split('x')[1][:-1]
        if _b[0]=='+':
            _b=_b[1:]
        #ax.text(left, top, f'a=${_a}$ \nb=${_b}$',
        #horizontalalignment='left',
        #verticalalignment='top',
        #transform=ax.transAxes)

        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        #ax.legend(frameon=False, fontsize=10)
    
    #leg = axs_sub[-1].legend(bbox_to_anchor=(1,1,), frameon=False)

    #legs.append(leg)


    
for ax in axs_sub:
    ax.set_xlabel(xlab)
sns.despine(fig) 




#for i, v_y in enumerate(varlistplot):
#    xlims = dic_lims[v_y]['xlims']
#    ax = ax_fits[i]
    
    #ax.set_xlim(xlims)
#    ax.set_ylim(dic_lims[v_y]['ylims'])
#    ax.set_yticklabels([])
#    ax.set_facecolor('#e9f2f9')##e5f8f8')
#    sns.despine(ax)
#ax.set_xlabel(xlab)
sns.despine(subfig2) 

subfig1.suptitle('SMEARII: Jul-Aug', size=16, y=1.05, zorder=100000)
subfig_up.suptitle('SMEARII: Jul-Aug', size=16, y=1., zorder=100000)
#subfig2.suptitle('F', size=16, y=1.05, c='w')
subfig3.suptitle(f'ATTO: {dic_season_nicename[season]}', size=16, y=1.05,zorder=100000)
subfig_up2.suptitle(f'ATTO: {dic_season_nicename[season]}', size=16, y=1.0,zorder=100000)

#axs_all = list(ax_fits.flatten())+list(axs_smr.flatten())+ list(axs_atto.flatten())
axs_all = list(list(axs_smr.flatten())+ list(axs_atto.flatten()))
for ax in axs_all:
    ax.grid(color = 'grey', linestyle = ':', linewidth = 0.5)



#for ax in axs_atto[:,0]:
#    ax.set_yticklabels([])
#    ax.set_ylabel('')

for ax in axs_atto[1:,:].flatten():
    ax.set_title('')


for ax in axs_smr[1:,:].flatten():
    ax.set_title('')

#ax_fits[0].set_title('.', color='w')
fn = make_fn_scat(f'together_{season}', v_x, 'Nx')
print(fn)
#plt.tight_layout()
plt.savefig(fn.with_suffix('.pdf'),bbox_inches='tight', )
plt.savefig(fn.with_suffix('.png'),bbox_inches='tight', dpi=200)

df = make_pd_of_dic(dic_fits)
df_log = df.copy()
df.to_csv(fn.with_suffix('.csv'))

plt.show()

# %%
display(fn.with_suffix('.pdf'))

# %% [markdown]
# #### Both linear

# %%
fig_main = plt.figure(constrained_layout=True,
                      figsize=[15,6],
                      )
spec2 = gridspec.GridSpec( nrows=2,ncols=3,
                           width_ratios=[4,.1,4],
                           height_ratios=[1,20],
                           figure=fig_main)

markersize = 5

subfig1 =  fig_main.add_subfigure(spec2[1, 0],frameon=True)
subfig2 =  fig_main.add_subfigure(spec2[1, 1])
subfig3 =  fig_main.add_subfigure(spec2[1, 2],frameon=True)
subfig_up =  fig_main.add_subfigure(spec2[0, 0])
subfig_up2 =  fig_main.add_subfigure(spec2[0, -1])

ax_dum = subfig_up.subplots(1)
ax_dum.axis('off')
ax_dum2 = subfig_up2.subplots(1)
ax_dum2.axis('off')


axs_smr = subfig1.subplots(3,5, sharex='col', sharey='row')
#ax_fits = subfig2.subplots(3,1, sharex='col', sharey='row')
axs_atto =subfig3.subplots(3,5, sharex='col', sharey='row')
#subfig2.set_facecolor('#e9f2f9')##e5f8f8')
# subfig3.set_facecolor('#fff4ea')

dic_fits = {}
dic_fits['SMR'] =dict()
dic_fits['ATTO'] =dict()

#subfig1.suptitle('SMEARII, Jul & Aug')
#subfig3.suptitle('ATTO, JFM')
#subfig2.suptitle('Fits')



divide_NorESM_by_factor = 8
divide_UKESM_by_factor = None


varlistplot = ['N50','N100','N200']
xlab = r'OA [$\mu$gm$^{-3}$]'
alpha_scatt = 0.4
source_list = models_and_obs[::-1]
v_x = 'OA'

## Settings
dic_lims = {
    'N50': {'xlims':[.01,12], 'ylims':[1,4000]},
    'N100': {'xlims':[.01,12], 'ylims':[1,2500]},
    'N200': {'xlims':[.01,12], 'ylims':[1,1500]},
    'N50-500': {'xlims':[.01,7], 'ylims':[1,1200]},
    'N100-500': {'xlims':[.01,7], 'ylims':[1,900]},
    'N200-500': {'xlims':[.01,7], 'ylims':[1,500]},

}

dic_ylabels = {
    'N50' : r'N$_{50}$  [cm$^{-3}$]',
    'N100' : r'N$_{100}$  [cm$^{-3}$]',
    'N200' : r'N$_{200}$  [cm$^{-3}$]',
    'N50-500' : r'N$_{50}$  [cm$^{-3}$]',
    'N100-500' : r'N$_{100}$  [cm$^{-3}$]',
    'N200-500' : r'N$_{200}$  [cm$^{-3}$]',

}




select_station = 'SMR'
season = 'JA'


xscale='linear'
yscale='linear'


# OBS:

dic_df_med = dic_df_med_SMR
axs_all = axs_smr
fig = subfig1


#fig, axs_all = plt.subplots(3,6,figsize=figsize, sharey='row', sharex='col')
## Settings
legends_smr = []
legends_atto = []
legs =[]

for i,v_y in enumerate(varlistplot):
    dic_fits[select_station][v_y] = dict()
    # Make plot
    ylab = dic_ylabels[v_y]
    ylims = dic_lims[v_y]['ylims']
    xlims = dic_lims[v_y]['xlims']

    axs_sub = axs_all[i,:]
    axs_sub[0].set_ylabel(ylab)
    make_plot2(v_x, v_y, xlims, ylims, season,
               xlab=xlab, ylab=ylab, alpha_scat=alpha_scatt,
               source_list = source_list, fig=fig,
               axs=axs_sub,
               xscale='linear', yscale='linear',
               dic_df_med = dic_df_med,
               select_station= select_station,
               markersize=markersize,
               )


    for mo, ax in zip(source_list, axs_sub):
        dic_fits[select_station][v_y][mo] = dict()

        df_s =  dic_df_med[mo]

        print(mo)
        mask_months = select_months(df_s, season=season)
        df_s = df_s[mask_months].copy()
        popt, pov, label, func = func_smr_lin(df_s,v_x,v_y)


        legends_smr.append(label)

        plot_fit(func, popt, mo, xlims, yscale, xscale, ax, label,)
        #plot_fit(func, popt, mo, xlims, yscale, xscale,  ax_fits[i],label,extra_plot=True)

        dic_fits[select_station][v_y][mo]['label'] = label
        dic_fits[select_station][v_y][mo]['popt'] = popt
        dic_fits[select_station][v_y][mo]['pcov'] = pov
        dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))

        dic_fits[select_station][v_y][mo]['func'] = func
        dic_fits[select_station][v_y][mo]['R2'] = get_r2(df_s,v_x,v_y, popt, func)
        dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
        print( get_r2(df_s,v_x,v_y, popt, func))
        dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])

        _a = label.split('x')[0][1:]
        _b = label.split('x')[1][:-1]
        if _b[0]=='+':
            _b=_b[1:]
        ax.text(left, top, f'a=${_a}$ \nb=${_b}$',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)


        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
    #leg = axs_sub[-1].legend(bbox_to_anchor=(1,1,), frameon=False)

    #legs.append(leg)



for ax in axs_sub:
    ax.set_xlabel(xlab)
sns.despine(fig)








varlistplot = ['N50-500','N100-500','N200-500']

select_station = 'ATTO'
season = 'FMA'

xscale='linear'
yscale='linear'


# OBS:

dic_df_med = dic_df_med_ATTO
axs_all = axs_atto
fig = subfig3
select_station = 'ATTO'


#fig, axs_all = plt.subplots(3,6,figsize=figsize, sharey='row', sharex='col')
## Settings

legs =[]

for i,v_y in enumerate(varlistplot):
    dic_fits[select_station][v_y] = dict()

    # Make plot
    ylab = dic_ylabels[v_y]
    ylims = dic_lims[v_y]['ylims']
    xlims = dic_lims[v_y]['xlims']
    axs_sub = axs_all[i,:]
    axs_sub[0].set_ylabel(ylab)

    make_plot2(v_x, v_y, xlims, ylims, season,
               xlab=xlab, ylab=ylab, alpha_scat=alpha_scatt,
               source_list = source_list, fig=fig,
               axs=axs_sub,
               xscale='linear', yscale='linear',
               dic_df_med = dic_df_med,
               select_station= select_station,
               divide_NorESM_by_factor=divide_NorESM_by_factor,
               divide_UKESM_by_factor = divide_UKESM_by_factor,
               markersize=markersize,
               )


    for mo, ax in zip(source_list, axs_sub):
        dic_fits[select_station][v_y][mo] = dict()

        df_s =  dic_df_med[mo]
        print(mo)
        mask_months = select_months(df_s, season=season)
        df_s = df_s[mask_months].copy()
        if (mo =='NorESM') &  (divide_NorESM_by_factor is not None):
            df_s = df_s/divide_NorESM_by_factor
            ax.set_facecolor('#fff6f6')
        if (mo =='UKESM') &  (divide_UKESM_by_factor is not None):
            df_s = df_s/divide_UKESM_by_factor
            ax.set_facecolor('#f0e3f2')
        popt, pov, label, func = func_atto(df_s,v_x,v_y)
        if (mo =='NorESM') &  (divide_NorESM_by_factor is not None):
            label,popt_out =  lin_lab_fix(popt, divide_NorESM_by_factor)
        elif (mo =='UKESM') &  (divide_UKESM_by_factor is not None):
            label,popt_out =  lin_lab_fix(popt, divide_UKESM_by_factor)
        else:
            popt_out = popt
        legends_atto.append(label)

        plot_fit(func, popt, mo, xlims, yscale, xscale, ax,label, linestyle='dashed')
        #plot_fit(func, popt, mo, xlims, yscale, xscale,  ax_fits[i],label,  extra_plot=True,linestyle='dashed',)
        ax.set_xlim(xlims)
        #dic_fits['ATTO'][v_y][mo]['label'] = label
        #dic_fits['ATTO'][v_y][mo]['popt'] = popt
        #dic_fits['ATTO'][v_y][mo]['pov'] = pov

        dic_fits[select_station][v_y][mo]['label'] = label
        dic_fits[select_station][v_y][mo]['popt'] = popt_out
        dic_fits[select_station][v_y][mo]['pcov'] = pov
        dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))

        #dic_fits[select_station][v_y][mo]['func'] = func
        dic_fits[select_station][v_y][mo]['R2'] = get_r2(df_s,v_x,v_y, popt, func)
        dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
        dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])

        _a = label.split('x')[0][1:]
        _b = label.split('x')[1][:-1]
        if _b[0]=='+':
            _b=_b[1:]
        ax.text(left, top, f'a=${_a}$ \nb=${_b}$',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)

        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        #ax.legend(frameon=False, fontsize=10)

    #leg = axs_sub[-1].legend(bbox_to_anchor=(1,1,), frameon=False)

    #legs.append(leg)



for ax in axs_sub:
    ax.set_xlabel(xlab)
sns.despine(fig)



sns.despine(subfig2)
subfig1.suptitle('SMEARII: Jul-Aug', size=16, y=1.05, zorder=100000)
subfig_up.suptitle('SMEARII: Jul-Aug', size=16, y=1., zorder=100000)
#subfig2.suptitle('F', size=16, y=1.05, c='w')
subfig3.suptitle(f'ATTO: {dic_season_nicename[season]}', size=16, y=1.05,zorder=100000)
subfig_up2.suptitle(f'ATTO: {dic_season_nicename[season]}', size=16, y=1.0,zorder=100000)

#axs_all = list(ax_fits.flatten())+list(axs_smr.flatten())+ list(axs_atto.flatten())
axs_all = list(list(axs_smr.flatten())+ list(axs_atto.flatten()))
for ax in axs_all:
    ax.grid(color = 'grey', linestyle = ':', linewidth = 0.5)

for ax in axs_atto[1:,:].flatten():
    ax.set_title('')


for ax in axs_smr[1:,:].flatten():
    ax.set_title('')

#ax_fits[0].set_title('.', color='w')
fn = make_fn_scat(f'together_{season}_bothlin', v_x, 'Nx')
print(fn)
plt.savefig(fn.with_suffix('.pdf'),bbox_inches='tight', dpi=200)
plt.savefig(fn.with_suffix('.png'),bbox_inches='tight', dpi=200)

df = make_pd_of_dic(dic_fits)
df_linear = df.copy()
df.to_csv(fn.with_suffix('.csv'))

plt.show()

# %% [markdown]
# ### Residual SMR ln

# %% tags=[]
select_station = 'SMR'

figsize=[7,10]
## Settings
alpha_scatt = 0.5


divide_NorESM_by_factor = 2


varlistplot = ['N50','N100']
xlab = r'OA [$\mu m^{-3}$]'
alpha_scatt = 0.4
source_list = models_and_obs[::-1]
v_x = 'OA'

## Settings
dic_lims = {
    'N50': {'xlims':[.01,10], 'ylims':[1,5000]},
    'N100': {'xlims':[.01,10], 'ylims':[1,3000]},
    'N200': {'xlims':[.01,10], 'ylims':[1,200]},
    'N50-500': {'xlims':[.01,10], 'ylims':[1,5000]},
    'N100-500': {'xlims':[.01,10], 'ylims':[1,3000]},
    'N200-500': {'xlims':[.01,10], 'ylims':[1,200]},

}

dic_ylabels = {
    'N50' : r'$\Delta$N$_{50}$  [cm$^{-3}$]',
    'N100' : r'$\Delta$N$_{100}$  [cm$^{-3}$]',
    'N200' : r'$\Delta$N$_{200}$  [cm$^{-3}$]',
    'N50-500' : r'$\Delta$N$_{50-500}$  [cm$^{-3}$]',
    'N100-500' : r'$\Delta$N$_{100-500}$  [cm$^{-3}$]',
    'N200-500' : r'$\Delta$N$_{200-500}$  [cm$^{-3}$]',

}






xscale='linear'
yscale='linear'


if select_station=='SMR':
    varlistplot = ['N50','N100']
    
    dic_df_med = dic_df_med_SMR
    season = 'JA'
    func_station = func_smr
    divide_NorESM_by_factor = None
    
elif select_station=='ATTO':
    dic_df_med = dic_df_med_ATTO
    season = 'JFM'
    func_station = func_atto
    
    divide_NorESM_by_factor = 2
    varlistplot = ['N50-500','N100-500','N200-500']


for i,v_y in enumerate(varlistplot):
    #dic_fits[select_station][v_y] = dict()
    fig, axs = plt.subplots(len(models_and_obs), sharex=True, sharey= True, figsize=figsize)
    
    for mo, ax in zip(models_and_obs, axs):

        df_s =  dic_df_med[mo]
        print(mo)
        
        mask_months = select_months(df_s, season=season)
        df_s = df_s[mask_months].copy()
        
        if (mo =='NorESM') &  (divide_NorESM_by_factor is not None):
            df_s = df_s/divide_NorESM_by_factor
            ax.set_facecolor('#fff6f6')
            title = f'{mo}/{divide_NorESM_by_factor}'
            print(title)
            ax.spines['bottom'].set_color('r')
            ax.spines['top'].set_color('r') 
            ax.spines['right'].set_color('red')
            ax.spines['left'].set_color('red')
            ax.set_title(title, c='r')
        else:
            title=mo
            
            ax.set_title(title, )
    
        popt, pov, label, func = func_station(df_s,v_x,v_y)
        #legends_atto.append(label)


        _ma = df_s[v_x].max() 
        _mi = df_s[v_x].min() 
        ax.scatter(df_s[v_x],df_s[v_y]-func(df_s[v_x],*popt),
                                    color=cdic_model[mo], 
                #alpha=alpha_scatt, 
                #facecolor='none',
               alpha=alpha_scatt,
               
                edgecolor=cdic_model[mo],

                label=label
                   )
        #plt.show()
        xlims = dic_lims[v_y]['xlims']
        ax.set_yscale(yscale)
        ax.set_xscale(xscale)


        ax.hlines(0, xmin=xlims[0],xmax=xlims[1], color='k', linewidth=1)
        ax.legend(frameon=False)
        ylab = dic_ylabels[v_y]
        ax.set_ylabel(ylab)
        ax.set_xlim(xlims)
        ax.legend(frameon=True)

        
    #fig.suptitle('Observations')
    axs[-1].set_xlabel(xlab)
    fig.suptitle(r'Residuals fits')

    sns.despine(fig)    
    
    fn = make_fn_scat(f'residual_ln_{season}_{select_station}', v_x, v_y)
    fig.savefig(fn, dpi=150)
    fig.savefig(fn.with_suffix('.pdf'), dpi=150)
    print(fn)
    plt.show()

# %% [markdown]
# ### Residual SMR linear

# %% tags=[]
select_station = 'SMR'

## Settings
alpha_scatt = 0.5


divide_NorESM_by_factor = 2


varlistplot = ['N50','N100','N200']
xlab = r'OA [$\mu m^{-3}$]'
alpha_scatt = 0.4
source_list = models_and_obs[::-1]
v_x = 'OA'

## Settings
dic_lims = {
    'N50': {'xlims':[.01,10], 'ylims':[1,5000]},
    'N100': {'xlims':[.01,10], 'ylims':[1,3000]},
    'N200': {'xlims':[.01,10], 'ylims':[1,200]},
    'N50-500': {'xlims':[.01,10], 'ylims':[1,5000]},
    'N100-500': {'xlims':[.01,10], 'ylims':[1,3000]},
    'N200-500': {'xlims':[.01,10], 'ylims':[1,200]},

}

dic_ylabels = {
    'N50' : r'$\Delta$N$_{50}$  [cm$^{-3}$]',
    'N100' : r'$\Delta$N$_{100}$  [cm$^{-3}$]',
    'N200' : r'$\Delta$N$_{200}$  [cm$^{-3}$]',
    'N50-500' : r'$\Delta$N$_{50-500}$  [cm$^{-3}$]',
    'N100-500' : r'$\Delta$N$_{100-500}$  [cm$^{-3}$]',
    'N200-500' : r'$\Delta$N$_{200-500}$  [cm$^{-3}$]',

}






xscale='linear'
yscale='linear'


if select_station=='SMR':
    varlistplot = ['N50','N100','N200']
    
    dic_df_med = dic_df_med_SMR
    season = 'JA'
    func_station = func_smr
    divide_NorESM_by_factor = None
    
elif select_station=='ATTO':
    dic_df_med = dic_df_med_ATTO
    season = 'JFM'
    func_station = func_atto
    
    divide_NorESM_by_factor = 2
    varlistplot = ['N50-500','N100-500','N200-500']


for i,v_y in enumerate(varlistplot):
    #dic_fits[select_station][v_y] = dict()
    fig, axs = plt.subplots(len(models_and_obs), sharex=True, sharey= True, figsize=figsize)
    
    for mo, ax in zip(models_and_obs, axs):

        df_s =  dic_df_med[mo]
        print(mo)
        
        mask_months = select_months(df_s, season=season)
        df_s = df_s[mask_months].copy()
        
        if (mo =='NorESM') &  (divide_NorESM_by_factor is not None):
            df_s = df_s/divide_NorESM_by_factor
            ax.set_facecolor('#fff6f6')
            title = f'{mo}/{divide_NorESM_by_factor}'
            print(title)
            ax.spines['bottom'].set_color('r')
            ax.spines['top'].set_color('r') 
            ax.spines['right'].set_color('red')
            ax.spines['left'].set_color('red')
            ax.set_title(title, c='r')
        else:
            title=mo
            
            ax.set_title(title, )
    
        #popt, pov, label, func = func_station(df_s,v_x,v_y)
        #legends_atto.append(label)
        popt, pov, label, func = get_odr_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True)
        


        _ma = df_s[v_x].max() 
        _mi = df_s[v_x].min() 
        ax.scatter(df_s[v_x],df_s[v_y]-func(df_s[v_x],*popt),
                                    color=cdic_model[mo], 
                #alpha=alpha_scatt, 
                #facecolor='none',
               alpha=alpha_scatt,
               
                edgecolor=cdic_model[mo],

                label=label
                   )
        #plt.show()
        xlims = dic_lims[v_y]['xlims']
        ax.set_yscale(yscale)
        ax.set_xscale(xscale)


        ax.hlines(0, xmin=xlims[0],xmax=xlims[1], color='k', linewidth=1)
        ax.legend(frameon=False)
        ylab = dic_ylabels[v_y]
        ax.set_ylabel(ylab)
        ax.set_xlim(xlims)
        ax.legend(frameon=True)

        
    #fig.suptitle('Observations')
    axs[-1].set_xlabel(xlab)
    fig.suptitle(r'Residuals fits')

    sns.despine(fig)    
    
    fn = make_fn_scat(f'residual_linear_{season}_{select_station}', v_x, v_y)
    fig.savefig(fn, dpi=150)
    fig.savefig(fn.with_suffix('.pdf'), dpi=150)
    print(fn)
    plt.show()

# %% [markdown]
# ### Residual ATTO linear

# %% tags=[]
select_station = 'ATTO'

## Settings
alpha_scatt = 0.5


divide_NorESM_by_factor = 4


xlab = r'OA [$\mu m^{-3}$]'
alpha_scatt = 0.4
source_list = models_and_obs[::-1]
v_x = 'OA'

## Settings
dic_lims = {
    'N50': {'xlims':[.01,10], 'ylims':[1,5000]},
    'N100': {'xlims':[.01,10], 'ylims':[1,3000]},
    'N200': {'xlims':[.01,10], 'ylims':[1,200]},
    'N50-500': {'xlims':[.01,6], 'ylims':[1,5000]},
    'N100-500': {'xlims':[.01,6], 'ylims':[1,3000]},
    'N200-500': {'xlims':[.01,6], 'ylims':[1,200]},

}

dic_ylabels = {
    'N50' : r'$\Delta$N$_{50}$  [cm$^{-3}$]',
    'N100' : r'$\Delta$N$_{100}$  [cm$^{-3}$]',
    'N200' : r'$\Delta$N$_{200}$  [cm$^{-3}$]',
    'N50-500' : r'$\Delta$N$_{50-500}$  [cm$^{-3}$]',
    'N100-500' : r'$\Delta$N$_{100-500}$  [cm$^{-3}$]',
    'N200-500' : r'$\Delta$N$_{200-500}$  [cm$^{-3}$]',

}



xscale='linear'
yscale='linear'


# OBS: 
if select_station=='SMR':
    varlistplot = ['N50','N100','N200']
    
    dic_df_med = dic_df_med_SMR
    season = 'JA'
    func_station = func_smr
    divide_NorESM_by_factor = None
    
elif select_station=='ATTO':
    dic_df_med = dic_df_med_ATTO
    season = 'FMA'
    func_station = func_atto
    
    divide_NorESM_by_factor = 4
    varlistplot = ['N50-500','N100-500','N200-500']



for i,v_y in enumerate(varlistplot):
    #dic_fits[select_station][v_y] = dict()
    fig, axs = plt.subplots(len(models_and_obs), sharex=True, sharey= True, figsize=figsize)
    
    for mo, ax in zip(models_and_obs, axs):

        df_s =  dic_df_med[mo]
        print(mo)
        
        mask_months = select_months(df_s, season=season)
        df_s = df_s[mask_months].copy()
        
        if (mo =='NorESM') &  (divide_NorESM_by_factor is not None):
            df_s = df_s/divide_NorESM_by_factor
            ax.set_facecolor('#fff6f6')
            title = f'{mo}/{divide_NorESM_by_factor}'
            print(title)
            ax.spines['bottom'].set_color('r')
            ax.spines['top'].set_color('r') 
            ax.spines['right'].set_color('red')
            ax.spines['left'].set_color('red')
            ax.set_title(title, c='r')
        else:
            title=mo
            
            ax.set_title(title, )
    
        popt, pov, label, func = func_station(df_s,v_x,v_y)
        #legends_atto.append(label)


        _ma = df_s[v_x].max() 
        _mi = df_s[v_x].min() 
        ax.scatter(df_s[v_x],df_s[v_y]-func(df_s[v_x],*popt),
                                    color=cdic_model[mo], 
                #alpha=alpha_scatt, 
                #facecolor='none',
               alpha=alpha_scatt,
               
                edgecolor=cdic_model[mo],

                label=label
                   )
        #plt.show()
        xlims = dic_lims[v_y]['xlims']
        ax.set_yscale(yscale)
        ax.set_xscale(xscale)


        ax.hlines(0, xmin=xlims[0],xmax=xlims[1], color='k', linewidth=1)
        ax.legend(frameon=False)
        ylab = dic_ylabels[v_y]
        ax.set_ylabel(ylab)
        ax.set_xlim(xlims)
        ax.legend(frameon=True)

        
    #fig.suptitle('Observations')
    axs[-1].set_xlabel(xlab)
    fig.suptitle(r'Residuals fits')

    sns.despine(fig)    
    
    fn = make_fn_scat(f'residual_ln_{season}_{select_station}', v_x, v_y)
    fig.savefig(fn, dpi=150)
    fig.savefig(fn.with_suffix('.pdf'), dpi=150)
    print(fn)
    plt.show()

# %% [markdown]
# ### Other seasons

# %%
popt

# %% [markdown] tags=[]
# #### JFM

# %%
fig_main = plt.figure(constrained_layout=True,
                  figsize=[15,6],
                 )
spec2 = gridspec.GridSpec( nrows=2,ncols=3, 
                          width_ratios=[4,.1,4],
                          height_ratios=[1,20], 
                          figure=fig_main)

markersize = 5

subfig1 =  fig_main.add_subfigure(spec2[1, 0],frameon=True)
subfig2 =  fig_main.add_subfigure(spec2[1, 1])
subfig3 =  fig_main.add_subfigure(spec2[1, 2],frameon=True)
subfig_up =  fig_main.add_subfigure(spec2[0, :])


axs_smr = subfig1.subplots(3,5, sharex='col', sharey='row')
#ax_fits = subfig2.subplots(3,1, sharex='col', sharey='row')
axs_atto =subfig3.subplots(3,5, sharex='col', sharey='row')
#subfig2.set_facecolor('#e9f2f9')##e5f8f8')
# subfig3.set_facecolor('#fff4ea')

dic_fits = {}
dic_fits['SMR'] =dict()
dic_fits['ATTO'] =dict()

#subfig1.suptitle('SMEARII, Jul & Aug')
#subfig3.suptitle('ATTO, JFM')
#subfig2.suptitle('Fits')

ax_dum = subfig_up.subplots(1)
ax_dum.axis('off')


divide_NorESM_by_factor = 4


varlistplot = ['N50','N100','N200']
xlab = r'OA [$\mu$gm$^{-3}$]'
alpha_scatt = 0.4
source_list = models_and_obs[::-1]
v_x = 'OA'

## Settings
dic_lims = {
    'N50': {'xlims':[.01,12], 'ylims':[1,4000]},
    'N100': {'xlims':[.01,12], 'ylims':[1,2500]},
    'N200': {'xlims':[.01,12], 'ylims':[1,1500]},
    'N50-500': {'xlims':[.01,5], 'ylims':[1,2000]},
    'N100-500': {'xlims':[.01,5], 'ylims':[1,1200]},
    'N200-500': {'xlims':[.01,5], 'ylims':[1,700]},

}

dic_ylabels = {
    'N50' : r'N$_{50}$  [cm$^{-3}$]',
    'N100' : r'N$_{100}$  [cm$^{-3}$]',
    'N200' : r'N$_{200}$  [cm$^{-3}$]',
    'N50-500' : r'N$_{50-500}$  [cm$^{-3}$]',
    'N100-500' : r'N$_{100-500}$  [cm$^{-3}$]',
    'N200-500' : r'N$_{200-500}$  [cm$^{-3}$]',

}




select_station = 'SMR'
season = 'JA'


xscale='linear'
yscale='linear'


# OBS: 

dic_df_med = dic_df_med_SMR
axs_all = axs_smr
fig = subfig1


#fig, axs_all = plt.subplots(3,6,figsize=figsize, sharey='row', sharex='col')
## Settings
legends_smr = []
legends_atto = []
legs =[]

for i,v_y in enumerate(varlistplot):
    dic_fits[select_station][v_y] = dict()
    # Make plot
    ylab = dic_ylabels[v_y]
    ylims = dic_lims[v_y]['ylims']
    xlims = dic_lims[v_y]['xlims']
    
    axs_sub = axs_all[i,:]
    axs_sub[0].set_ylabel(ylab)
    make_plot2(v_x, v_y, xlims, ylims, season, 
              xlab=xlab, ylab=ylab, alpha_scat=alpha_scatt,
             source_list = source_list, fig=fig, 
               axs=axs_sub,
              xscale='linear', yscale='linear',
              dic_df_med = dic_df_med,
           select_station= select_station,
               markersize=markersize,
         )

    
    for mo, ax in zip(source_list, axs_sub):
        dic_fits[select_station][v_y][mo] = dict()

        df_s =  dic_df_med[mo]
        
        print(mo)
        mask_months = select_months(df_s, season=season)
        df_s = df_s[mask_months].copy()
        popt, pov, label, func = func_smr(df_s,v_x,v_y)
            
            
        legends_smr.append(label)

        plot_fit(func, popt, mo, xlims, yscale, xscale, ax, label,)
        #plot_fit(func, popt, mo, xlims, yscale, xscale,  ax_fits[i],label,extra_plot=True)

        dic_fits[select_station][v_y][mo]['label'] = label
        dic_fits[select_station][v_y][mo]['popt'] = popt
        dic_fits[select_station][v_y][mo]['pcov'] = pov
        dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))
        
        dic_fits[select_station][v_y][mo]['func'] = func
        dic_fits[select_station][v_y][mo]['R2'] = get_r2(df_s,v_x,v_y, popt, func)
        dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
        print( get_r2(df_s,v_x,v_y, popt, func))
        dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])   
        
        
        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
    #leg = axs_sub[-1].legend(bbox_to_anchor=(1,1,), frameon=False)

    #legs.append(leg)


    
for ax in axs_sub:
    ax.set_xlabel(xlab)
sns.despine(fig) 








varlistplot = ['N50-500','N100-500','N200-500']

select_station = 'ATTO'
season = 'JFM'

xscale='linear'
yscale='linear'


# OBS: 

dic_df_med = dic_df_med_ATTO
axs_all = axs_atto
fig = subfig3
select_station = 'ATTO'


#fig, axs_all = plt.subplots(3,6,figsize=figsize, sharey='row', sharex='col')
## Settings

legs =[]

for i,v_y in enumerate(varlistplot):
    dic_fits[select_station][v_y] = dict()
    
    # Make plot
    ylab = dic_ylabels[v_y]
    ylims = dic_lims[v_y]['ylims']
    xlims = dic_lims[v_y]['xlims']
    axs_sub = axs_all[i,:]
    axs_sub[0].set_ylabel(ylab)

    make_plot2(v_x, v_y, xlims, ylims, season, 
              xlab=xlab, ylab=ylab, alpha_scat=alpha_scatt,
             source_list = source_list, fig=fig, 
               axs=axs_sub,
              xscale='linear', yscale='linear',
              dic_df_med = dic_df_med,
           select_station= select_station,
               divide_NorESM_by_factor=divide_NorESM_by_factor,
         )


    for mo, ax in zip(source_list, axs_sub):
        dic_fits[select_station][v_y][mo] = dict()
        
        df_s =  dic_df_med[mo]
        print(mo)
        mask_months = select_months(df_s, season=season)
        df_s = df_s[mask_months].copy()
        if (mo =='NorESM') &  (divide_NorESM_by_factor is not None):
            df_s = df_s/divide_NorESM_by_factor
            ax.set_facecolor('#fff6f6')
    
        popt, pov, label, func = func_atto(df_s,v_x,v_y)
        legends_atto.append(label)
        
        plot_fit(func, popt, mo, xlims, yscale, xscale, ax,label, linestyle='dashed')
        #plot_fit(func, popt, mo, xlims, yscale, xscale,  ax_fits[i],label,  extra_plot=True,linestyle='dashed',)
        ax.set_xlim(xlims)
        #dic_fits['ATTO'][v_y][mo]['label'] = label
        #dic_fits['ATTO'][v_y][mo]['popt'] = popt
        #dic_fits['ATTO'][v_y][mo]['pov'] = pov 
        
        dic_fits[select_station][v_y][mo]['label'] = label
        dic_fits[select_station][v_y][mo]['popt'] = popt
        dic_fits[select_station][v_y][mo]['pcov'] = pov
        dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))
        
        #dic_fits[select_station][v_y][mo]['func'] = func
        dic_fits[select_station][v_y][mo]['R2'] = get_r2(df_s,v_x,v_y, popt, func)
        dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
        dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])        

        _a = label.split('x')[0][1:]
        _b = label.split('x')[1][:-1]
        if _b[0]=='+':
            _b=_b[1:]
        ax.text(left, top, f'a=${_a}$ \nb=${_b}$',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        #ax.legend(frameon=False, fontsize=10)
    
    #leg = axs_sub[-1].legend(bbox_to_anchor=(1,1,), frameon=False)

    #legs.append(leg)


    
for ax in axs_sub:
    ax.set_xlabel(xlab)
sns.despine(fig) 




#for i, v_y in enumerate(varlistplot):
#    xlims = dic_lims[v_y]['xlims']
#    ax = ax_fits[i]
    
    #ax.set_xlim(xlims)
#    ax.set_ylim(dic_lims[v_y]['ylims'])
#    ax.set_yticklabels([])
#    ax.set_facecolor('#e9f2f9')##e5f8f8')
#    sns.despine(ax)
#ax.set_xlabel(xlab)
sns.despine(subfig2) 

subfig1.suptitle('SMEARII: Jul,Aug', size=16, y=1.05)
subfig2.suptitle('Fits', size=16, y=1.05, c='w')
subfig3.suptitle(f'ATTO: {dic_season_nicename[season]}', size=16, y=1.05)

#axs_all = list(ax_fits.flatten())+list(axs_smr.flatten())+ list(axs_atto.flatten())
axs_all = list(list(axs_smr.flatten())+ list(axs_atto.flatten()))
for ax in axs_all:
    ax.grid(color = 'grey', linestyle = ':', linewidth = 0.5)



#for ax in axs_atto[:,0]:
#    ax.set_yticklabels([])
#    ax.set_ylabel('')

for ax in axs_atto[1:,:].flatten():
    ax.set_title('')


for ax in axs_smr[1:,:].flatten():
    ax.set_title('')

#ax_fits[0].set_title('.', color='w')
fn = make_fn_scat(f'together_{season}', v_x, 'Nx')
print(fn)
plt.savefig(fn.with_suffix('.pdf'),bbox_inches='tight', dpi=200)
plt.savefig(fn.with_suffix('.png'),bbox_inches='tight', dpi=200)

df = make_pd_of_dic(dic_fits)
df.to_csv(fn.with_suffix('.csv'))

plt.show()

# %% [markdown]
# #### MAM

# %%
fig_main = plt.figure(constrained_layout=True,
                  figsize=[15,6],
                 )
spec2 = gridspec.GridSpec( nrows=2,ncols=3, 
                          width_ratios=[4,.1,4],
                          height_ratios=[1,20], 
                          figure=fig_main)

markersize = 5

subfig1 =  fig_main.add_subfigure(spec2[1, 0],frameon=True)
subfig2 =  fig_main.add_subfigure(spec2[1, 1])
subfig3 =  fig_main.add_subfigure(spec2[1, 2],frameon=True)
subfig_up =  fig_main.add_subfigure(spec2[0, :])


axs_smr = subfig1.subplots(3,5, sharex='col', sharey='row')
#ax_fits = subfig2.subplots(3,1, sharex='col', sharey='row')
axs_atto =subfig3.subplots(3,5, sharex='col', sharey='row')
#subfig2.set_facecolor('#e9f2f9')##e5f8f8')
# subfig3.set_facecolor('#fff4ea')

dic_fits = {}
dic_fits['SMR'] =dict()
dic_fits['ATTO'] =dict()

#subfig1.suptitle('SMEARII, Jul & Aug')
#subfig3.suptitle('ATTO, JFM')
#subfig2.suptitle('Fits')

ax_dum = subfig_up.subplots(1)
ax_dum.axis('off')


divide_NorESM_by_factor = 4


varlistplot = ['N50','N100','N200']
xlab = r'OA [$\mu$gm$^{-3}$]'
alpha_scatt = 0.4
source_list = models_and_obs[::-1]
v_x = 'OA'

## Settings
dic_lims = {
    'N50': {'xlims':[.01,12], 'ylims':[1,4000]},
    'N100': {'xlims':[.01,12], 'ylims':[1,2500]},
    'N200': {'xlims':[.01,12], 'ylims':[1,1500]},
    'N50-500': {'xlims':[.01,5], 'ylims':[1,2000]},
    'N100-500': {'xlims':[.01,5], 'ylims':[1,1200]},
    'N200-500': {'xlims':[.01,5], 'ylims':[1,700]},

}

dic_ylabels = {
    'N50' : r'N$_{50}$  [cm$^{-3}$]',
    'N100' : r'N$_{100}$  [cm$^{-3}$]',
    'N200' : r'N$_{200}$  [cm$^{-3}$]',
    'N50-500' : r'N$_{50-500}$  [cm$^{-3}$]',
    'N100-500' : r'N$_{100-500}$  [cm$^{-3}$]',
    'N200-500' : r'N$_{200-500}$  [cm$^{-3}$]',

}




select_station = 'SMR'
season = 'JA'


xscale='linear'
yscale='linear'


# OBS: 

dic_df_med = dic_df_med_SMR
axs_all = axs_smr
fig = subfig1


#fig, axs_all = plt.subplots(3,6,figsize=figsize, sharey='row', sharex='col')
## Settings
legends_smr = []
legends_atto = []
legs =[]

for i,v_y in enumerate(varlistplot):
    dic_fits[select_station][v_y] = dict()
    # Make plot
    ylab = dic_ylabels[v_y]
    ylims = dic_lims[v_y]['ylims']
    xlims = dic_lims[v_y]['xlims']
    
    axs_sub = axs_all[i,:]
    axs_sub[0].set_ylabel(ylab)
    make_plot2(v_x, v_y, xlims, ylims, season, 
              xlab=xlab, ylab=ylab, alpha_scat=alpha_scatt,
             source_list = source_list, fig=fig, 
               axs=axs_sub,
              xscale='linear', yscale='linear',
              dic_df_med = dic_df_med,
           select_station= select_station,
               markersize=markersize,
         )

    
    for mo, ax in zip(source_list, axs_sub):
        dic_fits[select_station][v_y][mo] = dict()

        df_s =  dic_df_med[mo]
        
        print(mo)
        mask_months = select_months(df_s, season=season)
        df_s = df_s[mask_months].copy()
        popt, pov, label, func = func_smr(df_s,v_x,v_y)
            
            
        legends_smr.append(label)

        plot_fit(func, popt, mo, xlims, yscale, xscale, ax, label,)
        #plot_fit(func, popt, mo, xlims, yscale, xscale,  ax_fits[i],label,extra_plot=True)

        dic_fits[select_station][v_y][mo]['label'] = label
        dic_fits[select_station][v_y][mo]['popt'] = popt
        dic_fits[select_station][v_y][mo]['pcov'] = pov
        dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))
        
        dic_fits[select_station][v_y][mo]['func'] = func
        dic_fits[select_station][v_y][mo]['R2'] = get_r2(df_s,v_x,v_y, popt, func)
        dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
        print( get_r2(df_s,v_x,v_y, popt, func))
        dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])   
        
        
        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
    #leg = axs_sub[-1].legend(bbox_to_anchor=(1,1,), frameon=False)

    #legs.append(leg)


    
for ax in axs_sub:
    ax.set_xlabel(xlab)
sns.despine(fig) 








varlistplot = ['N50-500','N100-500','N200-500']

select_station = 'ATTO'
season = 'MAM'

xscale='linear'
yscale='linear'


# OBS: 

dic_df_med = dic_df_med_ATTO
axs_all = axs_atto
fig = subfig3
select_station = 'ATTO'


#fig, axs_all = plt.subplots(3,6,figsize=figsize, sharey='row', sharex='col')
## Settings

legs =[]

for i,v_y in enumerate(varlistplot):
    dic_fits[select_station][v_y] = dict()
    
    # Make plot
    ylab = dic_ylabels[v_y]
    ylims = dic_lims[v_y]['ylims']
    xlims = dic_lims[v_y]['xlims']
    axs_sub = axs_all[i,:]
    axs_sub[0].set_ylabel(ylab)

    make_plot2(v_x, v_y, xlims, ylims, season, 
              xlab=xlab, ylab=ylab, alpha_scat=alpha_scatt,
             source_list = source_list, fig=fig, 
               axs=axs_sub,
              xscale='linear', yscale='linear',
              dic_df_med = dic_df_med,
           select_station= select_station,
               divide_NorESM_by_factor=divide_NorESM_by_factor,
         )


    for mo, ax in zip(source_list, axs_sub):
        dic_fits[select_station][v_y][mo] = dict()
        
        df_s =  dic_df_med[mo]
        print(mo)
        mask_months = select_months(df_s, season=season)
        df_s = df_s[mask_months].copy()
        if (mo =='NorESM') &  (divide_NorESM_by_factor is not None):
            df_s = df_s/divide_NorESM_by_factor
            ax.set_facecolor('#fff6f6')
    
        popt, pov, label, func = func_atto(df_s,v_x,v_y)
        legends_atto.append(label)
        
        plot_fit(func, popt, mo, xlims, yscale, xscale, ax,label, linestyle='dashed')
        #plot_fit(func, popt, mo, xlims, yscale, xscale,  ax_fits[i],label,  extra_plot=True,linestyle='dashed',)
        ax.set_xlim(xlims)
        #dic_fits['ATTO'][v_y][mo]['label'] = label
        #dic_fits['ATTO'][v_y][mo]['popt'] = popt
        #dic_fits['ATTO'][v_y][mo]['pov'] = pov 
        
        dic_fits[select_station][v_y][mo]['label'] = label
        dic_fits[select_station][v_y][mo]['popt'] = popt
        dic_fits[select_station][v_y][mo]['pcov'] = pov
        dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))
        
        #dic_fits[select_station][v_y][mo]['func'] = func
        dic_fits[select_station][v_y][mo]['R2'] = get_r2(df_s,v_x,v_y, popt, func)
        dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
        dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])        

        _a = label.split('x')[0][1:]
        _b = label.split('x')[1][:-1]
        if _b[0]=='+':
            _b=_b[1:]
        ax.text(left, top, f'a=${_a}$ \nb=${_b}$',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        #ax.legend(frameon=False, fontsize=10)
    
    #leg = axs_sub[-1].legend(bbox_to_anchor=(1,1,), frameon=False)

    #legs.append(leg)


    
for ax in axs_sub:
    ax.set_xlabel(xlab)
sns.despine(fig) 




#for i, v_y in enumerate(varlistplot):
#    xlims = dic_lims[v_y]['xlims']
#    ax = ax_fits[i]
    
    #ax.set_xlim(xlims)
#    ax.set_ylim(dic_lims[v_y]['ylims'])
#    ax.set_yticklabels([])
#    ax.set_facecolor('#e9f2f9')##e5f8f8')
#    sns.despine(ax)
#ax.set_xlabel(xlab)
sns.despine(subfig2) 

subfig1.suptitle('SMEARII: Jul,Aug', size=16, y=1.05)
subfig2.suptitle('Fits', size=16, y=1.05, c='w')
subfig3.suptitle(f'ATTO: {dic_season_nicename[season]}', size=16, y=1.05)

#axs_all = list(ax_fits.flatten())+list(axs_smr.flatten())+ list(axs_atto.flatten())
axs_all = list(list(axs_smr.flatten())+ list(axs_atto.flatten()))
for ax in axs_all:
    ax.grid(color = 'grey', linestyle = ':', linewidth = 0.5)



#for ax in axs_atto[:,0]:
#    ax.set_yticklabels([])
#    ax.set_ylabel('')

for ax in axs_atto[1:,:].flatten():
    ax.set_title('')


for ax in axs_smr[1:,:].flatten():
    ax.set_title('')

#ax_fits[0].set_title('.', color='w')
fn = make_fn_scat(f'together_{season}', v_x, 'Nx')
print(fn)
plt.savefig(fn.with_suffix('.pdf'),bbox_inches='tight', dpi=200)
plt.savefig(fn.with_suffix('.png'),bbox_inches='tight', dpi=200)

df = make_pd_of_dic(dic_fits)

df.to_csv(fn.with_suffix('.csv'))

plt.show()


# %% [markdown] tags=[]
# ## Make plot for T to OA

# %%
def make_plot(v_x, v_y, xlims, ylims, season, 
              xlab=None, ylab=None, alpha_scat=.3,
             source_list = models_and_obs, fig=None, ax=None, daxs=None, axs_extra=None,
              xscale='linear', yscale='linear',
              dic_df_med = dic_df_med,
              markersize=1,
              marker='.',

             ):
    if fig is None: 
        fig, ax, daxs, axs_extra = make_cool_grid3(ncols_extra=2, nrows_extra=3,)# w_ratio_sideplot=.5)

    if xlab is None: 
        if xlab in label_dic:
            xlab = label_dic[v_x]
    if ylab is None: 
        if ylab in label_dic:
            ylab = label_dic[v_y]

    make_scatter_plot(v_x, v_y, xlims, ylims, season, 
              xlab=xlab, ylab=ylab, alpha_scat=alpha_scat,
             source_list = source_list, fig=fig, ax=ax, daxs=daxs, axs_extra=axs_extra,
              xscale=xscale, yscale=yscale,
              dic_df_med = dic_df_med,
              markersize=markersize,
              marker=marker,)
        
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    fig.suptitle(f'ATTO, {season} season, 2012-2018', y=.95)
    xlim_dist = list(daxs['y'].get_xlim())
    for mo in source_list:

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

def make_scatter_plot(v_x, v_y, xlims, ylims, season, 
              xlab=None, ylab=None, alpha_scat=.3,
             source_list = models_and_obs, fig=None, ax=None, daxs=None, axs_extra=None,
              xscale='linear', yscale='linear',
              dic_df_med = dic_df_med,
              markersize=1,
              marker='.',):
    
    for mo, ax_ex in zip(source_list, axs_extra[:]):
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
                        marker=marker,
                        s=markersize,

                   )
        sns.scatterplot(x=v_x,y=v_y, 
                    data = df_s, 
                    color=cdic_model[mo], 
                    alpha=alpha_scatt+.1, 
                    label='__nolegend__',
                    ax = ax_ex,
                    #facecolor='none',
                    edgecolor=cdic_model[mo],
                        marker=marker,
                        s=markersize,
                    
                   )
        ax_ex.set_title(mo, y=.95)
    return
#### WET_mid

# %%
def make_cool_grid6(
    figsize=None,
    fig = None, 
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
    width_small_plot = size_big_plot/(num_subplots_per_big_plot)
    width_dist_ax = size_big_plot*frac_dist_axis_from_big

    if figsize is None:
        figsize = [size_big_plot + width_small_plot+ width_dist_ax,
                   size_big_plot + width_small_plot+ width_dist_ax,
                   ]
        print(figsize)
    #width_ratios = None
    add_gs_kw = None

    #if width_ratios is None:
    #    width_ratios = [1] * ncols + [w_cbar / w_plot] #+ [1]* ncols_extra
    if add_gs_kw is None:
        add_gs_kw = dict()


    if 'hspace' not in add_gs_kw.keys():
        add_gs_kw['hspace'] = 0
    if 'wspace' not in add_gs_kw.keys():
        add_gs_kw['wspace'] = 0
    if fig is None:
        fig = plt.figure(figsize=figsize,
                     dpi=100)

    w_r1 = [size_big_plot,size_big_plot*frac_dist_axis_from_big]
    h_r1 = [frac_dist_axis_from_big,1, ]
    
    width_ratio_big_small = [size_big_plot+width_dist_ax,width_small_plot][::-1]
    height_ratio_big_small = [size_big_plot+width_dist_ax,width_small_plot]
    
    gs0 = gridspec.GridSpec(2, 2, figure=fig, height_ratios=height_ratio_big_small ,
                            width_ratios = width_ratio_big_small)
    # Big plot:
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows+1, ncols+1, width_ratios=w_r1, height_ratios=h_r1, subplot_spec=gs0[0,1], **add_gs_kw)
    # for the small plots:
    gs01 = gridspec.GridSpecFromSubplotSpec(num_subplots_per_big_plot,2,width_ratios =[40,1], subplot_spec=gs0[:-1,0])#, **add_gs_kw)
    gs03 = gridspec.GridSpecFromSubplotSpec(2,num_subplots_per_big_plot+1, height_ratios =[1,40], subplot_spec=gs0[1,:])#, **add_gs_kw)
    # Axs for big plot and distribution axis:
    axs = gs00.subplots(sharex=sharex, sharey=sharey, )
    # Axes 
    axs_extra_sup = gs01.subplots(sharex=sharex, sharey=sharey, )
    axs_extra2_sup = gs03.subplots(sharex=sharex, sharey=sharey, )
    axs_extra = axs_extra_sup[:,0]
    axs_extra2 = axs_extra2_sup[1,:]
    for _ax in (list(axs_extra_sup[:,1].flatten())+list(axs_extra2_sup[0,:].flatten())) :
        _ax.axis("off")
        
    axs_extra = np.concatenate((axs_extra, axs_extra2[::-1],))
    
    axs[0,1].clear()
    axs[0,1].axis("off")
    daxs = dict(x=axs[0,0],y=axs[1,1])
    # distribution axis
    for a in daxs:
        _ax = daxs[a]
        sns.despine(bottom=False, left=False, ax=_ax)
        _ax.axis("off")
    # big plot!
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
make_cool_grid6()

# %%
from bs_fdbck.util.BSOA_datamanip.fits import get_least_square_fit_and_labs


# %%
def get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12]):
    v_log_y = f'ln({v_y})'
    df_s[v_log_y] = np.log(df_s[v_y])
    popt, pcov, label, func_lin = get_odr_fit_and_labs(df_s, v_x, v_log_y, fit_func = 'linear', return_func=True, beta0=beta0)
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

    return popt, pcov, label, func, func_lin, popt_lin


# %%
def get_r2_logy(df_s,v_x,x_y, popt, func):
    """
    Calculating the r2 value for a log scale fitting. 
    """
    v_log_y = f'ln({v_y})'
    df_s[v_log_y] = np.log(df_s[v_y])
    _df = df_s[[v_x, v_log_y]].dropna().copy()
    y_pred = func(_df[v_x].values, *popt)
    r2 =  r2_score(_df[v_log_y].values, y_pred)
    return r2


# %% [markdown]
# ## Paper figure 2 FMA

# %%
fx = .75
fig_main = plt.figure(constrained_layout=True,
                  figsize=[17*fx, 8.25*fx],
                 )
spec2 = gridspec.GridSpec( nrows=2,ncols=3, 
                          height_ratios=[1,30], 
                          width_ratios=[30,1,30], 
                          figure=fig_main)


markersize = 4

subfig_smr =  fig_main.add_subfigure(spec2[1, 0],frameon=True)
subfig_atto =  fig_main.add_subfigure(spec2[1, 2],frameon=True)
subfig_up_smr =  fig_main.add_subfigure(spec2[0, 0])
subfig_up_atto =  fig_main.add_subfigure(spec2[0, 2])



dic_fits = {}
dic_fits['SMR'] =dict()
dic_fits['ATTO'] =dict()



season_atto = 'FMA'
season_smr = 'JA'





fig, ax, daxs, axs_extra = make_cool_grid5(fig =subfig_smr )##ncols_extra=2, nrows_extra=2,)# w_ratio_sideplot=.5)

axs_extra = axs_extra.flatten()[::-1]








dic_df_med = dic_df_med_SMR
select_station = 'SMR'
dic_fits[select_station] = dict()
dic_fits[select_station][v_y] = dict()

## Settings
alpha_scatt = 0.6

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


linewidth=2
xlims = [5,30]
ylims = [.1,35]


season=season_smr
v_x = 'T_C'
v_y = 'OA'
dic_fits[select_station][v_y] = dict()


make_plot(v_x, v_y, xlims, ylims, season, 
              xlab, ylab, .3, models_and_obs, fig, ax, daxs, axs_extra,
          yscale='log',
          dic_df_med = dic_df_med,
          markersize = markersize
         )


for mo, ax_ex in zip(models_and_obs, axs_extra[:]):
    print(f'******{mo}*********')
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    #popt, pov, label, func = get_least_square_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    #popt, pov, label, func = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    popt, pov, label, func,func_lin, popt_lin = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])

    dic_fits[select_station][v_y][mo] = dict()
    dic_fits[select_station][v_y][mo]['label'] = label
    dic_fits[select_station][v_y][mo]['popt'] = popt
    dic_fits[select_station][v_y][mo]['pcov'] = pov
    dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))

    dic_fits[select_station][v_y][mo]['func'] = func
    dic_fits[select_station][v_y][mo]['R2'] = get_r2_logy(df_s,v_x,v_y, popt_lin, func_lin)
    dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
    print('R2*****')
    print(dic_fits[select_station][v_y][mo]['R2'])
    print(dic_fits[select_station][v_y][mo]['corr'])
    dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])   

    
    
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    if np.abs(dic_fits[select_station][v_y][mo]['corr'])<.1:
        continue
    ax.plot(x, func(x, *popt), c='w', linewidth=linewidth+2,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=linewidth+1, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=linewidth+1,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
               linewidth=linewidth,
              )
    ax_ex.set_yscale('log')
ax.set_yscale('log')
ax.grid(color='grey', linewidth=.5, linestyle=':', )
ax.set_xticks(np.arange(5,31,5))

for ax_ex in axs_extra:
    ax_ex.set_yticklabels([])
    ax_ex.set_xticks(np.arange(5,31,5))
    ax_ex.grid(color='grey', linewidth=.5, linestyle=':')
    ax_ex.set_ylim(ylims)
    ax_ex.set_xlim(xlims)

    
fn = make_fn_scat(f'exp1_{season}', v_x, v_y)
ax.legend(frameon=False)




## ATTO:
select_station = 'ATTO'
dic_fits[select_station] = dict()
dic_fits[select_station][v_y] = dict()


xlims = [20,40]



fig, ax, daxs, axs_extra = make_cool_grid5(fig =subfig_atto )##ncols_extra=2, nrows_extra=2,)# w_ratio_sideplot=.5)

axs_extra = axs_extra.flatten()[::-1]

dic_df_med = dic_df_med_ATTO

# Remove 2015 and 2016:
dic_df_med_adj = dict()

for k in dic_df_med.keys():
    _df  = dic_df_med[k].copy()
    _df = _df[~_df.index.year.isin([2015,2016])]
    dic_df_med_adj[k] = _df

## Settings
alpha_scatt = 0.6


linewidth=2
xlims = [20,40]


season = season_atto


make_plot(v_x, v_y, xlims, ylims, season, 
              xlab, ylab, .2, models_and_obs, fig, ax, daxs, axs_extra,
          yscale='log',
          dic_df_med = dic_df_med,
          markersize=markersize-2,
          marker='.',
         
         )
make_scatter_plot(v_x, v_y, xlims, ylims, season, 
              xlab, ylab, .2, models_and_obs, fig, ax, daxs, axs_extra,
          yscale='log',
          dic_df_med = dic_df_med_adj,
          markersize=markersize,
                  marker='*',
         
         )




for mo, ax_ex in zip(models_and_obs, axs_extra[:]):
    print(f'******{mo}********')
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    
    #popt, pov, label, func = get_least_square_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    #popt, pov, label, func = get_odr_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    popt, pov, label, func, func_lin, popt_lin = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
        
    dic_fits[select_station][v_y][mo] = dict()
    dic_fits[select_station][v_y][mo]['label'] = label
    dic_fits[select_station][v_y][mo]['popt'] = popt
    dic_fits[select_station][v_y][mo]['pcov'] = pov
    dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))

    dic_fits[select_station][v_y][mo]['func'] = func
    dic_fits[select_station][v_y][mo]['R2'] = get_r2_logy(df_s,v_x,v_y, popt_lin, func_lin)
    dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
    print('R2*****')
    print(dic_fits[select_station][v_y][mo]['R2'])
    print(dic_fits[select_station][v_y][mo]['corr'])
    
    dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])   

    
    
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    if np.abs(dic_fits[select_station][v_y][mo]['corr'])<.1:
        continue
    
    ax.plot(x, func(x, *popt), c='w', linewidth=linewidth+2,label='__nolegend__')
    ax.plot(x, func(x, *popt), linewidth=linewidth+1, c=cdic_model[mo],label=f'{mo}: {label}')

    ax_ex.plot(x, func(x, *popt), c='w', linewidth=linewidth+1,label=f'{mo}: {label}',
             )
    ax_ex.plot(x, func(x, *popt), c=cdic_model[mo],label=f'{mo}: {label}',
               linewidth=linewidth,
              )
    ax_ex.set_yscale('log')
    ax_ex.set_ylim(ylims)
    ax_ex.set_xlim(xlims)
    
#WITHOUT 2015-2016

select_station = 'ATTO-no2015/2016'
dic_fits[select_station] = dict()
dic_fits[select_station][v_y] = dict()

for mo, ax_ex in zip(models_and_obs, axs_extra[:]):
    print(f'******{mo}********')
    df_s =  dic_df_med_adj[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    
    #popt, pov, label, func = get_least_square_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    #popt, pov, label, func = get_odr_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    popt, pov, label, func, func_lin, popt_lin = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
        
    dic_fits[select_station][v_y][mo] = dict()
    dic_fits[select_station][v_y][mo]['label'] = label
    dic_fits[select_station][v_y][mo]['popt'] = popt
    dic_fits[select_station][v_y][mo]['pcov'] = pov
    dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))

    dic_fits[select_station][v_y][mo]['func'] = func
    dic_fits[select_station][v_y][mo]['R2'] = get_r2_logy(df_s,v_x,v_y, popt_lin, func_lin)
    dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
    print('R2*****')
    print(dic_fits[select_station][v_y][mo]['R2'])
    print(dic_fits[select_station][v_y][mo]['corr'])
    
    dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])   

    
    
    _mi = df_s[v_x].min()
    _ma = df_s[v_x].max() 
    _xlim = [_mi*.95, _ma*1.05]
    x = np.linspace(*_xlim)
    if np.abs(dic_fits[select_station][v_y][mo]['corr'])>.1:
    
        ax.plot(x, func(x, *popt), c='w', linewidth=linewidth+.2,label='__nolegend__')
        ax.plot(x, func(x, *popt), linestyle= '--',linewidth=linewidth+1, c=cdic_model[mo],label=f'{mo}: {label}')

        ax_ex.plot(x, func(x, *popt), c='w', linewidth=linewidth+.1,label=f'{mo}: {label}',
             )
        ax_ex.plot(x, func(x, *popt),linestyle= '--', c=cdic_model[mo],label=f'{mo}: {label}',
               linewidth=linewidth,
              )
    ax_ex.set_yscale('log')
    ax_ex.set_ylim(ylims)
    ax_ex.set_xlim(xlims)
ax.set_yscale('log')
ax.grid(color='grey', linewidth=.5, linestyle=':')

ax.set_xticks(np.arange(20,41,5))

for ax_ex in axs_extra.flatten():
    ax_ex.set_yticklabels([])
    ax_ex.set_xticks(np.arange(20,41,5))
    ax_ex.grid(color='grey', linewidth=.5, linestyle=':')

    
ax.legend(frameon=False)











ax_up_smr = subfig_up_smr.subplots()
ax_up_smr.axis('off')
subfig_up_smr.suptitle('SMEARII: July, Aug',size=16,)
ax_up_atto = subfig_up_atto.subplots()
ax_up_atto.axis('off')
subfig_up_atto.suptitle(f'ATTO: {dic_season_nicename[season_atto]}',size=16,)

fn = make_fn_scat(f'together_{season_smr}_{season_atto}', v_x, v_y)
print(fn)
fig_main.savefig(fn.with_suffix('.pdf'), dpi=200)
fig_main.savefig(fn.with_suffix('.png'), dpi=200)
plt.show()

df = make_pd_of_dic(dic_fits)
df_OA = df
df.to_csv(fn.with_suffix('.csv'))


# %% [markdown]
# ### Write all fits to file

# %%
df_all = pd.concat([df_log, df_linear, df_OA], axis=0)


df_all['full_fit'] = df_all['Fit'].copy()
df_all = df_all.reset_index().drop_duplicates()

for i in df_all.index:
    #print(i)
    #print(df_all.loc[i,'Fit'].values[0])
    if 'ln' in df_all.loc[i,'Fit']:
        df_all.loc[i,'Fit'] = '$a+b\ln{(c+x)}$'

    elif 'e^' in df_all.loc[i,'Fit']:
        df_all.loc[i,'Fit'] = '$a\cdot \exp{(bx)}$'
    else:
        df_all.loc[i,'Fit'] = '$ax +b$'
df_all = (df_all
          .sort_values(['station','variable','data source','Fit'], ascending=False)
          .set_index(['station','variable','data source','Fit'])        
         )
df_all.to_csv('Plots/Both_stations/all_fits.csv')

# %%
df_all.loc['ATTO'].loc['N200-500']

# %%
df_all.loc['ATTO'].loc['N100-500']

# %%
df_log.loc['SMR'].loc['N100']

# %%
df_linear.loc['SMR'].loc['N100']

# %%
df_all.loc['SMR'].loc['N100']

# %%
df_all.loc['ATTO']

# %%
df_all.loc['SMR']

# %%
df_all_rn = df_all.rename({'N100-500':'N100','N50-500':'N50','N200-500':'N200',})

# %%
df_all_rn.loc['ATTO'].drop_duplicates()

# %%
df_linear_rn = df_linear.rename({'N100-500':'N100','N50-500':'N50','N200-500':'N200',})
df_linear_rn
df_linear_rn['a_float']=df_linear_rn['a'].apply(lambda x: float(x.split(' ')[0]))
df_linear_rn['a_std_float']=df_linear_rn['a'].apply(lambda x: float(x.split(' ')[-1]))


# %%
_df = df_linear_rn[['a_float']].reset_index().drop_duplicates().set_index(['station','variable','data source'])


_df_smr = _df.reset_index().set_index(['station']).loc['SMR',:]
_df_atto = _df.reset_index().set_index(['station']).loc['ATTO',:]


_df1 = _df_smr.rename({'a_float':'a_smr'}, axis=1).reset_index().drop('station', axis=1)
_df2 = _df_atto.rename({'a_float':'a_atto'}, axis=1).reset_index().drop('station', axis=1)

_df_comb = pd.concat([_df1.set_index(['variable','data source']), _df2.set_index(['variable','data source'])], axis=1)

_df_comb

# %%
_df = df_linear_rn[['a_std_float']].reset_index().drop_duplicates().set_index(['station','variable','data source'])


_df_smr_std = _df.reset_index().set_index(['station']).loc['SMR',:]
_df_atto_std = _df.reset_index().set_index(['station']).loc['ATTO',:]


_df1_std = _df_smr_std.rename({'a_std_float':'a_std_smr'}, axis=1).reset_index().drop('station', axis=1)
_df2_std = _df_atto_std.rename({'a_std_float':'a_std_atto'}, axis=1).reset_index().drop('station', axis=1)

_df_comb_std = pd.concat([_df1_std.set_index(['variable','data source']), _df2_std.set_index(['variable','data source'])], axis=1)

_df_comb_std

# %%
_df = df_linear_rn[['a_float']].reset_index().drop_duplicates().set_index(['station','variable','data source'])


_df_smr = _df.reset_index().set_index(['station']).loc['SMR',:]
_df_atto = _df.reset_index().set_index(['station']).loc['ATTO',:]


_df1 = _df_smr.rename({'a_float':'a_smr'}, axis=1).reset_index().drop('station', axis=1)
_df2 = _df_atto.rename({'a_float':'a_atto'}, axis=1).reset_index().drop('station', axis=1)

_df_comb = pd.concat([_df1.set_index(['variable','data source']), _df2.set_index(['variable','data source'])], axis=1)

_df_comb

# %%
_df_comb = pd.concat([_df1.set_index(['variable','data source']), _df2.set_index(['variable','data source']),
                     _df1_std.set_index(['variable','data source']), _df2_std.set_index(['variable','data source'])], axis=1)


# %%
_df_comb

# %%
_df_comb.reset_index()

# %%
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib as mpl

# %%
styles = ['o','*','d']
f, ax = plt.subplots(1, figsize=[5,3,])
_df_combr = _df_comb.reset_index()
for v,st in zip(_df_combr['variable'].unique(), styles):
    _df = _df_combr[_df_combr['variable']==v]
    for m in _df_combr['data source'].unique():
        _df2 = _df[_df['data source'] ==m]
        c = cdic_model[m]
        _df2.plot.scatter(x = 'a_smr', y='a_atto', c = c, ax = ax, marker=st)#, yerr = 'a_std_atto', xerr = 'a_std_smr', linewidth=.5)
        ax.errorbar(_df2['a_smr'], _df2['a_atto'], c = c, marker=st, yerr = _df2['a_std_atto'], xerr = _df2['a_std_smr'], linewidth=1, alpha=.5)
plt.plot([20,700],[20,700], c='k', linestyle='--', alpha=.4, zorder=-100)
plt.xlim([20,800])
plt.ylim([20,800])



plt.xlabel('SMEAR: Slope OA to Nx')
plt.ylabel('ATTO: Slope OA to Nx')

custom_lines =  []
custom_labels = []
for m in models_and_obs[::-1]:
    c = cdic_model[m]
    li = Line2D([0], [0], color=c, lw=0, marker='o')
    custom_lines.append(li)
    custom_labels.append(m)
for v,st in zip(_df_combr['variable'].unique(), styles):
    li = Line2D([0], [0], color='k', lw=0, marker=st)
    custom_lines.append(li)
    custom_labels.append(v)
    
ax.legend(custom_lines, custom_labels, bbox_to_anchor=(1,1,))#, title='Number conc.')

#ax.grid()
sns.despine(f)
plt.yscale('log')
plt.xscale('log')
fn = make_fn_scat(f'slope_comparison_{season_smr}_{season_atto}', 'OA', 'Nx')
print(fn)
f.tight_layout()
f.savefig(fn.with_suffix('.png'), )
f.savefig(fn.with_suffix('.pdf'),)


# %%
from IPython.display import display, Image
display(Image(filename=fn))


# %%
styles = ['o','*','d']
f, ax = plt.subplots(1, figsize=[3,3,])
_df_combr = _df_comb.reset_index()
for v,st in zip(_df_combr['variable'].unique(), styles):
    _df = _df_combr[_df_combr['variable']==v]
    for m in _df_combr['data source'].unique():
        _df2 = _df[_df['data source'] ==m]
        c = cdic_model[m]
        _df2.plot.scatter(x = 'a_smr', y='a_atto', c = c, ax = ax, marker=st)#, yerr = 'a_std_atto', xerr = 'a_std_smr', linewidth=.5)
        ax.errorbar(_df2['a_smr'], _df2['a_atto'], c = c, marker=st, yerr = 2*_df2['a_std_atto'], xerr = 2*_df2['a_std_smr'], linewidth=1, alpha=.5)
plt.plot([20,700],[20,700], c='k', linestyle='--', alpha=.4, zorder=-100)
plt.xlim([20,800])
plt.ylim([20,800])



plt.xlabel('SMEAR: Slope OA to Nx')
plt.ylabel('ATTO: Slope OA to Nx')

custom_lines =  []
custom_labels = []
for m in models_and_obs[::-1]:
    c = cdic_model[m]
    li = Line2D([0], [0], color=c, lw=0, marker='o')
    custom_lines.append(li)
    custom_labels.append(m)
for v,st in zip(_df_combr['variable'].unique(), styles):
    li = Line2D([0], [0], color='k', lw=0, marker=st)
    custom_lines.append(li)
    custom_labels.append(v)
    
ax.legend(custom_lines, custom_labels, bbox_to_anchor=(1,1,))

plt.yscale('linear')
plt.xscale('linear')

# %%
hue_order = models_and_obs[::-1]
pal = sns.color_palette([cdic_model[h] for h in hue_order])

# %%
sns.relplot(x='a_smr', 
            y='a_atto', 
            hue='data source',
            style='variable',
            data = _df_comb.reset_index(),
            palette=pal,
            hue_order=hue_order,
            style_order=['OA','N50','N100','N200'],
            
            
           s=200)
plt.plot([0,700],[0,700], c='k', linestyle='--', alpha=.4, zorder=-100)
plt.xlim([10,800])
plt.ylim([10,800])
plt.xlabel('SMEAR: Slope OA to Nx')
plt.ylabel('ATTO: Slope OA to Nx')
plt.yscale('log')
plt.xscale('log')

# %%
_df = df_all_rn[['r$^2$']].reset_index().drop(['Fit'], axis=1).drop_duplicates().set_index(['station','variable','data source'])


_df_smr = _df.reset_index().set_index(['station']).loc['SMR',:]
_df_atto = _df.reset_index().set_index(['station']).loc['ATTO',:]


_df1 = _df_smr.rename({'r$^2$':'r2_smr'}, axis=1).reset_index().drop('station', axis=1)
_df2 = _df_atto.rename({'r$^2$':'r2_atto'}, axis=1).reset_index().drop('station', axis=1)

_df_comb = pd.concat([_df1.set_index(['variable','data source']), _df2.set_index(['variable','data source'])], axis=1)

_df_comb

# %%
sns.relplot(x='r2_smr', 
            y='r2_atto', 
            hue='data source',
            style='variable',
            data = _df_comb.reset_index(),
            palette=pal,
            hue_order=hue_order,
            style_order=['OA','N50','N100','N200'],
            
           s=200)
plt.plot([0,1],[0,1], c='k', linestyle='--', alpha=.4, zorder=-100)
plt.xlim([0,1])
plt.ylim([0,1])

# %%
sns.relplot(x='r2_smr', 
            y='r2_atto', 
            hue='data source',
            col='variable',
            data = _df_comb.reset_index(),
            palette=pal,
            hue_order=hue_order,
            col_order=['OA','N50','N100','N200'],
            
           s=200)

# %%
select_station = 'SMR'

if select_station=='SMR':
    dic_df_med = dic_df_med_SMR
    xlims = [5,30]
    season = 'JA'

    
elif select_station=='ATTO':
    dic_df_med = dic_df_med_ATTO
    xlims = [20,40]
    season = 'FMA'
    
## Settings
alpha_scatt = 0.5

figsize=[7,10]
xlab = r'T  [$^\circ$C]'
ylab = r'$\Delta$OA [$\mu m^{-3}$]'


#ylims = [1,700]

# OBS: 
v_y = 'OA'
v_x = 'T_C'


xscale='linear'
yscale='linear'

fig, axs = plt.subplots(len(models_and_obs), sharex=True, sharey= False, figsize=figsize)

## Settings
alpha_scatt = 0.6


for mo, ax in zip(models_and_obs, axs):
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    
    #popt, pov, label, func, out = get_odr_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True, return_out_obj=True)
    #popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    #popt, pov, label, func = get_least_square_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True)
    popt, pov, label, func, func_lin, popt_lin  = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])

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
    
fn = make_fn_scat(f'residual_exp_{season}_{select_station}', v_x, v_y)
ax.legend(frameon=False)
fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)
print(fn)

# %%
select_station = 'ATTO'

if select_station=='SMR':
    dic_df_med = dic_df_med_SMR
    xlims = [5,30]
    season = 'JA'

    
elif select_station=='ATTO':
    dic_df_med = dic_df_med_ATTO
    xlims = [20,40]
    season = 'FMA'
    
## Settings
alpha_scatt = 0.5

figsize=[7,10]
xlab = r'T  [$^\circ$C]'
ylab = r'$\Delta$OA [$\mu m^{-3}$]'


#ylims = [1,700]

# OBS: 
v_y = 'OA'
v_x = 'T_C'


xscale='linear'
yscale='linear'

fig, axs = plt.subplots(len(models_and_obs), sharex=True, sharey= False, figsize=figsize)

## Settings
alpha_scatt = 0.6


for mo, ax in zip(models_and_obs, axs):
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    
    #popt, pov, label, func, out = get_odr_fit_and_labs(df_s,v_x,v_y, fit_func='linear', return_func=True, return_out_obj=True)
    #popt, pov, label, func = get_log_fit_abc(df_s,v_x,v_y, return_func=True)
    #popt, pov, label, func = get_least_square_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True)
    popt, pov, label, func  , func_lin, popt_lin = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])

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
    
fn = make_fn_scat(f'residual_exp_{season}_{select_station}', v_x, v_y)
ax.legend(frameon=False)
fig.savefig(fn, dpi=150)
fig.savefig(fn.with_suffix('.pdf'), dpi=150)
print(fn)

# %%

# %% [markdown]
# ## JFMAM

# %%
fig_main = plt.figure(constrained_layout=True,
                  figsize=[17, 8.25],
                 )
spec2 = gridspec.GridSpec( nrows=2,ncols=3, 
                          height_ratios=[1,30], 
                          width_ratios=[30,1,30], 
                          figure=fig_main)

subfig_smr =  fig_main.add_subfigure(spec2[1, 0],frameon=True)
subfig_atto =  fig_main.add_subfigure(spec2[1, 2],frameon=True)
subfig_up_smr =  fig_main.add_subfigure(spec2[0, 0])
subfig_up_atto =  fig_main.add_subfigure(spec2[0, 2])



dic_fits = {}
dic_fits['SMR'] =dict()
dic_fits['ATTO'] =dict()



season_atto = 'JFMAM'
season_smr = 'JA'
markersize = 4





fig, ax, daxs, axs_extra = make_cool_grid5(fig =subfig_smr )##ncols_extra=2, nrows_extra=2,)# w_ratio_sideplot=.5)

axs_extra = axs_extra.flatten()[::-1]








dic_df_med = dic_df_med_SMR
select_station = 'SMR'
dic_fits[select_station] = dict()
dic_fits[select_station][v_y] = dict()

## Settings
alpha_scatt = 0.6

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


linewidth=2
xlims = [5,30]
ylims = [.1,35]


season=season_smr
v_x = 'T_C'
v_y = 'OA'
dic_fits[select_station][v_y] = dict()


make_plot(v_x, v_y, xlims, ylims, season, 
              xlab, ylab, .3, models_and_obs, fig, ax, daxs, axs_extra,
          yscale='log',
          dic_df_med = dic_df_med,
          markersize=markersize

         )


for mo, ax_ex in zip(models_and_obs, axs_extra[:]):
    print(f'******{mo}*********')
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    #popt, pov, label, func = get_least_square_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    #popt, pov, label, func = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    popt, pov, label, func,func_lin, popt_lin = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])

    dic_fits[select_station][v_y][mo] = dict()
    dic_fits[select_station][v_y][mo]['label'] = label
    dic_fits[select_station][v_y][mo]['popt'] = popt
    dic_fits[select_station][v_y][mo]['pcov'] = pov
    dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))

    dic_fits[select_station][v_y][mo]['func'] = func
    dic_fits[select_station][v_y][mo]['R2'] = get_r2_logy(df_s,v_x,v_y, popt_lin, func_lin)
    dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
    print('R2*****')
    print(dic_fits[select_station][v_y][mo]['R2'])
    print(dic_fits[select_station][v_y][mo]['corr'])
    dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])   

    
    
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
ax.grid(color='grey', linewidth=.5, linestyle=':')

for ax_ex in axs_extra:
    ax_ex.set_yticklabels([])
    ax_ex.grid(color='grey', linewidth=.5, linestyle=':')
    ax_ex.set_ylim(ylims)
    ax_ex.set_xlim(xlims)

    
fn = make_fn_scat(f'exp1_{season}', v_x, v_y)
ax.legend(frameon=False)




## ATTO:
select_station = 'ATTO'
dic_fits[select_station] = dict()
dic_fits[select_station][v_y] = dict()


xlims = [20,40]



fig, ax, daxs, axs_extra = make_cool_grid5(fig =subfig_atto )##ncols_extra=2, nrows_extra=2,)# w_ratio_sideplot=.5)

axs_extra = axs_extra.flatten()[::-1]

dic_df_med = dic_df_med_ATTO


## Settings
alpha_scatt = 0.6


linewidth=2
xlims = [20,40]


season = season_atto


make_plot(v_x, v_y, xlims, ylims, season, 
              xlab, ylab, .3, models_and_obs, fig, ax, daxs, axs_extra,
          yscale='log',
          dic_df_med = dic_df_med,
          markersize=markersize
         
         )


for mo, ax_ex in zip(models_and_obs, axs_extra[:]):
    print(f'******{mo}********')
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    
    #popt, pov, label, func = get_least_square_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    #popt, pov, label, func = get_odr_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    popt, pov, label, func, func_lin, popt_lin = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
        
    dic_fits[select_station][v_y][mo] = dict()
    dic_fits[select_station][v_y][mo]['label'] = label
    dic_fits[select_station][v_y][mo]['popt'] = popt
    dic_fits[select_station][v_y][mo]['pcov'] = pov
    dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))

    dic_fits[select_station][v_y][mo]['func'] = func
    dic_fits[select_station][v_y][mo]['R2'] = get_r2_logy(df_s,v_x,v_y, popt_lin, func_lin)
    dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
    print('R2*****')
    print(dic_fits[select_station][v_y][mo]['R2'])
    print(dic_fits[select_station][v_y][mo]['corr'])
    
    dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])   

    
    
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
    ax_ex.set_ylim(ylims)
    ax_ex.set_xlim(xlims)
ax.set_yscale('log')

ax.grid(color='grey', linewidth=.5, linestyle=':')

ax.set_xticks(np.arange(20,41,5))

for ax_ex in axs_extra.flatten():
    ax_ex.set_yticklabels([])
    ax_ex.set_xticks(np.arange(20,41,5))
    ax_ex.grid(color='grey', linewidth=.5, linestyle=':')

ax.legend(frameon=False)











ax_up_smr = subfig_up_smr.subplots()
ax_up_smr.axis('off')
subfig_up_smr.suptitle('SMEARII: July, Aug',size=16,)
ax_up_atto = subfig_up_atto.subplots()
ax_up_atto.axis('off')
#subfig_up_atto.suptitle('ATTO: Feb, Mar, Apr, May',size=16,)
subfig_up_atto.suptitle(f'ATTO: {dic_season_nicename[season_atto]}',size=16,)

fn = make_fn_scat(f'together_{season_smr}_{season_atto}', v_x, v_y)
print(fn)
fig_main.savefig(fn.with_suffix('.pdf'), dpi=200)
fig_main.savefig(fn.with_suffix('.png'), dpi=200)
plt.show()

df = make_pd_of_dic(dic_fits)
df.to_csv(fn.with_suffix('.csv'))


# %% [markdown]
# ## JFM

# %%
fig_main = plt.figure(constrained_layout=True,
                  figsize=[17, 8.25],
                 )
spec2 = gridspec.GridSpec( nrows=2,ncols=3, 
                          height_ratios=[1,30], 
                          width_ratios=[30,1,30], 
                          figure=fig_main)

subfig_smr =  fig_main.add_subfigure(spec2[1, 0],frameon=True)
subfig_atto =  fig_main.add_subfigure(spec2[1, 2],frameon=True)
subfig_up_smr =  fig_main.add_subfigure(spec2[0, 0])
subfig_up_atto =  fig_main.add_subfigure(spec2[0, 2])



dic_fits = {}
dic_fits['SMR'] =dict()
dic_fits['ATTO'] =dict()



season_atto = 'JFM'
season_smr = 'JA'





fig, ax, daxs, axs_extra = make_cool_grid5(fig =subfig_smr )##ncols_extra=2, nrows_extra=2,)# w_ratio_sideplot=.5)

axs_extra = axs_extra.flatten()[::-1]








dic_df_med = dic_df_med_SMR
select_station = 'SMR'
dic_fits[select_station] = dict()
dic_fits[select_station][v_y] = dict()

## Settings
alpha_scatt = 0.6

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


linewidth=2
xlims = [5,30]
ylims = [.1,35]


season=season_smr
v_x = 'T_C'
v_y = 'OA'
dic_fits[select_station][v_y] = dict()


make_plot(v_x, v_y, xlims, ylims, season, 
              xlab, ylab, .3, models_and_obs, fig, ax, daxs, axs_extra,
          yscale='log',
          dic_df_med = dic_df_med ,
          markersize=markersize
          
         )


for mo, ax_ex in zip(models_and_obs, axs_extra[:]):
    print(f'******{mo}*********')
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    #popt, pov, label, func = get_least_square_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    #popt, pov, label, func = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    popt, pov, label, func,func_lin, popt_lin = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])

    dic_fits[select_station][v_y][mo] = dict()
    dic_fits[select_station][v_y][mo]['label'] = label
    dic_fits[select_station][v_y][mo]['popt'] = popt
    dic_fits[select_station][v_y][mo]['pcov'] = pov
    dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))

    dic_fits[select_station][v_y][mo]['func'] = func
    dic_fits[select_station][v_y][mo]['R2'] = get_r2_logy(df_s,v_x,v_y, popt_lin, func_lin)
    dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
    print('R2*****')
    print(dic_fits[select_station][v_y][mo]['corr'])
    dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])   

    
    
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
ax.grid(color='grey', linewidth=.5, linestyle=':')

for ax_ex in axs_extra:
    ax_ex.set_yticklabels([])
    ax_ex.grid(color='grey', linewidth=.5, linestyle=':')
    ax_ex.set_ylim(ylims)
    ax_ex.set_xlim(xlims)

    
fn = make_fn_scat(f'exp1_{season}', v_x, v_y)
ax.legend(frameon=False)




## ATTO:
select_station = 'ATTO'
dic_fits[select_station] = dict()
dic_fits[select_station][v_y] = dict()


xlims = [20,40]



fig, ax, daxs, axs_extra = make_cool_grid5(fig =subfig_atto )##ncols_extra=2, nrows_extra=2,)# w_ratio_sideplot=.5)

axs_extra = axs_extra.flatten()[::-1]

dic_df_med = dic_df_med_ATTO


## Settings
alpha_scatt = 0.6


linewidth=2
xlims = [20,40]


season = season_atto


make_plot(v_x, v_y, xlims, ylims, season, 
              xlab, ylab, .3, models_and_obs, fig, ax, daxs, axs_extra,
          yscale='log',
          dic_df_med = dic_df_med,
          markersize=markersize
          
         
         )


for mo, ax_ex in zip(models_and_obs, axs_extra[:]):
    print(f'******{mo}********')
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    
    #popt, pov, label, func = get_least_square_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    #popt, pov, label, func = get_odr_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    popt, pov, label, func, func_lin, popt_lin = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
        
    dic_fits[select_station][v_y][mo] = dict()
    dic_fits[select_station][v_y][mo]['label'] = label
    dic_fits[select_station][v_y][mo]['popt'] = popt
    dic_fits[select_station][v_y][mo]['pcov'] = pov
    dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))

    dic_fits[select_station][v_y][mo]['func'] = func
    dic_fits[select_station][v_y][mo]['R2'] = get_r2_logy(df_s,v_x,v_y, popt_lin, func_lin)
    dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
    print('R2*****')
    print(dic_fits[select_station][v_y][mo]['R2'])
    print(dic_fits[select_station][v_y][mo]['corr'])
    
    dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])   

    
    
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
    ax_ex.set_ylim(ylims)
    ax_ex.set_xlim(xlims)
ax.set_yscale('log')
ax.grid(color='grey', linewidth=.5, linestyle=':')

ax.set_xticks(np.arange(20,41,5))

for ax_ex in axs_extra.flatten():
    ax_ex.set_yticklabels([])
    ax_ex.set_xticks(np.arange(20,41,5))
    ax_ex.grid(color='grey', linewidth=.5, linestyle=':')

    
ax.legend(frameon=False)











ax_up_smr = subfig_up_smr.subplots()
ax_up_smr.axis('off')
subfig_up_smr.suptitle('SMEARII: July, Aug',size=16,)
ax_up_atto = subfig_up_atto.subplots()
ax_up_atto.axis('off')
#subfig_up_atto.suptitle('ATTO: Feb, Mar, Apr, May',size=16,)
subfig_up_atto.suptitle(f'ATTO: {dic_season_nicename[season_atto]}',size=16,)

fn = make_fn_scat(f'together_{season_smr}_{season_atto}', v_x, v_y)
print(fn)
fig_main.savefig(fn.with_suffix('.pdf'), dpi=200)
fig_main.savefig(fn.with_suffix('.png'), dpi=200)
plt.show()

df = make_pd_of_dic(dic_fits)
df.to_csv(fn.with_suffix('.csv'))


# %% [markdown]
# ## MAM

# %%
fig_main = plt.figure(constrained_layout=True,
                  figsize=[17, 8.25],
                 )
spec2 = gridspec.GridSpec( nrows=2,ncols=3, 
                          height_ratios=[1,30], 
                          width_ratios=[30,1,30], 
                          figure=fig_main)

subfig_smr =  fig_main.add_subfigure(spec2[1, 0],frameon=True)
subfig_atto =  fig_main.add_subfigure(spec2[1, 2],frameon=True)
subfig_up_smr =  fig_main.add_subfigure(spec2[0, 0])
subfig_up_atto =  fig_main.add_subfigure(spec2[0, 2])



dic_fits = {}
dic_fits['SMR'] =dict()
dic_fits['ATTO'] =dict()



season_atto = 'MAM'
season_smr = 'JA'





fig, ax, daxs, axs_extra = make_cool_grid5(fig =subfig_smr )##ncols_extra=2, nrows_extra=2,)# w_ratio_sideplot=.5)

axs_extra = axs_extra.flatten()[::-1]








dic_df_med = dic_df_med_SMR
select_station = 'SMR'
dic_fits[select_station] = dict()
dic_fits[select_station][v_y] = dict()

## Settings
alpha_scatt = 0.6

xlab = r'T  [$^\circ$C]'
ylab = r'OA [$\mu g m^{-3}$]'


linewidth=2
xlims = [5,30]
ylims = [.1,35]


season=season_smr
v_x = 'T_C'
v_y = 'OA'
dic_fits[select_station][v_y] = dict()


make_plot(v_x, v_y, xlims, ylims, season, 
              xlab, ylab, .3, models_and_obs, fig, ax, daxs, axs_extra,
          yscale='log',
          dic_df_med = dic_df_med         ,
          markersize = markersize,
         )


for mo, ax_ex in zip(models_and_obs, axs_extra[:]):
    print(f'******{mo}*********')
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    #popt, pov, label, func = get_least_square_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    #popt, pov, label, func = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    popt, pov, label, func,func_lin, popt_lin = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])

    dic_fits[select_station][v_y][mo] = dict()
    dic_fits[select_station][v_y][mo]['label'] = label
    dic_fits[select_station][v_y][mo]['popt'] = popt
    dic_fits[select_station][v_y][mo]['pcov'] = pov
    dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))

    dic_fits[select_station][v_y][mo]['func'] = func
    dic_fits[select_station][v_y][mo]['R2'] = get_r2_logy(df_s,v_x,v_y, popt_lin, func_lin)
    dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
    print('R2*****')
    print(dic_fits[select_station][v_y][mo]['R2'])
    print(dic_fits[select_station][v_y][mo]['corr'])
    dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])   

    
    
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
ax.grid(color='grey', linewidth=.5, linestyle=':')

for ax_ex in axs_extra:
    ax_ex.set_yticklabels([])
    ax_ex.grid(color='grey', linewidth=.5, linestyle=':')
    ax_ex.set_ylim(ylims)
    ax_ex.set_xlim(xlims)

    
fn = make_fn_scat(f'exp1_{season}', v_x, v_y)
ax.legend(frameon=False)




## ATTO:
select_station = 'ATTO'
dic_fits[select_station] = dict()
dic_fits[select_station][v_y] = dict()


xlims = [20,40]



fig, ax, daxs, axs_extra = make_cool_grid5(fig =subfig_atto )##ncols_extra=2, nrows_extra=2,)# w_ratio_sideplot=.5)

axs_extra = axs_extra.flatten()[::-1]

dic_df_med = dic_df_med_ATTO


## Settings
alpha_scatt = 0.6


linewidth=2
xlims = [20,40]


season = season_atto


make_plot(v_x, v_y, xlims, ylims, season, 
              xlab, ylab, .3, models_and_obs, fig, ax, daxs, axs_extra,
          yscale='log',
          dic_df_med = dic_df_med,
          markersize=markersize,
         
         )


for mo, ax_ex in zip(models_and_obs, axs_extra[:]):
    print(f'******{mo}********')
    df_s =  dic_df_med[mo]
    print(mo)
    mask_months = select_months(df_s, season=season)
    df_s = df_s[mask_months].copy()
    
    #popt, pov, label, func = get_least_square_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    #popt, pov, label, func = get_odr_fit_and_labs(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
    popt, pov, label, func, func_lin, popt_lin = get_lin_log_fit(df_s, v_x, v_y, fit_func = 'exp', return_func=True, beta0=[0.01,.12])
        
    dic_fits[select_station][v_y][mo] = dict()
    dic_fits[select_station][v_y][mo]['label'] = label
    dic_fits[select_station][v_y][mo]['popt'] = popt
    dic_fits[select_station][v_y][mo]['pcov'] = pov
    dic_fits[select_station][v_y][mo]['standard_error'] = np.sqrt(np.diag(dic_fits[select_station][v_y][mo]['pcov']))

    dic_fits[select_station][v_y][mo]['func'] = func
    dic_fits[select_station][v_y][mo]['R2'] = get_r2_logy(df_s,v_x,v_y, popt_lin, func_lin)
    dic_fits[select_station][v_y][mo]['corr'] = get_corr(df_s,v_x,v_y)
    print('R2*****')
    print(dic_fits[select_station][v_y][mo]['R2'])
    print(dic_fits[select_station][v_y][mo]['corr'])
    
    dic_fits[select_station][v_y][mo] = calc_table_se(dic_fits[select_station][v_y][mo])   

    
    
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
    ax_ex.set_ylim(ylims)
    ax_ex.set_xlim(xlims)
ax.set_yscale('log')
ax.grid(color='grey', linewidth=.5, linestyle=':')

ax.set_xticks(np.arange(20,41,5))

for ax_ex in axs_extra.flatten():
    ax_ex.set_yticklabels([])
    ax_ex.set_xticks(np.arange(20,41,5))
    ax_ex.grid(color='grey', linewidth=.5, linestyle=':')

    
ax.legend(frameon=False)











ax_up_smr = subfig_up_smr.subplots()
ax_up_smr.axis('off')
subfig_up_smr.suptitle('SMEARII: July, Aug',size=16,)
ax_up_atto = subfig_up_atto.subplots()
ax_up_atto.axis('off')
#subfig_up_atto.suptitle('ATTO: Feb, Mar, Apr, May',size=16,)
subfig_up_atto.suptitle(f'ATTO: {dic_season_nicename[season_atto]}',size=16,)

fn = make_fn_scat(f'together_{season_smr}_{season_atto}', v_x, v_y)
print(fn)
fig_main.savefig(fn.with_suffix('.pdf'), dpi=200)
fig_main.savefig(fn.with_suffix('.png'), dpi=200)
plt.show()

df = make_pd_of_dic(dic_fits)
df.to_csv(fn.with_suffix('.csv'))


# %%

# %%
fig = plt.figure(constrained_layout=True, figsize=(10, 4),)
subfigs = fig.subfigures(1, 3, width_ratios=[4,1,4], wspace=0.07)


f1 = subfigs[0]
f1.subplots(3,5, sharex=True, sharey='row')
subfigs[2].subplots(3,5, sharex=True, sharey='row')
subfigs[1].subplots(3,1, sharex=True, sharey='row')
subfigs[0].set_facecolor('#e5f8f8')
subfigs[-1].set_facecolor('#fff4ea')
subfigs[0].suptitle('SMEARII, Jul & Aug')
subfigs[-1].suptitle('ATTO, JFM')
#subfigs[1].suptitle('Fits')
#fig.set_constrained_layout_pads(h_pad=1, w_pad = 1)
#fig.tight_layout()
plt.show()


# %%
fig, axs_all = plt.subplots(3,11,figsize=[20,10], sharey='row', sharex='col')

for ax in axs_all[:,:5].flatten():
    ax.set(facecolor = "#fff4ea",)#a4c8d6")
    #ax.patch.set_facecolor('orange')
for ax in axs_all[:,6:].flatten():
    ax.set(facecolor = "#e5f8f8",)#a4c8d6")
    #ax.patch.set_facecolor('orange')


# %%
fig, axs_all = plt.subplots(3,11,figsize=[20,10], sharey='row', sharex='col')

for ax in axs_all[:,:5].flatten():
    ax.set(facecolor = "#d0d6c4",)#a4c8d6")
    #ax.patch.set_facecolor('orange')
for ax in axs_all[:,6:].flatten():
    ax.set(facecolor = "#f1f8e3",)#a4c8d6")
    #ax.patch.set_facecolor('orange')


# %%
fig, axs_all = plt.subplots(3,6,figsize=[10,10], sharey='row', sharex='col')


# %%

# %%

# %%
