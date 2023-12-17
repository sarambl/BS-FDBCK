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
from bs_fdbck_clean.constants import path_measurement_data
import matplotlib.pyplot as plt

# %%
import pandas as pd
import numpy as np
import xarray as xr
xr.set_options(keep_attrs=True)

# %%
import datetime 

# %%
# %load_ext autoreload
# %autoreload 2

# %%
path_acsm = path_measurement_data / 'ACSM_DEFAULT.mat'

# %%
fn_pres = path_measurement_data / 'SMEARII'/ 'smeardata_20230307_pressure.csv'
fn_rad = path_measurement_data / 'SMEARII'/ 'smeardata_20221116_radiation.csv'
fn_temp4m = path_measurement_data / 'SMEARII'/'smeardata_20230307_temp4m.csv'

# %%
fl_meteo = ['smeardata_20221116_2012-2014.csv', 'smeardata_20221116_2014-2016.csv', 'smeardata_20221116_2016-2018.csv',
      'smeardata_20221116_2018-2019.csv']
fl_meteo = [path_measurement_data/'SMEARII'/f for f in fl_meteo]


# %% [markdown]
# ### Output data

# %%
postproc_data = path_measurement_data /'SMEARII'/'processed'
postproc_data.mkdir( parents = True, exist_ok=True)

# %% tags=[]

path_comb_data =postproc_data /'SMEARII_data_comb_daily.nc'
path_comb_data_full_time =postproc_data /'SMEAR_data_comb_hourly.csv'

# %% [markdown]
# ### Read in acsm data

# %% [markdown]
# ### Input data:
#

# %%
path_measurement_data

# %%
f = '/proj/bolinc/users/x_sarbl/analysis/BS-FDBCK/Data/ACSM_DEFAULT.mat'

# %% [markdown]
# ## Partilces 

# %%
xlist = [50, 80, 100, 150, 200, 250, 300]

# %% [markdown]
# DMPS data is downloaded from EBAS

# %%
_f = '/proj/bolinc/users/x_sarbl/analysis/BS-FDBCK/Data/EBAS/raw_data/SMR/FI0050R.20120101000000.20181205100800.dmps.particle_number_size_distribution.pm10.1y.1h.FI03L_UHEL_DMPS_HYY_01.FI03L__TRY_TDMPS.lev2.nc'

# %% [markdown]
# Data is downloaded from EBAS: 

# %% [markdown]
# https://ebas-data.nilu.no/DataSets.aspx?stations=FI0050R&nations=FI246FIN&InstrumentTypes=dmps&components=particle_number_size_distribution&fromDate=1970-01-01&toDate=2023-12-31

# %%
from bs_fdbck_clean.util.EBAS_data import get_ebas_dataset_Nx_daily_JA_median_df, get_ebas_dataset_with_Nx,get_station_ebas_data

ds_sizedist = get_station_ebas_data(station='SMR')

#ds_ebas_Nx = get_ebas_dataset_with_Nx()
ds_ebas_Nx = get_ebas_dataset_with_Nx(x_list=xlist, station = 'SMR', ds = ds_sizedist)#x_list = [90,100,110,120])
ds_ebas_Nx['time'].attrs['timezone'] = 'UTC'


# %% [markdown] tags=[]
# ## Timezone local time eastern winter time EET UTC+2

# %%
import datetime

# %%
from bs_fdbck_clean.util.BSOA_datamanip import standard_air_density

# %%
if ds_ebas_Nx['time'].attrs['timezone']=='UTC':
    
    ds_ebas_Nx['time_utc'] = ds_ebas_Nx['time'].copy()
    ds_ebas_Nx['time'] = pd.to_datetime(ds_ebas_Nx['time_utc']) + datetime.timedelta(hours=2)
    ds_ebas_Nx['time'].attrs = ds_ebas_Nx['time_utc'].attrs.copy()
    ds_ebas_Nx['time']=  ds_ebas_Nx['time'].assign_attrs(timezone='EEST')

# %%
ds_Nx_hourly = ds_ebas_Nx.resample(time='1h').median()[[f'N{n}' for n in xlist]]

ds_Nx_hourly_mean = ds_ebas_Nx.resample(time='1h').mean()[[f'N{n}' for n in xlist]]

df_ebas_Nx  = ds_Nx_hourly_mean.to_dataframe()
df_ebas_Nx

# %%
import pandas as pd


# %%
from bs_fdbck_clean.constants import path_measurement_data

# %%
fn = path_measurement_data / 'SourceData_Yli_Juuti2021.xls'

df_hyy_1 = pd.read_excel(fn, sheet_name=0, header=2, usecols=range(6))

df_hyy_1.head()

df_hyy_1['date'] = df_hyy_1.apply(lambda x: f'{x.year:.0f}-{x.month:02.0f}-{x.day:02.0f}', axis=1)

df_hyy_1['date'] = pd.to_datetime(df_hyy_1['date'] )



# %%

df_hyy_1 = df_hyy_1.set_index('date')

# %%
df_hyy_1.index = df_hyy_1.index.rename('time') 

# %%
R = 287.058
df_hyy_1['rho'] = 1e5/(R*(df_hyy_1['T (degree C)']+273.15))
df_hyy_1['N100 (cm^-3),STP'] = df_hyy_1['N100 (cm^-3)']*standard_air_density/df_hyy_1['rho']


# %%
import matplotlib.pyplot as plt

# %% [markdown] tags=[]
# ## Check integration against Yli-Juuti et al 

# %%

df_joint_hyy = pd.merge(df_ebas_Nx.resample('1D').median(), df_hyy_1, left_index=True, right_index=True)
(df_joint_hyy['N100']).loc['2014-07':'2014-09'].plot(label='mine')
(df_joint_hyy['N100 (cm^-3),STP']).loc['2014-07':'2014-09'].plot(label='orig')
plt.legend()
plt.show()



print(df_joint_hyy['N100'][df_joint_hyy['N100 (cm^-3)'].notnull()].mean()/df_joint_hyy['N100 (cm^-3)'].mean())
# %% [markdown] tags=[]
# ## Meteo data

# %% [markdown]
# Already in local time EET (UTC+2)

# %%
df_rad = pd.read_csv(fn_rad)

# %%
df_rad['time'] = pd.to_datetime(df_rad[['Year', 'Month', 'Day', 'Hour','Minute']].apply(lambda s : datetime.datetime(*s),axis = 1))

# %%
_df_june = df_rad[df_rad.Month==1]
_df=_df_june.groupby('Hour').mean()
_df['HYY_META.Glob'].plot()
plt.xticks(range(0,23));
plt.grid()
#plt.ylim([0,4])

# %%
def compute_u_v(ws, theta):
    theta_rad = theta/360*2*np.pi
    u = ws*np.cos(theta_rad)
    v = ws*np.sin(theta_rad)
    return u, v

def comp_ws_theta(u,v):
    ws = np.sqrt(u**2+v**2)
    theta =(np.arctan2(v, u))%(2*np.pi)*360/2/np.pi#*360/(2*np.pi)
    return ws, theta
    


# %% [markdown]
# ## Average wind by hour: 

# %% [markdown]
# First compute U,V, then average, and then recompute direction and strength
#

# %%
def make_hourly_wind(df_met_hr, var_ws, var_theta):
    df_met_hr['U'], df_met_hr['V'] = compute_u_v(df_met_hr[var_ws],df_met_hr[var_theta]) 

    df_met_hour = fix_time_meteo(df_met_hr)
    
    df_met_hour[var_ws],df_met_hour[var_theta] = comp_ws_theta(df_met_hour['U'], df_met_hour['V'])

    return df_met_hour

def fix_time_meteo(df_met_hr):
    df_met_hr['time'] = pd.to_datetime(df_met_hr[['Year', 'Month', 'Day', 'Hour','Minute']].apply(lambda s : datetime.datetime(*s),axis = 1))

    df_met_hr = df_met_hr.set_index('time')

    df_met_hour = df_met_hr.resample('h').mean()
    return df_met_hour



# %%
var_ws = 'HYY_META.WSU168'
var_theta = 'HYY_META.WDU168'
var_ws2 = 'HYY_META.WSU672'
var_theta2 = 'HYY_META.WDU672'
df_ls = list()
for f in fl_meteo:
    df = pd.read_csv(f)
    df_hourly = make_hourly_wind(df, var_ws, var_theta)
    df_hourly2 = make_hourly_wind(df, var_ws2, var_theta2)
    for v in [var_ws2, var_theta2]:
        df_hourly[v] = df_hourly2[v]
    
    df_ls.append(df_hourly)

# %%
df_meteo= pd.concat(df_ls)

# %%
df_hourly['HYY_META.WDU168'].plot.hist()

# %%
bins_dir = np.linspace(0,360, 100)

df_meteo['WDU168_mid']= pd.cut(df_meteo['HYY_META.WDU168'], bins_dir).apply(lambda x: x.mid)

df_freq = df_meteo.groupby('WDU168_mid').count()

df_mean = df_meteo.groupby('WDU168_mid').mean()

df_comb = pd.DataFrame(index=df_freq.index)

df_comb['WDU168_freq'] = df_freq['HYY_META.WDU168']

df_comb['WDU168_mean'] = df_freq['HYY_META.WSU168']

bins_dir = np.linspace(0,360)

df_meteo['WSU168_mid']= pd.cut(df_meteo['HYY_META.WSU168'], bins_dir).apply(lambda x: x.mid)

import plotly.express as px

df = df_meteo
fig = px.bar_polar(df_comb.reset_index(), r='WDU168_freq', theta='WDU168_mid',
                   color="WDU168_mean", 
                   template="plotly_dark",
                   color_discrete_sequence= px.colors.sequential.Plasma_r)
fig.show()

# %% [markdown]
# ### Read in pressure:

# %%
df_meteo_pres = pd.read_csv(fn_pres)
df_meteo_temp4m = pd.read_csv(fn_temp4m)

df_meteo_pres = fix_time_meteo(df_meteo_pres)
df_meteo_temp4m = fix_time_meteo(df_meteo_temp4m)

# %%
df_meteo_pres

# %%
df_meteo_others = pd.merge(df_meteo_pres['HYY_META.Pamb0'],df_meteo_temp4m,right_index=True,left_index=True, how = 'outer')

# %%
df_meteo_others

# %%

df_meteo = pd.merge(df_meteo, df_meteo_others[['HYY_META.Pamb0','HYY_META.T42']], right_index=True, left_index=True)

# %%
df_meteo = df_meteo.rename({'HYY_META.Pamb0_y':'HYY_META.Pamb0'}, axis=1)

# %%

df_meteo['HYY_META.T42'].plot()

# %%

df_meteo['HYY_META.Pamb0'].plot()

# %%
from bs_fdbck_clean.constants import path_measurement_data
import pandas as pd


# %%
    
def timeround10(dt):
    a, b = divmod(round(dt.minute, -1), 60)
    tdelta = datetime.timedelta(hours = (dt.hour+a), minutes=b)
    nh = (dt.hour+a)%24
    ndt = datetime.datetime(dt.year,dt.month, dt.day,) + tdelta
    #dt_o = datetime.datetime(dt.year,dt.month, dt.day, (dt.hour + a) % 24,b)
    return ndt



def fix_matlabtime(t):
    ind = pd.to_datetime(t-719529, unit='D')
    ind_s = pd.Series(ind)
    return ind_s.apply(timeround10)
    
    


# %%
fn_liine = path_measurement_data / 'ACSM_DEFAULT.mat'

# %%
varlist_acsm = ['time', 'Org','SO4','NO3','NH4','Chl']

# %%
import scipy.io as sio
test = sio.loadmat(fn_liine)

df_lii = pd.DataFrame(test['ACSM_DEFAULT'], columns=varlist_acsm)#.set_index('time')

df_lii['time'] = fix_matlabtime(df_lii['time'])# + datetime.timedelta(hours=1)

df_lii = df_lii.set_index('time')

df_lii['Org'].plot()

# %% [markdown]
# ## Combine datasets

# %%
df_lii

# %%
df_ebas_Nx

# %%
df_meteo

# %%
df_lii['Org'].plot()

# %%
df_all = pd.concat([df_meteo, df_lii,df_ebas_Nx], axis=1)

# %%
df_all['Org'].plot()

# %%

# %% [markdown] tags=[]
# ## Correct ACSM data to be standard temperature and pressure:

# %% [markdown]
# Converting by: 
#
# \begin{align}
# conc. OA_{amb}=\frac{m_{OA}}{V_{amb}} = & \frac{m_{OA}}{m_{air}} \cdot \frac{m_{air}}{V_{amb}} = w_{OA} \cdot \rho_{amb} 
# \end{align}
# and in the same way
# \begin{align}
# conc. OA_{STP}= & \frac{m_{OA}}{m_{air}} \cdot \frac{m_{air}}{V_{STP}} = w_{OA} \cdot \rho_{STP}
# \end{align}
#
# So finally
# \begin{align}
# conc. OA_{STP}= & conc. OA_{amb} \cdot \frac{\rho_{STP}}{\rho_{amb}}
# \end{align}

# %%
pressure = df_all['HYY_META.Pamb0']
temperature = df_all['HYY_META.T42']+273.15 # In Kelvin

# %%
R

# %%
df_all['density'] = (
    pressure*100   # hPa --> Pa
    /(temperature*R)
)

# %%
df_all['density'].plot()

# %%
df_all['Org'].plot()

# %%
standard_air_density

# %%
varlist_acsm

# %%

for v in varlist_acsm:
    if v not in df_all.columns: 
        continue
    # if already processes: 
    if f'{v}_amb' in df_all: 
        continue
    df_all = df_all.rename({v:f'{v}_amb'}, axis=1)
    df_all[f'{v}_STP'] = (df_all[f'{v}_amb']/df_all['density'])*standard_air_density


# %%

# %%
df_all['Org_STP'].plot()

df_all['Org_STP'].plot()

# %% [markdown]
# #### Check that we are not loosing values because of surface p and T

# %%
is_summer = df_all.index.month.isin([7,8])

# %%
df_all_summer = df_all[is_summer]

# %%
df_all_summer[['HYY_META.T42','HYY_META.Pamb0','Org_amb','HYY_META.T168']].dropna().count()

# %%
df_all_summer[['HYY_META.Pamb0','Org_amb','HYY_META.T168']].dropna().count()

# %%
df_all_summer[['Org_amb','HYY_META.T168']].dropna().count()

# %%
df_all_summer[['Org_amb']].dropna().count()

# %%
df_all[is_summer]['Org_STP'].plot.hist(bins=np.linspace(-2,15),alpha= .5)

df_all[is_summer&(df_all['Org_STP'].notnull())]['Org_amb'].plot.hist(bins=np.linspace(-2,15), alpha= .5)

# %% [markdown]
# ## Filter based on wind direction 

# %%
df_all['HYY_META.WDU168'].plot.hist(bins=100)


# %%
discard_wind = (df_all['HYY_META.WDU168']>=120) & (df_all['HYY_META.WDU168']<=140)

# %%
df_all = df_all[~discard_wind]

# %%
df_all['HYY_META.WDU168'].plot.hist(bins=100)


# %%
df_all['HYY_META.WDU672'].plot.hist()


# %%
bins_dir = np.linspace(0,360, 100)
_df_meteo = df_all.copy()
_df_meteo['WDU168_mid']= pd.cut(_df_meteo['HYY_META.WDU168'], bins_dir).apply(lambda x: x.mid)

df_freq = _df_meteo.groupby('WDU168_mid').count()

df_mean = _df_meteo.groupby('WDU168_mid').mean()

df_comb = pd.DataFrame(index=df_freq.index)

df_comb['WDU168_freq'] = df_freq['HYY_META.WDU168']

df_comb['WDU168_mean'] = df_freq['HYY_META.WSU168']

bins_dir = np.linspace(0,360)

_df_meteo['WSU168_mid']= pd.cut(_df_meteo['HYY_META.WSU168'], bins_dir).apply(lambda x: x.mid)

import plotly.express as px

df = _df_meteo
fig = px.bar_polar(df_comb.reset_index(), r='WDU168_freq', theta='WDU168_mid',
                   color="WDU168_mean", 
                   template="plotly_dark",
                   color_discrete_sequence= px.colors.sequential.Plasma_r)
fig.show()

# %% [markdown]
#

# %%

# %% [markdown]
# ## Write to file

# %%
df_all.to_csv(path_comb_data_full_time)

# %%
path_comb_data_full_time

# %%
