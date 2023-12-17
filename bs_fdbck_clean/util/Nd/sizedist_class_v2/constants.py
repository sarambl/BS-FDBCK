from bs_fdbck_clean.constants import path_data_info
import pandas as pd

path_Nd_var_name = path_data_info / 'Nd_bins.csv'

diameter_obs_df = pd.read_csv(path_Nd_var_name, index_col=0)

# %%
_diams = ['N25','N30','N50','N70', 'N60', 'N100', 'N150', 'N200', 'N250', 'N500']

diameters_observation = {d: list(diameter_obs_df.loc[d, ['from_diameter', 'to_diameter']].values) for d in _diams}
#    {'N30-50':[30,50], 'N50':[50,500],'N100':[100,500], 'N250':[250,500]}
# %%
