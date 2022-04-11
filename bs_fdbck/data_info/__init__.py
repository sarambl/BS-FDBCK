from pathlib import Path

import pandas as pd

from bs_fdbck.constants import path_data_info

path_area_defs = Path(path_data_info) / 'area_defs.csv'
path_case_overview = Path(path_data_info) / 'case_overview.csv'
print(path_area_defs)


# %%
def get_area_defs_pd():
    df_area: pd.DataFrame = pd.read_csv(path_area_defs, index_col=0)
    return df_area

def get_area_specs(area):
    df_area = get_area_defs_pd()
    if area in df_area.index:
        return df_area.loc[area]
    else:
        print(f'{area} not found in {path_area_defs}')
        return None


def get_nice_name_case(case):
    df_ov: pd.DataFrame = pd.read_csv(path_case_overview, index_col=0)
    if case in df_ov.index:
        nice_name = df_ov.loc[case,'nice_name']
        if not pd.isna(nice_name):
            return nice_name
    return case

def get_nice_name_area(area):
    df = get_area_specs(area)
    if df['nice_name'] is not None:
        if not pd.isna(df['nice_name']):
            return df['nice_name']
    return area
