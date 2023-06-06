from random import randint

import pandas as pd

from bs_fdbck_clean.constants import path_data_info
from bs_fdbck_clean.data_info import get_area_specs
import useful_scit.plot as ucp

path_case_overview = path_data_info / 'case_overview.csv'
case_overv = pd.read_csv(path_case_overview, index_col=0)


def get_case_col(case):
    if case in case_overv.index:
        if not pd.isnull(case_overv.loc[case, 'color']):
            return case_overv.loc[case, 'color']
    return ucp.cb[randint(0, len(ucp.cb))]


def get_area_col(area):
    df_area = get_area_specs(area)

    if df_area is not None:
        if not pd.isnull(df_area['color']):  # =='nan':#.isnan():
            return df_area['color']
    return ucp.cb[randint(0, len(ucp.cb))]
