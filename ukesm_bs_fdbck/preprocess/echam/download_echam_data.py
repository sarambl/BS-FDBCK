
from ukesm_bs_fdbck.constants import path_data_info, raw_data_path_echam
import pandas as pd
import sys


from subprocess import run

path_download = raw_data_path_echam /'SALSA_BSOA_feedback'

path_var_echam = path_data_info / 'echam_info' / 'output_variables.csv'


df_vars = pd.read_csv(path_var_echam, index_col=[0])
df_vars

# %%



def download_all_data(from_time='2012-01', to_time='2019-01', *varl):
    # %%
    print('hey')
    # %%
    from_time = pd.to_datetime(from_time)
    to_time = pd.to_datetime(to_time)

    if len(varl)==0:# is None:
        varl = df_vars.index


    # %%
    #pd.Series()
    month_list = pd.date_range(from_time, to_time,freq='1M', closed='left')#.tolist()

    # %%
    comms = []
    for v in varl:
        for t in month_list:

            ym =t.strftime('%Y%m')
            fn = f'SALSA_BSOA_feedback_{ym}_{v}.nc'
            fn_download = path_download /fn

            print(fn)
            url = f'https://a3s.fi/2002032-BSOA-feedback-ECHAM-SALSA/{fn}'

            co =f'wget {url}'
            if not fn_download.exists():
                print(co)
                comms.append(co)
    # %%
    for co in comms:
        print(co)
        run(co, shell=True, cwd=path_download)
    return



if __name__ == '__main__':
    args = sys.argv
    print(f'arguemnts: {args}')
    if len(args)>1:

        download_all_data(*args[1:])
    else:

        download_all_data()



