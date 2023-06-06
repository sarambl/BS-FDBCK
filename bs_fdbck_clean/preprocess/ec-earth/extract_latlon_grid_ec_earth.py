# %%
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import useful_scit.util.log as log

from bs_fdbck_clean.constants import get_input_datapath, path_extract_latlon_outdata
from bs_fdbck_clean.util.imports.import_fields_xr_ec_earth import get_file_subset_EC_Earth

log.ger.setLevel(log.log.INFO)

path_input_data = get_input_datapath()


log.ger.setLevel(log.log.INFO)

varlist_tm5 = [
#    'CCN0.20',
#    'CCN1.00',
#    'M_SO4NUS',
#    'M_SOANUS',
#    'M_BCAIS',
#    'M_POMAIS',
#    'M_SOAAIS',
#    'M_SO4ACS',
#    'M_BCACS',
#    'M_POMACS',
#    'M_SSACS',
#    'M_DUACS',
#    'M_SOAACS',
#    'M_SO4COS',
#    'M_BCCOS',
#    'M_POMCOS',
#    'M_SSCOS',
#    'M_DUCOS',
#    'M_SOACOS',
#    'M_BCAII',
#    'M_POMAII',
#    'M_SOAAII',
#    'M_DUACI',
#    'M_DUCOI',
#    'N_NUS',
#    'N_AIS',
#    'N_ACS',
#    'N_COS',
#    'N_AII',
#    'N_ACI',
#    'N_COI',
#    #    'GAS_O3',
#    #    'GAS_SO2',
#    #    'GAS_TERP',
#    #    'GAS_OH',
#    #    'GAS_ISOP',
#    'RWET_NUS',
#    'RWET_AIS',
#    'RWET_ACS',
#    'RWET_COS',
#    'RWET_AII',
#    'RWET_ACI',
#    'RWET_COI',
#    'RDRY_NUS',
#    'RDRY_AIS',
#    'RDRY_ACS',
#    'RDRY_COS',
#    #    'loadoa',
#    'od550aer',
#    'od550oa',
#    'od550soa',
#    'od440aer',
#    'od870aer',
#    'od350aer',
#    'loadsoa',
#    'emiterp',
#    'emiisop'
]
varlist_ifs_gg = [
    'var176',
    'var177',
    'var178',
    'var179',
    'var208',
    'var210',
    'var211',
    'var68',
    'var69',
    'var70',
    'var71',
    'var72',
    'var73',
    'var74',
    'var75',
    'var130',
    'var131',
    'var132',
    'var248',
    'var20',
    'var21',
    'var22',
    'var78',
    'var79',
    'var136',
    'var137',
    'var164'
]
varlist_ifs_t =[
    'var130',
]




varl = varlist_ifs_gg




# %%

def convert_lon_to_180(lon):
    return (lon + 180) % 360 - 180

def convert_lon_to_360(lon):
    return lon % 360


# %%


def check_nrproc(l_df: pd.DataFrame):
    _b = l_df['process'].isnull()
    return len(l_df[~_b])


def check_stat_proc(l_df):
    """
    Check nr of processes done/running or not running
    :param l_df:
    :return:
    """
    nrunning = l_df['status'][l_df['status'] == 'running'].count()
    ndone = l_df['status'][l_df['status'] == 'done'].count()
    nnrunning = l_df['status'][l_df['status'] == 'not running'].count()
    return nrunning, ndone, nnrunning


# %%
def update_stat_proc(r):
    """
    Update row if process is done
    :param r:
    :return:
    """
    if r['status'] == 'done':
        return 'done'
    if r['status'] == 'not running':
        return 'not running'
    else:
        p = r['process']
        p: subprocess.Popen
        if p.poll() is None:
            # (stdout_data, stderr_data) = p.communicate()
            # print('****')
            # print((stdout_data, stderr_data))
            # print('****')

            return 'running'

        else:
            return 'done'


# %%
def launch_ncks(comms, max_launches=5):
    """
    Launch a number of processes to calculate monthly files for file.
    :param comms: list of commands to run
    :param max_launches: maximum launched subprocesses at a time
    :return:
    """
    if len(comms) == 0:
        return
    # Setup dataframe to keep track of processes

    l_df = pd.DataFrame(index=comms, columns=['process', 'status'])
    l_df['status'] = 'not running'
    l_df['status'] = l_df.apply(update_stat_proc, axis=1)
    check_stat_proc(l_df)
    # pyf = sys.executable  # "/persistent01/miniconda3/envs/env_sec_v2/bin/python3"
    # file = package_base_path / 'bs_fdbck_clean'/'preprocess'/'noresm'/'subproc_station_output.py'
    # while loop:
    # mod_load  ='module load NCO/4.7.9-nsc5 &&'
    notdone = True
    while notdone:
        # Update status
        l_df['status'] = l_df.apply(update_stat_proc, axis=1)
        nrunning, ndone, nnrunning = check_stat_proc(l_df)
        # check if done, if so break
        notdone = len(l_df) != ndone
        if notdone is False:
            break
        # If room for one more process:
        if (nrunning < max_launches) and (nnrunning > 0):
            co = l_df[l_df['status'] == 'not running'].iloc[0].name
            print(co)
            # co.comm

            # Launch subprocess:
            p1 = subprocess.Popen([co], shell=True)
            # put subprocess in dataframe
            l_df.loc[co, 'process'] = p1
            l_df.loc[co, 'status'] = 'running'

        log.ger.info(l_df)
        time.sleep(5)


# %%

def extract_subset(case_name='ECE3_output_Sara',
                   from_time='2012-01-01',
                       to_time='2019-01-01',
                   which = 'IFS',
                   station='SMR',
                   model_name='EC-Earth',
                   lat_lims=None, lon_lims=None, out_folder=None, tmp_folder=None, history_field='.h1.', ):
    # %%

    if lat_lims is None:
        if station=='SMR':
            lat_lims = [60., 66.]
        elif station=='ATTO':
            lat_lims = [-8.,-1.]
    if lon_lims is None:
        if station=='SMR':
            lon_lims = [22., 30.]
        elif station == 'ATTO':
            lon_lims = [-67.,-52.]

    print(lat_lims)
    print(lon_lims)
    if which=='TM5':
        lon_lims = [convert_lon_to_180(lon_lims[0]), convert_lon_to_180(lon_lims[1])]
    else:
        lon_lims = [convert_lon_to_360(lon_lims[0]), convert_lon_to_360(lon_lims[1])]
    if out_folder is None:
        out_folder = Path(path_extract_latlon_outdata) / model_name / case_name
    if tmp_folder is None:
        tmp_folder = out_folder / 'tmp'

    if not out_folder.exists():
        out_folder.mkdir(parents=True)
    if not tmp_folder.exists():
        tmp_folder.mkdir(parents=True)
    path_input_data = get_input_datapath(model=model_name)
    input_folder = path_input_data
    # %%
    print(f'case_name: {case_name} \n from time: {from_time} \n to_time: {to_time} \n'
          f' lat_lims: {lat_lims} \n lon_lims_ {lon_lims} \n '
          f'out_folder: {str(out_folder)} \n tmp_folder: {str(tmp_folder)} \n'
          f'input_folder: {input_folder}'
          )
    # %%

    # %%
    #p = input_folder.glob(f'**/*{history_field}*')

    #files = [x for x in p if x.is_file()]
    #files.sort()
    #files = pd.Series(files)
    #print(files)

    # %%

    #files.sort()
    #files = pd.Series(files)
    #print(files)

    # %%
    fl_TM5 = list(input_folder.glob('*TM5*.nc'))
    fl_IFS_T = list(input_folder.glob('T_IFS*.nc'))
    fl_IFS_GG = list(input_folder.glob('IFS_GG*.nc'))
    fl_TM5.sort()
    fl_IFS_T.sort()
    fl_IFS_GG.sort()
    fl_TM5 = get_file_subset_EC_Earth(fl_TM5, from_time, to_time)
    fl_IFS_T = get_file_subset_EC_Earth(fl_IFS_T, from_time, to_time)
    fl_IFS_GG = get_file_subset_EC_Earth(fl_IFS_GG, from_time, to_time)
    if which == 'TM5':
        files = fl_TM5
    elif which == 'IFS':
        files = fl_IFS_GG # + fl_IFS_T
    elif which == 'IFS_T':
        files = fl_IFS_T

    else:
        files = fl_IFS_GG
    # %%
    for f in files:
        print(f)
        print(f.stem)
    # %%
    print(files)

    try:
        subprocess.run('module load NCO/4.7.9-nsc5', shell=True)
    except FileNotFoundError:
        print('could not load NCO')
    comms = []
    files_out = list()
    for f in files:
        print(f)
        f_s = f.stem
        fn_o = f_s + f'_{which}_{station}_tmp_subset.nc'
        fp_o = tmp_folder / fn_o
        files_out.append(fp_o)

        if fp_o.exists():
            size = fp_o.stat().st_size
            print(size)
            if size > 1e3:
                continue
        co = f'ncks -O -d lon,{lon_lims[0]},{lon_lims[1]} -d lat,{lat_lims[0]},{lat_lims[1]} {f} {fp_o}'
        # -v u10max,v10max
        comms.append(co)
    # %%
    for co in comms:
        print(co)
        #subprocess.run(co, shell=True)

    # %%

    #return
    launch_ncks(comms, max_launches=5)
    print('done')
    # %%
    # %%
    fn_out_final = out_folder / f'{case_name}_{which}_{from_time}-{to_time}_concat_subs_{lon_lims[0]}' \
                                f'-{lon_lims[1]}_{lat_lims[0]}-{lat_lims[1]}.nc'
    files_str = ''
    # fn_out =tmp_folder /f'{fn_out_final.stem}_tmp.nc'
    f = files_out[0]
    #for f in files_out:
    #    files_str += f' {f} '
        # %%
    # f = Path('/uno/dos/tres')
    # case_name='case_name'
    files_str_patt = f'{f.parent}/*_{which}_{station}_tmp_subset.nc'
    # files_str_patt
    # %%
    print(files_str_patt)
    com_concat = f'ncrcat {files_str_patt} {fn_out_final}'
    print(com_concat)
    subprocess.run(com_concat, shell=True)


# %%


if __name__ == '__main__':
    extract_subset(*sys.argv[1:])

