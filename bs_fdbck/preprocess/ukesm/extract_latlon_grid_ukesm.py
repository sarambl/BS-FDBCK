# %%
from pathlib import Path

import xarray as xr

from bs_fdbck.constants import get_input_datapath, path_extract_latlon_outdata, path_data_info

import subprocess
import sys
import time

import pandas as pd
import useful_scit.util.log as log

from bs_fdbck.util.imports.import_fields_xr_ec_earth import get_file_subset_EC_Earth
from bs_fdbck.util.imports.import_fields_xr_ukesm import filelist_ukesm, get_pathlist_and_rndic_ukesm

log.ger.setLevel(log.log.INFO)

path_input_data = get_input_datapath()

log.ger.setLevel(log.log.INFO)

varl = [
    'Reff_2d_distrib_x_weight',
    'Reff_2d_x_weight_warm_cloud',
    'area_cloud_fraction_in_each_layer',
    'bulk_cloud_fraction_in_each_layer',
    'cloud_ice_content_after_ls_precip',
    'dry_rho',
    'frozen_cloud_fraction_in_each_layer',
    'liq_cloud_fraction_in_each_layer',
    'qcf',
    'qcl',
    'supercooled_liq_water_content',
    'weight_Reff_2d_distrib',
    'weight_Reff_2d',
    'cdnc_top_cloud_x_weight',
    'weight_of_cdnc_top_cloud',

]


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
    # file = package_base_path / 'bs_fdbck'/'preprocess'/'subproc_station_output.py'
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

def extract_subset(case_name='AEROCOMTRAJ',
                   from_time='2012-01-01',
                   to_time='2019-01-01',
                   station='SMR',
                   model_name='UKESM',
                   lat_lims=None, lon_lims=None, out_folder=None, tmp_folder=None, history_field='.h1.', ):
    # %%

    if lat_lims is None:
        if station == 'SMR':
            lat_lims = [60., 66.]
        elif station == 'ATTO':
            lat_lims = [-8., -1.]
    if lon_lims is None:
        if station == 'SMR':
            lon_lims = [22., 30.]
        elif station == 'ATTO':
            lon_lims = [-67., -52.]

    print(lat_lims)
    print(lon_lims)
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
    # p = input_folder.glob(f'**/*{history_field}*')

    # files = [x for x in p if x.is_file()]
    # files.sort()
    # files = pd.Series(files)
    # print(files)

    # %%

    # files.sort()
    # files = pd.Series(files)
    # print(files)

    # %%
    path_input_data = get_input_datapath(model=model_name) /  case_name  # path to files

    from_time_dt = pd.to_datetime(from_time)
    to_time_dt = pd.to_datetime(to_time)

    # %%
    fl, rename_dic, dic_varname2file = get_pathlist_and_rndic_ukesm(from_time_dt, path_input_data, to_time_dt, varl)
    # %%
    fl.sort()

    files = fl

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
        fn_o = f_s + f'_{station}_tmp_subset.nc'
        fp_o = tmp_folder / fn_o
        files_out.append(fp_o)

        if fp_o.exists():
            size = fp_o.stat().st_size
            print(size)
            if size > 1e2:
                continue
        co = f'ncks -O -d lon,{lon_lims[0]},{lon_lims[1]} -d lat,{lat_lims[0]},{lat_lims[1]} {f} {fp_o}'
        # Make time record variable:
        co2 = f'ncks -O --mk_rec_dmn time {fp_o} {fp_o}'
        # -v u10max,v10max
        comms.append(co +'; '+ co2 )

    # %%
    for co in comms:
        print(co)
        # subprocess.run(co, shell=True)

    # %%

    # return
    launch_ncks(comms, max_launches=5)
    print('done')
    # %%
    # %%
    fn_out_final = out_folder / f'{case_name}_{from_time}-{to_time}_concat_subs_{lon_lims[0]}' \
                                f'-{lon_lims[1]}_{lat_lims[0]}-{lat_lims[1]}.nc'
    files_str = ''
    # fn_out =tmp_folder /f'{fn_out_final.stem}_tmp.nc'
    f = files_out[0]
    # for f in files_out:
    #    files_str += f' {f} '
    # %%
    # f = Path('/uno/dos/tres')
    # case_name='case_name'
    for v in varl:
        var = dic_varname2file[v]
        print(rename_dic)
        print(v, var)
        files_str_patt = f'{f.parent}/*{var}*_{station}_tmp_subset.nc'
        file_out = out_folder / f'{case_name}_{from_time}-{to_time}_{v}_concat_subs_{lon_lims[0]}' \
                                    f'-{lon_lims[1]}_{lat_lims[0]}-{lat_lims[1]}.nc'
        com_concat = f'ncrcat {files_str_patt} {file_out}'
        print(com_concat)
        # %%
        subprocess.run(com_concat, shell=True)
    # %%
    """
    print(files_str_patt)
    com_concat = f'ncrcat {files_str_patt} {fn_out_final}'
    print(com_concat)
    subprocess.run(com_concat, shell=True)
    """

# %%


if __name__ == '__main__':
    extract_subset(*sys.argv[1:])
