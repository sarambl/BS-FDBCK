# %%
import subprocess
import sys
import time

import pandas as pd
import useful_scit.util.log as log

from bs_fdbck_clean.constants import package_base_path


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
            return 'running'
        else:
            return 'done'


# %%
def launch_monthly_station_output_noresm(case, issectional, max_launches=3, from_time='2008-01-01', to_time='2009-01-01',
                                         history_field='.h1.'):
    """
    Launch a number of processes to calculate monthly files for file.
    :param history_field:
    :param case: Case to be run
    :param issectional: Whether sectional or not
    :param max_launches: maximum launched subprocesses at a time
    :param from_time: From time
    :param to_time: To time
    :return:
    """
    # Setup dataframe to keep track of processes
    l_df = setup_df_to_track_progress(from_time, to_time)
    pyf = sys.executable  # "/persistent01/miniconda3/envs/env_sec_v2/bin/python3"
    if history_field == '.h2.':
        file = package_base_path / 'bs_fdbck_clean' / 'preprocess' /'noresm'/ 'subproc_station_output_full_grid_h2.py'

    else:
        file = package_base_path / 'bs_fdbck_clean' / 'preprocess' / 'noresm'/'subproc_station_output_full_grid.py'
    # while loop:
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
            nr_row = l_df[l_df['status'] == 'not running'].iloc[0]
            # from time to time
            time1 = nr_row.name.strftime(format='%Y-%m-%d')
            time2 = (nr_row.name + pd.DateOffset(months=1)).strftime(format='%Y-%m-%d')
            # Launch subprocess:
            p1 = subprocess.Popen([pyf, file, str(time1), str(time2), case, str(issectional)])
            # put subprocess in dataframe
            l_df.loc[nr_row.name, 'process'] = p1
            l_df.loc[nr_row.name, 'status'] = 'running'

        log.ger.info(l_df)
        time.sleep(10)


def launch_monthly_station_output_ukesm(case, max_launches=3, from_time='2012-01-01', to_time='2019-01-01'):
    """
    Launch a number of processes to calculate monthly files for file.
    :param case: Case to be run
    :param max_launches: maximum launched subprocesses at a time
    :param from_time: From time
    :param to_time: To time
    :return:
    """
    print('HEY')

    # Setup dataframe to keep track of processes
    l_df = setup_df_to_track_progress(from_time, to_time)
    pyf = sys.executable  # "/persistent01/miniconda3/envs/env_sec_v2/bin/python3"
    file = package_base_path / 'bs_fdbck_clean' / 'preprocess' / 'ukesm' / 'subproc_station_output_full_grid_ukesm.py'
    print(f'running file:{file}')
    # while loop:
    notdone = True
    run_until_done(case, file, l_df, max_launches, notdone, pyf)


def launch_monthly_station_output_echam(case, max_launches=3, from_time='2008-01-01', to_time='2009-01-01'):
    """
    Launch a number of processes to calculate monthly files for file.
    :param case: Case to be run
    :param max_launches: maximum launched subprocesses at a time
    :param from_time: From time
    :param to_time: To time
    :return:
    """
    print('HEY')

    # Setup dataframe to keep track of processes
    l_df = setup_df_to_track_progress(from_time, to_time)
    pyf = sys.executable  # "/persistent01/miniconda3/envs/env_sec_v2/bin/python3"
    file = package_base_path / 'bs_fdbck_clean' / 'preprocess' / 'echam' / 'subproc_station_output_full_grid_echam.py'
    print(f'running file:{file}')
    # while loop:
    notdone = True
    run_until_done(case, file, l_df, max_launches, notdone, pyf)


def launch_monthly_station_output_ec_earth(case,
                                           max_launches=3,
                                           from_time='2012-01-01',
                                           to_time='2012-02-01'):
    """
    Launch a number of processes to calculate monthly files for file.
    :param case: Case to be run
    :param max_launches: maximum launched subprocesses at a time
    :param from_time: From time
    :param to_time: To time
    :return:
    """
    print('HEY')

    # Setup dataframe to keep track of processes
    l_df = setup_df_to_track_progress(from_time, to_time)
    pyf = sys.executable  # "/persistent01/miniconda3/envs/env_sec_v2/bin/python3"
    file = package_base_path / 'bs_fdbck_clean' / 'preprocess' / 'ec-earth' / 'subproc_station_output_full_grid_ec_earth.py'
    # while loop:
    print(f'running file:{file}')

    notdone = True
    run_until_done(case, file, l_df, max_launches, notdone, pyf)



def run_until_done(case, file, l_df, max_launches, notdone, pyf):
    while notdone:
        print('HEEEEEY********************************')
        # Update status
        l_df['status'] = l_df.apply(update_stat_proc, axis=1)
        nrunning, ndone, nnrunning = check_stat_proc(l_df)
        # check if done, if so break
        notdone = len(l_df) != ndone
        if notdone is False:
            break
        # If room for one more process:
        print(max_launches)
        print(nrunning, max_launches, nnrunning)
        if (nrunning < max_launches) and (nnrunning > 0):
            nr_row = l_df[l_df['status'] == 'not running'].iloc[0]
            # from time to time
            time1 = nr_row.name.strftime(format='%Y-%m-%d')
            time2 = (nr_row.name + pd.DateOffset(months=1)).strftime(format='%Y-%m-%d')
            # Launch subprocess:
            p1 = subprocess.Popen([pyf, file, str(time1), str(time2), case])
            # put subprocess in dataframe
            l_df.loc[nr_row.name, 'process'] = p1
            l_df.loc[nr_row.name, 'status'] = 'running'

        log.ger.info(l_df)
        time.sleep(4)





def setup_df_to_track_progress(from_time, to_time):
    _dr = pd.date_range(from_time, to_time, freq='MS')[:-1]  # DataFrame(range())
    l_df = pd.DataFrame(index=_dr, columns=['process', 'status'])
    l_df['status'] = 'not running'
    l_df['status'] = l_df.apply(update_stat_proc, axis=1)
    check_stat_proc(l_df)
    return l_df
