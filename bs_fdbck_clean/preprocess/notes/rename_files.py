

import os
# %%
import subprocess

case = 'OsloAero_intBVOC_f19_f19_mg17_incY_full'#sys.argv[1]
from_n = 'full'#sys.argv[2]
to_n = 'locations'#sys.argv[3]


run_path = '/proj/bolinc/users/x_sarbl/analysis/Output_data_BS-FDBCK/collocated_ds/NorESM/OsloAero_intBVOC_f19_f19_mg17_incY_full/'#'#sys.argv[4]

# %%

fl = os.listdir(run_path)

co_ls = []
for f in fl:
    last = f.split('_')[-1]
    if last.split('.')[0]==to_n:
        continue
    to_last = '_'.join(f.split('_')[:-1])# to_n

    fn = f'{to_last}_{to_n}.nc'
    co = f'mv {f} {fn}'
    co_ls.append(co)

# %%
for co in co_ls:
    print(co)
    subprocess.run(co, shell=True, cwd=run_path)



# %%
