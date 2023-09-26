import os
import pandas as pd
from pathlib import Path
import pathlib

## Root:
project_name = 'BS-FDBCK_clean'
project_name_output = 'BS-FDBCK'

tetralith_root = '/proj/bolinc/users/x_sarbl/'

# root:
project_base_path = Path(tetralith_root)
# analysis root: this is where the package is located and where the data folder will be created.
analysis_base_path = Path(project_base_path) / 'analysis'
# Path to project folder:
package_base_path = analysis_base_path / project_name
# %%

# INPUT model data:
# Models:
raw_data_path_NorESM = project_base_path / 'noresm_archive'
raw_data_path_echam = project_base_path / 'other_data' / 'BS-FDBCK' / 'ECHAM-SALSA'
raw_data_path_ukesm = project_base_path / 'other_data' / 'BS-FDBCK' / 'UKESM'
raw_data_path_ecEarth = Path('/proj/aerosol_esm_lund/users/x_casve/BIG_ISMO/output/ECE3_output_Sara/')

pathdic_raw_data = {'NorESM': raw_data_path_NorESM,
                    'ECHAM-SALSA': raw_data_path_echam,
                    'UKESM': raw_data_path_ukesm,
                    'EC-Earth': raw_data_path_ecEarth
                    }
def get_input_datapath(model='NorESM'):
    return pathdic_raw_data[model]


# Measuremens data:
path_measurement_data = package_base_path / 'Data'
path_eusaar_data = project_base_path / 'EUSAAR_data'
path_EBAS_data = analysis_base_path / project_name / 'Data' / 'EBAS'


# ****** No need to edit beyond this point ****

# %%
# Output paths:

# Plots path:
path_plots = analysis_base_path / f'Plots_{project_name}'



# Output data:

path_outdata = analysis_base_path / f'Output_data_{project_name_output}'

path_eusaar_outdata = path_eusaar_data / 'EUSAAR_data/'

latlon_path = path_outdata / 'latlon.nc'

path_extract_latlon_outdata = path_outdata / 'extracted_latlon_subset'


# Paths for preprocessed data
outpaths = dict(
    pressure_coords=path_outdata / 'fields_pressure_coordinates',
    original_coords=path_outdata / 'computed_fields_ng',
    computed_fields_ng=path_outdata / 'computed_fields_ng',  # native grid computed fields
    pressure_density_path=path_outdata / 'pressure_density',
    masks=path_outdata / 'means/masks/',
    area_means=path_outdata / 'means' / 'area_means/',
    map_means=path_outdata / 'means' / 'map_means/',
    levlat_means=path_outdata / 'means' / 'levlat_means/',
    profile_means=path_outdata / 'means' / 'profile_means/',
    sizedistrib_files=path_outdata / 'sizedistrib_files',
    collocated=path_outdata / 'collocated_ds/',
    eusaar=path_outdata / 'eusaar/',
)


def get_outdata_path(key):
    if key in outpaths:
        return outpaths[key]
    else:
        print('WARNING: key not found in outpaths, constants.py')
        return path_outdata / key


Path(path_outdata).mkdir(parents=True, exist_ok=True)

# data info
proj_lc = project_name.lower().replace('-', '_')

path_data_info = analysis_base_path / project_name / proj_lc / 'data_info/'
print(path_data_info)
# output locations:
path_locations_file = path_data_info / 'locations.csv'

if os.path.isfile(path_locations_file):
    collocate_locations = pd.read_csv(path_locations_file, index_col=0)
else:
    _dic = dict(Hyytiala={'lat': 61.51, 'lon': 24.17},
                Melpitz={'lat': 51.32, 'lon': 12.56},
                Amazonas={'lat': -3., 'lon': -63.},
                Beijing={'lat': 40, 'lon': 116})
    collocate_locations = pd.DataFrame.from_dict(_dic)
    collocate_locations.to_csv(path_locations_file)

path_echam_station = path_data_info / 'echam_info' / 'echam_stations.csv'

path_ec_earth_vars = path_data_info / 'ec_earth_info' / 'ec_earth_var_overview.csv'


def get_locations(model='NorESM'):
    locs = collocate_locations
    return locs

# %%
