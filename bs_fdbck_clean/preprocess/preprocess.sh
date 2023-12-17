python ec-earth/01-02-preprocess-model-output_station_bins_month_full_grid_ec-earth.py
python ec-earth/extract_latlon_grid_ec_earth.py ECE3_output_Sara 2012-01-01 2019-01-01 IFS SMR
python ec-earth/extract_latlon_grid_ec_earth.py ECE3_output_Sara 2012-01-01 2019-01-01 IFS ATTO


python echam/01-02-preprocess-model-output_station_bins_month_full_grid_echam.py
python echam/extract_latlon_grid_echam.py SALSA_BSOA_feedback 2012-01-01 2019-01-01 SMR
python echam/extract_latlon_grid_echam.py SALSA_BSOA_feedback 2012-01-01 2019-01-01 ATTO


python ukesm/x_ukesm_preproc.py
python ukesm/01-02-preprocess-model-output_station_bins_month_full_grid_ukesm.py
python ukesm/extract_latlon_grid_ukesm.py AEROCOMTRAJ 2012-01-01 2019-01-01 SMR
python ukesm/extract_latlon_grid_ukesm.py AEROCOMTRAJ 2012-01-01 2019-01-01 ATTO


python noresm/01-02-preprocess-model-output_station_bins_month_full_grid.py
python noresm/extract_latlon_grid_noresm.py OsloAero_intBVOC_f09_f09_mg17_full 2012-01-01 2015-01-01 SMR
python noresm/extract_latlon_grid_noresm.py OsloAero_intBVOC_f09_f09_mg17_full 2012-01-01 2015-01-01 ATTO
python noresm/extract_latlon_grid_noresm.py OsloAero_intBVOC_f09_f09_mg17_ssp245 2015-01-01 2019-01-01 SMR
python noresm/extract_latlon_grid_noresm.py OsloAero_intBVOC_f09_f09_mg17_ssp245 2015-01-01 2019-01-01 ATTO

