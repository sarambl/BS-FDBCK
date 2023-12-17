# BS-FDBCK
Analysis for BSOA paper:


# Setup:
### Download:
```bash
git clone git@github.com:sarambl/BS-FDBCK.git 
cd BS-FDBCK/
```


### Install environment: 
```bash
# install environment etc
conda env create -f environment.yml
conda activate env_bs_fdbck
conda develop .

cd ../
git clone https://git.nilu.no/ebas/ebas-io.git  cd
cd ebas-io/dist/
pip install ebas_io-3.6.1-py3-none-any.wh
```

## Download data:
- The SMEAR-II size distribution data was downloaded from the EBAS database and are available for download at [ebas-data.nilu.no](https://ebas-data.nilu.no/DataSets.aspx?stations=FI0050R&nations=FI246FIN&InstrumentTypes=dmps&components=particle_number_size_distribution&fromDate=1970-01-01&toDate=2023-12-31). 
    The temperature and wind measurements from SMEAR-II is available for download at: [https://smear.avaa.csc.fi](https://smear.avaa.csc.fi/download).

- The full ATTO aerosol measurement data sets can be downloaded in the ATTO data portal under [https://www.attodata.org/](https://www.attodata.org/).

- Temperature measurements from ATTO are available for download at [https://www.attodata.org/](attodata.org).

- Model data can be downloaded from: #TODO!

## Edit settings and paths: 
Edit paths at the top of [bs_fdbck_clean/constants.py](bs_fdbck/constants.py).

## Data organising: 




## To reproduce results:
### Preprocess model data:
```bash
cd bs_fdbck_clean/preprocess/
chmod +x preprocess.sh
./preprocess.sh
```


### Run notebooks:
Run the notebooks in [bs_fdbck_clean/notebooks](bs_fdbck_clean/notebooks) according to their ordering.

```bash
cd bs_fdbck_clean/notebooks/01-01-preprocess_station_data/
python 01-01-01-Preprocess_measurement_dataset_HYYTIALA.py
python 01-01-02-Preprocess_ACSM_meteo_sizedit_data_ATTO.py
python 01-01-03-Preprocess_dataset_MODELS_SMR.py
python 01-01-04-Preprocess_dataset_MODELS_ATTO.py

cd ../01-02-preprocess_satellite/
python 01-02-01-download_and_preproc_MODIS.py
python 01-02-02-produce_hyytiala_satellite_dataset.py
python 01-02-03_produce_ATTO_satellite_dataset.py

cd ../02-T2OA_OA2Nx/
python 02-01-relations_plots_TOANx_SMR.py
python 02-02-relation_plots_TOANx_ATTO.py
python 02-03-plot_both_stations_together.py
python 02-04-01_relations_plots_emissions_ATTO.py
python 02-04-01_relations_plots_emissions_SMR.py

cd ../03-cloud_properties/03-01-ATTO
python 03-01-01-create_file-ALL_year_new_version.py
python 03-01-03-01_confidence_interval_diff_median_my_data-ATTO_FMA.py
python 03-01-03-02_confidence_interval_diff_median_my_data-ATTO_Nx_FMA.py
python 03-01-04-01_confidence_interval_diff_median_my_data-ATTO_FMAM.py
python 03-01-05-01_confidence_interval_diff_median_my_data-ATTO_JFM.py
python 03-01-07-01_confidence_interval_diff_median_my_data-ATTO_JFMAM.py
python 03-01-08-01_confidence_interval_diff_median_my_data-ATTO_MAM.py


cd ../03-02-SMR
python 03-02-01-create_file.py
python 03-02-02-01_confidence_interval_diff_median_my_data-SMR_JA.py
python 03-02-02-02_confidence_interval_diff_median_my_data-SMR_JA-Nx.py

cd ../
python 03-03-compare_differentials.py

cd ../04-evaluation_measurements
python 04-01-OA_against_OA.py
python 04-02-OA_against_OA_ATTO.py
python 04-03-Nx_SMR.py
python 04-04-Nx_ATTO.py
python 04-05_NorESM_yield.ipynb

```
