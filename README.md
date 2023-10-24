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

##


## To reproduce results:
### Preprocess model data:
```bash
cd bs_fdbck_clean/preprocess/
chmod +x preprocess.sh
./preprocess.sh
```


### Run notebooks:
Run the notebooks in [bs_fdbck_clean/notebooks](bs_fdbck_clean/notebooks) according to their ordering. 