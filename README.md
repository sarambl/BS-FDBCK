# BS-FDBCK
Analysis for BSOA paper:


# Setup:
### Download:
```bash
git clone git@github.com:sarambl/BS-FDBCK.git 
cd BS-FDBCK/
git checkout ukesm_bs_fdbck
git pull origin ukesm_bs_fdbck
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
pip install  ebas_io-3.6.1-py3-none-any.wh


```

## Download data:


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
Run the notebooks in [bs_fdbck_clean/notebooks](ukesm_bs_fdbck/notebooks) according to their ordering. 