# BS-FDBCK
Analysis for BSOA paper:


# Setup:
### Download:
```bash
git clone git@github.com:sarambl/SECT_SENS.git 
cd SECT_SENS/
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
Edit paths at the top of [bs_fdbck/constants.py](bs_fdbck/constants.py). 


##


## To reproduce results:
### Create Nd datasets:
```bash
cd bs_fdbck/preprocess/
python Nd.py

```

### Run notebooks:
Run the notebooks in [bs_fdbck/notebooks](bs_fdbck/notebooks).