from useful_scit.imps import *
import datetime
import matplotlib as mpl
from ukesm_bs_fdbck.constants import path_EBAS_data
from pathlib import Path

raw_data_EBAS  = Path(path_EBAS_data) / 'raw_data'
out_data_EBAS = Path(path_EBAS_data) / 'reform_data'

def create_EBAS_sizedist(filename, place='SMR', perc=True, reform_dir_EBAS=out_data_EBAS):
    if place=='SMR':
        return EBAS_sizedist_SMR(filename, perc=perc, reform_dir_EBAS=reform_dir_EBAS)
    elif place == 'MPZ':
        return EBAS_sizedist_MPZ(filename,  perc=perc, reform_dir_EBAS=reform_dir_EBAS)


# noinspection PyUnresolvedReferences
class EBAS_sizedist_SMR:

    def __init__(self, filename, perc=False, reform_dir_EBAS=out_data_EBAS): # known special case of object.__init__
        self.input_filepath = Path(filename)
        self.inputfile_nb = Path(filename).name
        self.reform_dir_EBAS = Path(reform_dir_EBAS)
        # print(self.reform_dir_EBAS)
        self.dataset = self.load_data(perc=perc)

        return


    def plot_time_from_to_1(self, from_time='', to_time='', yscale='log', vmin=1e0, vmax=1e4):
        cba_kwargs={'label': r'dN/dlog$_{10}$D [#/cm$^3$]','aspect':8}
        if len(from_time)>0 and len(to_time)>0:
            self.dataset['dNdlog10D'].sel(time=slice(from_time, to_time)).plot(yscale = yscale, x='time',
                                                                             y='diameter',
                                                                             norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
                                                                             figsize=[20,4],ylim=[3,1e3],
                                                                             cbar_kwargs=cba_kwargs)
        else:
            self.dataset['dNdlog10D'].plot(yscale = yscale, x='time',
                                         y='diameter',
                                         norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
                                         figsize=[20,4],ylim=[3,1e3],
                                         cbar_kwargs=cba_kwargs)
        return

    def load_data(self, perc=False):
        dummy = xr.open_dataset(self.input_filepath)
        first_val_date = dummy.attrs['first_valid_date_of_data']
        first_val_date = datetime.datetime(first_val_date[0], first_val_date[1], first_val_date[2])
        days_since = dummy.days_from_file_reference_point.values
        dt_index = [(first_val_date + days * datetime.timedelta(days=1)) for days in days_since]
        self.original_dtset = dummy.copy()
        diameters = []
        mean_vars = []
        perc1587_vars = []
        perc8413_vars = []

        for var in dummy.data_vars:
            title = dummy[var].attrs['title']
            title_split = title.split(',')
            if len(title_split)==4 and 'numflag' not in title and 'particle_number_size_distribution' in title:
                dummy[var].attrs['units']=title_split[1]
                new_var_name=title_split[-2]+title_split[-1]
                dummy = dummy.rename({var:new_var_name})
                nm=title_split[-2].split('=')[1].split(' ')[0]
                if 'mean' in new_var_name and 'numflag' not in title:
                    mean_vars.append(new_var_name)
                elif 'percentile:15.87' in new_var_name:
                    perc1587_vars.append(new_var_name)
                elif 'percentile:84.13' in new_var_name:
                    perc8413_vars.append(new_var_name)
                if len(diameters)==0:
                    diameters.append(float(nm))
                elif float(nm) != diameters[-1]:
                    diameters.append(float(nm))

        dtset = xr.Dataset({'dNdlog10D':(['diameter','time'], dummy[mean_vars].to_array()),
                            'perc1587':(['diameter','time'], dummy[perc1587_vars].to_array()),
                            'perc8413':(['diameter','time'], dummy[perc8413_vars].to_array())},
                           coords={'diameter':diameters, 'time' : dt_index})
        dtset.to_netcdf(self.reform_dir_EBAS /(self.inputfile_nb[0:-3] + 'reformatted_smb.nc'))
        return dtset

    def plot_time_from_to(self, dataset = None, from_time='', to_time='', yscale='log', vmin=1e0, vmax=1e4, ax=None,
                          ylim=None, **kwargs):
        if ylim is None:
            ylim = [3, 1e3]
        if dataset is None:
            dataset = self.dataset
        if 'ylim' not in kwargs:
            kwargs['ylim'] = ylim
        if ax is not None:
            kwargs['ax']=ax
        else:
            kwargs['figsize']=[20,4]
        cba_kwargs={'label': r'dN/dlog$_{10}$D [#/cm$^3$]','aspect':8}
        if len(from_time)>0 and len(to_time)>0:
            dataset['dNdlog10D'].sel(time=slice(from_time, to_time)).plot(yscale = yscale, x='time',
                                                                             y='diameter',
                                                                             norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
                                                                             #figsize=[20,4],
                                                                             **kwargs,
                                                                             cbar_kwargs=cba_kwargs)
        else:
            dataset['dNdlog10D'].plot(yscale = yscale, x='time',
                                         y='diameter',
                                         norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
                                         #figsize=[20,4],
                                         **kwargs,
                                         cbar_kwargs=cba_kwargs)
        plt.ylabel('Diameter [nm]')
        return


class EBAS_sizedist_MPZ(EBAS_sizedist_SMR):

    def load_data(self, perc=False):
        # %%
        input_fn = self.input_filepath
        # %%
        #input_fn = raw_data_EBAS / 'DE0044R.20080101000000.20181205100800.dmps.particle_number_size_distribution.aerosol.1y.1h.DE08L_DMPS_IFT_MELPITZ02_until_20080818_.DE08L_IFT_DRY_TDMPS_until_20080818_DE08L_IFT_C.lev2.nc'
        #perc = False
        # %%
        dummy = xr.open_dataset(input_fn)
        # %%
        #dummy = xr.open_dataset(self.filename)
        #print(dummy)
        first_val_date = dummy.attrs['first_valid_date_of_data']
        first_val_date = datetime.datetime(first_val_date[0], first_val_date[1], first_val_date[2])
        days_since = dummy.days_from_file_reference_point.values
        dt_index = [(first_val_date + days * datetime.timedelta(days=1)) for days in days_since]
        # %%
        self.original_dtset= dummy.copy()
        # %%
        diameters=[]
        mean_vars=[]
        perc1587_vars=[]
        perc8413_vars=[]
        # %%
        if perc:
            for var in dummy.data_vars:
                title = dummy[var].attrs['title']
                title_split = title.split(',')
                if len(title_split)==4 and 'numflag' not in title and 'particle_number_size_distribution' in title:
                    dummy[var].attrs['units']=title_split[1]
                    new_var_name=title_split[-2]+title_split[-1]
                    dummy = dummy.rename({var:new_var_name})
                    nm=title_split[-2].split('=')[1].split(' ')[0]
                    if 'mean' in new_var_name and 'numflag' not in title:
                        mean_vars.append(new_var_name)
                    elif 'percentile:15.87' in new_var_name:
                        perc1587_vars.append(new_var_name)
                    elif 'percentile:84.13' in new_var_name:
                        perc8413_vars.append(new_var_name)
                    if len(diameters)==0:
                        diameters.append(float(nm))
                    elif float(nm) != diameters[-1]:
                        diameters.append(float(nm))

            dtset = xr.Dataset({#'dNdlog10D':(['diameter','time'], dummy[mean_vars].to_array()),
                'perc1587':(['diameter','time'], dummy[perc1587_vars].to_array()),
                'perc8413':(['diameter','time'], dummy[perc8413_vars].to_array())},
                coords={'diameter':diameters, 'time' : dt_index})
        else:

            for var in dummy.data_vars:
                title = dummy[var].attrs['title']
                title_split = title.split(',')
                if len(title_split)==3:
                    #print(title_split)
                    dummy[var].attrs['units']=title_split[1]
                    new_var_name=title_split[-1]
                    dummy = dummy.rename({var:new_var_name})
                    nm=title_split[-1].split('=')[1].split(' ')[0]
                    mean_vars.append(new_var_name)
                    if len(diameters)==0:
                        diameters.append(float(nm))
                    elif float(nm) != diameters[-1]:
                        diameters.append(float(nm))




            dtset = xr.Dataset({'dNdlog10D':(['diameter','time'], dummy[mean_vars].to_array())},
                               coords={'diameter':diameters, 'time' : dt_index})
        # %%
        dtset
        # %%
        return dtset


