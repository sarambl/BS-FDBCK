import os

from bs_fdbck.util.Nd.sizedist_class_v2.SizedistributionSurface import SizedistributionSurface
from bs_fdbck.util.collocate.collocate import CollocateModel
from bs_fdbck.util.filenames import get_filename_pressure_coordinate_field, get_filename_ng_field

from bs_fdbck.data_info.variable_info import sized_varListNorESM
from bs_fdbck.util.imports.get_pressure_coord_fields import get_pressure_coord_fields
from bs_fdbck.util.Nd.sizedist_class_v2 import Sizedistribution
from bs_fdbck import constants
# import bs_fdbck.util.Nd.sizedist_class_v2.constants as constants_sizedist
import bs_fdbck.util.Nd.sizedist_class_v2.constants as constants_sizedist
from bs_fdbck.util.practical_functions import make_folders
# from bs_fdbck.util.naming_conventions.var_info import get_varname_Nd
import numpy as np
from scipy.stats import lognorm
import useful_scit.util.log as log
import xarray as xr
# %%

class SizedistributionBins(Sizedistribution):
    """

    """
    #nr_of_levels = 3

    savepath_pressure_coordinates = constants.get_outdata_path('pressure_coords')
    def __init__(self,*vars, diameters = constants_sizedist.diameter_obs_df, **kwargs):
        self.diameters= diameters
        return super().__init__(*vars, **kwargs)

    def compute_Nd_vars(self, diameters=None, overwrite=False):
        """

        :param diameters: dataframe with index: var_name and rows 'from_diameter' and 'to_diameter'
        :param overwrite:
        :return:
        """
        if diameters is None:
            diameters=self.diameters
        # Get variable list needed:
        varl = self.get_varlist_input()
        # Get input data:
        input_ds = get_pressure_coord_fields(self.case_name,
                                             varl,
                                             self.from_time,
                                             self.to_time,
                                             self.history_field,
                                             model=self.model_name)
        #print(diameters)
        #print(type(diameters))

        for key in diameters.index:
            fromd = float(diameters.loc[key]['from_diameter'])
            tod = float(diameters.loc[key]['to_diameter'])
            out_varn =key# get_varname_Nd(fromd, tod)
            log.ger.info('Calculating %s'%out_varn)
            da_Nd = calc_Nd_interval_NorESM(input_ds, fromd, tod, out_varn )
            fn = self.get_Nd_output_name(out_varn)
            if os.path.isfile(fn) and not overwrite:
                continue
            #da_Nd.attrs['nice_name'] = get_N_nice_name_Nd(out_varn)
            #da_Nd.attrs['fancy_name'] = get_N_nice_name_Nd(out_varn)
            #self.to_netcdf(da_Nd.to_dataset(), fn)
            self.to_netcdf(da_Nd, fn)
    def get_Nd_output_name(self, out_varn):
        fn = get_filename_pressure_coordinate_field(out_varn, self.model_name, self.case_name, self.from_time,
                                                    self.to_time)
        return fn


class SizedistributionSurfaceBins(SizedistributionSurface):
    """

    """
    #nr_of_levels = 3

    savepath_pressure_coordinates = constants.outpaths['computed_fields_ng']
    def __init__(self,*vars, diameters = constants_sizedist.diameter_obs_df, **kwargs):
        self.diameters= diameters
        return super().__init__(*vars, **kwargs)


    def compute_Nd_vars(self, diameters=None, overwrite=False):
        if diameters is None:
            diameters=self.diameters
        varl =[]
        #print(self.get_varlist_input_sec())
        if self.isSectional:
            varl = varl + self.get_varlist_input_sec()
            #print(varl)
        for key in sized_varListNorESM:
            varl = varl + sized_varListNorESM[key]

        input_ds = self.get_input_data( varl)

        for key in diameters.index:
            # Get dins
            fromd = float(diameters.loc[key]['from_diameter'])
            tod = float(diameters.loc[key]['to_diameter'])
            out_varn =key# get_varname_Nd(fromd, tod)
            log.ger.info('Calculating %s'%out_varn)
            fn = self.get_Nd_output_name(out_varn)
            if os.path.isfile(fn) and not overwrite:
                continue
            da_Nd = calc_Nd_interval_NorESM(input_ds, fromd, tod, out_varn )
            self.to_netcdf(da_Nd, fn)

    def return_Nd_ds(self, diameters=None):
        if diameters is None:
            diameters=self.diameters

        fl = []
        comp_vars=False
        for key in diameters.index:
            out_varn =key# get_varname_Nd(fromd, tod)
            fn = self.get_Nd_output_name(out_varn)
            if not os.path.isfile(fn):
                comp_vars=True
            fl.append(fn)
        if comp_vars:
            self.compute_Nd_vars(diameters=diameters)

        return xr.open_mfdataset(fl, combine='by_coords')



    def get_Nd_output_name(self, out_varn):
        fn = get_filename_ng_field(out_varn, self.model_name, self.case_name, self.from_time, self.to_time)
        return fn


class SizedistributionStationBins(SizedistributionSurfaceBins):
    """

    """
    #nr_of_levels = 3

    savepath_pressure_coordinates = constants.outpaths['collocated']
    def __init__(self,*vars, diameters = constants_sizedist.diameter_obs_df, **kwargs):
        self.diameters= diameters
        super().__init__(*vars, **kwargs)
        self.space_resolution='locations'
        self.coll_mod = CollocateModel(self.case_name, self.from_time, self.to_time,
                                                   self.isSectional,
                                                   self.time_resolution,
                                                   space_res=self.space_resolution,
                                                   model_name=self.model_name,
                                                   history_field=self.history_field,
                                                   raw_data_path=self.raw_data_path,
                                                   # locations = constants.collocate_locations,
                                                   # chunks = self.chunks,
                                                   use_pressure_coords=self.use_pressure_coords,
                                                   )

        # %%
        #self.variables=[get_varname_Nd()]
        # %%
    """
    def get_Nd_vars(self, diameters=None, overwrite=False):
        if diameters is None:
            diameters=self.diameters
        varl =[]
        print(self.get_varlist_input_sec())
        if self.isSectional:
            varl = varl + self.get_varlist_input_sec()
            print(varl)
        for key in sized_varListNorESM:
            varl = varl + sized_varListNorESM[key]
        # savepath_coll_ds(var_name)
        #input_ds = self.get_input_data( varl)


        print(diameters)
        print(type(diameters))
        if type(diameters) is dict:
            for key in diameters:
                fromd = diameters[key][0]
                tod = diameters[key][1]

                out_varn = get_varname_Nd(fromd, tod)
                log.ger.info('Calculating %s'%out_varn)
                fn = cm.savepath_coll_ds(out_varn)#(out_varn, self.model_name, self.case_name, self.from_time,
                                                  #          self.to_time)
                if os.path.isfile(fn) and not overwrite:
                    continue
                da_Nd = calc_Nd_interval_NorESM(input_ds, fromd, tod, out_varn )
                da_Nd.attrs['nice_name'] = get_N_nice_name_Nd(out_varn)
                da_Nd.attrs['fancy_name'] = get_N_nice_name_Nd(out_varn)
                #self.to_netcdf(da_Nd.to_dataset(), fn)
                self.to_netcdf(da_Nd, fn)
    """

    def get_input_data(self, varlist):#(self, variables=None, redo=False, return_cm=False):
        ds = self.coll_mod.get_collocated_dataset(varlist)#set_input_datset(self.get_sizedist_var(var_names=varlist))
        return ds
    def get_Nd_output_name(self, out_varn):
        fn = self.coll_mod.savepath_coll_ds(out_varn)#(out_varn, self.model_name, self.case_name, self.from_time, self.to_time)
        make_folders(fn)
        return fn





def calc_Nd_interval_NorESM(input_ds, fromNd, toNd, varNameN):
    varN = 'NCONC%02.0f' % 1
    da_Nd = input_ds[varN] * 0.  # keep dimensions, zero value
    da_Nd.name = varNameN
    da_Nd.attrs['long_name'] ='N$_{%.0f-%.0f}$'%(fromNd,toNd)
    varsNCONC = sized_varListNorESM['NCONC']
    varsNMR = sized_varListNorESM['NMR']*2 #radius --> diameter
    varsSIG = sized_varListNorESM['SIGMA']
    for varN, varSIG, varNMR in zip(varsNCONC, varsSIG, varsNMR):
        NCONC = input_ds[varN].values  # *10**(-6) #m-3 --> cm-3
        SIGMA = input_ds[varSIG].values  # case[varSIG][lev]#*10**6
        NMR = input_ds[varNMR].values * 2  # *1e9 #case[varNMR][lev]*2  #  radius --> diameter

        # nconc_ab_nlim[case][model]+=logR*NCONC*lognorm.pdf(logR, np.log(SIGMA),scale=NMR)
        if fromNd > 0:
            dummy = NCONC * (lognorm.cdf(toNd, np.log(SIGMA), scale=NMR)) - NCONC * (
                    lognorm.cdf(fromNd, np.log(SIGMA), scale=NMR))
        else:
            dummy = NCONC * (lognorm.cdf(toNd, np.log(SIGMA), scale=NMR))
        # if NMR=0 --> nan values. We set these to zero:
        dummy[NMR == 0] = 0.
        dummy[NCONC == 0] = 0.
        dummy[np.isnan(NCONC)] = np.nan
        da_Nd+= dummy
    return da_Nd

def get_N_nice_name_Nd(N_name):
    splt=N_name.split('_')
    if len(splt)==2:
        N_fancy_name='N$_{d<%s}$'%splt[1]
    else:
        N_fancy_name= 'N$_{%s<d<%s}$'%(splt[0][1::],splt[2])
    return N_fancy_name

#def path_to_binned_data(var_name, pressure_coordinates, )
#def get_filename_Nd(caseName, from_year, model_name, pressure_adjust, to_year, from_diam, to_diam):
#    if pressure_adjust:
#        filen = dataset_path_Nd + '/' + model_name +  '/%s_%s_%s_%s_dmin%d_maxd%d_press_adj.nc' % (
#            model_name, caseName, from_year, to_year, from_diam, to_diam)
#    else:
#        filen = dataset_path_Nd + '/' + model_name +  '/%s_%s_%s_%s_dmin%d_maxd%d.nc' % (
#            model_name, caseName, from_year, to_year, from_diam, to_diam)
#    return filen


