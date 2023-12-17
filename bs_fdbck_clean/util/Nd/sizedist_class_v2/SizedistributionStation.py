from useful_scit.util.make_folders import make_folders

from bs_fdbck_clean.util.Nd.sizedist_class_v2.SizedistributionSurface import SizedistributionSurface
from bs_fdbck_clean.util.collocate.collocate import CollocateModel


class SizedistributionStation(SizedistributionSurface):
    def __init__(self,*vars, **kwargs):
        #self.diameters= diameter
        super().__init__(*vars, **kwargs)
        self.space_resolution='locations'
        if 'locations' in kwargs:
            locs = kwargs['locations']
        else:
            locs=None
        self.coll_mod = CollocateModel(self.case_name, self.from_time, self.to_time,
                                                   self.isSectional,
                                                   self.time_resolution,
                                                   space_res=self.space_resolution,
                                                   model_name=self.model_name,
                                                   history_field=self.history_field,
                                                   raw_data_path=self.raw_data_path,
                                                   locations = locs,# constants.collocate_locations,
                                                   # chunks = self. chunks,
                                                   use_pressure_coords=self.use_pressure_coords,
                                                   )


    def dataset_savepath_var(self, var_name, case_name, model_name):
        """
        Returns filename of dataset
        :param var_name:
        :param case_name:
        :param model_name:
        :return:
        """
        fn = self.coll_mod.savepath_coll_ds(var_name) #(out_varn, self.model_name, self.case_name, self.from_time, self.to_time)
        make_folders(fn)
        return fn

        #case_name = case_name.replace(' ', '_')
        #_sp = self.default_savepath_root
        #st = '%s/%s/%s/%s_%s' % (_sp, model_name, case_name, var_name, case_name)
        #st = st + '_%s_%s' % (self.from_time, self.to_time)
        #st = st + '_%s_%s' % (self.time_resolution, self.space_resolution)
        #fn = st + '.nc'
        #make_folders(fn)
        #return fn
    def get_input_data(self, varlist):#(self, variables=None, redo=False, return_cm=False):
        ds = self.coll_mod.get_collocated_dataset(varlist, parallel=True, chunks={'time':48})#set_input_datset(self.get_sizedist_var(var_names=varlist))
        return ds

    def get_collocated_dataset(self, variables=None, redo=False, return_cm=False,
                               parallel=False, chunks=None):
        if variables is None:
            variables = self.varl_sizedist_final()
        cm = CollocateModel(self.case_name, self.from_time, self.to_time,
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
        #    varl = []
        if return_cm:
            return cm.get_collocated_dataset(var_names=variables), cm
        return cm.get_collocated_dataset(var_names=variables, parallel=parallel, chunks=chunks)  # , chunks=self.chunks)
