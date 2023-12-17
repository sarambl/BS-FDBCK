from bs_fdbck_clean.util.imports.fix_xa_dataset_v2 import xr_fix
from bs_fdbck_clean.util.collocate.collocate import CollocateModel
from bs_fdbck_clean.util.Nd.sizedist_class_v2 import Sizedistribution, get_nrSEC_varname, _get_nconc_varname, \
    _get_nmr_varname, _get_sig_varname
from bs_fdbck_clean.util.imports.import_fields_xr_v2 import xr_import_NorESM
from bs_fdbck_clean.util.plot.lineplots import plot_seasonal_plots
from bs_fdbck_clean.util.practical_functions import make_folders
from bs_fdbck_clean import constants
import xarray as xr
import matplotlib.pyplot as plt


class SizedistributionSurface(Sizedistribution):
    """

    """
    surface = 'surface/'
    nr_of_levels = 3
    default_savepath_root = constants.get_outdata_path('sizedistrib_files') / surface

    def __init__(self,*vars, **kwargs):
        if 'use_pressure_coords' in kwargs.keys():
            del kwargs['use_pressure_coords']#='False'

        return super().__init__(*vars, **kwargs, use_pressure_coords=False)



    #def dataset_savepath(self, case_name, model_name):
    #    """
    #    Returns filename of dataset
    #    :param case_name:
    #    :param model_name:
    #    :return:
    #    """
    #
    #    case_name = case_name.replace(' ', '_')
    #    _savep = self.default_savepath_root
    #    st = '%s/%s/%s/%s' % (_savep, model_name, case_name, case_name)
    #    st = st + '_%s_%s' % (self.from_time, self.to_time)
    #    st = st + '_%s_%s' % (self.time_resolution, self.space_resolution)
    #    fn = st + '.nc'
    #    make_folders(fn)
    #    return fn

    def dataset_savepath_var(self, var_name, case_name, model_name):
        """
        Returns filename of dataset
        :param var_name:
        :param case_name:
        :param model_name:
        :return:
        """
        case_name = case_name.replace(' ', '_')
        _sp = self.default_savepath_root
        st = '%s/%s/%s/%s_%s' % (_sp, model_name, case_name, var_name, case_name)
        st = st + '_%s_%s' % (self.from_time, self.to_time)
        st = st + '_%s_%s' % (self.time_resolution, self.space_resolution)
        fn = st + '.nc'
        make_folders(fn)
        return fn

    def _calc_sizedist_sec_var_nr(self, var_name, num):
        if num is None:
            num = var_name[-2:]
        varl = get_nrSEC_varname(num)
        input_ds = self.get_input_data(varl)
        #input_ds = xr_import_NorESM(self.case_name, varl, self.from_time, self.to_time, model=self.model_name,
        #                            history_fld=self.history_field, comp='atm')
        #from_lev = len(input_ds['lev']) - self.nr_of_levels
        #input_ds = input_ds.isel(lev=slice(from_lev, None))
        #input_ds = xr_fix(input_ds, model_name=self.model_name)
        ds_dNdlogD = self._compute_dNdlogD_sec(var_name, self.diameter, input_ds, num)
        input_ds.close()
        del input_ds
        return ds_dNdlogD


    def _calc_sizedist_mod_var_nr(self, var_name, num=None):
        if num is None:
            num = var_name[-2:]
        varN = _get_nconc_varname(num)
        varNMR = _get_nmr_varname(num)
        varSIG = _get_sig_varname(num)
        varl = [varN, varNMR, varSIG]
        input_ds =self.get_input_data(varl)# xr_import_NorESM(self.case_name, varl, self.from_time, self.to_time, model=self.model_name,
        #                            history_fld=self.history_field, comp='atm')
        #from_lev = len(input_ds['lev']) - self.nr_of_levels
        #input_ds = input_ds.isel(lev=slice(from_lev, None))
        #input_ds = xr_fix(input_ds, model_name=self.model_name)
        ds_dNdlogD = self._compute_dNdlogD_mod(var_name, self.diameter, input_ds, varN, varNMR, varSIG)
        input_ds.close()
        del input_ds
        return ds_dNdlogD

    def get_collocated_dataset(self, variables=None, redo=False, return_cm=False):
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
        cm.set_input_datset(self.get_sizedist_var(var_names=variables))
        cm.collocate_dataset_vars(var_names=variables, redo=redo)
        #if self.isSectional:
        #    varl = []
        if return_cm:
            return cm.get_collocated_dataset(var_names=variables), cm
        return cm.get_collocated_dataset(var_names=variables)  # , chunks=self.chunks)

    def get_input_data(self, varlist):
        input_ds = xr_import_NorESM(self.case_name, varlist, self.from_time, self.to_time, model=self.model_name,
                                    history_fld=self.history_field, comp='atm')
        from_lev = len(input_ds['lev']) - self.nr_of_levels
        input_ds = input_ds.isel(lev=slice(from_lev, None))
        input_ds = xr_fix(input_ds, model_name=self.model_name)
        return input_ds


    def plot_location(self, variables: list=None,
                      seasons=None,
                      axs=None,
                      sharex: bool =True,
                      sharey: bool = True,
                      figsize=None,
                      legend: bool=True,
                      apply2lev=None,
                      apply_method=None,
                      loc = None,
                      **kwargs
                      ):

        _defkwargs = dict(xscale='log', yscale='log', ylim=[1, None], xlim=[3, 1e3])
        for key in _defkwargs:
            if key not in kwargs:
                kwargs[key] = _defkwargs[key]
        if 'dNdlogD' in variables:
            comp_tot = True

            _vars = list(set(variables + self.final_sizedist_vars)-{'dNdlogD'})
        else:
            _vars = variables
            comp_tot=False
        ds = self.get_collocated_dataset(variables=_vars)
        if comp_tot:
            self.make_total_sizedist(ds)
        if apply2lev is None:
            apply2lev = xr.Dataset.mean
        _ds = apply2lev(ds, 'lev')
        if variables is None:
            variables = ds.data_vars
        if loc is None:
            locations =ds.coords['location'].values
            loc_non = True
        else:
            locations=[loc]
            loc_non = False
        for loc in locations:
            if loc_non and axs is None:
                fig, axs = plt.subplots(2, 2, figsize=figsize, sharey=sharey, sharex=sharex)
            for var in variables:
                _da = _ds.sel(location=loc)
                plot_seasonal_plots(_da,
                                    varname=var,
                                    x='diameter',
                                    label=self.case_name_nice,
                                    seasons=seasons,
                                    axs=axs,
                                    title=str(loc),
                                    sharex=sharex,
                                    sharey=sharey,
                                    figsize=figsize,
                                    legend=legend,
                                    apply_method=apply_method,
                                    **kwargs
                                    )
            if loc_non: plt.show()
