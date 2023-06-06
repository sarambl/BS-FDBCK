import os
import os.path
import time

import numpy as np
import useful_scit.util.log as log
import xarray as xr
from dask.diagnostics import ProgressBar

import bs_fdbck_clean.data_info.variable_info
import bs_fdbck_clean.util.eusaar_data.distc_var as distc_var
from bs_fdbck_clean import constants
from bs_fdbck_clean.data_info import get_nice_name_case
from bs_fdbck_clean.data_info.variable_info import sized_varListNorESM
from bs_fdbck_clean.util.imports.get_pressure_coord_fields import get_pressure_coord_fields
from bs_fdbck_clean.util.practical_functions import make_folders

D_NDLOG_D_SEC = 'dNdlogD_sec'
D_NDLOG_D_MOD = 'dNdlogD_mod'
TOTAL_VARS = [D_NDLOG_D_MOD, D_NDLOG_D_SEC]

varListNorESM = bs_fdbck_clean.data_info.variable_info.sized_varListNorESM
default_savepath = constants.get_outdata_path('sizedistrib_files')


def update_dic(dic_from, dic_to):
    for key in dic_from:
        dic_to[key] = dic_from[key]


def append2dic(ds_append, ds_add):
    """

    :param ds_append:
    :param ds_add:
    """
    for key in ds_add.attrs.keys():
        if key not in ds_append.attrs:
            ds_append.attrs[key] = ds_add.attrs[key]
    return ds_append


def sum_vars(ds, varlist4sum, varname_sum, long_name=None):
    """

    :param ds:
    :param varlist4sum:
    :param varname_sum:
    :param long_name:
    :return:
    """
    keep_coords = list(ds[varlist4sum[0]].dims)
    drop_l = list(set(ds.variables) - set(varlist4sum + keep_coords))
    _ds = ds.drop(drop_l)
    da = _ds.to_array(dim='variable', name=varname_sum)  # 'dNdlogD_mod')
    da = da.sum('variable')
    if long_name is not None:
        da.attrs['long_name'] = long_name
    _at = _ds[varlist4sum[0]].attrs
    if 'units' in _at:
        da.attrs['units'] = _at['units']
    return da


class Sizedistribution:
    """
    Class to calculate and read sizedistribution dataset
    """
    default_savepath_root = constants.get_outdata_path('sizedistrib_files')

    # noinspection PyTypeChecker
    def __init__(self,
                 case_name,
                 from_time, to_time,
                 dlim_sec = None,
                 isSectional = False,
                 time_res ='hour',
                 raw_data_path=constants.get_input_datapath(), space_res='full',
                 nr_bins=5, print_stat=False, model_name='NorESM', history_field='.h0.',
                 locations=None,#constants.locations,
                 chunks=None, use_pressure_coords=True,
                 use_eusaar_diam=True):
        """

        :param case_name:
        :param from_time:
        :param to_time:
        :param raw_data_path:
        :param dlim_sec:
        :param isSectional:
        :param time_res: 'month', 'year', 'hour'
        :param space_res: 'full', 'locations'
        :param print_stat:
        :param model_name:
        """
        if dlim_sec is None:
            dlim_sec = [5,39.6]
        if chunks is None:
            chunks = {'diameter': 20}
        self.chunks = chunks
        self.dmin_sec = dlim_sec[0]
        self.dmax_sec = dlim_sec[1]
        self.nr_bins = nr_bins

        bin_diam, bin_diam_int = self.get_sectional_params()

        self.bin_diameter_int = bin_diam_int
        self.bin_diameter = bin_diam
        if use_eusaar_diam:
            d_arr = distc_var.get_diameter_sized()
        else:
            d_arr = np.logspace(np.log10(3), 4, 50)  # np.logspace(0, 4, 50)
        self.diameter = xr.DataArray(d_arr,
                                     name='diameter',
                                     coords=[d_arr],
                                     dims='diameter',
                                     attrs={'units': 'nm'})
        # self.read_from_file = read_from_file
        self.model_name = model_name
        # self.case_plotting_name = model_name
        self.dataset = None
        self.use_pressure_coords = use_pressure_coords
        self.case_name_nice = get_nice_name_case(case_name)
        self.case_name = case_name
        self.raw_data_path = raw_data_path
        self.from_time = from_time
        self.to_time = to_time
        self.time_resolution = time_res
        self.space_resolution = space_res
        self.history_field = history_field
        self.locations = locations
        self.isSectional = isSectional
        self.final_sizedist_vars = self.varl_sizedist_final()

        # self.savepath_sizedist = self.dataset_savepath(case_name, model_name)
        self.print = print_stat
        self.attrs_ds = dict(raw_data_path=str(self.raw_data_path),
                             model=self.model_name, model_name=self.model_name,
                             case_name=self.case_name, case=self.case_name,
                             case_name_nice=self.case_name_nice,
                             isSectional=str(self.isSectional),
                             from_time=self.from_time,
                             to_time=self.to_time,
                             time_resolution=self.time_resolution,
                             history_field=self.history_field,
                             pressure_coords=str(self.use_pressure_coords)
                             )
        # self.size_dtset = self.get_sizedistrib_dataset()

        # self.attrs = vars(self        self.dmin_sec = dlim_sec[0]
        return

    def varl_sizedist_final(self):
        if self.isSectional:
            return [D_NDLOG_D_SEC, D_NDLOG_D_MOD]
        else:
            return [D_NDLOG_D_MOD]

    def get_sectional_params(self):
        """
        Set sectional parameters.
        :return:
        """
        max_diameter = self.dmax_sec
        min_diameter = self.dmin_sec
        nr_bins = self.nr_bins
        bin_diam, bin_diam_int = get_bin_diameter(nr_bins, min_diameter, max_diameter)
        return bin_diam, bin_diam_int

    def to_netcdf(self, ds, fn):
        """
        Saves dataset to netcdf with attributes
        :param ds:
        :param fn:
        :return:
        """
        atts = self.attrs_ds
        attrs_to = ds.attrs
        update_dic(atts, attrs_to)
        ds = ds.assign_attrs(attrs_to)
        delayed_obj = ds.to_netcdf(fn, compute=False)  # , chunks={'diameter':1})
        with ProgressBar():
            results = delayed_obj.compute()

    def get_sizedist_var(self, var_names=None, CHUNKS=None):
        """

        :param var_names:
        :param CHUNKS:
        :return:
        """
        if var_names is None:
            if not self.isSectional:
                var_names = [D_NDLOG_D_MOD]
            else:
                var_names = [D_NDLOG_D_MOD, D_NDLOG_D_SEC]
        if CHUNKS is None:
            CHUNKS = self.chunks
        fn_list = []
        for var_name in var_names:
            # fn = self.dataset_savepath_var(var_name, self.case_name, self.model_name)
            fn = self.make_sure_sizedist_varfile_exists(var_name)
            fn_list.append(fn)
        log.ger.info('Opening: [' + 'j'.join(fn_list) + ']')
        ds = xr.open_mfdataset(fn_list, combine='by_coords', chunks=CHUNKS)
        make_tot = True
        for var in TOTAL_VARS:
            if var not in ds.data_vars:
                make_tot = False

        if make_tot:
            self.make_total_sizedist(ds)
        elif D_NDLOG_D_MOD in ds.data_vars and not self.isSectional:
            ds['dNdlogD'] = ds[D_NDLOG_D_MOD].copy()
            ds['dNdlogD'].attrs = ds[D_NDLOG_D_MOD].attrs
            # ds['dNdlogD'].attrs['long_name'] = 'dNdlogD'

        return ds

    def make_total_sizedist(self, ds):
        if self.isSectional:
            ds['dNdlogD'] = ds[D_NDLOG_D_SEC] + ds[D_NDLOG_D_MOD]
        else:
            ds['dNdlogD'] = ds[D_NDLOG_D_MOD].copy()
        ds['dNdlogD'].attrs = ds[D_NDLOG_D_MOD].attrs
        ds['dNdlogD'].attrs['long_name'] = 'dNdlogD'

    # def get_var(self, var_name, CHUNKS={'diameter': 20}):
    #    fn = self.dataset_savepath_var(var_name, self.case_name, self.model_name)

    #   return xr.open_dataset(fn, chunks=CHUNKS)

    # def dataset_savepath(self, case_name, model_name):
    #     """
    #     Returns filename of dataset
    #     :param case_name:
    #     :param model_name:
    #     :return:
    #     """
    #
    #     case_name = case_name.replace(' ', '_')
    #     _savep = self.default_savepath_root
    #     st = '%s/%s/%s/%s' % (_savep, model_name, case_name, case_name)
    #     st = st + '_%s_%s' % (self.from_time, self.to_time)
    #     st = st + '_%s_%s' % (self.time_resolution, self.space_resolution)
    #     fn = st + '.nc'
    #     make_folders(fn)
    #     return fn

    def dataset_savepath_var(self, var_name, case_name, model_name):
        """
        Returns filename of dataset
        :param var_name:
        :param case_name:
        :param model_name:
        :return:
        """
        case_name = case_name.replace(' ', '_')
        _savep = self.default_savepath_root

        st = '%s/%s/%s/%s_%s' % (_savep, model_name, case_name, var_name, case_name)
        st = st + '_%s_%s' % (self.from_time, self.to_time)
        st = st + '_%s_%s' % (self.time_resolution, self.space_resolution)
        fn = st + '.nc'
        make_folders(fn)
        return fn

    def compute_sizedist_var(self, var_name):
        """
        Compute sizedistribution variable.
        :param var_name:
        :return:
        """
        if 'sec' in var_name:
            ds = self.compute_sizedist_sec_var(var_name)
            return ds
        else:
            ds = self.compute_sizedist_mod_var(var_name)
            return ds

    def compute_sizedist_mod_var(self, var_name):
        """
        Compute modal file
        :param var_name: The variable to be computed
        :return:
        """
        num = var_name[-2:]
        if num.isdigit():
            return self._calc_sizedist_mod_var_nr(var_name, num=num)
        else:
            return self.compute_sizedist_mod_tot()

    def make_sure_sizedist_varfile_exists(self, var_name):
        """
        Check that var is computed and if not compute the var
        :param var_name: the variable in question
        :return: filename of file
        """
        fn = self.dataset_savepath_var(var_name, self.case_name, self.model_name)
        if not os.path.isfile(fn):
            log.ger.debug('computing file for %s: \n %s' % (var_name, fn))
            t1 = time.time()
            ds = self.compute_sizedist_var(var_name)
            if not os.path.isfile(fn):
                self.to_netcdf(ds, fn)
                ds.close()
                del ds
            t2 = time.time()
            log.ger.info('computed %s in time : %s m' % (fn, (t2 - t1) / 60.))

        return fn

    def compute_sizedist_tot(self):
        if self.isSectional:
            ds_sec = self.compute_sizedist_sec_tot()
            ds_mod = self.compute_sizedist_mod_tot()
            return xr.merge([ds_sec,ds_mod])
        return self.compute_sizedist_mod_tot()

    def compute_sizedist_mod_tot(self):
        """
        Compute total of modal, i.e. the sum of all modes.
        :return:
        """
        dNdlogD_var = D_NDLOG_D_MOD
        fn_final = self.dataset_savepath_var(dNdlogD_var, self.case_name, self.model_name)
        if os.path.isfile(fn_final):
            log.ger.info('Modal tot file found %s' % fn_final)
            return xr.open_dataset(fn_final)
        else:
            log.ger.info('Computing file %s' % fn_final)
        vs_NCONC = varListNorESM['NCONC']
        fl = []
        l_vars_dNdlogD = []
        for var in vs_NCONC:
            dNdlogD_var_nr = _varname_mod_nr(var)
            l_vars_dNdlogD.append(dNdlogD_var_nr)
            _f = self.make_sure_sizedist_varfile_exists(dNdlogD_var_nr)
            fl.append(_f)
            log.ger.debug('Modal tot file found %s' % var)
        log.ger.debug(fl)
        start = time.time()

        CHUNKS = {'diameter': 20}
        ds = xr.open_mfdataset(fl, combine='by_coords', parallel=True,
                               chunks=CHUNKS)  # .isel(diameter=slice(0,30))#.squeeze()

        da = sum_vars(ds, l_vars_dNdlogD, dNdlogD_var, long_name='dN/dlogD (modal)')
        self.to_netcdf(da, fn_final)  # , compute=False)  # , chunks={'diameter':1})
        # delayed_obj = self.to_netcdf(da, fn_final)#, compute=False)  # , chunks={'diameter':1})
        # with ProgressBar():
        #    results = delayed_obj.compute()
        end = time.time()
        log.ger.debug('Time elapsed: %f' % (end - start))
        ds.close()
        da.close()
        del da
        del ds
        return xr.open_dataset(fn_final)

    def _calc_sizedist_mod_var_nr(self, var_name, num=None):
        if num is None:
            num = var_name[-2:]
        varN = _get_nconc_varname(num)
        varNMR = _get_nmr_varname(num)
        varSIG = _get_sig_varname(num)
        varl = [varN, varNMR, varSIG]
        input_ds = get_pressure_coord_fields(self.case_name, varl, self.from_time, self.to_time,
                                             self.history_field, model=self.model_name)
        ds_dNdlogD = self._compute_dNdlogD_mod(var_name, self.diameter, input_ds, varN, varNMR, varSIG)
        input_ds.close()
        del input_ds
        return ds_dNdlogD

    def _compute_dNdlogD_mod(self, dNdlogD_var, diameter, input_ds, varN, varNMR, varSIG):
        size_dtset = xr.Dataset(coords={**input_ds.coords, 'diameter': self.diameter})
        log.ger.debug(varN)  # varListNorESM['NCONC'][i])
        # varNMR = varListNorESM['NMR'][i]
        NCONC = input_ds[varN]  # [::]*10**(-6) #m-3 --> cm-3
        SIGMA = input_ds[varSIG]  # [::]#*10**6
        NMR = input_ds[varNMR] * 2.  # radius --> diameter
        # number:
        size_dtset[dNdlogD_var] = dNdlogD_modal(NCONC, NMR, SIGMA, diameter)
        size_dtset[dNdlogD_var].attrs['units'] = 'cm-3'
        size_dtset[dNdlogD_var].attrs['long_name'] = 'dN/dlogD (mode' + dNdlogD_var[-2:] + ')'
        return size_dtset

    def compute_sizedist_sec_tot(self, chunks=None):
        """
        Compute total of sizedistribution sectional total
        :param chunks:
        :return:
        :return:
        """
        if chunks is None:
            chunks = self.chunks
        dNdlogD_var = D_NDLOG_D_SEC
        fn_final = self.dataset_savepath_var(dNdlogD_var, self.case_name, self.model_name)
        if os.path.isfile(fn_final):
            log.ger.info('opening :%s' % fn_final)
            return xr.open_dataset(fn_final)
        # vs_NCONC = varListNorESM['NCONC']
        if not self.isSectional:
            return
        input_vars = self.get_varlist_input_sec()
        #
        start = time.time()
        l_vars_dNdlogD = []
        fl = []
        for var in input_vars:
            dNdlogD_var_nr = _varname_sec_nr(var)
            fl.append(self.make_sure_sizedist_varfile_exists(dNdlogD_var_nr))
            l_vars_dNdlogD.append(dNdlogD_var_nr)
        fl = list(dict.fromkeys(fl))
        ds = xr.open_mfdataset(fl, combine='by_coords', chunks=chunks, parallel=True)
        da = sum_vars(ds, l_vars_dNdlogD, dNdlogD_var, long_name='dN/dlogD (sectional)')
        # ex_var = l_vars_dNdlogD[0]
        # keep_coords = list(ds[ex_var].dims)
        # drop_l = list(set(ds.variables) - set(l_vars_dNdlogD + keep_coords))
        # ds = ds.drop(drop_l)
        # log.ger.warning(ds)
        # da = ds.to_array(dim='variable', name=dNdlogD_var)  # 'dNdlogD_mod')
        # da = da.sum('variable')
        self.to_netcdf(da, fn_final)  # da.to_netcdf(fn_final, compute=False)  # , chunks={'diameter':1})
        # delayed_obj =self.to_netcdf(da,fn_final)# da.to_netcdf(fn_final, compute=False)  # , chunks={'diameter':1})
        # with ProgressBar():
        #    results = delayed_obj.compute()
        end = time.time()
        log.ger.debug('Time elapsed: %f' % (end - start))

        ds.close()
        da.close()
        del da
        del ds
        return xr.open_dataset(fn_final)

        # return #ds

    def compute_sizedist_sec_var(self, var_name):
        """
        Compute sectional dNdlogD variable
        :param var_name:
        :return:
        """
        num = var_name[-2:]
        if num.isdigit():

            return self._calc_sizedist_sec_var_nr(var_name, num=num)
        else:
            return self.compute_sizedist_sec_tot()

    def _calc_sizedist_sec_var_nr(self, var_name, num):
        if num is None:
            num = var_name[-2:]
        varl = get_nrSEC_varname(num)
        input_ds = get_pressure_coord_fields(self.case_name,
                                             varl,
                                             self.from_time,
                                             self.to_time,
                                             self.history_field,
                                             model=self.model_name)

        ds_dNdlogD = self._compute_dNdlogD_sec(var_name, self.diameter, input_ds, num)
        input_ds.close()
        del input_ds
        return ds_dNdlogD

    def get_input_data(self, varlist):
        if self.isSectional:
            get_pressure_coord_fields(self.case_name,
                                      varlist,
                                      self.from_time,
                                      self.to_time,
                                      self.history_field,
                                      model=self.model_name)
        else:
            log.ger.warning('NOT IMPLEMENTED FOR NATIVE LEV COORD')

    def get_varlist_input_sec(self):
        """
        returns necessary input vars for sectional
        :return:
        """
        nr_bins = self.nr_bins
        vl = []
        for i in range(1, nr_bins + 1):
            vl = vl + get_nrSEC_varname(i)
        return vl

    def get_varlist_input(self):
        vl = []
        if self.isSectional:
            vl = self.get_varlist_input_sec()
        for key in sized_varListNorESM:
            vl = vl + sized_varListNorESM[key]
        return vl

    def _compute_dNdlogD_sec(self, dNdlogD_var, diameter, input_ds, num):
        varnSOA = get_nrSEC_varname(num)[0]
        varnSO4 = get_nrSEC_varname(num)[1]

        size_dtset = xr.Dataset(coords={**input_ds.coords, 'diameter': self.diameter})
        SOA = input_ds[varnSOA]  # [::]*10**(-6) #m-3 --> cm-3
        SO4 = input_ds[varnSO4]  # [::]#*10**6
        # number:
        bin_diameter_int = self.bin_diameter_int
        size_dtset[dNdlogD_var] = dNdlogD_sec(diameter, SOA, SO4, num, bin_diameter_int)
        size_dtset[dNdlogD_var].attrs['units'] = 'cm-3'
        size_dtset[dNdlogD_var].attrs['long_name'] = 'dNdlogD (sectional' + dNdlogD_var[-2:] + ')'

        return size_dtset
    # def save_folder(self):
    #    """
    #    Returns folder for output
    #    :   return:
    #    """
    #    st = self.savepath_sizedist
    #    return extract_path_from_filepath(st)


def dNdlogD_sec(diameter, SOA, SO4, num, bin_diameter_int):
    """
    Calculate dNdlogD sectional for individual bin
    :param diameter:
    :param SOA:
    :param SO4:
    :param num:
    :param bin_diameter_int:
    :return:
    """
    # SECnr = self.nr_bins
    # binDiam_l = self.bin_diameter_int  # binDiameter_l
    if type(num) is str:
        num = int(num)

    diam_u = bin_diameter_int[num]  # bin upper lim
    diam_l = bin_diameter_int[num - 1]  # bin lower lim

    SOA, dum = xr.broadcast(SOA, diameter)  # .values  # *1e-6
    SO4, dum = xr.broadcast(SO4, diameter)  # .values  # *1e-6
    dNdlogD = (SOA + SO4) / (np.log(diam_u / diam_l))  #
    in_xr = dict(diameter=diameter[(diameter >= diam_l) & (diameter < diam_u)])
    out_da = xr.DataArray(np.zeros_like(dNdlogD.values), coords=dNdlogD.coords)  # dNdlogD*0.#, diameter)
    out_da.loc[in_xr] = dNdlogD.loc[in_xr]
    return out_da


def get_nrSEC_varname(num):
    """
    Get necessary input nr sec variables
    :param num: number e.g. 02
    :return: list of variables.
    """
    SOA = 'nrSOA_SEC'
    SO4 = 'nrSO4_SEC'
    if type(num) is int:
        num = '%02.0f' % num
    return [SOA + num, SO4 + num]
    # fl = ['nrSOA%s']


def dNdlogD_modal(NCONC, NMR, SIGMA, diameter):
    """

    :param NCONC:
    :param NMR: in diameter!!
    :param SIGMA:
    :param diameter:
    :return:
    """
    da = NCONC / (np.log(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
        -(np.log(diameter) - np.log(NMR)) ** 2 / (2 * np.log(SIGMA) ** 2))
    return da


def _varname_sec_nr(varN):
    dNdlogD_var = 'dNdlogD_sec%s' % (varN[-2::])
    return dNdlogD_var


def _varname_mod_nr(varN):
    dNdlogD_var = 'dNdlogD_mode%s' % (varN[-2::])
    return dNdlogD_var


def _get_sig_varname(num):
    return 'SIGMA%s' % num


def _get_nmr_varname(num):
    return 'NMR%s' % num


def _get_nconc_varname(num):
    return 'NCONC%s' % num


def get_bin_diameter(nr_bins, min_diameter=5.0,
                     max_diameter=39.6):  # minDiameter=3.0e-9,maxDiameter=23.6e-9):
    """
    Set sectional parameters.
    :param nr_bins:
    :param min_diameter:
    :param max_diameter:
    :return:
    :return:
    """
    # %%

    d_rat = (max_diameter / min_diameter) ** (1 / nr_bins)
    binDiam = np.zeros(nr_bins)
    binDiam_in = np.zeros(nr_bins + 1)
    # binDiam_h = np.zeros(nr_bins)
    binDiam[0] = min_diameter
    binDiam_in[0] = (2 / (1 + d_rat)) * binDiam[0]
    # binDiam_h[0] = d_rat * binDiam[0] * (2 / (1 + d_rat))
    for i in np.arange(1, nr_bins):
        binDiam[i] = binDiam[i - 1] * d_rat
        binDiam_in[i] = (2 / (1 + d_rat)) * binDiam[i]
        # binDiam_h[i] = (2 / (1 + d_rat)) * d_rat * binDiam[i]
    binDiam_in[nr_bins] = max_diameter
    # %%
    return binDiam, binDiam_in
