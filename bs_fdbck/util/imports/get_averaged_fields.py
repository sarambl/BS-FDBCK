import bs_fdbck.util.slice_average.avg_pkg as avg_pkg
import bs_fdbck.util.imports.get_pressure_coord_fields
from bs_fdbck.util.slice_average.avg_pkg import maps
import bs_fdbck.util.slice_average.avg_pkg.levlat
import bs_fdbck.util.slice_average.avg_pkg.one_value
import bs_fdbck.util.slice_average.avg_pkg.profiles
from bs_fdbck.util.imports.get_fld_fixed import get_field_fixed
from bs_fdbck.util.imports.import_fields_xr_v2 import import_constants
import numpy as np
import xarray as xr

from bs_fdbck.constants import get_input_datapath


# %%
def get_monthly_fields(cases, varlist, from_time, to_time,
                       pressure_adjust=True,
                       raw_data_path=get_input_datapath(),
                       model='NorESM'):
    """
    Get monthly fields of files (i.e. not averaged)
    :param cases: list of cases to load
    :param varlist: variables
    :param from_time:
    :param to_time:
    :param pressure_adjust: if true, returns var in pressure coordinates
    :param raw_data_path: where to find the raw data
    :param model: model name
    :return:
    """
    return_dic = {}
    for case in cases:
        dummy = get_field_fixed(case,
                                varlist,
                                from_time, to_time,
                                raw_data_path=raw_data_path,
                                pressure_adjust=pressure_adjust)
        ds_constants = import_constants(case,
                                        path=raw_data_path,
                                        model=model)
        dummy = xr.merge([dummy, ds_constants])
        return_dic[case] = dummy.copy()
        del dummy
    return return_dic


def get_levlat_cases(cases, varlist, startyear, endyear,
                     pressure_adjust=True,
                     raw_data_path=get_input_datapath(),
                     model='NorESM', recalculate=False,
                     time_mask=None):
    """
    Get cases averaged over lon and time so dimensions are level and latitude
    :param cases:
    :param varlist:
    :param startyear:
    :param endyear:
    :param pressure_adjust:
    :param raw_data_path:
    :param model:
    :param recalculate: if True, loads and calculates again instead of reading the pre-calculated
    :param time_mask:
    :return:
    """
    dum_dic = get_levlat_nested([model], cases, varlist, startyear, endyear,
                                pressure_adjust=pressure_adjust,
                                raw_data_path=raw_data_path,
                                recalculate=recalculate,
                                time_mask=time_mask)
    return dum_dic[model]


def get_levlat_nested(models, cases, varlist, from_time, to_time,
                      pressure_adjust=True,
                      raw_data_path=get_input_datapath(),
                      recalculate=False,
                      time_mask=None):
    """
    Load or calculate avg maps
    :param recalculate:
    :param time_mask:
    :param models:
    :param cases:
    :param varlist:
    :param from_time:
    :param to_time:
    :param pressure_adjust:
    :param raw_data_path:
    :return: dict [models][cases]:dataset
    """
    save_ds = None
    nested_ds = {}
    ds_levlat = None
    for model in models:
        dummy_dic = {}
        for case in cases:
            first = True
            for var in varlist:
                # Check if data already calculated:
                var_loaded = False
                if not recalculate:
                    ds_levlat, var_loaded = avg_pkg.levlat.load_average2levlat(model, case,
                                                                               var, from_time,
                                                                               to_time,
                                                                               pressure_adjust,
                                                                               time_mask=time_mask)
                # If not, load data and calculate average:
                if not var_loaded or recalculate:
                    # If weighted mean, needs to import 2 variables to create!
                    var_subl = avg_pkg.get_fields4weighted_avg(var)
                    dummy = get_field_fixed(case,
                                            var_subl,
                                            from_time, to_time,
                                            raw_data_path=raw_data_path,
                                            pressure_adjust=pressure_adjust)
                    ds_constants = import_constants(case,
                                                    path=raw_data_path,
                                                    model=model)
                    dummy = xr.merge([dummy, ds_constants])
                    ds_levlat = avg_pkg.levlat.get_average_levlat(dummy, [var], case,
                                                                  from_time, to_time,
                                                                  pressure_coord=pressure_adjust,
                                                                  model_name=model,
                                                                  recalculate=recalculate,
                                                                  time_mask=time_mask)
                    # because of slight (e-14) difference from noresm2 to 1
                ds_levlat['lat'] = np.round(ds_levlat['lat'].values, decimals=5)
                ds_levlat['lon'] = np.round(ds_levlat['lon'].values, decimals=5)
                if first:
                    save_ds = ds_levlat.copy()
                    first = False
                else:
                    if var in ds_levlat:
                        save_ds[var] = ds_levlat[var]  # .copy()
            dummy_dic[case] = save_ds.copy()
            del save_ds
        nested_ds[model] = dummy_dic.copy()
    return nested_ds


def get_maps_cases(cases,
                   varlist,
                   startyear,
                   endyear,
                   avg_over_lev=False,
                   pmin=None,
                   p_level=None,
                   pressure_adjust=True,
                   raw_data_path=get_input_datapath(),
                   model='NorESM', recalculate=False,
                   time_mask=None):
    """
    Load averages in maps, i.e. latitude, longitude and averaged over time/lev
    :param cases: list of cases
    :param varlist: list of variables
    :param startyear: from which time e.g. '2008-01'
    :param endyear: to which time e.g. '2009-12'
    :param avg_over_lev: if True, averages over levels
    :param pmin: Lowest pressure level to include in average over levels
    :param p_level: If not average over level, uses this to pick out which lev to return
    :param pressure_adjust: In pressure coordinates
    :param raw_data_path: path to raw data
    :param model: model name
    :param recalculate:
    :param time_mask: e.g. 'JJA', if None, all year
    :return: dictionary with case names as keys and xr.Dataset as elements.
    """
    dum_dic = get_maps_nested([model], cases, varlist, startyear, endyear,
                              avg_over_lev=avg_over_lev,
                              pmin=pmin,
                              p_level=p_level,
                              pressure_adjust=pressure_adjust,
                              raw_data_path=raw_data_path,
                              recalculate=recalculate,
                              time_mask=time_mask)
    return dum_dic[model]


def get_maps_nested(models, cases, varlist, from_time, to_time,
                    avg_over_lev=False,
                    pmin=None,
                    p_level=None,
                    pressure_adjust=True,
                    raw_data_path=get_input_datapath(),
                    recalculate=False,
                    time_mask=None):
    """
    Load or calculate avg maps. Return map averages over lev/time.
    Outputs dictionary with dic[model_name][case_name] = xr.Dataset
    :param models:
    :param cases:
    :param varlist:
    :param from_time:
    :param to_time:
    :param avg_over_lev:
    :param pmin:
    :param p_level:
    :param pressure_adjust:
    :param raw_data_path:
    :param recalculate:
    :param time_mask:
    :return: Dictionary with dic[model_name][case_name] = xr.Dataset
    """
    ds_map = None
    save_ds = None
    if avg_over_lev and pmin is None:
        pmin = 850.  # hPa
    if not avg_over_lev and p_level is None:
        p_level = 1013.
    nested_ds = {}

    for model in models:
        dummy_dic = {}
        for case in cases:
            first = True
            for var in varlist:
                # Check if data already calculated:
                var_loaded = False
                if not recalculate:
                    ds_map, var_loaded = maps.load_average2map(model, case,
                                                               var, from_time,
                                                               to_time,
                                                               avg_over_lev,
                                                               pmin, p_level,
                                                               pressure_adjust,
                                                               time_mask=time_mask)
                # If not, load data and calculate average:
                if not var_loaded or recalculate:
                    # If weighted mean, needs to import 2 variables to create!
                    var_subl = avg_pkg.get_fields4weighted_avg(var)
                    # .get_vars_for_computed_vars(var_subl, model)#+[var]
                    dummy = get_field_fixed(case,
                                            var_subl,
                                            from_time, to_time,
                                            raw_data_path=raw_data_path,
                                            pressure_adjust=pressure_adjust)
                    ds_constants = import_constants(case,
                                                    path=raw_data_path,
                                                    model=model)
                    dummy = xr.merge([dummy, ds_constants])
                    ds_map = avg_pkg.maps.get_average_map2(dummy, [var], case,
                                                           from_time, to_time,
                                                           avg_over_lev=avg_over_lev,
                                                           pmin=pmin, p_level=p_level,
                                                           pressure_coord=pressure_adjust,
                                                           model_name=model,
                                                           recalculate=recalculate,
                                                           time_mask=time_mask)
                if first:
                    save_ds = ds_map.copy()
                    first = False
                else:
                    if var in ds_map:
                        save_ds[var] = ds_map[var]  # .copy()
            dummy_dic[case] = save_ds.copy()
            del save_ds
        nested_ds[model] = dummy_dic
    return nested_ds


def get_profiles(cases, varlist, from_time, to_time,
                 area='Global',
                 pressure_adjust=True,
                 raw_data_path=get_input_datapath(),
                 recalculate=False,
                 model='NorESM', time_mask=None):
    """
    Load average profiles for cases in cases
    :param cases: list of case names
    :param varlist: list of vars
    :param from_time:
    :param to_time:
    :param area:
    :param pressure_adjust: if True: pressure coordinates
    :param raw_data_path:
    :param recalculate: if True, recalculates average
    :param model: model name
    :param time_mask: e.g. 'JJA'
    :return:
    """
    save_ds = None
    ds_profile = None
    dic_cases = {}
    for case in cases:
        first = True
        for var in varlist:
            # Check if data already calculated:
            var_loaded = False
            if not recalculate:
                ds_profile = avg_pkg.profiles.load_profile(case, var, area,
                                                           from_time, to_time,
                                                           pressure_adjust,
                                                           model=model,
                                                           time_mask=time_mask)
                var_loaded = ds_profile is not None
            # If not, load data and calculate average:
            if not var_loaded or recalculate:
                var_subl = avg_pkg.get_fields4weighted_avg(var)  # .get_vars_for_computed_vars(var_subl, model)#+[var]
                dummy = get_field_fixed(case,
                                        var_subl,
                                        from_time, to_time,
                                        raw_data_path=raw_data_path,
                                        pressure_adjust=pressure_adjust)
                ds_constants = import_constants(case,
                                                path=raw_data_path,
                                                model=model)
                dummy = xr.merge([dummy, ds_constants])
                ds_profile = avg_pkg.profiles.get_average_profile2(dummy, var,
                                                                   model, case,
                                                                   area, pressure_adjust,
                                                                   from_time, to_time,
                                                                   look_for_file=(not recalculate),
                                                                   time_mask=time_mask)

            if first:
                save_ds = ds_profile.copy()
                first = False
            else:
                if var in ds_profile:
                    save_ds[var] = ds_profile[var]  # .copy()
        dic_cases[case] = save_ds.copy()
        del save_ds
    return dic_cases


def get_profiles_nested(models, cases, varlist, startyear, endyear, area,
                        pressure_adjust=True,
                        raw_data_path=get_input_datapath(),
                        time_mask=None):
    nested_ds = {}
    for model in models:
        _dic_cases = get_profiles(cases, varlist, startyear, endyear,
                                  area=area,
                                  pressure_adjust=pressure_adjust,
                                  raw_data_path=raw_data_path,
                                  model=model, time_mask=time_mask)
        nested_ds[model] = _dic_cases
    return nested_ds


def get_area_avg_dic(cases,
                     varlist,
                     area,
                     from_time,
                     to_time,
                     avg_over_lev=False,
                     pmin=None,
                     p_level=None,
                     pressure_adjust=True,
                     raw_data_path=get_input_datapath(), model_name='NorESM'):
    """
    Load or calculate avg maps
    :param area:
    :param model_name:
    :param cases:
    :param varlist:
    :param from_time:
    :param to_time:
    :param avg_over_lev:
    :param pmin:
    :param p_level:
    :param pressure_adjust:
    :param raw_data_path:
    :return: dict [models][cases]:dataset
    """
    dic = get_area_avg_nested([model_name], cases, varlist, area, from_time, to_time,
                              avg_over_lev=avg_over_lev,
                              pmin=pmin,
                              p_level=p_level,
                              pressure_adjust=pressure_adjust,
                              raw_data_path=raw_data_path)
    return dic[model_name]


def get_area_avg_nested(models, cases, varlist, area, from_time, to_time,
                        avg_over_lev=False,
                        pmin=None,
                        p_level=None,
                        pressure_adjust=True,
                        raw_data_path=get_input_datapath()):
    """
    Load or calculate avg maps
    :param area:
    :param models:
    :param cases:
    :param varlist:
    :param from_time:
    :param to_time:
    :param avg_over_lev:
    :param pmin:
    :param p_level:
    :param pressure_adjust:
    :param raw_data_path:
    :return: dict [models][cases]:dataset
    """
    save_ds = None
    nested_ds = {}
    for model in models:
        dummy_dic = {}
        for case in cases:
            first = True
            for var in varlist:
                # Check if data already calculated:
                ds_avg, var_loaded = bs_fdbck.util.slice_average.avg_pkg.one_value.load_area_file_var(model,
                                                                                                       case,
                                                                                                       var,
                                                                                                       area,
                                                                                                       from_time,
                                                                                                       to_time,
                                                                                                       avg_over_lev,
                                                                                                       pmin,
                                                                                                       p_level,
                                                                                                       pressure_adjust
                                                                                                       )
                # If not, load data and calculate average:
                if not var_loaded:
                    var_subl = avg_pkg.get_fields4weighted_avg(var)
                    dummy = get_field_fixed(case,
                                            var_subl,
                                            from_time, to_time,
                                            raw_data_path=raw_data_path,
                                            pressure_adjust=pressure_adjust)

                    ds_constants = import_constants(case,
                                                    path=raw_data_path,
                                                    model=model)
                    dummy = xr.merge([dummy, ds_constants])

                    ds_avg = bs_fdbck.util.slice_average.avg_pkg.one_value.get_average_area(dummy,
                                                                                             [var], case,
                                                                                             area, from_time,
                                                                                             to_time, model, pmin,
                                                                                             avg_over_lev, p_level,
                                                                                             pressure_adjust,
                                                                                             look_for_file=False)

                if first:
                    save_ds = ds_avg.copy()
                    first = False
                else:
                    if var in ds_avg:
                        save_ds[var] = ds_avg[var]  # .copy()
            dummy_dic[case] = save_ds.copy()
            del save_ds
        nested_ds[model] = dummy_dic.copy()
    return nested_ds
