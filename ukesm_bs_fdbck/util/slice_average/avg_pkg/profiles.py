import os

import xarray as xr

from ukesm_bs_fdbck.util import practical_functions
from ukesm_bs_fdbck.util.filenames import filename_profile_avg, filename_profile_avg_allvars
from ukesm_bs_fdbck.util.slice_average.avg_pkg import get_fields4weighted_avg, average_model_var, \
    compute_weighted_averages, is_weighted_avg_var


def get_average_profile2(xr_ds: xr.Dataset, var: str, model: str, case: str, area: str
                         , pressure_coord: bool, from_time, to_time
                         , look_for_file: bool = True,
                         time_mask=None) -> xr.Dataset:
    """
    :param time_mask:
    :param to_time:
    :param from_time:
    :param var:
    :param xr_ds:
    :param model:
    :param case:
    :param area:
    :param pressure_coord:
    :param look_for_file:
    :return:
    """
    # df = pd.DataFrame(columns=[model_name], index = varList)
    xr_out = xr_ds
    # print(xr_out)
    # found_area_mean = False
    ds_area_avg = load_profile(case, var, area, from_time, to_time, pressure_coord, time_mask=time_mask)
    if ds_area_avg is not None and look_for_file:
        return ds_area_avg

    else:
        # avg over all except lev.
        sub_varL = get_fields4weighted_avg(var)
        dummy = xr_ds.copy()
        for svar in sub_varL:
            dummy = average_model_var(dummy, svar, area=area,
                                      dim = list(set(xr_ds[svar].dims)-{'lev'}),
                                      time_mask=time_mask)
        dummy = compute_weighted_averages(dummy, var, model)
        dummy[var].attrs['Calc_weight_mean']=str(is_weighted_avg_var(var)) + 'profile area: %s'%area

        filen = filename_profile_avg(var, case, area, from_time,
                                     to_time, pressure_coord,
                                     model=model,
                                     time_mask=time_mask)
        practical_functions.make_folders(filen)

        practical_functions.save_dataset_to_netcdf(dummy, filen)
        xr_out[var] = dummy[var]
    return xr_out


def get_average_profile(xr_ds: xr.Dataset, varList: list, model_name: str, case_name: str, area: str
                    , pressure_adj: bool
                    , look_for_file: bool = True) -> xr.Dataset:
    """
    :param xr_ds:
    :param varList:
    :param model_name:
    :param case_name:
    :param area:
    :param pressure_adj:
    :param look_for_file:
    :return:
    """
    # df = pd.DataFrame(columns=[model_name], index = varList)
    startyear = xr_ds.attrs['startyear']
    endyear = xr_ds.attrs['endyear']
    xr_out = xr_ds.copy()
    for var in varList:
        print(var)
        found_area_mean = False
        ds_area_avgs, filen = load_profile_file_allvars(model_name, case_name, var, area, startyear, endyear, pressure_adj)
        if ds_area_avgs is not None:
            if var in ds_area_avgs:
                found_area_mean=True
                xr_out[var] = ds_area_avgs[var]

        if not found_area_mean or not look_for_file:

            # avg over all except lev.
            sub_varL = get_fields4weighted_avg(var)
            dummy = xr_ds.copy()
            for svar in sub_varL:
                dummy = average_model_var(dummy, svar, area=area, dim = list(set(xr_ds[svar].dims)-{'lev'})) #\
            dummy = compute_weighted_averages(dummy, var, model_name)
            dummy[var].attrs['Calc_weight_mean'] = str(is_weighted_avg_var(var)) + 'profile area: %s'%area
            practical_functions.make_folders(filen)
            if ds_area_avgs is None:
                practical_functions.save_dataset_to_netcdf(dummy, filen)
            else:
                ds_area_avgs[var] = dummy[var]
                practical_functions.save_dataset_to_netcdf(ds_area_avgs, filen)
            xr_out[var] = dummy[var]
    return xr_out


def load_profile(case, var, area, from_time, to_time, pressure_coord, model='NorESM',
                 time_mask=None):
    filen = filename_profile_avg(var, case, area, from_time, to_time, pressure_coord,
                                 model=model, time_mask=time_mask)
    if os.path.isfile(filen):
        print('Loading file %s' % filen)
        xr_ds = xr.open_dataset(filen).load().copy()
        return xr_ds
    else:
        print('Did not find profile mean with filename: %s ' % filen)
        return None


def load_profile_file_allvars(model, case, var, area,  startyear, endyear, pressure_adj):
    filen = filename_profile_avg_allvars(model, case, area, startyear, endyear, pressure_adj)
    if os.path.isfile(filen):
        print('Loading file %s' % filen)
        xr_ds = xr.open_dataset(filen).load().copy()
        return xr_ds, filen
    else:
        print('Did not find map mean with filename: %s ' % filen)
        return None, filen