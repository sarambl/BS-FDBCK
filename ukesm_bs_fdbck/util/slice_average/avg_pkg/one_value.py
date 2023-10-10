import os

import xarray as xr

from ukesm_bs_fdbck.util import practical_functions
from ukesm_bs_fdbck.util.filenames import filename_area_avg
from ukesm_bs_fdbck.util.slice_average.avg_pkg import get_fields4weighted_avg, average_model_var, \
    compute_weighted_averages, is_weighted_avg_var


def get_average_area(xr_ds: xr.Dataset, varList: list, case_name: str, area: str, from_time,
                     to_time, model_name: str, pmin: float,
                     avg_over_lev: bool, p_level: float, pressure_coords: bool, look_for_file: bool = True, ) -> xr.Dataset:
    """
    :param area:
    :param from_time:
    :param to_time:
    :param xr_ds:
    :param varList:
    :param model_name:
    :param pmin:
    :param case_name:
    :param avg_over_lev:
    :param p_level:
    :param pressure_coords:
    :param look_for_file:
    :return:
    """
    from_time = from_time #xr_ds.attrs['startyear']
    to_time = to_time #xr_ds.attrs['endyear']

    xr_out = xr_ds.copy()
    for var in varList:
        print(var)
        found_area_mean = False
        ds_area_avgs, file_found = load_area_file_var(model_name, case_name, var, area, from_time, to_time, avg_over_lev,
                                                      pmin, p_level, pressure_coords)
        if file_found:
            xr_out[var] = ds_area_avgs[var]
        if not found_area_mean or not look_for_file:

            sub_varL = get_fields4weighted_avg(var)

            dummy = xr_ds.copy()
            for svar in sub_varL:
                lev_was_coord = False
                if 'lev' in xr_ds[svar].dims:
                    lev_was_coord=True


                if avg_over_lev:
                    dummy = average_model_var(dummy, svar, area=area, minp=pmin, dim = list(xr_ds[svar].dims)) #\
                else:
                    dummy = average_model_var(dummy,
                                          svar, area=area,
                                              dim = list(set(xr_ds[svar].dims)-{'lev'}))#.sel(lev=p_level, method='nearest')
                    if 'lev' in dummy.dims:
                        dummy = dummy.sel(lev=p_level, method='nearest')
            dummy = compute_weighted_averages(dummy, var, model_name)
            dummy[var].attrs['Calc_weight_mean']=str(is_weighted_avg_var(var)) + ' area: %s'%area
            filen = filename_area_avg(var, model_name, case_name, area, from_time, to_time, avg_over_lev, pmin, p_level, pressure_coords,
                                      lev_was_coord=lev_was_coord)
            practical_functions.make_folders(filen)
            if ds_area_avgs is None:
                practical_functions.save_dataset_to_netcdf(dummy, filen)
            else:
                ds_area_avgs[var] = dummy[var]
                practical_functions.save_dataset_to_netcdf(ds_area_avgs, filen)
            xr_out[var] = dummy[var]
    return xr_out


def load_area_file_var(model,
                       case,
                       var,
                       area,
                       startyear,
                       endyear,
                       avg_over_lev,
                       pmin,
                       p_level,
                       pressure_adj
                       ):
    filen_had_lev = filename_area_avg(var, model, case, area, startyear, endyear, avg_over_lev, pmin, p_level, pressure_adj,
                             lev_was_coord=True)
    filen_no_lev = filename_area_avg(var, model, case, area, startyear, endyear, avg_over_lev, pmin, p_level, pressure_adj,
                               lev_was_coord=False)
    if os.path.isfile(filen_had_lev):
        filen=filen_had_lev
        file_found=True
    elif os.path.isfile(filen_no_lev):
        filen = filen_no_lev
        file_found=True
    else:
        file_found=False
    if file_found:
        #print('Loading file %s' % filen)
        ds = xr.open_dataset(filen).copy()
    else:
        print('Did not find mean with filename: %s or  %s' % (filen_had_lev, filen_no_lev))
        ds=None
    return ds, file_found
