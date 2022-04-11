import os

import xarray as xr

from bs_fdbck.util import practical_functions
from bs_fdbck.util.filenames import filename_map_avg
from bs_fdbck.util.slice_average.avg_pkg import get_fields4weighted_avg, average_model_var, \
    compute_weighted_averages, is_weighted_avg_var


def get_average_map2(ds, varlist,case_name, from_time, to_time,
                    avg_over_lev =True, pmin=850., p_level=1013., pressure_coord=True, save_avg=True,
                     recalculate=False, model_name='NorESM', time_mask=None):
    """

    :param ds:
    :param varlist:
    :param case_name:
    :param from_time:
    :param to_time:
    :param avg_over_lev:
    :param pmin:
    :param p_level:
    :param pressure_coord:
    :param save_avg:
    :param recalculate:
    :param model_name:
    :param time_mask:
    :return:
    """

    xr_out = ds.copy()
    for var in varlist:
        found_map_avg = False
        # Look for file:
        if not recalculate:
           ds_copy, found_map_avg = load_average2map(model_name, case_name, var, from_time, to_time, avg_over_lev,
                                                      pmin, p_level, pressure_coord, time_mask=time_mask)
        # if file not found:
        if not found_map_avg:
            # some fields require other fields for weighted average:
            sub_varL = get_fields4weighted_avg(var)


            ds_copy = ds.copy()
            for svar in sub_varL:
                if 'lev' in ds[svar].dims:
                    had_lev_coord=True
                else: had_lev_coord=False
                if avg_over_lev:
                    ds_copy = average_model_var(ds_copy, svar, area='Global',
                                                dim=list(set(ds.dims)-{'lat', 'lon'}), minp=pmin,
                                                time_mask=time_mask)
                else:
                    ds_copy = average_model_var(ds_copy, svar, area='Global',
                                              dim=list(set(ds.dims)-{'lat', 'lon', 'lev'}),
                                                time_mask=time_mask)
                    if 'lev' in ds_copy[svar].dims:
                        ds_copy[svar] = ds_copy[svar].sel(lev=p_level, method='nearest')

            ds_copy = compute_weighted_averages(ds_copy, var, model_name)
            ds_copy[var].attrs['Calc_weight_mean']=str(is_weighted_avg_var(var)) + ' map'
            if save_avg:
                filen = filename_map_avg(model_name, case_name, var, from_time, to_time, avg_over_lev, pmin, p_level,
                                         pressure_coord, lev_was_coord=had_lev_coord, time_mask=time_mask)
                practical_functions.make_folders(filen)
                practical_functions.save_dataset_to_netcdf(ds_copy, filen)
        xr_out[var] = ds_copy[var]
    return xr_out


def load_average2map(model, case, var, startyear, endyear, avg_over_lev, pmin, p_level, pressure_adj,
                     time_mask=None):
    """
    Loads avg to map
    :param model:
    :param case:
    :param var:
    :param startyear:
    :param endyear:
    :param avg_over_lev:
    :param pmin:
    :param p_level:
    :param pressure_adj:
    :param time_mask: optional

    :return:
    """
    #var_info_df = var_overview_sql.open_and_fetch_var_case(model, case, var)
    #if len(var_info_df) > 0:
    #    had_lev_coord = bool(var_info_df['lev_is_dim'].values)
    #else:
    #    had_lev_coord = True
    filen_had_lev = filename_map_avg(model, case, var, startyear, endyear, avg_over_lev, pmin, p_level, pressure_adj,
                                     lev_was_coord=True, time_mask=time_mask)
    filen_no_lev = filename_map_avg(model, case, var, startyear, endyear, avg_over_lev, pmin, p_level, pressure_adj,
                                    lev_was_coord=False, time_mask=time_mask)
    if os.path.isfile(filen_had_lev):
        filen=filen_had_lev
        file_found=True
    elif os.path.isfile(filen_no_lev):
        filen = filen_no_lev
        file_found=True
    else:
        file_found=False
    if file_found:
        print('Loading file %s' % filen)
        ds = xr.open_dataset(filen).copy()
    else:
        print('Did not find map mean with filename: %s or  %s' % (filen_had_lev, filen_no_lev))
        ds=None
    return ds, file_found