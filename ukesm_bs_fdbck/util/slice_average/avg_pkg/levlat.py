import os

import xarray as xr

from ukesm_bs_fdbck.util import practical_functions
from ukesm_bs_fdbck.util.filenames import filename_levlat_avg
from ukesm_bs_fdbck.util.slice_average.avg_pkg import get_fields4weighted_avg, average_model_var, \
    compute_weighted_averages, is_weighted_avg_var


def get_average_levlat(ds, varlist, case_name, from_time, to_time,
                       pressure_coord=True, save_avg=True,
                       recalculate=False, model_name='NorESM', time_mask=None):
    """

    :param ds:
    :param varlist:
    :param case_name:
    :param from_time:
    :param to_time:
    :param pressure_coord:
    :param save_avg:
    :param recalculate:
    :param model_name:
    :param time_mask: e.g. JJA, DJF etc.
    :return:
    """

    xr_out = ds.copy()
    for var in varlist:
        found_levlat_avg = False
        # Look for file:
        if not recalculate:
            ds_copy, found_levlat_avg = load_average2levlat(model_name, case_name,
                                                            var,
                                                            from_time,
                                                            to_time,
                                                            pressure_coord,
                                                            time_mask=time_mask)
        # if file not found:
        if not found_levlat_avg:
            # some fields require other fields for weighted average:
            sub_varl = get_fields4weighted_avg(var)

            ds_copy = ds.copy()
            for svar in sub_varl:
                ds_copy = average_model_var(ds_copy, svar, area='Global',
                                            dim=list(set(ds.dims) - {'lat', 'lev'}), minp=0,
                                            time_mask=time_mask)

            ds_copy = compute_weighted_averages(ds_copy, var, model_name)
            ds_copy[var].attrs['Calc_weight_mean'] = str(is_weighted_avg_var(var)) + ' levlat'
            if save_avg:
                filen = filename_levlat_avg(model_name, case_name, var, from_time, to_time,
                                            pressure_coord, time_mask=time_mask)
                practical_functions.make_folders(filen)
                practical_functions.save_dataset_to_netcdf(ds_copy, filen)
        xr_out[var] = ds_copy[var]
    return xr_out


def load_average2levlat(model, case, var, startyear, endyear, pressure_adj,
                        time_mask=None):
    """
    Loads avg to map
    :param model:
    :param case:
    :param var:
    :param startyear:
    :param endyear:
    :param pressure_adj:
    :param time_mask: optional

    :return:
    """
    filen_had_lev = filename_levlat_avg(model, case, var,
                                        startyear,
                                        endyear,
                                        pressure_adj,
                                        time_mask=time_mask)
    if os.path.isfile(filen_had_lev):
        filen = filen_had_lev
        file_found = True
    else:
        file_found = False
    if file_found:
        print('Loading file %s' % filen)
        ds = xr.open_dataset(filen).copy()
    else:
        print('Did not find levlat mean with filename: %s ' % filen_had_lev)
        ds = None
    return ds, file_found
