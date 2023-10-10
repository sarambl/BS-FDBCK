import pathlib

import numpy as np
from ukesm_bs_fdbck import constants

path_to_global_avg = constants.get_outdata_path('area_means')
path_to_map_avg = constants.get_outdata_path('map_means')
path_to_levlat_avg = constants.get_outdata_path('levlat_means')
path_to_profile_avg = constants.get_outdata_path('profile_means')

default_save_pressure_coordinates = constants.get_outdata_path('pressure_coords')  # 'Data/Fields_pressure_coordinates'
default_save_original_coordinates = constants.get_outdata_path('original_coords')


def get_filename_pressure_coordinate_field(var, model, case,
                                           from_time, to_time,
                                           path_savePressCoord=default_save_pressure_coordinates):
    """
    Filename for field in pressure coordinates
    :param var:
    :param model:
    :param case:
    :param from_time:
    :param to_time:
    :param path_savePressCoord:
    :return:
    """
    try:
        from_time = np.datetime_as_string(from_time, unit='M')  # from_time.strftime('%Y-%m')
        to_time = np.datetime_as_string(to_time, unit='M')  # to_time.strftime('%Y-%m')
    except:
        pass
    filename_m = path_savePressCoord/ model / case/ f'{var}_{model}_{case}_{from_time}-{to_time}.nc'
    #filename_m = path_savePressCoord / '/%s/%s/%s_%s_%s_%s-%s.nc' % (model, case, var, model, case, from_time, to_time)
    #filename_m = filename_m.replace(" ", "_")
    return filename_m


def get_filename_constants(case, model):
    filen = default_save_original_coordinates / f'/{model}/{case}/constants.nc'
    return filen


def get_filename_ng_field(var, model, case, from_time, to_time,
                          path_save_ng=default_save_original_coordinates):
    """
    Get field name for native grid fields
    :param var:
    :param model:
    :param case:
    :param from_time:
    :param to_time:
    :param path_save_ng:
    :return:
    """
    try:
        from_time = np.datetime_as_string(from_time, unit='M')  # from_time.strftime('%Y-%m')
        to_time = np.datetime_as_string(to_time, unit='M')  # to_time.strftime('%Y-%m')
    except:
        pass
    path_save_ng = pathlib.Path(path_save_ng)

    filename_m = path_save_ng /model / case/ f'{var}_{model}_{case}_{from_time}-{to_time}_hybsig.nc' #% ( var, model, case, from_time, to_time)
    #filename_m = filename_m.replace(" ", "_")
    return filename_m


def filename_map_avg(model, case, var,
                     startyear, endyear,
                     avg_over_lev, pmin, p_lev, pres_adj,
                     lev_was_coord=True, path_to_data=path_to_map_avg,
                     time_mask=None):
    """
    filename for map averages
    :param model:
    :param case:
    :param var:
    :param startyear:
    :param endyear:
    :param avg_over_lev:
    :param pmin:
    :param p_lev:
    :param pres_adj:
    :param lev_was_coord:
    :param path_to_data:
    :param time_mask:
    :return:
    """
    filen = str(path_to_data) + '/%s/%s/%s_%s-%s' % (model, case, var, startyear, endyear)
    if time_mask is not None:
        filen = filen + '_%s' % time_mask
    if not lev_was_coord:
        filen = filen + '_lev_not_dim'
    elif avg_over_lev:
        filen = filen + '_avg2lev%.0f' % pmin
    else:
        filen = filen + '_atlev%.0f' % p_lev
    if not pres_adj:
        filen = filen + 'not_pres_adj.nc'
    else:
        filen = filen + '.nc'
    return filen.replace(' ', '_')


def filename_levlat_avg(model, case, var,
                        startyear, endyear,
                        pres_adj,
                        path_to_data=path_to_levlat_avg,
                        time_mask=None):
    """
    filename for map averages
    :param model:
    :param case:
    :param var:
    :param startyear:
    :param endyear:
    :param pres_adj:
    :param path_to_data:
    :param time_mask:
    :return:
    """
    path = str(path_to_data) / model / case
    filen = f'{var}_{startyear}-{endyear}'  # '/%s/%s/%s_%s-%s' % (model, case, var, startyear, endyear)
    if time_mask is not None:
        filen = filen + '_%s' % time_mask
    if not pres_adj:
        filen = filen + 'not_pres_adj.nc'
    else:
        filen = filen + '.nc'
    filen = filen.replace(' ', '_')
    return path / filen


def filename_area_avg(var, model, case, area,
                      startyear, endyear,
                      avg_over_lev, pmin,
                      p_lev, pres_adj,
                      lev_was_coord=True,
                      path_to_data=path_to_global_avg):
    """
    Field name for average over area
    :param var:
    :param model:
    :param case:
    :param area:
    :param startyear:
    :param endyear:
    :param avg_over_lev:
    :param pmin:
    :param p_lev:
    :param pres_adj:
    :param lev_was_coord:
    :param path_to_data:
    :return:
    """
    filen = str(
        path_to_data) + f'/{model}/{case}/{area}/{var}_{startyear}-{endyear}'  # % (model, case, area, var, startyear, endyear)
    if not lev_was_coord:
        filen = filen + 'lev_not_dim'
    elif avg_over_lev:
        filen = filen + '_avg2lev%.0f' % pmin
    else:
        filen = filen + '_atlev%.0f' % p_lev
    if not pres_adj:
        filen = filen + 'not_pres_adj'
    filen = filen + '.nc'
    return filen.replace(' ', '_')


def filename_profile_avg_allvars(model, case, area,
                                 startyear, endyear,
                                 pres_adj, path_to_data=path_to_profile_avg):
    """
    Depricated??
    :param model:
    :param case:
    :param area:
    :param startyear:
    :param endyear:
    :param pres_adj:
    :param path_to_data:
    :return:
    """
    filen = str(path_to_data) + '/%s/%s/%s/%.0f-%.0f' % (model, case, area, startyear, endyear)
    if not pres_adj:
        filen = filen + 'not_pres_adj.nc'
    else:
        filen = filen + '.nc'
    return filen.replace(' ', '_')


def filename_profile_avg(var, case, area,
                         from_time, to_time, pressure_coord, model='NorESM',
                         path_to_data=path_to_profile_avg, time_mask=None):
    """
    Filename for profile averages
    :param var:
    :param case:
    :param area:
    :param from_time:
    :param to_time:
    :param pressure_coord:
    :param model:
    :param path_to_data:
    :param time_mask:
    :return:
    """
    filen = str(path_to_data) + '/%s/%s/%s/%s_%s-%s' % (model, case, area, var, from_time, to_time)
    if time_mask is not None:
        filen = filen + '_%s' % time_mask
    if not pressure_coord:
        filen = filen + 'not_pres_adj.nc'
    else:
        filen = filen + '.nc'
    return filen.replace(' ', '_')
