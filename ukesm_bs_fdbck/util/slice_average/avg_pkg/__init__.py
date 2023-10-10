import numpy as np
import xarray as xr
# import analysis_tools.area_pkg_sara
from ukesm_bs_fdbck.util.imports.get_fld_fixed import get_field_fixed
from ukesm_bs_fdbck.util.imports.import_fields_xr_v2 import import_constants
from ukesm_bs_fdbck.util.slice_average import area_mod
# from ukesm_bs_fdbck.util.slice_average.avg_pkg import maps
# import analysis_tools.var_overview_sql
# from analysis_tools import area_pkg_sara, practical_functions
from ukesm_bs_fdbck import constants
from useful_scit.util import log as log
import sys

path_to_global_avg = constants.get_outdata_path('area_means')  # 'Data/area_means/'
path_to_map_avg = constants.get_outdata_path('map_means')
path_to_profile_avg = constants.get_outdata_path('profile_means')

# Fields that should be weighted:
fields4weighted_avg = {'AREL_incld': ['AREL', 'FREQL'],
                       'AWNC_incld': ['AWNC', 'FREQL'],
                       'ACTREL_incld': ['ACTREL', 'FCTL'],
                       'ACTNL_incld': ['ACTNL', 'FCTL'],
                       'Smax_w': ['Smax', 'Smax_supZero'],
                       }

"""
Recomended: only use average_model_var
"""


def masked_average(xa: xr.DataArray,
                   dim=None,
                   weights: xr.DataArray = None,
                   mask: xr.DataArray = None):
    """
    This function will average
    :param xa: dataArray
    :param dim: dimension or list of dimensions. e.g. 'lat' or ['lat','lon','time']
    :param weights: weights (as xarray)
    :param mask: mask (as xarray), True where values to be masked.
    :return: masked average xarray
    """
    # lest make a copy of the xa
    xa_copy: xr.DataArray = xa.copy()

    if mask is not None:
        xa_weighted_average = __weighted_average_with_mask(
            dim, mask, weights, xa, xa_copy
        )
    elif weights is not None:
        xa_weighted_average = __weighted_average(
            dim, weights, xa, xa_copy
        )
    else:
        xa_weighted_average = xa.mean(dim)

    return xa_weighted_average


def __weighted_average(dim, weights, xa, xa_copy):
    """helper function for masked_average"""
    _, weights_all_dims = xr.broadcast(xa, weights)  # broadcast to all dims
    x_times_w = xa_copy * weights_all_dims
    xw_sum = x_times_w.sum(dim)
    x_tot = weights_all_dims.where(xa_copy.notnull()).sum(dim=dim)
    xa_weighted_average = xw_sum / x_tot
    return xa_weighted_average


def __weighted_average_with_mask(dim, mask, weights, xa, xa_copy):
    """helper function for masked_average"""
    _, mask_all_dims = xr.broadcast(xa, mask)  # broadcast to all dims
    xa_copy = xa_copy.where(np.logical_not(mask))
    if weights is not None:
        _, weights_all_dims = xr.broadcast(xa, weights)  # broadcast to all dims
        weights_all_dims = weights_all_dims.where(~mask_all_dims)
        x_times_w = xa_copy * weights_all_dims
        xw_sum = x_times_w.sum(dim=dim)
        x_tot = weights_all_dims.where(xa_copy.notnull()).sum(dim=dim)
        xa_weighted_average = xw_sum / x_tot
    else:
        xa_weighted_average = xa_copy.mean(dim)
    return xa_weighted_average


def average_model_var(xr_ds, var, area='Global', dim=None, minp=850., time_mask=None):
    """
    Example:
    average_model_var(dtset, var, dim=['lon','time'], area='Global', minp=850.)
    :param time_mask:
    :param xr_ds:
    :param var:
    :param area:
    :param dim:
    :param minp:
    :return:
    """
    get_lev_weights = True
    if dim is not None:
        if 'lev' not in dim:
            get_lev_weights = False
    wg_matrix_lat = get_lat_wgts_matrix(var, xr_ds)
    if 'lev' in xr_ds[var].dims and get_lev_weights:
        wg_matrix_press = get_pres_wgts_matrix(var, xr_ds)
        # Let weights be product of weighted by lat and by pressure different.
        wgts_matrix = wg_matrix_lat * wg_matrix_press
    else:
        wgts_matrix = wg_matrix_lat.copy()
    mask = get_mask_ds(area, dim, minp, time_mask, var, xr_ds)
    # If lev or lat not in xr_ds, mask/weights have too many dims
    mask, wgts_matrix = fix_dims_mask_stupid(mask, var, wgts_matrix, xr_ds)
    # wgts_matrix = wgts_matrix.isel(lev=29
    if dim is None:
        dim = list(xr_ds[var].dims)

    mean = masked_average(xr_ds[var], dim=list(set(xr_ds[var].dims).intersection(dim)), weights=wgts_matrix, mask=mask)

    if 'startyear' in xr_ds[var].attrs:
        mean.attrs['startyear'] = xr_ds[var].attrs['startyear']
    else:
        mean.attrs['startyear'] = xr_ds['time.year'].min().values
    if 'endyear' in xr_ds[var].attrs:
        mean.attrs['endyear'] = xr_ds[var].attrs['endyear']
    else:
        mean.attrs['startyear'] = xr_ds['time.year'].max().values
    mean.attrs = xr_ds[var].attrs
    out_ds = xr_ds.copy()
    out_ds[var] = mean
    out_ds[var].attrs['area'] = area
    if time_mask is not None:
        out_ds[var].attrs['time_mask'] = time_mask
    return out_ds


def get_mask_ds(area, dim, minp, time_mask, var, xr_ds):
    mask1, area_masked = area_mod.get_4d_area_mask_xa(area, xr_ds, var)  # get 3D mask for area
    mask_l = [mask1]
    if 'lev' in xr_ds[var].dims:
        dummy, pressure = xr.broadcast(xr_ds[var], xr_ds['lev'])
        if dim is None:
            mask2 = pressure <= minp
            mask_l.append(mask2)
        elif 'lev' in dim:
            mask2 = pressure <= minp
            mask_l.append(mask2)
    # time:
    mask3, found_time_m = get_time_mask(time_mask, xr_ds, var)
    if found_time_m: mask_l.append(mask3)
    mask = mask_l[0]
    for m_add in mask_l[1:]:
        mask = np.logical_or(mask, m_add)
    return mask


def get_time_mask(period_name, ds, var):
    if period_name is None:
        return None, False
    if 'time' not in ds[var].dims:
        print('get_time_mask: time not in dims')
        return None, False
    if period_name in ['DJF', 'MAM', 'JJA', 'SON']:
        time_mask = ds['time.season'] != period_name
        _, mask_out = xr.broadcast(ds[var], time_mask)
        return mask_out, True
    else:
        log.ger.error('Did not find period name %s' % period_name)
        sys.exit('Did not find period name %s' % period_name)


def fix_dims_mask_stupid(mask, var, wgts_matrix, xr_ds):
    if 'lev' not in xr_ds[var].dims and 'lev' in mask.dims:
        mask = mask.isel(lev=29)  # ; wgts_matrix = wgts_matrix.isel(lev=29)
    if ('lev' not in xr_ds[var].dims) and ('lev' in wgts_matrix.dims):
        wgts_matrix = wgts_matrix.isel(lev=29)
    if 'lat' not in xr_ds[var].dims and 'lat' in mask.dims:
        mask = mask.isel(lat=0)  # ; wgts_matrix = wgts_matrix.isel(lat=0)
    if 'lat' not in xr_ds[var].dims and 'lat' in wgts_matrix.dims:
        wgts_matrix = wgts_matrix.isel(lat=0)
    return mask, wgts_matrix


def get_lat_wgts_matrix(var, xr_ds):
    """
    Get latitude weights for gaussian grid
    :param var:
    :param xr_ds:
    :return:
    """
    if 'lat_wg' in xr_ds:
        if 'time' in xr_ds['lat_wg'].dims:
            xr_ds['lat_wg'] = xr_ds['lat_wg'].isel(time=0)
        wgts_ = xr_ds['lat_wg'] / xr_ds['lat_wg'].sum()  # Get latitude weights
    else:
        wgts_ = xr.DataArray(area_mod.get_wghts(xr_ds['lat'].values),
                             # define xarray with weights and dimension lat
                             dims={'lat': xr_ds['lat']})
        wgts_ = wgts_ / wgts_.sum()
    dummy, wg_matrix = xr.broadcast(xr_ds[var], wgts_)  # broadcast one 1D weights to all dimensions in dtset[var]

    return wg_matrix


def get_pres_wgts_matrix(var, xr_ds):
    """
    Get pressure weight by pressure difference (weight of grid box)
    :param var: Variable
    :param xr_ds:
    :return:
    """
    # Pressure difference:

    ilev = xr_ds['ilev'].values
    lev = xr_ds['lev'].values
    # avg over pressure:
    pres_diff_1d = np.zeros_like(lev) * np.nan
    if len(lev) > 1:
        if lev[0] > lev[1]:  # lev oriented upwards:
            pres_diff_1d[:] = ilev[0:-1] - ilev[1::]
        else:
            pres_diff_1d[:] = ilev[1::] - ilev[0:-1]
    else:
        pres_diff_1d[:] = 1
    press_diff_xr1d = xr.DataArray(pres_diff_1d,  # define xarray with weights and dimension lat
                                   dims={'lev': lev})
    # Broadcast to same coordinates:
    dummy, press_diff_matrix = xr.broadcast(xr_ds[var], press_diff_xr1d)
    return press_diff_matrix


# Weighted averages ############################################

def is_weighted_avg_var(var):
    return var in fields4weighted_avg


def get_fields4weighted_avg(var):
    if var in fields4weighted_avg:
        return fields4weighted_avg[var]
    else:
        return [var]


def compute_weighted_averages(xr_ds, var, model):
    if model == 'NorESM':
        if var == 'AREL_incld':
            xr_ds[var] = xr_ds['AREL'] / xr_ds['FREQL']
            xr_ds[var] = xr_ds[var].where((xr_ds['FREQL'] != 0))
            xr_ds[var].attrs['units'] = xr_ds['AREL'].attrs['units']
        if var == 'AWNC_incld':
            xr_ds[var] = xr_ds['AWNC'] / xr_ds['FREQL']  # "*1.e-6
            xr_ds[var] = xr_ds[var].where((xr_ds['FREQL'] != 0))
            xr_ds[var].attrs['units'] = xr_ds['AWNC'].attrs['units']
        if var == 'ACTNL_incld':
            xr_ds[var] = xr_ds['ACTNL'] / xr_ds['FCTL']  # *1.e-6
            xr_ds[var] = xr_ds[var].where((xr_ds['FCTL'] != 0))
            xr_ds[var].attrs['units'] = xr_ds['ACTNL'].attrs['units']
        if var == 'ACTREL_incld':
            xr_ds[var] = xr_ds['ACTREL'] / xr_ds['FCTL']  # *1.e-6
            xr_ds[var] = xr_ds[var].where((xr_ds['FCTL'] != 0))
            xr_ds[var].attrs['units'] = xr_ds['ACTREL'].attrs['units']
        if var == 'Smax_w':
            xr_ds[var] = xr_ds['Smax'] / xr_ds['Smax_supZero']
            xr_ds[var] = xr_ds[var].where((xr_ds['Smax_supZero'] != 0))
            xr_ds[var].attrs['units'] = xr_ds['Smax'].attrs['units']

    xr_ds[var].attrs['Calc_weight_mean'] = str(is_weighted_avg_var(var))
    return xr_ds


def yearly_mean_dic(varl,
                    cases,
                    startyear,
                    endyear,
                    pmin,
                    pressure_adjust,
                    model='NorESM',
                    avg_over_lev=True,
                    groupby='time.year',
                    dims=None,
                    area='Global'
                    ):
    """
    Get mean over groupby. If groupby is None, then will return simple average.
    :param model:
    :param pressure_adjust:
    :param pmin:
    :param endyear:
    :param startyear:
    :param varl:
    :param cases:
    :param avg_over_lev:
    :param groupby:
    :param dims:
    :param area:
    :return:
    """
    if dims is None:
        dims = {'lev', 'lat', 'lon'}
    dummy_dic = {}
    for case in cases:
        save_ds = yearly_mean_case(area, avg_over_lev, case, dims, groupby, varl, startyear,
                                   endyear,
                                   pmin,
                                   pressure_adjust,
                                   model=model)
        dummy_dic[case] = save_ds.copy()
        del save_ds
    return dummy_dic


def yearly_mean_case(
        area,
        avg_over_lev,
        case,
        dims,
        groupby,
        varl,
        startyear,
        endyear,
        pmin,
        pressure_adjust,
        model='NorESM'

):
    """
    Get mean over groupby. If groupby is None, then will return simple average.
    :param model:
    :param pressure_adjust:
    :param pmin:
    :param endyear:
    :param startyear:
    :param area:
    :param avg_over_lev:
    :param case:
    :param dims:
    :param groupby:
    :param varl:
    :return:
    """
    save_ds = None
    first = True
    for var in varl:
        dummy = yearly_mean_var(area, avg_over_lev, case, dims, groupby, var,
                                startyear,
                                endyear,
                                pmin,
                                pressure_adjust,
                                model=model)

        if first:
            save_ds = dummy.copy()
            first = False
        else:
            if var in dummy:
                save_ds[var] = dummy[var]  # .copy()
    return save_ds


def yearly_mean_var(
        area,
        avg_over_lev,
        case,
        dims,
        groupby,
        var,
        startyear,
        endyear,
        pmin,
        pressure_adjust,
        model='NorESM'
):
    """
    Get mean over groupby. If groupby is None, then will return simple average.

    :param model:
    :param pressure_adjust:
    :param pmin:
    :param endyear:
    :param startyear:
    :param area:
    :param avg_over_lev:
    :param case:
    :param dims:
    :param groupby:
    :param var:
    :return:
    """
    var_subl = get_fields4weighted_avg(var)
    log.ger.debug('Getting fields fixed input fields:')

    dummy = get_field_fixed(case,
                            var_subl,
                            startyear,
                            endyear,
                            pressure_adjust=pressure_adjust)
    print(f'averaging case {case}:')
    print(var_subl)
    ds_constants = import_constants(case, model=model)
    dummy = xr.merge([dummy, ds_constants])
    if groupby is not None:
        dummy = dummy.groupby(groupby).mean()
    for svar in var_subl:
        dims_s = set(dims).intersection(set(dummy[svar].dims))
        if not avg_over_lev:
            dims_s = set(dims_s) - {'lev'}
        if avg_over_lev:
            dummy = average_model_var(dummy, svar, area=area, minp=pmin, dim=list(dims_s))  # \
        else:
            dummy = average_model_var(dummy, svar, area=area,
                                      dim=list(dims_s))
    dummy = compute_weighted_averages(dummy, var, model)
    return dummy
