# This code is the main program that plots a map of a specific region studied

# Importing necessary stuff
import numpy as np  # For scientific computing
import matplotlib.pyplot as plt
import sys
import socket
import cartopy.crs as ccrs
import xarray as xr
from bs_fdbck import constants
from bs_fdbck.data_info import get_area_specs

seconds_in_year = 365 * 24 * 60 * 60
seconds_in_month = 30 * 24 * 60 * 60
Earth_surface = 5.101e14  # in m
Terra_fact = 1.e12 * 1.e-3  # to convert to terra grams

path_to_mask_data = constants.get_outdata_path('masks')  # + '/Data/masks/'


def get_square_area_TF(dtset, area):
    """
    Returns square are with True in square False outside
    :param dtset:
    :param area:
    :return:
    """
    min_lat, max_lat, min_lon, max_lon, found_TF = get_limits(area)
    lat = dtset['lat']
    lon = dtset['lon']
    if min(lon) >= -1:
        min_lon = lon360_2lon180(min_lon)
        max_lon = lon360_2lon180(max_lon)

    if max_lon < min_lon:
        lon_T = np.logical_or(lon <= max_lon, lon >= min_lon)
    else:
        lon_T = np.logical_and((lon <= max_lon), lon >= min_lon)
    lat_T = np.logical_and((lat <= max_lat), (lat >= min_lat))
    # print(lat_T)
    lat_T_matrix = np.repeat(lat_T.values[:, np.newaxis], len(lon), axis=1)
    lon_T_matrix = np.repeat(lon_T.values[np.newaxis, :], len(lat), axis=0)

    return np.logical_and(lat_T_matrix, lon_T_matrix), found_TF


# convert from 0-360 to -180 - 180
def lon360_2lon180(lon):
    """
    convert from 0-360 to -180 - 180
    :param lon:
    :return:
    """
    return (lon + 180) % 360 - 180


def lon180_2lon360(lon):
    """
    Convert from -180 - 180 to 0 - 360
    :param lon:
    :return:
    """
    return lon % 360


def plot_my_area(plt_dt, TF, var, area):
    """
    Plot where TF is True
    :param plt_dt: plot dataset
    :param TF: True/False
    :param var: The variable to plot
    :param area: The area name
    :return: None
    """
    assert isinstance(var, str)
    if 'nird' in socket.gethostname():
        print('Doesnt attempt to plot on nird')
        return
    if 'lev' in plt_dt:
        if plt_dt['lev'][0] > 900:
            lev_ind = 0
        else:
            lev_ind = len(plt_dt['lev']) - 1
    # print(plt_dt)
    # print(plt_dt.values.shape)
    if isinstance(plt_dt, xr.Dataset):
        plt_dt = plt_dt[var]
    plt_var = plt_dt.where(TF).copy()
    try:
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())  # Orthographic(10, 0))
        if 'lev' in plt_dt:
            if 'time' in plt_dt:
                plt_var.isel(time=0, lev=lev_ind).plot.contourf(ax=ax, transform=ccrs.PlateCarree(), robust=True)
            else:
                plt_var.isel(lev=lev_ind).plot.contourf(ax=ax, transform=ccrs.PlateCarree(), robust=True)
        elif 'time' in plt_dt:
            plt_var.isel(time=0).plot.contourf(ax=ax, transform=ccrs.PlateCarree(), robust=True)
        else:
            plt_var.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), robust=True)
        ax.set_global()
        ax.coastlines()
        plt.savefig('AREA_PLOT' + area + '.png')
    except:
        print('couldnt print')
        print(sys.exc_info())

    return


# This function returns max and min values for each area. To reduce risk of typos in other files.
def get_limits(area):
    """
    Get limits of square areas
    :param area:
    :return:
    """
    area_specs = get_area_specs(area)
    if area_specs is not None:
        min_lat = area_specs['min_lat']
        max_lat = area_specs['max_lat']
        min_lon = area_specs['min_lon']
        max_lon = area_specs['max_lon']
        return min_lat, max_lat, min_lon, max_lon, True
    if area == 'Arctic':
        max_lat = 90.
        min_lat = 60.
        max_lon = 180.
        min_lon = -180.

    elif area == 'Tropics':
        max_lat = 20.
        min_lat = -20.
        max_lon = 130.
        min_lon = 90.
    elif area == 'CCN Siberia':
        max_lat = 70.
        min_lat = 50.
        max_lon = 180.
        min_lon = 90.
    elif area == 'South East Europe':
        max_lat = 50.
        min_lat = 40.
        max_lon = 30.
        min_lon = 20.
    elif area == 'Mediterranean':
        max_lat = 40.
        min_lat = 30.
        max_lon = 40.
        min_lon = -10.
    elif area == 'Boreal forest':
        max_lat = 70.
        min_lat = 55.
        max_lon = 180.
        min_lon = -180.
    elif area == 'NH':
        max_lat = 90.
        min_lat = 0.
        max_lon = 180.
        min_lon = -180.
    elif area == 'SH':
        max_lat = 0.
        min_lat = -90.
        max_lon = 180.
        min_lon = -180.
    elif area == 'SH mid lat':
        max_lat = -30.
        min_lat = -55.
        max_lon = 180.
        min_lon = -180.
    elif area == 'NH mid lat':
        max_lat = 55.
        min_lat = 30.
        max_lon = 180.
        min_lon = -180.
    elif area == 'SH high lat':
        max_lat = -55.
        min_lat = -90.
        max_lon = 180.
        min_lon = -180.
    elif area == 'NH high lat':
        max_lat = 90.
        min_lat = 55.
        max_lon = 180.
        min_lon = -180.
    elif area == 'All Tropics':
        max_lat = 30.
        min_lat = -30.
        max_lon = 180.
        min_lon = -180.
    elif area == 'Equatorial Africa':
        max_lat = 4.0
        min_lat = -6.
        max_lon = 26.
        min_lon = 17.5
    elif area == 'Pacific':
        max_lat = 30.
        min_lat = -30.
        max_lon = -120.
        min_lon = 170.
    elif area == 'Amazonas and surroundings':
        max_lat = 2.
        min_lat = -16.
        max_lon = -50.
        min_lon = -74.

    else:
        print('Error:', area, ' misses predefined coordinates')
        return 0, 0, 0, 0, False

    return min_lat, max_lat, min_lon, max_lon, True


def get_land_only(dtset, tolerance=0.5):
    """
    Returns True where land.
    :param tolerance:
    :param dtset:
    :return:
    """
    if 'LANDFRAC' in dtset:
        mask = dtset['LANDFRAC'] > tolerance
        if 'time' in mask.dims:
            return mask.isel(time=0).squeeze()
        else:
            return mask
    else:
        print('LANDFRAC not in dataset')
    """
    print('DEPRICATED: DOESNT WORK ANYMORE!! -- find better solution')

    bm = Basemap()
    lat = dtset['lat']
    lon = dtset['lon']
    land = np.empty([len(lat), len(lon)])
    # land=list(land)
    map = Basemap(llcrnrlon=-180, llcrnrlat=-90., urcrnrlon=180, urcrnrlat=90, projection='cyl')
    for i in np.arange(len(lat)):
        for j in np.arange(len(lon)):
            land[i, j] = map.is_land(lon[j], lat[i])
    island = (land == 1)
    return island
    """


def get_4d_area_mask(area, dtset, lev, test_var, time):
    """
    Returns True in mask, False for designated ares. Returns 4D area mask for area and dtset using test_var for
    dimensions.
    :param area: area name
    :param dtset: dtset
    :param lev: levels
    :param test_var: str test variable for dimensions
    :param time: time dimension
    :return:
    """
    area_masked = False
    if area != 'Global':
        if area == 'landOnly':
            mask_area = np.logical_not(get_land_only(dtset))
            area_masked = True

        elif area == 'notLand':
            mask_area = get_land_only(dtset)
            area_masked = True
        else:

            area_masked = True
            mask_area, found_TF = get_square_area_TF(dtset, area)
            mask_area = np.logical_not(mask_area)
    if area_masked:
        mask_area = np.repeat(mask_area[np.newaxis, :, :], len(time), axis=0)
        mask_area = np.repeat(mask_area[:, np.newaxis, :, :], len(lev), axis=1)
        mask = np.logical_or(np.isnan(dtset[test_var].values), mask_area)
    else:
        mask = np.isnan(dtset[test_var].values)
    return mask, area_masked


def get_4d_area_mask_xa(area, dtset, test_var):
    """
    Returns True if mask, False if in designated area. Returns 4D area mask for area and dtset using test_var for dimensions.
    :param area:
    :param dtset:
    :param test_var:
    :return:
    """
    area_masked = False
    if area != 'Global':
        if area == 'landOnly':
            mask_area = np.logical_not(get_land_only(dtset))
            area_masked = True

        elif area == 'notLand':
            mask_area = get_land_only(dtset)
            area_masked = True
        else:

            area_masked = True
            mask_area, found_TF = get_square_area_TF(dtset, area)
            mask_area = np.logical_not(mask_area)
            if not found_TF:
                mask_area = get_single_grid_mask(dtset, area)
                mask_area = np.logical_not(mask_area)
    else:
        mask_area = np.zeros((len(dtset['lat']), len(dtset['lon'])), dtype=bool)
    masked_area_xr = xr.DataArray(mask_area, dims={'lat': dtset['lat'], 'lon': dtset['lon']})
    dummy, masked_area_xr = xr.broadcast(dtset[test_var], masked_area_xr)
    dtset[area] = masked_area_xr  # xr.DataArray(mask, coords=dtset[test_var].coords)
    return dtset[area], area_masked


def get_xd_area_mask(area, dtset, lev, test_var, time, model='NorESM'):
    """
    Get area mask with unknown dimensions. Returns True for values to be masked (outside of designated area).
    :param area:
    :param dtset:
    :param lev:
    :param test_var:
    :param time:
    :param model:
    :return:
    """
    if 'model' in dtset.attrs:
        model = dtset.attrs['model']
    area_masked = False
    if area != 'Global':
        if area == 'landOnly':
            mask_area = np.logical_not(get_land_only(dtset))
            area_masked = True

        elif area == 'notLand':
            mask_area = get_land_only(dtset)
            area_masked = True
        else:

            area_masked = True
            mask_area, found_TF = get_square_area_TF(dtset, area)
            mask_area = np.logical_not(mask_area)

            if not found_TF:
                mask_area = get_mask_from_file(area, model)
                area_masked = True
            if mask_area is None:
                mask_area = get_single_grid_mask(dtset, area)
                mask_area = np.logical_not(mask_area)
                area_masked = True
    if area_masked:
        dummy, mask_area = xr.broadcast(dtset[test_var], mask_area)
        # if 'lev' in dtset[test_var].coords:
        #    mask_area = np.repeat(mask_area[np.newaxis, :, :], len(lev), axis=0)
        #    if 'time' in dtset[test_var].coords:
        #        mask_area = np.repeat(mask_area[np.newaxis, :, :, :], len(time), axis=0)
        # elif 'time' in dtset[test_var].coords:
        #    mask_area = np.repeat(mask_area[np.newaxis, :, :], len(time), axis=0)

        # mask_area = np.repeat(mask_area[:, np.newaxis, :, :], len(lev), axis=1)
        mask = np.logical_or(np.isnan(dtset[test_var].values), mask_area)
    else:
        mask = np.isnan(dtset[test_var].values)
    return mask, area_masked


def get_wghts(lat):
    """
    get latitude weights for gaussian grid.
    :param lat: latitudes
    :return:
    """
    latr = np.deg2rad(lat)  # convert to radians
    weights = np.cos(latr)  # calc weights
    return weights


def get_wghts_v2(ds):
    """
    get latitude weights for gaussian grid.
    :param ds:
    :return:
    """
    if 'gw' in ds:
        return ds['gw'].values
    lat = ds['lat'].values
    latr = np.deg2rad(lat)  # convert to radians
    weights = np.cos(latr)  # calc weights
    return weights


def get_mask_from_file(area, model):
    """
    Read mask from file
    :param area:
    :param model:
    :return:
    """
    path = path_to_mask_data + model + '/'
    filen = path + area
    try:
        mask = xr.open_dataarray(filen)
        mask_out = mask.copy()

        mask_out.values[mask.values == 1] = True
        mask_out.values[mask.values == 0] = False
    except:
        return None

    return mask_out.values


locations = {'Hyytiala': {'lat': 61.85, 'lon': 24.29}, 'Melpitz': {'lat': 51.32, 'lon': 12.56}}


def get_single_grid_mask(dtset, location):
    if location in locations:
        lat = dtset['lat'];
        lon = dtset['lon']
        latlon = lat * lon
        d, lat_m = xr.broadcast(latlon, lat)
        d, lon_m = xr.broadcast(latlon, lon)
        dummy = dtset.sel(lat=locations[location]['lat'], lon=locations[location]['lon'], method='nearest')
        ma = np.logical_and(lat_m == dummy['lat'], lon_m == dummy['lon'])
        print('Found location %s' % location)
        return ma
    else:
        return None
