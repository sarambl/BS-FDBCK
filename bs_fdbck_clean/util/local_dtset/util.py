def get_coord_from_location(location):
    li = location.split('_')
    if li[1][-1] in ['W', 'w']:
        longitude_coor = -float(li[1][0:-1])
    else:
        longitude_coor = int(li[1][0:-1])
    if li[3][-1] in ['S', 's']:
        latitude_coor = -float(li[3][0:-1])
    else:
        latitude_coor = int(li[3][0:-1])
    return [longitude_coor, latitude_coor]