import pathlib
neptune_proj_root = '/home/ubuntu/mnts/nird/projects/'
saturn_proj_root = '/home/ubuntu/mnts/nird/projects/'
tetralith_root = '/proj/bolinc/users/x_sarbl/'
path_proj = {'neptune': neptune_proj_root,
             'saturn' : saturn_proj_root,
             'tetralith1.nsc.liu.se':tetralith_root,
             'tetralith2.nsc.liu.se':tetralith_root,
             'n623':tetralith_root,
             }
def get_project_base(hostname, nird_project_code=None, testmode=False):

    if testmode:
        return str(pathlib.Path(__file__).parent.absolute())


    if hostname in path_proj.keys():
        return path_proj[hostname]
    else:
        return tetralith_root




