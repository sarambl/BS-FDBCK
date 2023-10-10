from IPython import get_ipython
from useful_scit.util import log

from ukesm_bs_fdbck.constants import get_locations
from ukesm_bs_fdbck.util.collocate.collocate import CollocateModel, collocate_dataset
from ukesm_bs_fdbck.util.imports.fix_xa_dataset_v2 import xr_fix
from ukesm_bs_fdbck.util.imports.import_fields_xr_ec_earth import xr_import_EC_earth

log.ger.setLevel(log.log.INFO)

# noinspection PyBroadException
try:
    _ipython = get_ipython()
    _magic = _ipython.magic
    _magic('load_ext autoreload')
    _magic('autoreload 2')
except:
    pass


# %%

# %%


# %%
class CollocateModelECEarth(CollocateModel):
    """
    collocate a model to a list of locations
    """

    def __init__(self,  *_vars, **kwargs):
        print(_vars, kwargs)
        if 'which' not in kwargs:
            kwargs['which'] = 'TM5'
        kwargs['model_name'] = 'EC-Earth'
        self.which = kwargs['which']
        #del kwargs['which']
        if 'which' in kwargs:
            del kwargs['which']
        super().__init__(*_vars, **kwargs)

    def load_raw_ds(self, varlist, chunks=None):
        if chunks is None:
            chunks = {}

        print('Loading input dataset from raw file:')
        ds = xr_import_EC_earth(self.case_name,
                                self.from_time,
                                self.to_time,
                                which=self.which,
                                chunks=chunks,

                                )

        ds = xr_fix(ds, model_name=self.model_name)

        self.input_dataset = ds
        return ds

    def set_input_datset(self, ds):
        self.input_dataset = ds

    def get_new_instance(self, *_vars, **kwargs):
        kwargs['which'] = self.which
        return CollocateModelECEarth(*_vars, **kwargs)


# %%

