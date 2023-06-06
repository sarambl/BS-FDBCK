from IPython import get_ipython
from useful_scit.util import log

from bs_fdbck_clean.constants import get_locations
from bs_fdbck_clean.util.collocate.collocate import CollocateModel, collocate_dataset
from bs_fdbck_clean.util.imports.fix_xa_dataset_v2 import xr_fix
from bs_fdbck_clean.util.imports.import_fields_xr_ukesm import xr_import_ukesm

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
class CollocateModelUkesm(CollocateModel):
    """
    collocate a model to a list of locations
    """
    def __init__(self,*_vars, **kwargs):
        print(_vars, kwargs)
        kwargs['model_name'] = 'UKESM'
        super().__init__(*_vars,**kwargs)

    def load_raw_ds(self, varlist, chunks=None):
        if chunks is None:
            chunks = {}

        print('Loading input dataset from raw file:')
        ds = xr_import_ukesm(self.case_name, self.from_time, self.to_time, varlist, model=self.model_name,
                             chunks=chunks)
        # %%
        ds = xr_fix(ds, model_name=self.model_name)

        self.input_dataset = ds
        return ds

    def set_input_datset(self, ds):
        self.input_dataset = ds

    def get_new_instance(self, *_vars, **kwargs):
        #kwargs['which'] = self.which
        return CollocateModelUkesm(*_vars, **kwargs)


# %%

