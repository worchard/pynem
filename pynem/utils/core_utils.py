from collections import defaultdict
import numpy as np
import anndata as ad
from pynem.custom_types import *

def defdict2dict(defdict, keys):
    """
    Convert a given default dictionary, ``defdict``, into a dictionary with keys from collection ``keys``.
    Parameters
    ----------
    defdict:
        Default dictionary to be converted
    keys:
        Iterable of keys to be present in the output dictionary
    Examples
    --------
    >>> from collections import defaultdict
    >>> dd = defaultdict(set)
    >>> dd[1] = 2
    >>> dd[3] = 4
    >>> d = defdict2dict(dd, [1,3,5])
    >>> d
    {1: 2, 3: 4, 5: set()}
    """
    factory = defdict.default_factory
    defdict_keys = defdict.keys()
    d = {k: defdict[k] if k in defdict_keys else factory() for k in keys}
    return d

def get_unique_name(name, name_universe):
  if isinstance(name, int):
    while name in name_universe:
        name += 1
  else:
    i = 1
    orig_name = name
    while name in name_universe:
        name = f"{orig_name}_{i}"
        i += 1
  return name

def preprocess_adata(adata: ad.AnnData, data_key = 'data', data_type_key = 'binary',
                     names_array_key = 'names', groups: list = list(), genes: list = list()):
    raise NotImplementedError
    gene_names = list(adata.var.index)
    if not np.all(genes in gene_names):
        raise ValueError("Not all provided genes could be found in the data")
    group_names = list(adata.uns[data_key][data_type_key].dtype.names)
    if not np.all(groups in group_names):
        raise ValueError("Not all provided groups could be found in the data")
    names_array = adata.uns[data_key][names_array_key]
    name_sort_array = np.argsort(np.array(names_array.tolist()).T)
    data_array = np.array(adata.uns[data_key][data_type_key].tolist())
    name_sorted_data_array = np.empty(data_array.shape)
    for i in range(len(group_names)):
        sort_idx = name_sort_array[i]
        name_sorted_data_array[:,i] = data_array[:,i][sort_idx]
    if genes:
        pass