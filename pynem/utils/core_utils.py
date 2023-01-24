from collections import defaultdict
import numpy as np
import anndata as ad
from pandas import DataFrame
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
                     names_array_key = 'names', groups: list = None, genes: list = None) -> DataFrame:
    data_array = np.array(adata.uns[data_key][data_type_key].tolist())
    names_array = np.array(adata.uns[data_key][names_array_key].tolist()).T
    
    gene_names = list(adata.var.index)
    if genes is not None:
        if not np.all(np.isin(genes, gene_names)):
            raise ValueError("Not all provided genes could be found in the data")

    group_names = np.array(adata.uns[data_key][data_type_key].dtype.names)
    if groups is not None:
        if not np.all(np.isin(groups, group_names)):
            raise ValueError("Not all provided groups could be found in the data")
        group_mask = np.isin(group_names, groups)
        data_array = data_array[:,group_mask]
        names_array = names_array[group_mask]
        group_names = group_names[group_mask]

    name_sort_array = np.argsort(names_array)
    gene_names = names_array[0][name_sort_array[0]]
    
    out_array = np.zeros(data_array.shape)
    for i in range(data_array.shape[1]):
        sort_idx = name_sort_array[i]
        out_array[:,i] = data_array[:,i][sort_idx]
    
    out_df = DataFrame(out_array, gene_names, group_names)

    if genes is not None:
        out_df = out_df.loc[genes]

    if groups is not None:
        out_df = out_df.loc[:,groups]
    
    return out_df


def preprocess_adata2(adata: ad.AnnData, data_key = 'data', data_type_key = 'binary',
                     names_array_key = 'names', groups: list = None, genes: list = None) -> DataFrame:
    data_array = np.array(adata.uns[data_key][data_type_key].tolist())
    names_array = np.array(adata.uns[data_key][names_array_key].tolist()).T
    
    gene_names = list(adata.var.index)
    if genes is not None:
        genes = np.array(genes)
        if not np.all(np.isin(genes, gene_names)):
            raise ValueError("Not all provided genes could be found in the data")

    group_names = np.array(adata.uns[data_key][data_type_key].dtype.names)
    if groups is not None:
        groups = np.array(groups)
        if not np.all(np.isin(groups, group_names)):
            raise ValueError("Not all provided groups could be found in the data")
        group_mask = np.isin(group_names, groups)
        data_array = data_array[:,group_mask]
        names_array = names_array[group_mask]
        group_names = group_names[group_mask]

    name_sort_array = np.argsort(names_array)
    gene_names = names_array[0][name_sort_array[0]]
    
    out_array = np.zeros(data_array.shape)
    for i in range(data_array.shape[1]):
        sort_idx = name_sort_array[i]
        out_array[:,i] = data_array[:,i][sort_idx]

    if genes is not None:
        gene_mask = np.isin(gene_names, genes)
        out_array = out_array[gene_mask][np.argsort(np.argsort(genes))]
        gene_names = genes

    if groups is not None:
        group_names_sort = np.argsort(group_names)
        out_array = out_array[:,group_names_sort][:,np.argsort(np.argsort(groups))]
        group_names = groups
    
    return (out_array, gene_names, group_names)