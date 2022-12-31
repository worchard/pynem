from collections import defaultdict
import numpy as np
from pynem.custom_types import *

def defdict2dict(defdict, keys):
    factory = defdict.default_factory
    defdict_keys = defdict.keys()
    d = {k: defdict[k] if k in defdict_keys else factory() for k in keys}
    return d

def name2idx(property_array, name):
    return np.nonzero(property_array['name'] == name)[0][0]

def names2idx(property_array, name_array):
    full_name_array = property_array['name']
    sorter = full_name_array.argsort()
    return sorter[np.searchsorted(full_name_array, name_array, sorter=sorter)]

def edgeNames2idx(property_array, edges):
    edge_tuples = [*zip(*edges)]
    sources = names2idx(property_array, edge_tuples[0])
    sinks = names2idx(property_array, edge_tuples[1])
    return [*zip(sources, sinks)]