from collections import defaultdict
import numpy as np
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

def name2idx(property_array, name):
    """
    Convert a given node ``name`` to its corresponding index according to the ``property_array``
    of an AugmentedGraph object.
    Parameters
    ----------
    property_array:
        Property array of an AugmentedGraph object
    name:
        Node name to convert to an index. Note this name must appear in the name column of the property array.
    Examples
    --------
    >>> from pynem import AugmentedGraph
    >>> from pynem.utils import core_utils
    >>> ag = AugmentedGraph(signals = ['S1', 'S2', 'S3'], effects = ['E1', 'E2', 'E3'])
    >>> core_utils.name2idx(ag.property_array, 'S1')
    0
    """
    return np.nonzero(property_array['name'] == name)[0][0]

def names2idx(property_array, name_array):
    """
    Convert node names given in a 1D ndarray ``name_array`` to a corresponding array of node indices 
    according to the ``property_array`` of an AugmentedGraph object.
    Parameters
    ----------
    property_array:
        Property array of an AugmentedGraph object
    name_array:
        ndarray of node names to convert to indices. Note all names must appear in the name column of the property array.
    Examples
    --------
    >>> from pynem import AugmentedGraph
    >>> from pynem.utils import core_utils
    >>> ag = AugmentedGraph(signals = ['S1', 'S2', 'S3'], effects = ['E1', 'E2', 'E3'])
    >>> core_utils.names2idx(ag.property_array, np.array(['S1', 'S3']))
    array([0, 2])
    """
    full_name_array = property_array['name']
    sorter = full_name_array.argsort()
    return sorter[np.searchsorted(full_name_array, name_array, sorter=sorter)]

def edgeNames2idx(property_array, edges):
    """
    Convert an iterable of edges referring to nodes by name to a corresponding list 
    of edges referring nodes by their indices, according to the ``property_array``
    of an AugmentedGraph object.
    Parameters
    ----------
    property_array:
        Property array of an AugmentedGraph object
    edges:
        Iterable of edges to convert. Note all node names must appear in the name column of the property array.
    Examples
    --------
    >>> from pynem import AugmentedGraph
    >>> from pynem.utils import core_utils
    >>> ag = AugmentedGraph(signals = ['S1', 'S2', 'S3'], effects = ['E1', 'E2', 'E3'], \
        edges = [('S1', 'S2'), ('S2', 'S3'), ('S1', 'S3')])
    >>> core_utils.edgeNames2idx(ag.property_array, [('S1', 'S2'), ('S2', 'S3')])
    [(0, 1), (1, 2)]
    """
    edge_tuples = [*zip(*edges)]
    sources = names2idx(property_array, edge_tuples[0])
    sinks = names2idx(property_array, edge_tuples[1])
    return [*zip(sources, sinks)]