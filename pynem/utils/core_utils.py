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