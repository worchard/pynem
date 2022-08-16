from collections import defaultdict
from pynem.custom_types import *

def defdict2dict(defdict, keys):
    factory = defdict.default_factory
    defdict_keys = defdict.keys()
    d = {k: defdict[k] if k in defdict_keys else factory() for k in keys}
    return d