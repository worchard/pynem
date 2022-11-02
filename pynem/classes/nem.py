from audioop import mul
from collections import defaultdict
import itertools as itr
from typing import Hashable, Set, Union, Tuple, Any, Iterable, Dict, FrozenSet, List

from pynem.utils import core_utils
from pynem.custom_types import *
from pynem import SignalGraph, EffectAttachments

import anndata as ad
import numpy as np

class NestedEffectsModel():
    """
    Class uniting the separate elements of a Nested Effects Model in order to facilitate scoring and learning.
    """

    def __init__(self, adata: ad.AnnData, signals: Set = set(), effects: Set = set(),
                controls: Set = {'control'}, signals_column: str = 'signals', 
                signal_graph: SignalGraph = None, effect_attachments: EffectAttachments = None, 
                nem = None):

        if nem is not None:
            pass
        else:
            self._adata = adata
            self._signals = set(adata.obs[signals_column]).difference(controls)
            self._effects = set(adata.var.index)
            if signals:
                self._signals.intersection_update(signals)
            if effects:
                self._effects.intersection_update(effects)
            self._signalgraph = SignalGraph(nodes = self._signals)
            self._effectattachments = EffectAttachments.fromeffects(effects, signals = self._signals)

    def predict(self):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def transitive_closure(self):
        raise NotImplementedError

    def transitive_reduction(self):
        raise NotImplementedError