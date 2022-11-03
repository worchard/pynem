from typing import Hashable, Set, Union, Tuple, Any, Iterable, Dict, FrozenSet, List

from pynem.utils import core_utils
from pynem.custom_types import *
from pynem.classes import SignalGraph, EffectAttachments

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
        self._controls = controls
        self._signalscolumn = signals_column
        if nem is not None:
            self._adata = nem._adata
            self._signals = nem._signals
            self._effects = nem._effects
            self._signalgraph = nem._signalgraph
            self._effectattachments = nem._effectattachments
        else:
            self._adata = adata
            self._signals = set(adata.obs[signals_column]).difference(controls)
            self._effects = set(adata.var.index)
            
            if signals:
                self._signals.intersection_update(signals)
            if effects:
                self._effects.intersection_update(effects)
            
            if signal_graph:
                self._signalgraph = signal_graph
                self._signals = signal_graph.nodes
            else:
                self._signalgraph = SignalGraph(nodes = self._signals)
            if effect_attachments:
                self._effectattachments = effect_attachments
                self._effects = effect_attachments.effects()
            else:
                self._effectattachments = EffectAttachments.fromeffects(effects, signals = self._signals)

            assert self._signalgraph.nodes == self._effectattachments.signals.difference({None}), \
                "Nodes of the signal graph must match the signals of the effect attachments"

    def predict(self, return_array: bool = False) -> Union[dict, np.ndarray]:
        """
        Predict which effect reporters will be affcted by the perturbation of each signal, given a signal graph and effect attachments.
        """
        
        pass

    def score(self):
        raise NotImplementedError

    def copy(self):
        """
        Return a copy of the current NestedEffectsModel.
        """
        return NestedEffectsModel(nem=self)

    def transitive_closure(self):
        raise NotImplementedError

    def transitive_reduction(self):
        raise NotImplementedError