from typing import Hashable, Set, Union, Tuple, Any, Iterable, Dict, FrozenSet, List
from os import linesep

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
        self._score = None
        if nem is not None:
            self._adata = nem._adata
            self._signals = nem._signals
            self._effects = nem._effects
            self._signal_graph = nem._signal_graph
            self._effect_attachments = nem._effect_attachments
        else:
            self._adata = adata
            self._signals = set(adata.obs[signals_column]).difference(controls)
            self._effects = set(adata.var.index)
            
            if signals:
                self._signals.intersection_update(signals)
            if effects:
                self._effects.intersection_update(effects)
            
            if signal_graph:
                self._signal_graph = signal_graph
                self._signals = signal_graph.nodes
            else:
                self._signal_graph = SignalGraph(nodes = self._signals)
            if effect_attachments:
                self._effect_attachments = effect_attachments
                self._effects = effect_attachments.effects()
            else:
                self._effect_attachments = EffectAttachments.fromeffects(self._effects, signals = self._signals)

            assert self._signal_graph.nodes == self._effect_attachments.signals.difference({None}), \
                "Nodes of the signal graph must match the signals of the effect attachments"

    def predict(self, return_array: bool = False) -> Union[dict, np.ndarray]:
        """
        Predict which effect reporters will be affcted by the perturbation of each signal, given a signal graph and effect attachments.
        """
        pass

    def score_model(self):
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
    
    def __str__(self) -> str:
        scr = self._score
        dat = self._adata.__str__()
        sg = self._signal_graph.__str__()
        ea = self._effect_attachments.__str__()
        out_string = "Nested Effects Model object" + linesep + f"Score: {scr}" + linesep + \
        f"Data: {dat}" + linesep + f"Signal graph: {sg}" + linesep + f"Effect Attachments: {ea}"
        return out_string
    
    def __repr__(self) -> str:
        return self.__str__()

    # === PROPERTIES

    @property
    def signals(self) -> Set[Node]:
        return set(self._signals)
    
    @property
    def effects(self) -> Set[Node]:
        return set(self._effects)

    @property
    def signal_graph(self) -> SignalGraph:
        return self._signal_graph.copy()
    
    @property
    def effect_attachments(self) -> EffectAttachments:
        return self._effect_attachments.copy()
    
    @property
    def adata(self) -> ad.AnnData:
        return self._adata.copy()

    @property
    def score(self) -> float:
        return self._score