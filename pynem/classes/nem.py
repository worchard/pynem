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

    def __init__(self, adata: ad.AnnData = ad.AnnData(), signals: Set = set(), effects: Set = set(),
                controls: Iterable = {'control'}, signals_column: str = 'signals', 
                signal_graph: SignalGraph = None, effect_attachments: EffectAttachments = None, 
                nem = None):
        self._controls = controls
        self._signals_column = signals_column
        self._score = None
        if nem is not None:
            self._adata = nem.adata
            self._signals = nem.signals
            self._effects = nem.effects
            self._signal_graph = nem.signal_graph
            self._effect_attachments = nem.effect_attachments
        else:
            self._adata = adata.copy()
            if adata:
                self._signals = set(adata.obs[signals_column]).difference(set(controls))
                self._effects = set(adata.var.index)
            else:
                self._signals = signals
                self._effects = effects
            
            if signals:
                self._signals.intersection_update(signals)
            if effects:
                self._effects.intersection_update(effects)
            
            if signal_graph:
                self._signal_graph = signal_graph.copy()
                self._signals = self._signal_graph.nodes
            else:
                self._signal_graph = SignalGraph(nodes = self._signals)
            if effect_attachments:
                self._effect_attachments = effect_attachments.copy()
                self._signals = self._effect_attachments.signals
                self._effects = self._effect_attachments.effects()
            else:
                self._effect_attachments = EffectAttachments.fromeffects(self._effects, signals = self._signals)

            assert self._signal_graph.nodes == self._effect_attachments.signals.difference({None}), \
                "Nodes of the signal graph must match the signals of the effect attachments"

    def predict_dict(self) -> dict:
        """
        Predict which effect reporters will be affcted by the perturbation of each signal, given a signal graph and effect attachments, and return
        dictionary where each key is a signal, and each value is the set of predicted perturbed effects.
        ----------
        See Also
        --------
        predict_array
        Example
        --------
        """
        raise NotImplementedError

    def predict_array(self) -> np.ndarray:
        """
        Predict which effect reporters will be affcted by the perturbation of each signal, given a signal graph and effect attachments, and return
        a tuple containing a prediction array, M, and lists indexing the rows (signals) and columns (effects) of the array. M_ij = 1 if effect j is 
        predicted to be dependent on the perturbation of signal i, otherwise M_ij = 0. 
        ----------
        See Also
        --------
        predict_dict
        Return
        ------
        (M, signal_list, effect_list)
        Example
        --------
        """
        raise NotImplementedError

    def score_model(self):
        raise NotImplementedError

    def copy(self):
        """
        Return a copy of the current NestedEffectsModel.
        """
        return NestedEffectsModel(adata = self.adata, nem=self)

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


    # # === SignalGraph methods
    # def add_node(self, node: Node):
    #     """
    #     Add ``node`` to the SignalGraph.
    #     Parameters
    #     ----------
    #     node:
    #         a hashable Python object
    #     See Also
    #     --------
    #     add_nodes_from
    #     Examples
    #     --------
    #     >>> from pynem import NestedEffectsModel
    #     >>> nem = SignalGraph()
    #     >>> g.add_node(1)
    #     >>> g.add_node(2)
    #     >>> len(g.nodes)
    #     2
    #     """
