from collections import defaultdict
from typing import Hashable, Set, Union, Tuple, Any, Iterable, Dict, FrozenSet, List

from pynem.utils import core_utils
from pynem.custom_types import *

import numpy as np
import scipy.sparse as sps

class AugmentedGraph:
    """
    Base class for graphs over Nested Effects Models, combining both the graph over perturbed 'signal' nodes and downstream
    'effect' nodes
    """
    # def __init__(self, nodes: Set[Node] = set(), edges: Set[Edge] = set(), signal_graph = None):
    #     self._amat_tuple = None
    #     if signal_graph is not None:
    #         self._nodes = set(signal_graph._nodes)
    #         self._edges = set(signal_graph._edges)
    #         self._parents = defaultdict(set)
    #         for node, par in signal_graph._parents.items():
    #             self._parents[node] = set(par)
    #         self._children = defaultdict(set)
    #         for node, ch in signal_graph._children.items():
    #             self._children[node] = set(ch)
    #         self._amat_tuple = signal_graph.amat_tuple
            
    #     else:
    #         self._nodes = set(nodes)
    #         self._edges = set()
    #         self._parents = defaultdict(set)
    #         self._children = defaultdict(set)
    #         self.add_edges_from(edges)

    def __init__(self, signals: Iterable[Node] = list(), effects: Iterable[Node] = list(), 
                 edges: Iterable[Edge] = list(), graph = None):
        if graph is not None:
            pass
        else:
            len_signals = len(signals)
            len_effects = len(effects)
            
            self._nnodes = len_signals + len_effects
            self._nodes = np.array(range(self._nnodes))
            self._signals = self._nodes[:len_signals]
            self._effects = self._nodes[len_signals:]
            
            #initialise and populate property array
            self._property_array = np.empty(self._nnodes, dtype={'names':('name', 'is_signal'), 'formats': ('object', 'b')})
            self._property_array['name'] = np.array(list(signals) + list(effects))
            self._property_array['is_signal'] = np.array([True]*len_signals + [False]*len_effects)

            self._amat = np.zeros((self._nnodes, self._nnodes))
            self.add_edges_from(edges)
        
    def _add_edge(self, i: int, j: int):
        self._amat[i, j] = 1
    
    def _add_edges_from(self, edges: Iterable[Edge]):
        if len(edges) == 0:
            return
        self._amat[[*zip(*edges)]] = 1

    def add_edge(self, i: Node, j: Node):
        """
        Add the edge ``i`` -> ``j`` to the AugmentedGraph
        Parameters
        ----------
        i:
            source node of the edge
        j:
            target node of the edge
        
        See Also
        --------
        add_edges_from
        Examples
        --------
        """
        i = core_utils.name2idx(self._property_array, i)
        j = core_utils.name2idx(self._property_array, j)
        self._add_edge(i, j)
    
    def add_edges_from(self, edges: Iterable[Edge]):
        """
        Add edges to the graph from the collection ``edges``.
        Parameters
        ----------
        edges:
            collection of edges to be added.

        See Also
        --------
        add_edge
        Examples
        --------
        """
        if len(edges) == 0:
            return
        edges_idx = core_utils.edgeNames2idx(self._property_array, edges)
        self._add_edges_from(edges_idx)