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
            self._signals = self._nodes[:len_signals] ##TODO Do I need this line and below?
            self._effects = self._nodes[len_signals:]
            
            #initialise and populate property array
            self._property_array = np.empty(self._nnodes, dtype={'names':('name', 'is_signal'), 'formats': ('object', 'b')})
            self._property_array['name'] = np.array(list(signals) + list(effects))
            self._property_array['is_signal'] = np.array([True]*len_signals + [False]*len_effects)

            self._amat = np.zeros((self._nnodes, self._nnodes))
            self.add_edges_from(edges)

    def __eq__(self, other):
        if not isinstance(other, AugmentedGraph):
            return False
        return np.array_equal(self._property_array, other._property_array) and np.array_equal(self._amat, other._amat)
    
    def _add_edge(self, i: int, j: int):
        self._amat[i, j] = 1
    
    def _add_edges_from(self, edges: Iterable[Edge]):
        if len(edges) == 0:
            return
        self._amat[(*zip(*edges),)] = 1

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

    # === BASIC METHODS

    def nsignals(self) -> int:
        return self._property_array['is_signal'].sum()
    
    def neffects(self) -> int:
        return self._property_array.shape[0] - self.nsignals()
    
    def signals_idx(self) -> np.ndarray:
        return np.array(range(self.nsignals()))
    
    def signals(self) -> np.ndarray:
        return self._property_array['name'][:self.nsignals()].copy()
    
    def edges_idx(self) -> list:
        return [*zip(*self._amat.nonzero())]

    def edges(self) -> list:
        edge_array = self._amat.nonzero()
        sources = self._property_array['name'][edge_array[0]]
        sinks = self._property_array['name'][edge_array[1]]
        return [*zip(sources, sinks)]
    
    # === KEY METHODS

    def add_signal(self, signal_name = None):
        raise NotImplementedError
        nsignals = self.nsignals()

    def _signal_amat(self) -> np.ndarray:
        nsignals = self.nsignals()
        return self._amat[:nsignals, :nsignals]

    def signal_amat(self) -> Tuple[np.ndarray, np.ndarray]:
        nsignals = self.nsignals()
        signal_amat = self._signal_amat().copy()
        signal_array = self._property_array['name'][:nsignals].copy()
        return (signal_amat, signal_array)
    
    def _effect_attachments(self) -> np.ndarray:
        nsignals = self.nsignals()
        return self._amat[:nsignals, nsignals:]
    
    def effect_attachments(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nsignals = self.nsignals()
        effect_attachments = self._effect_attachments().copy()
        signal_array = self._property_array['name'][:nsignals].copy()
        effect_array = self._property_array['name'][nsignals:].copy()
        return (effect_attachments, signal_array, effect_array)

    # === UTILITY METHODS

    def name2idx(self, name) -> int:
        """
        Convert a given node ``name`` to its corresponding index according to the ``property_array``
        of an AugmentedGraph object.
        Parameters
        ----------
        name:
            Node name to convert to an index. Note this name must appear in the name column of the property array.
        Examples
        --------
        >>> from pynem import AugmentedGraph
        >>> ag = AugmentedGraph(signals = ['S1', 'S2', 'S3'], effects = ['E1', 'E2', 'E3'])
        >>> ag.name2idx('S1')
        0
        """
        return np.nonzero(self._property_array['name'] == name)[0][0]

    def names2idx(self, name_array) -> np.ndarray:
        """
        Convert node names given in a 1D ndarray ``name_array`` to a corresponding array of node indices 
        according to the ``property_array`` of an AugmentedGraph object.
        Parameters
        ----------
        name_array:
            ndarray of node names to convert to indices. Note all names must appear in the name column of the property array.
        Examples
        --------
        >>> from pynem import AugmentedGraph
        >>> ag = AugmentedGraph(signals = ['S1', 'S2', 'S3'], effects = ['E1', 'E2', 'E3'])
        >>> ag.names2idx(np.array(['S1', 'S3']))
        array([0, 2])
        """
        full_name_array = self._property_array['name']
        sorter = full_name_array.argsort()
        return sorter[np.searchsorted(full_name_array, name_array, sorter=sorter)]

    def edgeNames2idx(self, edges) -> list:
        """
        Convert an iterable of edges referring to nodes by name to a corresponding list 
        of edges referring nodes by their indices, according to the ``property_array``
        of an AugmentedGraph object.
        Parameters
        ----------
        edges:
            Iterable of edges to convert. Note all node names must appear in the name column of the property array.
        Examples
        --------
        >>> from pynem import AugmentedGraph
        >>> ag = AugmentedGraph(signals = ['S1', 'S2', 'S3'], effects = ['E1', 'E2', 'E3'], \
            edges = [('S1', 'S2'), ('S2', 'S3'), ('S1', 'S3')])
        >>> ag.edgeNames2idx([('S1', 'S2'), ('S2', 'S3')])
        [(0, 1), (1, 2)]
        """
        edge_tuples = [*zip(*edges)]
        sources = self.names2idx(edge_tuples[0])
        sinks = self.names2idx(edge_tuples[1])
        return [*zip(sources, sinks)]

    # === PROPERTIES

    @property
    def property_array(self) -> np.ndarray:
        return self._property_array.copy()