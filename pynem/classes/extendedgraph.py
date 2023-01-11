from collections import defaultdict
from typing import Hashable, Set, Union, Tuple, Any, Iterable, Dict, FrozenSet, List
import warnings
from itertools import chain

from pynem.utils import core_utils
from pynem.custom_types import *

import numpy as np
import scipy.sparse as sps

class ExtendedGraph:
    """
    Base class for graphs over Nested Effects Models, combining both the graph over perturbed 'signal' nodes and downstream
    'effect' nodes
    """
    def __init__(self, signals: Iterable[Node] = set(), effects: Iterable[Node] = set(), 
                 edges: Iterable[Edge] = set(), attachments: Iterable[Edge] = set(), 
                 graph = None):
        if graph is not None:
            pass
        else:
            signals = set(signals)
            effects = set(effects)
            signals.update(set(chain(*edges)))
            effects.update(set(chain(*attachments)))
            self._nsignals = len(signals)
            self._neffects = len(effects)
            nnodes = self._nsignals + self._neffects
            
            #initialise and populate property array
            self._property_array = np.empty(nnodes, dtype={'names':('name', 'is_signal'), 'formats': ('object', 'B')})
            self._property_array['name'] = np.array(list(signals) + list(effects))
            self._property_array['is_signal'] = np.array([True]*self._nsignals + [False]*self._neffects)

            self._parents = defaultdict(set)
            self._children = defaultdict(set)

            self._amat = np.zeros((self._nsignals, nnodes), dtype='B')
            np.fill_diagonal(self._signal_amat(), 1)
            self.add_edges_from(edges)
            self.attach_effects_from(attachments)
    
    # === CORE CLASS METHODS

    def __eq__(self, other):
        if not isinstance(other, ExtendedGraph):
            return False
        return np.array_equal(self._property_array, other._property_array) and np.array_equal(self._amat, other._amat)
    
    # === BASIC METHODS
    
    def signals_idx(self) -> np.ndarray:
        return np.array(range(self._nsignals))
    
    def signals(self) -> np.ndarray:
        return self._property_array['name'][:self._nsignals].copy()
    
    def effects_idx(self) -> np.ndarray:
        return np.array(range(self._nsignals, self._property_array.shape[0]))
    
    def effects(self) -> np.ndarray:
        return self._property_array['name'][self._nsignals:].copy()
    
    def edges_idx(self) -> list:
        return [*zip(*self._signal_amat().nonzero())]
    
    def attachments_idx(self) -> list:
        return [*zip(*self._attachment_amat().nonzero())]
    
    def edges(self) -> list:
        edge_array = self._signal_amat().nonzero()
        sources = self._property_array['name'][edge_array[0]]
        sinks = self._property_array['name'][edge_array[1]]
        return [*zip(sources, sinks)]
    
    def attachments(self) -> list:
        edge_array = self._attachment_amat().nonzero()
        sources = self._property_array['name'][edge_array[0]]
        sinks = self._property_array['name'][edge_array[1]]
        return [*zip(sources, sinks)]
    
    def _signal_amat(self) -> np.ndarray:
        return self._amat[:self._nsignals, :self._nsignals]

    def signal_amat(self) -> Tuple[np.ndarray, np.ndarray]:
        signal_amat = self._signal_amat().copy()
        signal_array = self.signals()
        return (signal_amat, signal_array)
    
    def _attachment_amat(self) -> np.ndarray:
        return self._amat[:self._nsignals, self._nsignals:]
    
    def attachment_amat(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        attachment_amat = self._attachment_amat().copy()
        signal_array = self.signals()
        effect_array = self.effects()
        return (attachment_amat, signal_array, effect_array)
    
    # === RELATION MANIPULATION METHODS PRIVATE

    def _add_edge(self, i: int, j: int):
        if i == j:
            warnings.warn("Self loops are present by default so adding them does nothing!")
            return
        self._signal_amat()[i, j] = 1
        self._parents[j].add(i)
        self._children[i].add(j)
    
    def _add_edges_from(self, edges: Iterable[Edge]):
        if len(edges) == 0:
            return
        for i, j in edges:
            self._add_edge(i, j)

    def _remove_edge(self, i: int, j: int, ignore_error: bool = False):
        if i == j:
            warnings.warn("Self loops are present by default and cannot be removed!")
            return
        self._signal_amat()[i, j] = 0
        try:
            self._parents[j].remove(i)
            self._children[i].remove(j)
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e
    
    def _remove_edges_from(self, edges: Iterable[Edge], ignore_error: bool = False):
        if len(edges) == 0:
            return
        for i, j in edges:
            self._remove_edge(i, j, ignore_error)

    def _attach_effect(self, signal: int, effect: int):
        self._detach_effect(effect)
        self._attachment_amat()[signal, effect - self._nsignals] = 1
    
    def _attach_effects_from(self, attachments: Iterable[Edge]):
        if len(attachments) == 0:
            return
        for i, j in attachments:
            self._attach_effect(i, j)
    
    def _detach_effect(self, effect: int):
        self._attachment_amat()[:, effect - self._nsignals] = 0
    
    def _detach_effects_from(self, effects: Iterable[Node]):
        if len(effects) == 0:
            return
        for effect in effects:
            self._detach_effect(effect)
        
    # def _join_signals(self, i: Node, j: Node):
    #     self.add_signal(name = frozenset({i, j}))

    # === RELATION MANIPULATION METHODS PUBLIC

    def add_edge(self, i: Node, j: Node):
        """
        Add the edge from signal ``i`` to signal ``j`` to the ExtendedGraph
        Parameters
        ----------
        i:
            source signal node of the edge
        j:
            target signal node of the edge
        
        See Also
        --------
        add_edges_from
        Examples
        --------
        """
        i = self.name2idx(i)
        j = self.name2idx(j)
        self._add_edge(i, j)
    
    def remove_edge(self, i: Node, j: Node):
        """
        Remove the edge from signal ``i`` to signal ``j`` to the ExtendedGraph
        Parameters
        ----------
        i:
            source signal node of the edge
        j:
            target signal node of the edge
        
        See Also
        --------
        add_edges_from
        Examples
        --------
        """
        i = self.name2idx(i)
        j = self.name2idx(j)
        self._remove_edge(i, j)
    
    def add_edges_from(self, edges: Iterable[Edge]):
        """
        Add edges between signals to the graph from the collection ``edges``.
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
        edges_idx = self.edgeNames2idx(edges)
        self._add_edges_from(edges_idx)
    
    def remove_edges_from(self, edges: Iterable[Edge]):
        """
        Remove edges between signals to the graph from the collection ``edges``.
        Parameters
        ----------
        edges:
            collection of edges to be removed.
        See Also
        --------
        add_edge
        Examples
        --------
        """
        if len(edges) == 0:
            return
        edges_idx = self.edgeNames2idx(edges)
        self._remove_edges_from(edges_idx)
    
    def attach_effect(self, signal: Node, effect: Node):
        signal = self.name2idx(signal)
        effect = self.name2idx(effect, is_signal=False)
        self._attach_effect(signal, effect)
    
    def detach_effect(self, effect: Node):
        effect = self.name2idx(effect, is_signal=False)
        self._detach_effect(effect)

    def attach_effects_from(self, attachments: Iterable[Edge]):
        if len(attachments) == 0:
            return
        attachments_idx = self.edgeNames2idx(attachments, is_signal=False)
        self._attach_effects_from(attachments_idx)
    
    def detach_effects_from(self, effects: Iterable[Node]):
        if len(effects) == 0:
            return
        effects_idx = self.names2idx(effects, is_signal=False)
        self._detach_effects_from(effects_idx)

    # === NODE MANIPULATION METHODS

    def add_signal(self, name = "as_index"):
        #First redo the adjacency matrix
        new_amat = np.zeros((self._amat.shape[0] + 1, self._amat.shape[1] + 1), dtype='B')
        orig_cols = np.append(range(self.nsignals), range(self.nsignals+1, self.nnodes + 1))
        if self.nsignals > 0:
            new_amat[:self.nsignals, orig_cols] = self._amat
        self._amat = new_amat
        self._amat[self.nsignals, self.nsignals] = 1 #add self-loop

        #Then the property array
        if name == "as_index":
            name = self.nsignals
        new_name = core_utils.get_unique_name(name, self.signals())
        if new_name != name:
            warnings.warn(f"Signal name changed to {new_name} to avoid a clash with an existing signal")
        ### --- following line needs modifying if new properties become supported --- ###
        self._property_array = np.insert(self._property_array, self.nsignals, (new_name, 1))

        #finally update nsignals
        self._nsignals += 1
    
    def add_effect(self, name = "as_index"):
        self._amat = np.c_[self._amat, np.zeros(self.nsignals, dtype='B')]
        if name == "as_index":
            name = self.nnodes
        new_name = core_utils.get_unique_name(name, self.effects())
        if new_name != name:
            warnings.warn(f"Effect name changed to {new_name} to avoid a clash with an existing effect")
        new_row = np.array((new_name, 0), dtype={'names': ('name', 'is_signal'), 'formats': ('object', 'B')})
        self._property_array = np.r_[self._property_array, new_row]
        self._neffects += 1
    
    def _remove_signal(self, signal):
        if signal not in self.signals_idx():
            raise KeyError("Signal not in graph")
        orig_cols = np.append(range(signal), range(signal+1, self.nnodes)).astype('B')
        self._amat = self._amat[orig_cols[:self.nsignals - 1]][:, orig_cols]
        self._property_array = np.delete(self._property_array, signal, 0)
        self._nsignals -= 1
        self._parents.pop(signal, None)
        self._children.pop(signal, None)
    
    def remove_signal(self, signal):
        signal = self.name2idx(signal)
        self._remove_signal(signal)
    
    def _remove_effect(self, effect):
        if effect not in self.effects_idx():
            raise KeyError("Effect not in graph")
        orig_cols = np.append(range(effect), range(effect+1, self.nnodes)).astype('B')
        self._amat = self._amat[:, orig_cols]
        self._property_array = np.delete(self._property_array, effect, 0)
        self._neffects -= 1
    
    def remove_effect(self, effect):
        effect = self.name2idx(effect, is_signal=False)
        self._remove_effect(effect)

    # === PROPERTIES

    @property
    def nsignals(self) -> int:
        return self._nsignals
    
    @property
    def neffects(self) -> int:
        return self._neffects
    
    @property
    def nnodes(self) -> int:
        return self._amat.shape[1]
    
    @property
    def property_array(self) -> np.ndarray:
        return self._property_array.copy()
    
    @property
    def amat(self) -> np.ndarray:
        return (self._amat.copy(), self.signals(), self._property_array['name'].copy())

    # === UTILITY METHODS

    def name2idx(self, name, is_signal: bool = True) -> int:
        """
        Convert a given node ``name`` to its corresponding index according to the ``property_array``
        of an ExtendedGraph object.
        Parameters
        ----------
        name:
            Node name to convert to an index. Note this name must appear in the name column of the property array.
        is_signal:
            Boolean indicating whether the name is of a signal node or an effect node.
        Examples
        --------
        >>> from pynem import ExtendedGraph
        >>> eg = ExtendedGraph(signals = ['S1', 'S2', 'S3'], effects = ['E1', 'E2', 'E3'])
        >>> eg.name2idx('S1')
        0
        """
        mask = self._property_array['is_signal'] == is_signal
        if is_signal:
            return np.nonzero(self._property_array['name'][mask] == name)[0][0]
        else:
            return np.nonzero(self._property_array['name'][mask] == name)[0][0] + self._nsignals

    def names2idx(self, name_array, is_signal: bool = True) -> np.ndarray:
        """
        Convert node names given in a 1D ndarray ``name_array`` to a corresponding array of node indices 
        according to the ``property_array`` of an ExtendedGraph object.
        Parameters
        ----------
        name_array:
            ndarray of node names to convert to indices. Note all names must appear in the name column of the property array.
        is_signal:
            Boolean indicating whether the names in ``name_array`` are of signal nodes or effect nodes.
        Examples
        --------
        >>> from pynem import ExtendedGraph
        >>> eg = ExtendedGraph(signals = ['S1', 'S2', 'S3'], effects = ['E1', 'E2', 'E3'])
        >>> eg.names2idx(np.array(['S1', 'S3']))
        array([0, 2])
        """
        mask = self._property_array['is_signal'] == is_signal
        full_name_array = self._property_array['name'][mask]
        sorter = full_name_array.argsort()
        if is_signal:
            return sorter[np.searchsorted(full_name_array, name_array, sorter=sorter)]
        else:
            return sorter[np.searchsorted(full_name_array, name_array, sorter=sorter)] + self._nsignals

    def edgeNames2idx(self, edges, is_signal: bool = True) -> list:
        """
        Convert an iterable of edges referring to nodes by name to a corresponding list 
        of edges referring nodes by their indices, according to the ``property_array``
        of an ExtendedGraph object.
        Parameters
        ----------
        edges:
            Iterable of edges to convert. Note all node names must appear in the name column of the property array.
        is_signal:
            Boolean indicating whether the edges in ``edges`` are between signals or are attaching signals to effects.
        Examples
        --------
        >>> from pynem import ExtendedGraph
        >>> eg = ExtendedGraph(signals = ['S1', 'S2', 'S3'], effects = ['E1', 'E2', 'E3'], \
            edges = [('S1', 'S2'), ('S2', 'S3'), ('S1', 'S3')])
        >>> eg.edgeNames2idx([('S1', 'S2'), ('S2', 'S3')])
        [(0, 1), (1, 2)]
        """
        edge_tuples = [*zip(*edges)]
        sources = self.names2idx(edge_tuples[0])
        sinks = self.names2idx(edge_tuples[1], is_signal=is_signal)
        return [*zip(sources, sinks)]