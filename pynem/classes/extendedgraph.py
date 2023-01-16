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
    def __init__(self, signals: List[Node] = list(), effects: List[Node] = list(), 
                 edges: Iterable[Edge] = set(), attachments: Iterable[Edge] = set(), 
                 signal_amat: np.ndarray = None, attachment_amat: np.ndarray = None, 
                 extended_graph = None):
        if extended_graph is not None:
            self._nsignals = extended_graph._nsignals
            self._neffects = extended_graph._neffects
            self._property_array = extended_graph.property_array
            self._signal_amat = extended_graph._signal_amat.copy()
            self._attachment_amat = extended_graph._attachment_amat.copy()
            self._join_array = extended_graph._join_array.copy()
        else:
            signals = list(signals)
            effects = list(effects)
            if signal_amat is None:
                if edges:
                    signals = signals + list(set(chain(*edges)).difference(signals))
                self._nsignals = len(signals)
                self._signal_amat = np.zeros((self._nsignals, self._nsignals), dtype = 'B')
            else:
                self._nsignals = signal_amat.shape[0]
                if signals and len(signals) != self._nsignals:
                    raise ValueError("Dimensions of signals list and signal_amat do not match!")
                self._signal_amat = signal_amat.copy()
            np.fill_diagonal(self._signal_amat, 1)
                
            if attachment_amat is None:
                if attachments:
                    effects = effects + list({i[1] for i in attachments}.difference(effects))
                self._neffects = len(effects)
                self._attachment_amat = np.zeros((self._nsignals, self._neffects), dtype = 'B')
            else:
                self._neffects = attachment_amat.shape[1] - self._nsignals
                if (effects and len(effects) != self._neffects) or attachment_amat.shape[0] != self._nsignals:
                    raise ValueError("Dimensions of the signal and/or effects list do not match the attachment_amat!")
                self._attachment_amat  = attachment_amat.copy()

            if not signals:
                signals = list(range(self._nsignals))
            if not effects:
                effects = list(range(self._neffects))

            #initialise and populate property array
            self._property_array = np.empty(self._nsignals + self._neffects, dtype={'names':('name', 'is_signal'), 'formats': ('object', 'B')})
            self._property_array['name'] = np.array(signals + effects)
            self._property_array['is_signal'] = np.array([True]*self._nsignals + [False]*self._neffects)

            self._join_array = np.zeros((self._nsignals, self._nsignals), dtype='bool')
            
            self.add_edges_from(edges)
            self.attach_effects_from(attachments)
    
    # === CORE CLASS METHODS

    def __eq__(self, other):
        if not isinstance(other, ExtendedGraph):
            return False
        return np.array_equal(self._property_array, other._property_array) and np.array_equal(self._signal_amat, other._signal_amat) \
            and np.array_equal(self._attachment_amat, other._attachment_amat)
    
    def copy(self):
        return ExtendedGraph(extended_graph=self)
    
    # === BASIC METHODS
    
    def signals_idx(self) -> np.ndarray:
        return np.array(range(self._nsignals))
    
    def signals(self) -> np.ndarray:
        return self._property_array['name'][:self._nsignals].copy()
    
    def effects_idx(self) -> np.ndarray:
        return np.array(range(self._nsignals, self.nnodes))
    
    def effects(self) -> np.ndarray:
        return self._property_array['name'][self._nsignals:].copy()
    
    def edges_idx(self) -> list:
        return [*zip(*self._signal_amat.nonzero())]
    
    def attachments_idx(self) -> list:
        return [*zip(*self._attachment_amat.nonzero())]
    
    def edges(self) -> list:
        edge_array = self._signal_amat.nonzero()
        sources = self._property_array['name'][edge_array[0]]
        sinks = self._property_array['name'][edge_array[1]]
        return [*zip(sources, sinks)]
    
    def attachments(self) -> list:
        edge_array = self._attachment_amat.nonzero()
        sources = self._property_array['name'][edge_array[0]]
        sinks = self._property_array['name'][edge_array[1]]
        return [*zip(sources, sinks)]

    def _full_amat(self) -> np.ndarray:
        return np.c_[self._signal_amat, self._attachment_amat]
    
    def full_amat(self) -> Tuple[np.ndarray, np.ndarray]:
        signal_array = self.signals()
        node_array = np.r_[signal_array, self.effects()]
        return (self._full_amat(), signal_array, node_array)
    
    def _parents_of(self, signals: Union[int, List[int]]) -> Set[Node]:
        """
        Return all signals that are parents of the signals in the list ``signals``.
        Parameters
        ----------
        signals
            A list of signals.
        See Also
        --------
        children_of
        Examples
        """
        return set(self._signal_amat[:,signals].nonzero()[0])

    def _children_of(self, signals: List[int]) -> Set[Node]:
        """
        Return all signals that are children of the signals in the list ``signals``.
        Parameters
        ----------
        signals
            A list of signals.
        See Also
        --------
        parents_of
        Examples
        --------
        """
        return set(self._signal_amat[signals].nonzero()[1])
    
    def _joined_to(self, signal: int) -> np.ndarray:
        return self._join_array[signal].nonzero()[0].astype('B') #if a lil_matrix the index here needs to be switched back to [1], but otherwise same
    
    # === RELATION MANIPULATION METHODS PRIVATE

    def _add_edge(self, i: int, j: int):
        if i == j:
            warnings.warn("Self loops are present by default so adding them does nothing!")
            return
        i_joined_to = np.append(i, self._joined_to(i))
        j_joined_to = np.append(j, self._joined_to(j))
        for i in i_joined_to:
            self._signal_amat[i, j_joined_to] = 1
    
    def _add_edges_from(self, edges: Iterable[Edge]):
        if len(edges) == 0:
            return
        for i, j in edges:
            self._add_edge(i, j)

    def _remove_edge(self, i: int, j: int):
        if i == j:
            warnings.warn("Self loops are present by default and cannot be removed!")
            return
        i_joined_to = np.append(i, self._joined_to(i))
        j_joined_to = np.append(j, self._joined_to(j))
        for i in i_joined_to:
            self._signal_amat[i, j_joined_to] = 0
    
    def _remove_edges_from(self, edges: Iterable[Edge]):
        if len(edges) == 0:
            return
        for i, j in edges:
            self._remove_edge(i, j)

    def _attach_effect(self, signal: int, effect: int):
        self._detach_effect(effect)
        self._attachment_amat[signal, effect - self._nsignals] = 1
    
    def _attach_effects_from(self, attachments: Iterable[Edge]):
        if len(attachments) == 0:
            return
        for i, j in attachments:
            self._attach_effect(i, j)
    
    def _detach_effect(self, effect: int):
        self._attachment_amat[:, effect - self._nsignals] = 0
    
    def _detach_effects_from(self, effects: Iterable[Node]):
        if len(effects) == 0:
            return
        for effect in effects:
            self._detach_effect(effect)

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
        raise NotImplementedError
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
        self._attachment_amat = np.c_[self._attachment_amat, np.zeros(self.nsignals, dtype='B')]
        if name == "as_index":
            name = self.nnodes
        new_name = core_utils.get_unique_name(name, self.effects())
        if new_name != name:
            warnings.warn(f"Effect name changed to {new_name} to avoid a clash with an existing effect")
        new_row = np.array((new_name, 0), dtype={'names': ('name', 'is_signal'), 'formats': ('object', 'B')})
        self._property_array = np.r_[self._property_array, new_row]
        self._neffects += 1
    
    def _remove_signal(self, signal: int):
        raise NotImplementedError
        if signal not in self.signals_idx():
            raise ValueError("Signal not in graph")
        orig_cols = np.append(range(signal), range(signal+1, self.nnodes)).astype('B')
        self._amat = self._amat[orig_cols[:self.nsignals - 1]][:, orig_cols]
        self._property_array = np.delete(self._property_array, signal, 0)
        self._nsignals -= 1
    
    def remove_signal(self, signal):
        raise NotImplementedError
        signal = self.name2idx(signal)
        self._remove_signal(signal)
    
    def _remove_signals_from(self, signals: List[int]):
        raise NotImplementedError
        if not np.all(np.isin(signals, self.signals_idx())):
            raise ValueError("All signals being removed must be in the graph")
        orig_cols = [signal for signal in range(self.nnodes) if signal not in signals]
        self._amat = self._amat[orig_cols[:self.nsignals - len(signals)]][:,orig_cols]
        self._property_array = np.delete(self._property_array, signals, 0)
        self._nsignals -= len(signals)
    
    def remove_signals_from(self, signals: List[int]):
        raise NotImplementedError
        signals = self.names2idx(np.array(signals))
        self._remove_signals_from(signals)
    
    def _remove_effect(self, effect):
        if effect not in self.effects_idx():
            raise ValueError("Effect not in graph")
        effect -= self.nsignals
        orig_cols = np.append(range(effect), range(effect+1, self.neffects)).astype('B')
        self._attachment_amat = self._attachment_amat[:, orig_cols]
        self._property_array = np.delete(self._property_array, effect, 0)
        self._neffects -= 1
    
    def remove_effect(self, effect: Node):
        effect = self.name2idx(effect, is_signal=False)
        self._remove_effect(effect)
    
    def _join_signals(self, i: int, j: int):
        to_join = np.append(j, self._joined_to(j))
        self._join_array[i, to_join] = 1
        self._join_array[to_join, i] = 1
    
    def _split_signals(self, i: int, j: int):
        if not self._join_array[i,j]:
            return
        j_joined_to = np.append(j, self._joined_to(j))
        self._join_array[i, j_joined_to] = 0
        self._join_array[j_joined_to, i] = 0
        self._signal_amat[j_joined_to, i] = 0
        self._signal_amat[i, i] = 1
    
    # def _join_signals(self, i: int, j: int):
    #     """
    #     Join the nodes ``i`` and ``j`` into a single multi-node.
    #     Parameters
    #     ----------
    #     i:
    #         index of first signal to join
    #     j:
    #         index of second signal to join
    #     See Also
    #     --------
    #     split_node
    #     Examples
    #     --------
    #     """
    #     if not (i < self.nsignals and j < self.nsignals):
    #         raise ValueError("Both signals must be in the graph")
    #     i_name = self._property_array['name'][i]
    #     j_name = self._property_array['name'][j]
    #     join_generator = (n if isinstance(n, frozenset) else frozenset({n}) for n in (i_name,j_name))
    #     joined_signal_name = frozenset.union(*join_generator)
    #     self.add_signal(joined_signal_name)

    #     new_row = np.logical_or(self._amat[i], self._amat[j])
    #     new_col = np.logical_or(self._amat[:,i], self._amat[:,j])

    #     self._amat[self._nsignals - 1] = new_row
    #     self._amat[:, self._nsignals - 1] = new_col
    #     self._amat[self._nsignals - 1, self._nsignals - 1] = 1

    #     self._remove_signals_from([i,j])
    
    # def join_signals(self, i: Node, j: Node):
    #     i = self.name2idx(i)
    #     j = self.name2idx(j)
    #     self._join_signals(i,j)

    # === PROPERTIES

    @property
    def nsignals(self) -> int:
        return self._nsignals
    
    @property
    def neffects(self) -> int:
        return self._neffects
    
    @property
    def nnodes(self) -> int:
        return self._property_array.shape[0]
    
    @property
    def property_array(self) -> np.ndarray:
        return self._property_array.copy()
    
    @property
    def signal_amat(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self._signal_amat.copy(), self.signals())
    
    @property
    def attachment_amat(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self._attachment_amat.copy(), self.signals(), self.effects())

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