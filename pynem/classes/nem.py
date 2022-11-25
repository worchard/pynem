from typing import Hashable, Set, Union, Tuple, Any, Iterable, Dict, FrozenSet, List
from os import linesep
from operator import itemgetter

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
                self._signals = set(signals)
                self._effects = set(effects)
            
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
        dictionary where each key is a signal, and each value is a tuple of predicted perturbed effects.
        ----------
        See Also
        --------
        predict_array
        Example
        --------
        >>> from pynem import SignalGraph, EffectAttachments, NestedEffectsModel
        >>> sg = SignalGraph(edges={(0, 1), (0, 2), (1, 2)})
        >>> sg.add_node(3)
        >>> ea = EffectAttachments({'E0': 0, 'E1': 1, 'E2': 2}, signals = {3})
        >>> nem = NestedEffectsModel(signal_graph = sg, effect_attachments = ea)
        >>> nem.predict_dict()
        {0: ('E0', 'E1', 'E2'), 1: ('E1', 'E2'), 2: 'E2', 3: ()}
        """
        flipped_ea = {s: e for e, s in self._effect_attachments.items()}
        pre_dict = {s: (itemgetter(s, *self._signal_graph.children_of(s))(flipped_ea)) for s in flipped_ea.keys()}
        for signal in self._signal_graph._nodes:
            if signal in pre_dict.keys():
                pass
            else:
                pre_dict[signal] = ()
        
        return pre_dict

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
    
    @property
    def amat_tuple(self) -> Tuple[np.ndarray, list]:
        return self._amat_tuple

    # === NEM manipulation
    def add_signal(self, signal: Node):
        """
        Add ``signal`` to the SignalGraph and EffectAttachments.
        Parameters
        ----------
        signal:
            a hashable Python object
        See Also
        --------
        add_signals_from
        Examples
        --------
        >>> from pynem import NestedEffectsModel
        >>> nem = NestedEffectsModel()
        >>> nem.add_signal('S1')
        >>> nem.signals
        {'S1'}
        """
        self._signal_graph.add_node(signal)
        self._effect_attachments.add_signal(signal)
        self._signals.add(signal)

    def add_signals_from(self, signals: Iterable):
        """
        Add signals to the SignalGraph and EffectAttachments from the collection ``signals``.
        Parameters
        ----------
        signals:
            collection of signals to be added.
        See Also
        --------
        add_signal
        Examples
        --------
        >>> from pynem import NestedEffectsModel
        >>> nem = NestedEffectsModel()
        >>> nem.add_signals_from({'S1', 'S2'})
        >>> nem.add_signals_from(range(3, 6))
        >>> nem.signals
        {3, 4, 5, 'S2', 'S1'}
        """
        for signal in signals:
            self.add_signal(signal)
        
    ########
    ########

    def parents_of(self, signals: NodeSet) -> Set[Node]:
        """
        Return all signals that are parents of the signal or set of signals ``signals``.
        Parameters
        ----------
        signals
            A signal or set of signals.
        See Also
        --------
        children_of
        Examples
        --------
        >>> from pynem import SignalGraph, NestedEffectsModel
        >>> g = SignalGraph(edges={(1, 2), (2, 3)})
        >>> nem = NestedEffectsModel(signal_graph = g)
        >>> nem.parents_of(2)
        {1}
        >>> nem.parents_of({2, 3})
        {1, 2}
        """
        return self._signal_graph.parents_of(signals)

    def children_of(self, signals: NodeSet) -> Set[Node]:
        """
        Return all signals that are children of the signal or set of signals ``signals``.
        Parameters
        ----------
        nodes
            A signal or set of signals.
        See Also
        --------
        parents_of
        Examples
        --------
        >>> from pynem import SignalGraph, NestedEffectsModel
        >>> g = SignalGraph(edges={(1, 2), (2, 3)})
        >>> nem = NestedEffectsModel(signal_graph = g)
        >>> nem.children_of(1)
        {2}
        >>> nem.children_of({1, 2})
        {2, 3}
        """
        return self._signal_graph.children_of(signals)

    def remove_signal(self, signal: Node, ignore_error=False):
        """
        Remove the signal ``signal`` from consideration in the model.
        Parameters
        ----------
        signal:
            signal to be removed.
        ignore_error:
            if True, ignore the KeyError raised when node is not in the SignalGraph.
        Examples
        --------
        >>> from pynem import SignalGraph, NestedEffectsModel
        >>> g = SignalGraph(edges={(1, 2)})
        >>> nem = NestedEffectsModel(signal_graph = g)
        >>> nem.remove_signal(2)
        >>> nem.signal_graph.nodes
        {1}
        >>> nem
        Nested Effects Model object
        Score: None
        Data: AnnData object with n_obs × n_vars = 0 × 0
        Signal graph: Signal graph of 1 nodes and 0 edges
        Effect Attachments: {}, signals = {1}
        """
        try:
            self._signal_graph.remove_node(signal)
            self._effect_attachments._signals.remove(signal)
            self._signals.remove(signal)
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def add_edge(self, i: Node, j: Node):
        """
        Add the edge ``i`` -> ``j`` to the SignalGraph
        Parameters
        ----------
        i:
            source signal of the edge
        j:
            target signal of the edge
        
        See Also
        --------
        add_edges_from
        Examples
        --------
        >>> from pynem import SignalGraph, NestedEffectsModel
        >>> g = SignalGraph({1, 2})
        >>> nem = NestedEffectsModel(signal_graph = g)
        >>> nem.add_edge(1, 2)
        >>> nem.signal_graph.edges
        {(1, 2)}
        """
        self._signal_graph.add_edge(i, j)

        if None in self._signals:
            self._signals = set(self._signal_graph._nodes)
            self._signals.add(None)
            self._effect_attachments._signals = set(self._signal_graph._nodes)
            self._effect_attachments._signals.add(None)
        else:
            self._signals = set(self._signal_graph._nodes)
            self._effect_attachments._signals = set(self._signal_graph._nodes)

    def add_edges_from(self, edges: Union[Set[Edge], Iterable[Edge]]):
        """
        Add edges to the SignalGraph from the collection ``edges``.
        Parameters
        ----------
        edges:
            collection of edges to be added.

        See Also
        --------
        add_edge
        Examples
        --------
        >>> from pynem import SignalGraph, NestedEffectsModel
        >>> g = SignalGraph(edges={(1, 2)})
        >>> nem = NestedEffectsModel(signal_graph = g)
        >>> nem.add_edges_from({(1, 3), (2, 3)})
        >>> nem.signal_graph.edges
        {(2, 3), (1, 2), (1, 3)}
        """
        self._signal_graph.add_edges_from(edges)

        if None in self._signals:
            self._signals = set(self._signal_graph._nodes)
            self._signals.add(None)
            self._effect_attachments._signals = set(self._signal_graph._nodes)
            self._effect_attachments._signals.add(None)
        else:
            self._signals = set(self._signal_graph._nodes)
            self._effect_attachments._signals = set(self._signal_graph._nodes)
    
    def remove_edge(self, i: Node, j: Node, ignore_error=False):
        """
        Remove the edge ``i`` -> ``j``.
        Parameters
        ----------
        i:
            source of edge to be removed.
        j:
            target of edge to be removed.
        ignore_error:
            if True, ignore the KeyError raised when edge is not in the SignalGraph.
        Examples
        --------
        >>> from pynem import SignalGraph, NestedEffectsModel
        >>> g = SignalGraph(edges={(1, 2)})
        >>> nem = NestedEffectsModel(signal_graph = g)
        >>> nem.remove_edge(1, 2)
        >>> nem.signal_graph.edges
        set()
        """
        self._signal_graph.remove_edge(i, j, ignore_error)

    def remove_edges_from(self, edges: Iterable, ignore_error=False):
        """
        Remove each edge in ``edges`` from the SignalGraph.
        Parameters
        ----------
        edges
            The edges to be removed from the SignalGraph.
        ignore_error:
            if True, ignore the KeyError raised when an edges is not in the SignalGraph.
        Examples
        --------
        >>> from pynem import SignalGraph, NestedEffectsModel
        >>> g = SignalGraph(edges={(1, 2), (2, 3), (3, 4)})
        >>> nem = NestedEffectsModel(signal_graph = g)
        >>> nem.remove_edges_from({(1, 2), (2, 3)})
        >>> nem.signal_graph.edges
        {(3, 4)}
        """
        for i, j in edges:
            self._signal_graph.remove_edge(i, j, ignore_error=ignore_error)

    def join_signals(self, signals_to_join: Set[Hashable]):
        """
        Join the signals in the set ``signals_to_join`` into a single multi-node.
        Parameters
        ----------
        signals_to_join:
            set of signals to be joined

        See Also
        --------
        split_signal
        Examples
        --------
        >>> from pynem import SignalGraph, NestedEffectsModel
        >>> g = SignalGraph(edges={(1, 2), (2, 3)})
        >>> nem = NestedEffectsModel(signal_graph = g)
        >>> nem.join_signals({1, 2})
        >>> nem.signals
        {3, frozenset({1, 2})}
        """
        self._signal_graph.join_nodes(signals_to_join)
        
        if None in self._signals:
            self._signals = set(self._signal_graph._nodes)
            self._signals.add(None)
            self._effect_attachments._signals = set(self._signal_graph._nodes)
            self._effect_attachments._signals.add(None)
        else:
            self._signals = set(self._signal_graph._nodes)
            self._effect_attachments._signals = set(self._signal_graph._nodes)
    
    def split_signal(self, node: Node, multinode: FrozenSet[Node], direction: str = 'up'):
        """
        Split ``node`` off from ``multinode`` either 'up' (so that ``node`` is made a parent of the new node) or
        'down' (so that ``node`` is made a child of the new node). Both nodes resulting from the split inherit all
        the parents and children of the original multinode.
        Parameters
        ----------
        node:
            node to be split off

        multinode:
            multi-node from which the node is being split off
        
        direction:
            either 'up' or 'down' resulting in either ``node`` becoming parent of the newly split multinode or becoming
            the child, respectively. Defaults to 'up'.

        See Also
        --------
        join_signals
        Examples
        --------
        >>> from pynem import SignalGraph, NestedEffectsModel
        >>> g = SignalGraph(nodes = {frozenset({1,2}), 3}, edges = {(frozenset({1,2}), 3)})
        >>> nem = NestedEffectsModel(signal_graph = g)
        >>> nem.split_signal(node = 1, multinode = frozenset({1,2}), direction = 'up')
        >>> nem.signals, nem.signal_graph.edges
        ({1, 2, 3}, {(2, 3), (1, 2), (1, 3)})
        """
        self._signal_graph.split_node(node, multinode, direction)
        if None in self._signals:
            self._signals = set(self._signal_graph._nodes)
            self._signals.add(None)
            self._effect_attachments._signals = set(self._signal_graph._nodes)
            self._effect_attachments._signals.add(None)
        else:
            self._signals = set(self._signal_graph._nodes)
            self._effect_attachments._signals = set(self._signal_graph._nodes)
    
    def to_adjacency(self, signal_list: List[Node] = list(), effect_list: List[Node] = list(), 
    save: bool = False) -> Tuple[np.ndarray, list]:
        """
        Return the adjacency matrix for the full graph, including both SignalGraph and EffectAttachments.
        Signals always index the rows and columns first, followed by the effects.
        Parameters
        ----------
        signal_list:
            List indexing the first len(signal_list) rows/columns of the matrix.
        effect_list:
            List indexing the next len(effect_list) rows/columns of the matrix.
        save:
            Boolean indicating whether the adjacency matrix and associated node_list should be saved as the ``NestedEffectsModel.amat_tuple`` attribute
        Return
        ------
        (adjacency_matrix, node_list)
        Example
        -------
        >>> from pynem import SignalGraph, EffectAttachments, NestedEffectsModel
        >>> sg = SignalGraph(edges={('S1', 'S2'), ('S1', 'S3'), ('S2', 'S3')})
        >>> ea = EffectAttachments({'E1':'S1', 'E2':'S2', 'E3':'S3'})
        >>> nem = NestedEffectsModel(signal_graph = sg, effect_attachments = ea)
        >>> adjacency_matrix, node_list = nem.to_adjacency(signal_list = ['S1', 'S2', 'S3'], effect_list = ['E1', 'E2', 'E3'])
        >>> adjacency_matrix
        array([[1, 1, 1, 1, 0, 0],
               [0, 1, 1, 0, 1, 0],
               [0, 0, 1, 0, 0, 1],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])
        >>> node_list
        ['S1', 'S2', 'S3', 'E1', 'E2', 'E3']
        """
        sg_amat, signal_list = self._signal_graph.to_adjacency(node_list=signal_list)
        ea_amat, signal_list, effect_list = self._effect_attachments.to_adjacency(signal_list=signal_list, effect_list=effect_list)
        zero_array = np.zeros((ea_amat.shape[0], sg_amat.shape[1] + ea_amat.shape[1]), dtype=int)

        tmp_array = np.hstack([sg_amat, ea_amat])
        adjacency_matrix = np.vstack([tmp_array, zero_array])

        node_list = signal_list + effect_list

        if save:
            self._amat_tuple = (adjacency_matrix, node_list)

        return adjacency_matrix, node_list