from typing import Hashable, Set, Union, Tuple, Any, Iterable, Dict, FrozenSet, List
from collections import defaultdict
from os import linesep
from operator import itemgetter
from itertools import chain

from pynem.utils import core_utils
from pynem.custom_types import *
from pynem.classes import SignalGraph, EffectAttachments

import anndata as ad
import numpy as np

class NestedEffectsModel2:
    """
    Class uniting the data, graph and learning algorithms to facilitate scoring and learning of Nested Effects Models.
    """
    def __init__(self, adata: ad.AnnData = ad.AnnData(), signals_column: str = 'signals', controls: Iterable = {'control'},
                signals: Set = set(), effects: Set = set(), structure_prior: Union[Iterable[Edge], np.ndarray] = None,
                attachments_prior: Union[Iterable[Edge], np.ndarray] = None, alpha: float = 0.13, beta: float = 0.05,
                lambda_reg: float = 0, delta: float = 1, signal_graph: Union[Iterable[Edge], np.ndarray] = None,
                effect_attachments: Union[Iterable[Edge], np.ndarray] = None, nem = None):
        pass

class NestedEffectsModel():
    """
    Class uniting the separate elements of a Nested Effects Model in order to facilitate scoring and learning.
    """

    def __init__(self, adata: ad.AnnData = ad.AnnData(), signals: Set = set(), effects: Set = set(),
                controls: Iterable = {'control'}, signals_column: str = 'signals', 
                signal_graph: SignalGraph = None, effect_attachments: EffectAttachments = None, 
                alpha: float = 0.13, beta: float = 0.05, lambda_reg: float = 0, delta: float = 1, nem = None):
        self._controls = controls
        self._signals_column = signals_column
        self._score = None
        if not (alpha >= 0 and alpha <= 1):
                raise ValueError("alpha must be between 0 and 1")
        if not (beta >= 0 and beta <= 1):
                raise ValueError("beta must be between 0 and 1")
        if lambda_reg < 0:
            raise ValueError("lambda_reg cannot be negative")
        if delta < 0:
            raise ValueError("delta cannot be negative")
        self._alpha = alpha
        self._beta = beta
        self._lambda_reg = lambda_reg
        self._delta = delta
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

    def __setitem__(self, key, value):
        self.add_signal(value)
        self._effects.add(key)
        self._effect_attachments[key] = value

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
        {0: {'E0', 'E1', 'E2'}, 1: {'E1', 'E2'}, 2: {'E2'}, 3: set()}
        """
        flipped_ea = defaultdict(set)
        for e, s in self._effect_attachments.items():
            flipped_ea[s].add(e)
        
        """This next line takes each signal, looks up both its effects and its children's effects using the flipped effect attachments,
        puts them together into a single set, and then assigns them as a value to a dictionary with each signal as the key"""
        pre_dict = {s_key: set().union(*[flipped_ea[s] for s in [s_key, *self._signal_graph.children_of(s_key)]]) for s_key in self._signal_graph._nodes}
        for signal in self._signal_graph._nodes:
            if signal in pre_dict.keys():
                pass
            else:
                pre_dict[signal] = set()
        
        return pre_dict

    def predict_array(self, replicates: Union[int, Dict[Node, int]] = 1, signal_list: list = list(), 
                    effect_list: list = list(), replicates_from_adata: bool = False) -> Tuple[np.ndarray, list, list]:
        """
        Predict which effect reporters will be affcted by the perturbation of each signal, given a signal graph and effect attachments, and return
        a tuple containing a prediction array, F, and lists indexing the rows (signals) and columns (effects) of the array. F_ij = 1 if effect j is 
        predicted to be dependent on the perturbation of signal i, otherwise F_ij = 0.
        Parameters
        ----------
        replicates:
            A dictionary with signals as keys and the number of replicates of perturbations of each signal as its corresponding value.
            If an integer is provided, it is assumed all signals have ``replicates`` number of replicates.
        signal_list:
            List indexing the rows of the output prediction array
        effect_list:
            List indexing the columns of the output prediction array
        replicates_from_adata:
            Boolean indicating whether ``replicates`` argument should be taken from the internal nem.adata object
        See Also
        --------
        predict_dict
        Return
        ------
        (prediction_array, signal_list, effect_list)
        Example
        --------
        >>> from pynem import SignalGraph, EffectAttachments, NestedEffectsModel
        >>> sg = SignalGraph(edges={(0, 1), (0, 2), (1, 2)})
        >>> sg.add_node(3)
        >>> ea = EffectAttachments({'E0': 0, 'E1': 1, 'E2': 2}, signals = {3})
        >>> nem = NestedEffectsModel(signal_graph = sg, effect_attachments = ea)
        >>> F, signal_list, effect_list = nem.predict_array(replicates = 2, signal_list = [2,1,0], effect_list = ['E1', 'E2', 'E0'])
        >>> F
        array([[0, 1, 0],
               [0, 1, 0],
               [1, 1, 0],
               [1, 1, 0],
               [1, 1, 1],
               [1, 1, 1]])
        >>> signal_list
        [2, 2, 1, 1, 0, 0]
        """
        prediction_dict = self.predict_dict()
        if not signal_list:
            signal_list = list(self._signal_graph._nodes)
        if not effect_list:
            effect_list = list(self._effects)
        
        if isinstance(replicates, int):
            replicates = dict.fromkeys(signal_list, replicates)
        
        if replicates_from_adata:
            replicates = dict(self._adata.obs[self._signals_column].value_counts())

        prediction_lists = [np.tile([e in prediction_dict[s] for e in effect_list], (replicates[s], 1)) for s in signal_list]
        prediction_array = np.vstack(prediction_lists).astype(int)

        if prediction_array.shape[0] != len(signal_list):
            signal_list = list(chain.from_iterable([s]*replicates[s] for s in signal_list))

        return (prediction_array, signal_list, effect_list)

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
        alp = self._alpha
        bet = self._beta
        dat = self._adata.__str__()
        sg = self._signal_graph.__str__()
        ea = self._effect_attachments.__str__()
        out_string = "Nested Effects Model object" + linesep + f"Score: {scr}" + linesep + \
        f"alpha: {alp}, beta: {bet}" + linesep + f"Data: {dat}" + linesep + \
        f"Signal graph: {sg}" + linesep + f"Effect Attachments: {ea}"
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
    def alpha(self) -> float:
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        if not (value >= 0 and value <= 1):
            raise ValueError("alpha must be between 0 and 1")
        self._alpha = value
    
    @property
    def beta(self) -> float:
        return self._beta
    
    @beta.setter
    def beta(self, value):
        if not (value >= 0 and value <= 1):
            raise ValueError("beta must be between 0 and 1")
        self._beta = value
    
    @property
    def lambda_reg(self) -> float:
        return self._lambda_reg
    
    @lambda_reg.setter
    def lambda_reg(self, value):
        if value < 0:
            raise ValueError("lambda_reg cannot be negative")
        self._lambda_reg = value
    
    @property
    def delta(self) -> float:
        return self._delta
    
    @delta.setter
    def delta(self, value):
        if value < 0:
            raise ValueError("delta cannot be negative")
        self._delta = value
    
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
        if signal != None:
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
        alpha: None, beta: None
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