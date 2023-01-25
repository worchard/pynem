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
    Base class for graphs over Nested Effects Models, combining both the graph over perturbations, 'action' nodes, and downstream
    'effect' nodes. Actions are connected by 'edges' to each other and by 'attachments' to effects.
    """
    def __init__(self, actions: List[Node] = list(), effects: List[Node] = list(), 
                 edges: Iterable[Edge] = set(), attachments: Iterable[Edge] = set(), 
                 actions_amat: np.ndarray = None, attachments_amat: np.ndarray = None, 
                 extended_graph = None):
        if extended_graph is not None:
            self._nactions = extended_graph._nactions
            self._neffects = extended_graph._neffects
            self._property_array = extended_graph.property_array
            self._actions_amat = extended_graph._actions_amat.copy()
            self._attachments_amat = extended_graph._attachments_amat.copy()
            self._join_array = extended_graph._join_array.copy()
        else:
            actions = list(actions)
            effects = list(effects)
            if actions_amat is None:
                if edges:
                    actions = actions + list(set(chain(*edges)).difference(actions))
                self._nactions = len(actions)
                self._actions_amat = np.zeros((self._nactions, self._nactions), dtype = 'B')
            else:
                self._nactions = actions_amat.shape[0]
                if actions and len(actions) != self._nactions:
                    raise ValueError("Dimensions of actions list and actions_amat do not match!")
                self._actions_amat = actions_amat.copy()
            np.fill_diagonal(self._actions_amat, 1)
                
            if attachments_amat is None:
                if attachments:
                    effects = effects + list({i[1] for i in attachments}.difference(effects))
                self._neffects = len(effects)
                self._attachments_amat = np.zeros((self._nactions, self._neffects), dtype = 'B')
            else:
                self._neffects = attachments_amat.shape[1] - self._nactions
                if (effects and len(effects) != self._neffects) or attachments_amat.shape[0] != self._nactions:
                    raise ValueError("Dimensions of the actions and/or effects list do not match the attachments_amat!")
                self._attachments_amat  = attachments_amat.copy()

            if not actions:
                actions = list(range(self._nactions))
            if not effects:
                effects = list(range(self._neffects))

            #initialise and populate property array
            self._property_array = np.empty(self._nactions + self._neffects, dtype={'names':('name', 'is_action'), 'formats': ('object', 'B')})
            self._property_array['name'] = np.array(actions + effects)
            self._property_array['is_action'] = np.array([True]*self._nactions + [False]*self._neffects)

            self._join_array = np.eye(self._nactions, dtype='bool')
            
            self.add_edges_from(edges)
            self.attach_effects_from(attachments)
    
    # === CORE CLASS METHODS

    def __eq__(self, other):
        if not isinstance(other, ExtendedGraph):
            return False
        return np.array_equal(self._property_array, other._property_array) and np.array_equal(self._actions_amat, other._actions_amat) \
            and np.array_equal(self._attachments_amat, other._attachments_amat)
    
    def copy(self):
        return ExtendedGraph(extended_graph=self)
    
    # === BASIC METHODS
    
    def actions_idx(self) -> np.ndarray:
        return np.array(range(self._nactions))
    
    def actions(self) -> np.ndarray:
        return self._property_array['name'][:self._nactions].copy()
    
    def effects_idx(self) -> np.ndarray:
        return np.array(range(self._nactions, self.nnodes))
    
    def effects(self) -> np.ndarray:
        return self._property_array['name'][self._nactions:].copy()
    
    def edges_idx(self) -> list:
        return [*zip(*self._actions_amat.nonzero())]
    
    def attachments_idx(self) -> list:
        return [*zip(*self._attachments_amat.nonzero())]
    
    def edges(self) -> list:
        edge_array = self._actions_amat.nonzero()
        sources = self._property_array['name'][edge_array[0]]
        sinks = self._property_array['name'][edge_array[1]]
        return [*zip(sources, sinks)]
    
    def attachments(self) -> list:
        attachment_array = self._attachments_amat.nonzero()
        sources = self._property_array['name'][attachment_array[0]]
        sinks = self._property_array['name'][attachment_array[1]]
        return [*zip(sources, sinks)]

    def _full_amat(self) -> np.ndarray:
        return np.c_[self._actions_amat, self._attachments_amat]
    
    def full_amat(self) -> Tuple[np.ndarray, np.ndarray]:
        action_array = self.actions()
        node_array = np.r_[action_array, self.effects()]
        return (self._full_amat(), action_array, node_array)
    
    def _parents_of(self, actions: Union[int, List[int]]) -> Set[Node]:
        """
        Return all actions that are parents of the actions in the list ``actions``.
        Parameters
        ----------
        actions
            A list of actions.
        See Also
        --------
        children_of
        Examples
        """
        return set(self._actions_amat[:,actions].nonzero()[0])

    def _children_of(self, actions: List[int]) -> Set[Node]:
        """
        Return all actions that are children of the actions in the list ``actions``.
        Parameters
        ----------
        actions
            A list of actions.
        See Also
        --------
        parents_of
        Examples
        --------
        """
        return set(self._actions_amat[actions].nonzero()[1])
    
    def _joined_to(self, action: int) -> np.ndarray:
        return self._join_array[action].nonzero()[0].astype('B') #if a lil_matrix the index here needs to be switched back to [1], but otherwise same
    
    # === RELATION MANIPULATION METHODS PRIVATE

    def _add_edge(self, i: int, j: int, inplace: bool = True):
        if i == j:
            warnings.warn("Self loops are present by default so adding them does nothing!")
            return
        if inplace:
            for i in self._joined_to(i):
                self._actions_amat[i, self._joined_to(j)] = 1
        else:
            actions_amat = self._actions_amat.copy()
            for i in self._joined_to(i):
                actions_amat[i, self._joined_to(j)] = 1
            return actions_amat
        
    
    def _add_edges_from(self, edges: Iterable[Edge], inplace: bool = True):
        if len(edges) == 0:
            return
        if inplace:
            for i, j in edges:
                self._add_edge(i, j)
        else:
            actions_amat = self._actions_amat.copy()
            for i, j in edges:
                for i_join in self._joined_to(i):
                    actions_amat[i_join, self._joined_to(j)] = 1
            return actions_amat


    def _remove_edge(self, i: int, j: int, inplace: bool = True):
        if i == j:
            warnings.warn("Self loops are present by default and cannot be removed!")
            return
        if inplace:
            for i in self._joined_to(i):
                self._actions_amat[i, self._joined_to(j)] = 0
        else:
            actions_amat = self._actions_amat.copy()
            for i in self._joined_to(i):
                actions_amat[i, self._joined_to(j)] = 0
    
    def _remove_edges_from(self, edges: Iterable[Edge], inplace: bool = True):
        if len(edges) == 0:
            return
        if inplace:
            for i, j in edges:
                self._remove_edge(i, j)
        else:
            actions_amat = self._actions_amat.copy()
            for i, j in edges:
                for i_join in self._joined_to(i):
                    actions_amat[i_join, self._joined_to(j)] = 0
            return actions_amat

    def _attach_effect(self, action: int, effect: int, inplace: bool = True):
        if inplace:
            self._detach_effect(effect)
            self._attachments_amat[action, effect - self._nactions] = 1
        else:
            attachments_amat = self._detach_effect(effect, inplace = False)
            attachments_amat[action, effect - self._nactions] = 1
            return attachments_amat
    
    def _attach_effects_from(self, attachments: Iterable[Edge], inplace: bool = True):
        if len(attachments) == 0:
            return
        if inplace:
            for action, effect in attachments:
                self._attach_effect(action, effect)
        else:
            for action, effect in attachments:
                attachments_amat = self._detach_effect(effect, inplace = False)
                attachments_amat[action, effect - self._nactions] = 1
            return attachments_amat
    
    def _detach_effect(self, effect: int, inplace: bool = True):
        if inplace:
            self._attachments_amat[:, effect - self._nactions] = 0
        else:
            attachments_amat = self._attachments_amat.copy()
            attachments_amat[:, effect - self._nactions] = 0
            return attachments_amat
    
    def _detach_effects_from(self, effects: Iterable[Node], inplace: bool = True):
        if len(effects) == 0:
            return
        if inplace:
            for effect in effects:
                self._detach_effect(effect)
        else:
            attachments_amat = self._attachments_amat.copy()
            for effect in effects:
                attachments_amat[:, effect - self._nactions] = 0
            return attachments_amat

    # === RELATION MANIPULATION METHODS PUBLIC

    def add_edge(self, i: Node, j: Node, inplace: bool = True):
        """
        Add the edge from action ``i`` to action ``j`` to the ExtendedGraph
        Parameters
        ----------
        i:
            source action node of the edge
        j:
            target action node of the edge
        
        See Also
        --------
        add_edges_from
        Examples
        --------
        """
        i = self.name2idx(i)
        j = self.name2idx(j)
        return self._add_edge(i, j, inplace)
    
    def remove_edge(self, i: Node, j: Node, inplace: bool = True):
        """
        Remove the edge from action ``i`` to action ``j`` to the ExtendedGraph
        Parameters
        ----------
        i:
            source action node of the edge
        j:
            target action node of the edge
        
        See Also
        --------
        add_edges_from
        Examples
        --------
        """
        i = self.name2idx(i)
        j = self.name2idx(j)
        return self._remove_edge(i, j, inplace)
    
    def add_edges_from(self, edges: Iterable[Edge], inplace: bool = True):
        """
        Add edges between actions to the graph from the collection ``edges``.
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
        return self._add_edges_from(edges_idx, inplace)
    
    def remove_edges_from(self, edges: Iterable[Edge], inplace: bool = True):
        """
        Remove edges between actions to the graph from the collection ``edges``.
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
        return self._remove_edges_from(edges_idx, inplace)
    
    def attach_effect(self, action: Node, effect: Node, inplace: bool = True):
        action = self.name2idx(action)
        effect = self.name2idx(effect, is_action=False)
        return self._attach_effect(action, effect, inplace)
    
    def detach_effect(self, effect: Node, inplace: bool = True):
        effect = self.name2idx(effect, is_action=False)
        return self._detach_effect(effect, inplace)

    def attach_effects_from(self, attachments: Iterable[Edge], inplace: bool = True):
        if len(attachments) == 0:
            return
        attachments_idx = self.edgeNames2idx(attachments, is_action=False)
        return self._attach_effects_from(attachments_idx, inplace)
    
    def detach_effects_from(self, effects: Iterable[Node], inplace: bool = True):
        if len(effects) == 0:
            return
        effects_idx = self.names2idx(effects, is_action=False)
        self._detach_effects_from(effects_idx, inplace)

    # === NODE MANIPULATION METHODS

    def add_action(self, name = "as_index"):
        raise NotImplementedError
        #First redo the adjacency matrix
        new_amat = np.zeros((self._amat.shape[0] + 1, self._amat.shape[1] + 1), dtype='B')
        orig_cols = np.append(range(self.nactions), range(self.nactions+1, self.nnodes + 1))
        if self.nactions > 0:
            new_amat[:self.nactions, orig_cols] = self._amat
        self._amat = new_amat
        self._amat[self.nactions, self.nactions] = 1 #add self-loop

        #Then the property array
        if name == "as_index":
            name = self.nactions
        new_name = core_utils.get_unique_name(name, self.actions())
        if new_name != name:
            warnings.warn(f"Signal name changed to {new_name} to avoid a clash with an existing action")
        ### --- following line needs modifying if new properties become supported --- ###
        self._property_array = np.insert(self._property_array, self.nactions, (new_name, 1))

        #finally update nactions
        self._nactions += 1
    
    def add_effect(self, name = "as_index"):
        self._attachments_amat = np.c_[self._attachments_amat, np.zeros(self.nactions, dtype='B')]
        if name == "as_index":
            name = self.nnodes
        new_name = core_utils.get_unique_name(name, self.effects())
        if new_name != name:
            warnings.warn(f"Effect name changed to {new_name} to avoid a clash with an existing effect")
        new_row = np.array((new_name, 0), dtype={'names': ('name', 'is_action'), 'formats': ('object', 'B')})
        self._property_array = np.r_[self._property_array, new_row]
        self._neffects += 1
    
    def _remove_action(self, action: int):
        raise NotImplementedError
        if action not in self.actions_idx():
            raise ValueError("Signal not in graph")
        orig_cols = np.append(range(action), range(action+1, self.nnodes)).astype('B')
        self._amat = self._amat[orig_cols[:self.nactions - 1]][:, orig_cols]
        self._property_array = np.delete(self._property_array, action, 0)
        self._nactions -= 1
    
    def remove_action(self, action):
        raise NotImplementedError
        action = self.name2idx(action)
        self._remove_action(action)
    
    def _remove_actions_from(self, actions: List[int]):
        raise NotImplementedError
        if not np.all(np.isin(actions, self.actions_idx())):
            raise ValueError("All actions being removed must be in the graph")
        orig_cols = [action for action in range(self.nnodes) if action not in actions]
        self._amat = self._amat[orig_cols[:self.nactions - len(actions)]][:,orig_cols]
        self._property_array = np.delete(self._property_array, actions, 0)
        self._nactions -= len(actions)
    
    def remove_actions_from(self, actions: List[int]):
        raise NotImplementedError
        actions = self.names2idx(np.array(actions))
        self._remove_actions_from(actions)
    
    def _remove_effect(self, effect):
        if effect not in self.effects_idx():
            raise ValueError("Effect not in graph")
        effect -= self.nactions
        orig_cols = np.append(range(effect), range(effect+1, self.neffects)).astype('B')
        self._attachments_amat = self._attachments_amat[:, orig_cols]
        self._property_array = np.delete(self._property_array, effect, 0)
        self._neffects -= 1
    
    def remove_effect(self, effect: Node):
        effect = self.name2idx(effect, is_action=False)
        self._remove_effect(effect)
    
    def _join_actions(self, i: int, j: int, inplace: bool = True):
        i_joined_to = self._joined_to(i)
        j_joined_to = self._joined_to(j)
        if inplace:
            for i in i_joined_to:
                self._join_array[i, j_joined_to] = 1
                self._join_array[j_joined_to, i] = 1
                self._actions_amat[i, j_joined_to] = 1
                self._actions_amat[j_joined_to, i] = 1
        else:
            join_array = self._join_array.copy()
            actions_amat = self._actions_amat.copy()
            for i in i_joined_to:
                join_array[i, j_joined_to] = 1
                join_array[j_joined_to, i] = 1
                actions_amat[i, j_joined_to] = 1
                actions_amat[j_joined_to, i] = 1
            return (actions_amat, join_array)
    
    def join_actions(self, i: Node, j: Node, inplace: bool = True):
        i_idx = self.name2idx(i)
        j_idx = self.name2idx(j)
        return self._join_actions(i_idx, j_idx, inplace)
    
    def _splitoff_actions(self, actions: list, inplace: bool = True):
        """
        Splits actions in ``actions`` from any joined node they are participating in to produce
        two joined nodes: one containing actions in ``actions`` as a parent to the joined node
        containing the remaining actions left in the original joined node.
        """
        i = actions[0]
        i_joined_to = self._joined_to(i)
        if len(actions) == i_joined_to.size:
            return
        if not np.all(np.isin(actions, i_joined_to)):
            raise ValueError("Not all actions being split are joined")
        split_from = np.setdiff1d(i_joined_to, actions)
        if inplace:
            for a in actions:
                self._join_array[a, split_from] = 0
                self._join_array[split_from, a] = 0
                self._actions_amat[split_from, a] = 0
        else:
            join_array = self._join_array.copy()
            actions_amat = self._actions_amat.copy()
            for a in actions:
                join_array[a, split_from] = 0
                join_array[split_from, a] = 0
                actions_amat[split_from, a] = 0
            return (actions_amat, join_array)
    
    def splitoff_actions(self, actions: List[Node], inplace: bool = True):
        actions_idx = self.names2idx(actions)
        self._splitoff_actions(actions_idx, inplace)

    # === PROPERTIES

    @property
    def nactions(self) -> int:
        return self._nactions
    
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
    def actions_amat(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self._actions_amat.copy(), self.actions())
    
    @property
    def attachments_amat(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self._attachments_amat.copy(), self.actions(), self.effects())

    # === UTILITY METHODS

    def name2idx(self, name, is_action: bool = True) -> int:
        """
        Convert a given node ``name`` to its corresponding index according to the ``property_array``
        of an ExtendedGraph object.
        Parameters
        ----------
        name:
            Node name to convert to an index. Note this name must appear in the name column of the property array.
        is_action:
            Boolean indicating whether the name is of a action node or an effect node.
        Examples
        --------
        >>> from pynem import ExtendedGraph
        >>> eg = ExtendedGraph(actions = ['S1', 'S2', 'S3'], effects = ['E1', 'E2', 'E3'])
        >>> eg.name2idx('S1')
        0
        """
        mask = self._property_array['is_action'] == is_action
        if is_action:
            return np.nonzero(self._property_array['name'][mask] == name)[0][0]
        else:
            return np.nonzero(self._property_array['name'][mask] == name)[0][0] + self._nactions

    def names2idx(self, name_array, is_action: bool = True) -> np.ndarray:
        """
        Convert node names given in a 1D ndarray ``name_array`` to a corresponding array of node indices 
        according to the ``property_array`` of an ExtendedGraph object.
        Parameters
        ----------
        name_array:
            ndarray of node names to convert to indices. Note all names must appear in the name column of the property array.
        is_action:
            Boolean indicating whether the names in ``name_array`` are of action nodes or effect nodes.
        Examples
        --------
        >>> from pynem import ExtendedGraph
        >>> eg = ExtendedGraph(actions = ['S1', 'S2', 'S3'], effects = ['E1', 'E2', 'E3'])
        >>> eg.names2idx(np.array(['S1', 'S3']))
        array([0, 2])
        """
        mask = self._property_array['is_action'] == is_action
        full_name_array = self._property_array['name'][mask]
        sorter = full_name_array.argsort()
        if is_action:
            return sorter[np.searchsorted(full_name_array, name_array, sorter=sorter)]
        else:
            return sorter[np.searchsorted(full_name_array, name_array, sorter=sorter)] + self._nactions

    def edgeNames2idx(self, edges, is_action: bool = True) -> list:
        """
        Convert an iterable of edges referring to nodes by name to a corresponding list 
        of edges referring nodes by their indices, according to the ``property_array``
        of an ExtendedGraph object.
        Parameters
        ----------
        edges:
            Iterable of edges to convert. Note all node names must appear in the name column of the property array.
        is_action:
            Boolean indicating whether the edges in ``edges`` are between actions or are attaching actions to effects.
        Examples
        --------
        >>> from pynem import ExtendedGraph
        >>> eg = ExtendedGraph(actions = ['S1', 'S2', 'S3'], effects = ['E1', 'E2', 'E3'], \
            edges = [('S1', 'S2'), ('S2', 'S3'), ('S1', 'S3')])
        >>> eg.edgeNames2idx([('S1', 'S2'), ('S2', 'S3')])
        [(0, 1), (1, 2)]
        """
        edge_tuples = [*zip(*edges)]
        sources = self.names2idx(edge_tuples[0])
        sinks = self.names2idx(edge_tuples[1], is_action=is_action)
        return [*zip(sources, sinks)]