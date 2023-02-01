from typing import Hashable, Set, Union, Tuple, Any, Iterable, Dict, FrozenSet, List
from collections import defaultdict
from os import linesep
from operator import itemgetter
from itertools import chain

from pynem.utils import core_utils
from pynem.custom_types import *
from pynem.classes import ExtendedGraph

import numpy as np

class NestedEffectsModel(ExtendedGraph):
    """
    Class uniting the data, graph and learning algorithms to facilitate scoring and learning of Nested Effects Models.
    """
    def __init__(self, data: np.ndarray = np.array([]), col_data: list = list(), row_data: list = list(),
                actions: List = list(), effects: List = list(), structure_prior: np.ndarray = None,
                attachments_prior: np.ndarray = None, alpha: float = 0.13, beta: float = 0.05,
                lambda_reg: float = 0, delta: float = 1, actions_graph: Union[Iterable[Edge], np.ndarray] = None,
                effect_attachments: Union[Iterable[Edge], np.ndarray] = None, nem = None):
        if nem is not None:
            #NEM specific
            self._score = nem.score
            self._alpha = nem.alpha
            self._beta = nem.beta
            self._lambda_reg = nem.lambda_reg
            self._delta = nem.delta
            if nem._structure_prior is not None:
                self._structure_prior = nem._structure_prior.copy()
            else:
                self._structure_prior = None
            if nem._attachments_prior is not None:
                self._attachments_prior = nem._attachments_prior.copy()
            else:
                self._attachments_prior = None
            self._data = nem.copy()
            self._col_data = nem._col_data.copy()
            self._row_data = nem._row_data.copy()
            #ExtendedGraph
            self._nactions = nem._nactions
            self._neffects = nem._neffects
            self._property_array = nem.property_array
            self._actions_amat = nem._actions_amat.copy()
            self._attachments_amat = nem._attachments_amat.copy()
            self._join_array = nem._join_array.copy()
        else:
            #misc and hyper-parameters
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

            self._data = data.copy()
            self._col_data = np.array(col_data)
            self._row_data = np.array(row_data)

            actions = np.array(actions)
            effects = np.array(effects)

            if self._data.size > 0:
                if self._col_data.size > 0:
                    if not self._data.shape[1] == self._col_data.size:
                        raise ValueError("Dimensions of data and col_data do not match!")
                    if actions.size > 0:
                        if not np.all(np.isin(actions, self._col_data)):
                            raise ValueError("Not all provided actions found in col_data!")
                    else:
                        _, idx = np.unique(self._col_data, return_index=True)
                        actions = self._col_data[idx]
                else:
                    if actions.size > 0:
                        if not self._data.shape[1] == actions.size:
                            raise ValueError("Dimensions of data and actions do not match!")
                        else:
                            self._col_data = actions
                    else:
                        actions = np.array(range(self._data.shape[1]))
                        self._col_data = actions
                if self._row_data.size > 0:
                    if not self._data.shape[0] == self._row_data.size:
                        raise ValueError("Dimensions of data and row_data do not match!")
                    if effects.size > 0:
                        if not np.all(np.isin(effects, self._row_data)):
                            raise ValueError("Not all provided effects found in row_data!")
                    else:
                        effects = self._row_data
                else:
                    if effects.size > 0:
                        if not self._data.shape[0] == effects.size:
                            raise ValueError("Dimensions of data and effects do not match!")
                        else:
                            self._row_data = effects
                    else:
                        effects = np.array(range(self._data.shape[0]))
                        self._row_data = effects

            self._structure_prior = None
            if structure_prior is not None:
                if actions.size > 0:
                    if structure_prior.shape[0] == actions.size:
                        self._structure_prior = structure_prior.copy()
                    else:
                        raise ValueError("Dimenions of structure_prior and actions do not match!")
                else:
                    self._structure_prior = structure_prior.copy()
                    actions = np.array(range(self._structure_prior.shape[0]))
            
            self._attachments_prior = None
            if attachments_prior is not None:
                if effects.size > 0:
                    if attachments_prior.shape[0] == effects.size:
                        self._attachments_prior = attachments_prior.copy()
                    else:
                        raise ValueError("Dimenions of attachments_prior and actions do not match!")
                else:
                    self._attachments_prior = attachments_prior.copy()
                    effects = np.array(range(self._attachments_prior.shape[0]))

            edges = set()
            actions_amat = None
            if actions_graph is not None:
                if isinstance(actions_graph[0], tuple):
                    edges = actions_graph.copy()
                elif isinstance(actions_graph, np.ndarray):
                    actions_amat = actions_graph.copy()
                else:
                    raise ValueError("actions_graph needs to either be an iterable of edges or an adjacency matrix")

            attachments = set()
            attachments_amat = None
            if effect_attachments is not None:
                if isinstance(effect_attachments[0], tuple):
                    attachments = effect_attachments.copy()
                elif isinstance(effect_attachments, np.ndarray):
                    attachments_amat = effect_attachments.copy()
                else:
                    raise ValueError("effect_attachments needs to either be an iterable of attachments or an adjacency matrix")

            super().__init__(actions=actions, effects=effects, actions_amat=actions_amat, attachments_amat=attachments_amat,
                            edges=edges, attachments=attachments)
    
    # === BASIC METHODS

    def copy(self):
        """
        Return a copy of the current NestedEffectsModel.
        """
        return NestedEffectsModel(nem=self)
    
    def __str__(self) -> str:
        return NotImplemented
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
    def data(self) -> np.ndarray:
        return self._data.copy()

    @property
    def col_data(self) -> np.ndarray:
        return self._col_data.copy()

    @property
    def row_data(self) -> np.ndarray:
        return self._row_data.copy()
    
    @property
    def data_tuple(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.data, self.row_data, self.col_data)

    @property
    def structure_prior(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self._structure_prior.copy(), self.actions())
    
    @property
    def attachments_prior(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self._attachments_prior.copy(), self.actions(), self.effects())

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

    # === CORE METHODS

    def _can_add_edge(self, i: int, j: int):
        """
        Checks whether one can add the edge i -> j and preserve transitivity
        """
        out = self._parents(i).issubset(self._parents(j)) \
            and self._children(j).issubset(self._children(i))
        return out
    
    def _can_remove_edge(self, i: int, j:int):
        """
        Checks whether one can remove the edge i -> j and preserve transitivity
        """
        return len(self._children(i).intersection(self._parents(j))) == 0
    
    def _can_join_actions(self, i: int, j:int):
        """
        Checks whether one can join the action ``i`` (along with the actions it is joined to) to the action ``j`` 
        (along with the actions it is joined to) and preserve transitivity
        """
        i_parents = self._parents(i).difference({j})
        j_parents = self._parents(j)
        i_children = self._children(i)
        j_children = self._children(j).difference({i})
        
        check = i_parents.issubset(j_parents) and j_children.issubset(i_children)
        return bool(self._actions_amat[j,i] and check)

    def learn(self):
        raise NotImplementedError

    def _data2summaries(self):
        if self._nactions == self._col_data.size:
            self._D1 = (self._data == 1).astype('int64')
            self._D0 = (self._data == 0).astype('int64')
            self._DNaN = np.isnan(self._data).astype('int64')
        else:
            self._D1 = np.array([np.sum(self._data.T[a == self._col_data] == 1, axis = 0) for a in self.actions()]).T
            self._D0 = np.array([np.sum(self._data.T[a == self._col_data] == 0, axis = 0) for a in self.actions()]).T
            self._DNaN = np.array([np.sum(np.isnan(self._data.T[a == self._col_data]), axis = 0) for a in self.actions()]).T
        
    def transitive_closure(self):
        raise NotImplementedError

    def transitive_reduction(self):
        raise NotImplementedError