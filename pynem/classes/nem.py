from typing import Hashable, Set, Union, Tuple, Any, Iterable, Dict, FrozenSet, List
from collections import defaultdict
from os import linesep
from operator import itemgetter
from itertools import chain

from pynem.utils import core_utils
from pynem.custom_types import *
from pynem.classes import ExtendedGraph

import numpy as np
from scipy.special import logsumexp
from scipy.stats import gamma
import matplotlib.pyplot as plt
import seaborn as sns

class nem:
    def __init__(self, data: np.ndarray = np.array([]), row_data: list = list(),
                 col_data: list = list(), actions: list = list(), effects: list = list(),
                 structure_prior: np.ndarray = None, attachments_prior: np.ndarray = None, 
                 alpha: float = 0.13, beta: float = 0.05, lambda_reg: float = 0, delta: float = 1,
                 effects_selection: str = 'regularisation'):
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
        if effects_selection not in {'regularisation', 'iterative'}:
            raise ValueError("effects_selection must be either 'regularisation' or 'iterative'")
        self._alpha = alpha
        self._beta = beta
        self._lambda_reg = lambda_reg
        self._delta = delta
        self._effects_selection = effects_selection

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
                    actions = self._col_data[np.sort(idx)]
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
        
        self._nactions = len(actions)
        self._neffects = len(effects)

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
        else:
            if self._lambda_reg > 0:
                self._structure_prior = np.eye(self._nactions)
        
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
        else:
            self._attachments_prior = np.full((self._neffects, self._nactions), 1/self._nactions)
        
        self._actions = actions
        self._effects = effects
        
        #This adds the null actions to the attachments prior
        if self._attachments_prior.shape[1] != self._nactions + 1:
            self._attachments_prior = np.c_[self._attachments_prior, np.full(self._neffects, 1/self._nactions)]
            self._attachments_prior[:,self._nactions] = self._delta/self._nactions
            self._attachments_prior = self._attachments_prior/self._attachments_prior.sum(axis=1)[:, None]
        
        #This computes the log likelihood ratio matrix (from binary input data in this case)
        if self._nactions == self._col_data.size:
            self._D1 = (self._data == 1).astype('int64')
            self._D0 = (self._data == 0).astype('int64')
            DNaN = np.isnan(self._data).astype('int64')
        else:
            self._D1 = np.array([np.sum(self._data.T[a == self._col_data] == 1, axis = 0) for a in self._actions]).T
            self._D0 = np.array([np.sum(self._data.T[a == self._col_data] == 0, axis = 0) for a in self._actions]).T
            DNaN = np.array([np.sum(np.isnan(self._data.T[a == self._col_data]), axis = 0) for a in self._actions]).T
        #following two lines means that if there is a NaN, it has likelihood = 1, hence doesn't affect score
        self._D1 += DNaN
        self._D0 += DNaN

        null_mat = np.log(self._alpha)*self._D1 + np.log(1-self._alpha)*self._D0
        #self._null_const = np.sum(null_mat, axis = 1) this line is essentially never necessary
        self._R = np.log(1-self._beta)*self._D1 + np.log(self._beta)*self._D0 - null_mat
    
    def _phi_distr(self, model: np.ndarray, a=1, b=0.1):
        d = np.abs(model - self._structure_prior)
        pPhi = a/(2*b)*(1 + d/b)**(-a-1)
        return np.sum(np.log(pPhi))
    
    def _incorporate_structure_prior(self, model: np.ndarray):
        if self._lambda_reg != 0:
            return -self._lambda_reg*np.sum(np.abs(model - self._structure_prior)) + \
                np.log(self._lambda_reg*0.5)*2**self.nactions
        else:
            return self._phi_distr(model)

    def _logmarginalposterior(self, model: np.ndarray):
        LL = self._R @ model
        self._LLP = LL+np.log(self._attachments_prior)
        self._LLP_sums = logsumexp(self._LLP,axis=1)
        score = np.sum(self._LLP_sums)
        if self._structure_prior is not None:
            score += self._incorporate_structure_prior(model[:self._nactions, :self._nactions])
        return score

class nemcmc:
    def __init__(self, nem: nem, init: np.ndarray, n: int = 1e5, burn_in: int = 1e4):
        
        self._nem = nem
        self._nactions = nem._nactions
        self._curr = init.copy()
        np.fill_diagonal(self._curr, 0) #This is so parents/children/neighbours are computed properly below
        self._n = int(n)
        self._burn_in = int(burn_in)
        
        self._neighbours = set()

        self._parents = defaultdict(set)
        self._children = defaultdict(set)

        self._out = np.zeros(self._curr.shape)
        self._arratio = []
        nedges = self._curr.sum()
        self._avg_nedges = [nedges]

        for i in range(self._curr.shape[0]):
            self._children[i] = set(self._curr[i].nonzero()[0])
            self._parents[i] = set(self._curr.T[i].nonzero()[0])

        for i in range(self._nactions):
            for j in range(self._nactions):
                if i == j:
                    continue
                if self.can_insert(i,j):
                        self._neighbours.add((i,j,1))
                if self.can_delete(i,j):
                        self._neighbours.add((i,j,0))

        #This line adds a null action column, currently by default
        self._curr = np.c_[self._curr, np.zeros((self._nactions, 1))]
        
        curr_post = nem._logmarginalposterior(self._curr)
        i = 0
        accepts = 0
        while i < self._n:
            neigh_list = list(self._neighbours)
            change = neigh_list[np.random.choice(range(len(neigh_list)))]
            unif = np.random.uniform()
            proposal = self._curr.copy()
            proposal[change[0], change[1]] = change[2]
            prop_post = nem._logmarginalposterior(proposal)
            if unif <= self.accept(prop_post, curr_post):
                curr_post = prop_post
                self._curr = proposal
                self.update_current(change)
                accepts += 1
                if change[2] == 1:
                    nedges += 1
                else:
                    nedges -= 1
            self._arratio.append(accepts/(i+1))
            self._avg_nedges.append((self._avg_nedges[i]*(i+1) + nedges)/(i+2))
            if i >= self._burn_in:
                self._out += self._curr[:, :self._nactions]
            i += 1
        
        self._out /= (self._n - self._burn_in)
        np.fill_diagonal(self._out, 1)

    def update_current(self, change: tuple):
        self._neighbours.discard(change)
        i, j, t = change
        if t == 1:
            self._children[i].add(j)
            self._parents[j].add(i)
        else:
            self._children[i].remove(j)
            self._parents[j].remove(i)
        self.do_checks(i)
        self.do_checks(j)

    def do_checks(self, node: int):
        for j in range(self._nactions):
            if j == node:
                continue
            if self.can_insert(node,j):
                self._neighbours.add((node,j,1))
            else:
                self._neighbours.discard((node,j,1))
            if self.can_insert(j,node):
                self._neighbours.add((j,node,1))
            else:
                self._neighbours.discard((j,node,1))
            if self.can_delete(node,j):
                self._neighbours.add((node,j,0))
            else:
                self._neighbours.discard((node,j,0))
            if self.can_delete(j,node):
                self._neighbours.add((j,node,0))
            else:
                self._neighbours.discard((j,node,0))

    def accept(self, proposal: float, current: float):
        return min(1, np.exp(proposal - current))

    def can_insert(self, i:int ,j:int):
        return not self._curr[i,j] and not self._curr[j,i] and \
            self._parents[i].issubset(self._parents[j]) and self._children[j].issubset(self._children[i])
    
    def can_delete(self, i: int, j: int):
        return self._curr[i,j] and len(self._children[i].intersection(self._parents[j])) == 0
    
    def convergence_plots(self, up_to = None):
        if up_to is None:
            up_to = self._n
        up_to = min(self._n, up_to)
        x = np.arange(up_to)
        fig, axs = plt.subplots(1,2)

        axs[0].plot(x, self._arratio[:up_to])
        axs[0].axvspan(xmin = 0, xmax = self._burn_in, color='lightgray', alpha=0.5, linewidth=0)
        axs[0].set_xlabel('Iteration number')
        axs[0].set_ylabel('Accept/reject ratio')

        axs[1].plot(x,self._avg_nedges[:up_to])
        axs[1].axvspan(xmin = 0, xmax = self._burn_in, color='lightgray', alpha=0.5, linewidth=0)
        axs[1].set_xlabel('Iteration number')
        axs[1].set_ylabel('Moving average number of edges')

        plt.tight_layout()
        plt.show()
    
    def heatmap(self, data: np.ndarray, row_labels: list = None, col_labels: list = None):
        if row_labels is None:
            row_labels = col_labels
        if col_labels is None:
            col_labels = row_labels

        ax = plt.gca()

        # Plot the heatmap
        sns.heatmap(data, annot=True, cmap='viridis', fmt='.2f', cbar=True, ax=ax)

        # Set row and column labels
        if row_labels is not None and col_labels is not None:
            ax.set_xticklabels(col_labels, rotation=45, ha='right')
            ax.set_yticklabels(row_labels, rotation=0, ha='right')

        plt.show()

class JointNEMCMC:
    def __init__(self, nem_list: List[nem], init_graphs: List[np.ndarray] = None, init_meta: np.ndarray = None, init_nus: List[float] = None,
                 n: int = 1e5, burn_in: int = 1e4, meta_prior: np.ndarray = None, lambda_reg: float = 0, sigma: float = 10, 
                 shape: float = 1, rate: float = 2, compare: List[Tuple[int,int]] = None):
        self._sigma = sigma
        self._shape = shape
        self._rate = rate
        self._lambda_reg = lambda_reg
        self._K = len(nem_list)
        self._nactions = nem_list[0]._nactions
        self._compare = None
        if compare is not None:
            self._compare = compare.copy()
        if meta_prior is None:
            if self._lambda_reg > 0:
                self._meta_prior = np.eye(self.nactions)
            else:
                self._meta_prior = None
        elif meta_prior.shape[0] == self._nactions and meta_prior.shape[1] == meta_prior.shape[0]:
            self._meta_prior = meta_prior.copy()
        else:
            raise ValueError("Dimensions of meta_prior don't match input NEMs!")
        #The next line copies each of the inputs and adds a null action column to each
        if init_graphs is None:
            self._curr_graphs = [np.zeros((self._nactions,self._nactions+1)) for k in range(self._K)]
        else:
            self._curr_graphs = [np.c_[init, np.zeros((init.shape[0],1))] for init in init_graphs]
        if init_meta is None:
            self._curr_meta = np.zeros((self._nactions, self._nactions))
        else:
            self._curr_meta = init_meta.copy()
        if init_nus is None:
            self._curr_nus = np.random.gamma(shape=self._shape, scale=1/self._rate, size=self._K)
        else:
            self._curr_nus = init_nus.copy()
        for c in self._curr_graphs:
            np.fill_diagonal(c, 0)
        np.fill_diagonal(self._curr_meta, 0)
        self._n = int(n)
        self._burn_in = int(burn_in)

        self._neighbours_list = [set() for k in range(self._K)]
        self._parents_list = [defaultdict(set) for k in range(self._K)]
        self._children_list = [defaultdict(set) for k in range(self._K)]

        self._neighbours_meta = set()
        self._parents_meta = defaultdict(set)
        self._children_meta = defaultdict(set)

        self._out_graphs = [np.zeros(self._curr_meta.shape) for k in range(self._K)]
        self._out_meta = np.zeros(self._curr_meta.shape)
        self._out_nus = [[] for k in range(self._K)]
        self._out_comparisons = None
        if self._compare is not None:
            self._out_comparisons = [np.zeros(self._curr_meta.shape) for i in range(len(self._compare))]

        self._hamming_dists = [np.abs(m[:,:self._nactions]-self._curr_meta).sum() for m in self._curr_graphs]

        for k in range(self._K):
            for i in range(self._nactions):
                self._children_list[k][i] = set(self._curr_graphs[k][i].nonzero()[0])
                self._parents_list[k][i] = set(self._curr_graphs[k].T[i].nonzero()[0])
        
        for i in range(self._nactions):
            self._children_meta[i] = set(self._curr_meta[i].nonzero()[0])
            self._parents_meta[i] = set(self._curr_meta.T[i].nonzero()[0])
        
        for k in range(self._K):
            for i in range(self._nactions):
                for j in range(self._nactions):
                    if i == j:
                        continue
                    if self.can_insert(i,j,k):
                        self._neighbours_list[k].add((i,j,1))
                    if self.can_delete(i,j,k):
                        self._neighbours_list[k].add((i,j,0))
        
        for i in range(self._nactions):
            for j in range(self._nactions):
                if i == j:
                    continue
                if self.can_insert_meta(i,j):
                    self._neighbours_meta.add((i,j,1))
                if self.can_delete_meta(i,j):
                    self._neighbours_meta.add((i,j,0))
        
        for c in self._curr_graphs:
            np.fill_diagonal(c, 1)
        np.fill_diagonal(self._curr_meta, 1)
        
        curr_graph_posts = [nem_list[k]._logmarginalposterior(self._curr_graphs[k])-self._curr_nus[k]*self._hamming_dists[k] for k in range(self._K)]
        curr_nu_probs = [self.laplace(self._curr_nus[k], self._hamming_dists[k])+self.nu_gamma(self._curr_nus[k]) for k in range(self._K)]
        curr_meta_prob = 0
        for k in range(self._K):
            curr_meta_prob += self.laplace(self._curr_nus[k], self._hamming_dists[k])
        if self._meta_prior:
            curr_meta_prob += self.compute_meta_prior(self._curr_meta)

        self._graph_arratios = [[] for k in range(self._K)]
        self._nu_arratios = [[] for k in range(self._K)]
        self._meta_arratio = []

        nedges_graphs = [self._curr_graphs[k].sum() for k in range(self._K)]
        nedges_meta = self._curr_meta.sum()
        self._avg_nedges_graphs = [[nedges_graphs[k]] for k in range(self._K)]
        self._avg_nedges_meta = [nedges_meta]

        graph_accepts = [0 for k in range(self._K)]
        nu_accepts = [0 for k in range(self._K)]
        meta_accepts = 0

        i = 0
        while i < self._n:
            #Proposals for graphs
            for k in range(self._K):
                neigh_list = list(self._neighbours_list[k])
                change = neigh_list[np.random.choice(range(len(neigh_list)))]
                unif = np.random.uniform()
                proposal = self._curr_graphs[k].copy()
                proposal[change[0], change[1]] = change[2]
                if proposal[change[0], change[1]] == self._curr_meta[change[0], change[1]]:
                    prop_hamming = self._hamming_dists[k] - 1
                else:
                    prop_hamming = self._hamming_dists[k] + 1
                prop_post = nem_list[k]._logmarginalposterior(proposal) - self._curr_nus[k]*prop_hamming
                if unif <= min(1, np.exp(prop_post - curr_graph_posts[k])):
                    curr_graph_posts[k] = prop_post
                    curr_nu_probs[k] += self._curr_nus[k]*(self._hamming_dists[k] - prop_hamming)
                    curr_meta_prob += self._curr_nus[k]*(self._hamming_dists[k] - prop_hamming)
                    self._curr_graphs[k] = proposal
                    self._hamming_dists[k] = prop_hamming
                    self.update_current(change, k)
                    graph_accepts[k] += 1
                    if change[2] == 1:
                        nedges_graphs[k] += 1
                    else:
                        nedges_graphs[k] -= 1
                self._graph_arratios[k].append(graph_accepts[k]/(i+1))
                self._avg_nedges_graphs[k].append((self._avg_nedges_graphs[k][i]*(i+1) + nedges_graphs[k])/(i+2))
                if i >= self._burn_in:
                    self._out_graphs[k] += self._curr_graphs[k][:,:self._nactions]
            
            if i >= self._burn_in:
                if self._compare is not None:
                        for j in range(len(self._compare)):
                            g1, g2 = self._compare[j]
                            self._out_comparisons[j] += np.abs(self._curr_graphs[g1][:,:self._nactions]\
                                                               -self._curr_graphs[g2][:,:self._nactions])

            #Proposals for nu parameters
            for k in range(self._K):
                unif = np.random.uniform()
                proposal = np.exp(np.random.normal(np.log(self._curr_nus[k]), self._sigma))
                prop_prob = self.laplace(proposal,self._hamming_dists[k]) + self.nu_gamma(proposal)
                if unif <= min(1, np.exp(prop_prob - curr_nu_probs[k])):
                    curr_nu_probs[k] = prop_prob
                    curr_graph_posts[k] += self._hamming_dists[k]*(self._curr_nus[k] - proposal)
                    curr_meta_prob -= self.laplace(self._curr_nus[k], self._hamming_dists[k])
                    curr_meta_prob += self.laplace(proposal, self._hamming_dists[k])
                    self._curr_nus[k] = proposal
                    nu_accepts[k] += 1
                self._nu_arratios[k].append(nu_accepts[k]/(i+1))
                if i>= self._burn_in:
                    self._out_nus[k].append(self._curr_nus[k])
            
            neigh_list = list(self._neighbours_meta)
            change = neigh_list[np.random.choice(range(len(neigh_list)))]
            unif = np.random.uniform()
            proposal = self._curr_meta.copy()
            proposal[change[0], change[1]] = change[2]
            prop_hamming_dists = []
            prop_prob = 0
            for k in range(self._K):
                if proposal[change[0], change[1]] == self._curr_graphs[k][change[0], change[1]]:
                    prop_hamming_dists.append(self._hamming_dists[k]-1)
                else:
                    prop_hamming_dists.append(self._hamming_dists[k]+1)
                prop_prob += self.laplace(self._curr_nus[k], prop_hamming_dists[k])
            if self._meta_prior:
                prop_prob += self.compute_meta_prior(self._curr_meta)
            if unif <= min(1,np.exp(prop_prob-curr_meta_prob)):
                curr_meta_prob = prop_prob
                self._curr_meta = proposal
                for k in range(self._K):
                    curr_graph_posts[k] += self._curr_nus[k]*(self._hamming_dists[k] - prop_hamming_dists[k])
                    curr_nu_probs[k] += self._curr_nus[k]*(self._hamming_dists[k] - prop_hamming_dists[k])
                self._hamming_dists = prop_hamming_dists
                self.update_current_meta(change)
                meta_accepts += 1
                if change[2] == 1:
                    nedges_meta += 1
                else:
                    nedges_meta -= 1
            self._meta_arratio.append(meta_accepts/(i+1))
            self._avg_nedges_meta.append((self._avg_nedges_meta[i]*(i+1) + nedges_meta)/(i+2))
            if i >= self._burn_in:
                self._out_meta += self._curr_meta[:,:self._nactions]

            i += 1
        
        for k in range(self._K):
            self._out_graphs[k] /= (self._n - self._burn_in)
            np.fill_diagonal(self._out_graphs[k], 1)
        self._out_meta /= (self._n - self._burn_in)
        np.fill_diagonal(self._out_meta, 1)
        
        if self._compare is not None:
            for i in range(len(self._compare)):
                self._out_comparisons[i] /= (self._n - self._burn_in)

    def _phi_distr(self, model: np.ndarray, a=1, b=0.1):
        d = np.abs(model - self._meta_prior)
        pPhi = a/(2*b)*(1 + d/b)**(-a-1)
        return np.sum(np.log(pPhi))
    
    def compute_meta_prior(self, model: np.ndarray):
        if self._lambda_reg != 0:
            return -self._lambda_reg*np.sum(np.abs(model - self._meta_prior)) + \
                np.log(self._lambda_reg*0.5)*2**self._nactions
        else:
            return self._phi_distr(model)
    
    def laplace(self, nu, hamming_dist):
        return -1*nu*hamming_dist-self._nactions*(self._nactions-1)*np.log(1+np.exp(-1*nu))
    
    def nu_gamma(self,nu):
        return gamma.logpdf(nu,a = self._shape, scale = 1/self._rate) + nu
    
    def update_current(self, change: tuple, k: int):
        self._neighbours_list[k].discard(change)
        i, j, t = change
        if t == 1:
            self._children_list[k][i].add(j)
            self._parents_list[k][j].add(i)
        else:
            self._children_list[k][i].remove(j)
            self._parents_list[k][j].remove(i)
        self.do_checks(i, k)
        self.do_checks(j, k)

    def do_checks(self, node: int, k: int):
        for j in range(self._nactions):
            if j == node:
                continue
            if self.can_insert(node,j,k):
                self._neighbours_list[k].add((node,j,1))
            else:
                self._neighbours_list[k].discard((node,j,1))
            if self.can_insert(j,node,k):
                self._neighbours_list[k].add((j,node,1))
            else:
                self._neighbours_list[k].discard((j,node,1))
            if self.can_delete(node,j,k):
                self._neighbours_list[k].add((node,j,0))
            else:
                self._neighbours_list[k].discard((node,j,0))
            if self.can_delete(j,node,k):
                self._neighbours_list[k].add((j,node,0))
            else:
                self._neighbours_list[k].discard((j,node,0))
    
    def update_current_meta(self, change: tuple):
        self._neighbours_meta.discard(change)
        i, j, t = change
        if t == 1:
            self._children_meta[i].add(j)
            self._parents_meta[j].add(i)
        else:
            self._children_meta[i].remove(j)
            self._parents_meta[j].remove(i)
        self.do_checks_meta(i)
        self.do_checks_meta(j)

    def do_checks_meta(self, node: int):
        for j in range(self._nactions):
            if j == node:
                continue
            if self.can_insert_meta(node,j):
                self._neighbours_meta.add((node,j,1))
            else:
                self._neighbours_meta.discard((node,j,1))
            if self.can_insert_meta(j,node):
                self._neighbours_meta.add((j,node,1))
            else:
                self._neighbours_meta.discard((j,node,1))
            if self.can_delete_meta(node,j):
                self._neighbours_meta.add((node,j,0))
            else:
                self._neighbours_meta.discard((node,j,0))
            if self.can_delete_meta(j,node):
                self._neighbours_meta.add((j,node,0))
            else:
                self._neighbours_meta.discard((j,node,0)) 

    def can_insert(self, i: int, j: int, k: int):
        return not self._curr_graphs[k][i,j] and not self._curr_graphs[k][j,i] and \
                self._parents_list[k][i].issubset(self._parents_list[k][j]) and \
                    self._children_list[k][j].issubset(self._children_list[k][i])
    
    def can_insert_meta(self, i: int, j: int):
        return not self._curr_meta[i,j] and not self._curr_meta[j,i] and \
            self._parents_meta[i].issubset(self._parents_meta[j]) and self._children_meta[j].issubset(self._children_meta[i])
    
    def can_delete(self, i: int, j: int, k: int):
        return self._curr_graphs[k][i,j] and len(self._children_list[k][i].intersection(self._parents_list[k][j])) == 0
    
    def can_delete_meta(self, i: int, j: int):
        return self._curr_meta[i,j] and len(self._children_meta[i].intersection(self._parents_meta[j])) == 0
    
    def nu_trace_plots(self, start: int = 0, end: int = None):
        if end is None:
            end = len(self._out_nus[0])
        end = min(len(self._out_nus[0]),end)
        x = np.arange(start,end)
        fig, axs = plt.subplots(1,self._K)

        for k in range(self._K):
            axs[k].plot(x,self._out_nus[k][start:end])
            axs[k].set_xlabel('Iteration number')
            axs[k].set_ylabel(f'nu {k}')
        
        plt.tight_layout()
        plt.show()
    
    def nu_density_plots(self, start: int = 0, end: int = None):
        if end is None:
            end = len(self._out_nus[0])
        end = min(len(self._out_nus[0]),end)

        for k in range(self._K):
            sns.kdeplot(self._out_nus[k][start:end],fill=True,label=f'nu {k}')
        
        plt.xlabel('nu')
        plt.legend()
        plt.show()

    def graph_ar_plots(self, start: int = 0, end: int = None):
        if end is None:
            end = self._n
        end = min(self._n,end)
        x = np.arange(start,end)
        fig, axs = plt.subplots(1,self._K+1)

        for k in range(self._K):
            axs[k].plot(x,self._graph_arratios[k][start:end])
            if end > self._burn_in:
                axs[k].axvspan(xmin=start,xmax=self._burn_in,color='lightgray', alpha = 0.5, linewidth = 0)
                axs[k].set_xlabel('Iteration number')
                axs[k].set_ylabel(f'Accept/reject ratio for graph {k}')
        
        axs[self._K].plot(x,self._meta_arratio[start:end])
        if end > self._burn_in:
            axs[self._K].axvspan(xmin=start,xmax=self._burn_in,color='lightgray', alpha = 0.5, linewidth = 0)
            axs[self._K].set_xlabel('Iteration number')
            axs[self._K].set_ylabel(f'Accept/reject ratio for meta graph')

        plt.tight_layout()
        plt.show()

    def nu_ar_plots(self, start: int = 0, end: int = None):
        if end is None:
            end = self._n
        end = min(self._n,end)
        x = np.arange(start,end)
        fig, axs = plt.subplots(1,self._K)

        for k in range(self._K):
            axs[k].plot(x,self._nu_arratios[k][start:end])
            if end > self._burn_in:
                axs[k].axvspan(xmin=start,xmax=self._burn_in,color='lightgray', alpha = 0.5, linewidth = 0)
                axs[k].set_xlabel('Iteration number')
                axs[k].set_ylabel(f'Accept/reject ratio for nu {k}')

        plt.tight_layout()
        plt.show()
    
    def average_edge_number_plots(self, start: int = 0, end: int = None):
        if end is None:
            end = self._n
        end = min(self._n,end)
        x = np.arange(start,end)
        fig, axs = plt.subplots(1,self._K+1)
        for k in range(self._K):
            axs[k].plot(x,self._avg_nedges_graphs[k][start:end])
            if end > self._burn_in:
                axs[k].axvspan(xmin=start,xmax=self._burn_in,color='lightgray', alpha = 0.5, linewidth = 0)
                axs[k].set_xlabel('Iteration number')
                axs[k].set_ylabel(f'Moving average number of edges for graph {k}')
        
        axs[self._K].plot(x,self._avg_nedges_meta[start:end])
        if end > self._burn_in:
            axs[self._K].axvspan(xmin=start,xmax=self._burn_in,color='lightgray', alpha = 0.5, linewidth = 0)
            axs[self._K].set_xlabel('Iteration number')
            axs[self._K].set_ylabel(f'Moving average number of edges for meta graph')

        plt.tight_layout()
        plt.show()
    
    def heatmap(self, data: np.ndarray, row_labels: list = None, col_labels: list = None):
        if row_labels is None:
            row_labels = col_labels
        if col_labels is None:
            col_labels = row_labels

        ax = plt.gca()

        # Plot the heatmap
        sns.heatmap(data, annot=True, cmap='viridis', fmt='.2f', cbar=True, ax=ax)

        # Set row and column labels
        if row_labels is not None and col_labels is not None:
            ax.set_xticklabels(col_labels, rotation=45, ha='right')
            ax.set_yticklabels(row_labels, rotation=0, ha='right')

        plt.show()

class NestedEffectsModel(ExtendedGraph):
    """
    Class uniting the data, graph and learning algorithms to facilitate scoring and learning of Nested Effects Models.
    """
    def __init__(self, data: np.ndarray = np.array([]), col_data: list = list(), row_data: list = list(),
                actions: List = list(), effects: List = list(), structure_prior: np.ndarray = None,
                attachments_prior: np.ndarray = None, alpha: float = 0.13, beta: float = 0.05,
                lambda_reg: float = 0, delta: float = 1, actions_graph: Union[Iterable[Edge], np.ndarray] = None,
                effect_attachments: Union[Iterable[Edge], np.ndarray] = None, 
                effects_selection: str = 'regularisation', nem = None):
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
            self._effects_selection = nem._effects_selection
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
            if effects_selection not in {'regularisation', 'iterative'}:
                raise ValueError("effects_selection must be either 'regularisation' or 'iterative'")
            self._alpha = alpha
            self._beta = beta
            self._lambda_reg = lambda_reg
            self._delta = delta
            self._effects_selection = effects_selection

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
                        actions = self._col_data[np.sort(idx)]
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
        raise NotImplementedError
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
        if self._attachments_prior.shape[1] == self.nactions + 1:
            cols = np.append(self.actions(), 'null')
        else:
            cols = self.actions()
        return (self._attachments_prior.copy(), self.effects(), cols)

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
    def effects_selection(self):
        return self._effects_selection

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

    def _assign_priors(self):
        if self._attachments_prior is None:
            self._attachments_prior = np.full((self.neffects, self.nactions), 1/self.nactions)
        if self._structure_prior is None:
            if self._lambda_reg > 0:
                self._structure_prior = np.eye(self.nactions)

    def _add_null_action(self):
        if self._attachments_prior is None:
            raise ValueError("Provide an attachments_prior first or else run nem._assign_priors")
        if self._attachments_prior.shape[1] == self.nactions + 1:
            return
        self._attachments_prior = np.c_[self._attachments_prior, np.full(self.neffects, 1/self.nactions)]
        self._attachments_prior[:,self.nactions] = self.delta/self.nactions
        self._attachments_prior = self._attachments_prior/self._attachments_prior.sum(axis=1)[:, None]
        self._actions_amat = np.c_[self._actions_amat, np.zeros(self.nactions)]
            
    def _initialise_learn(self):
        #Generate priors if necessary
        self._assign_priors()
        #Add null action if regularisation is being applied
        if self._effects_selection == "regularisation":
            self._add_null_action()
        #Split data into counts for 1s, 0s and NaNs to facilitate scoring
        self._gen_d0_d1()
        #Score initial model (empty graph by default)
        self._score_current_mLL()
        #Store initial action equivalence class representatives
        self._areps = self.action_reps_idx()
    
    def _phi_distr(self, actions_amat: np.ndarray, a=1, b=0.1):
        d = np.abs(actions_amat - self._structure_prior)
        pPhi = a/(2*b)*(1 + d/b)**(-a-1)
        return np.sum(np.log(pPhi))
    
    def _incorporate_structure_prior(self, actions_amat: np.ndarray):
        if self._lambda_reg != 0:
            return -self._lambda_reg*np.sum(np.abs(actions_amat - self._structure_prior)) + \
                np.log(self._lambda_reg*0.5)*2**self.nactions
        else:
            return self._phi_distr(actions_amat)

    def _gpo_forward_search(self):
        #Get actions_amat over action reps
        actions_reps_amat = self._actions_amat[self._areps][:,self._areps]
        #Store possible edge additions for forward search
        nonzeros = np.array((1 - actions_reps_amat).nonzero()).T
        possible_add = np.zeros((nonzeros.shape[0], 5)).astype('int')
        possible_add[:,0] = np.arange(possible_add.shape[0])
        possible_add[:,1:3] = nonzeros
        current_score = self.score
        current_proposal = None
        last_change = None
        while True:
            self._check_forward_proposals(possible_add, last_change)
            add_to_score = possible_add[possible_add[:,3].astype('bool'),0:3]
            join_to_score = possible_add[possible_add[:,4].astype('bool'),0:3]
            if add_to_score.shape[0] + join_to_score.shape[0] == 0:
                print("No possible edges to add or actions to join!")
                break
            for i in range(add_to_score.shape[0]):
                d = self._add_edge(*add_to_score[i,1:3], inplace=False)
                proposal = self._score_proposal_mLL(**d)
                if proposal['score'] > current_score:
                    current_score = proposal['score']
                    current_proposal = proposal
                    update_idx = add_to_score[i,0]
                    change_type = "add"
            for i in range(join_to_score.shape[0]):
                d = self._join_actions(*join_to_score[i,1:3], inplace=False)
                proposal = self._score_proposal_mLL(actions_amat=d['actions_amat'], targets = d['j_targets'])
                if proposal['score'] > current_score:
                    current_score = proposal['score']
                    current_proposal = proposal
                    update_idx = join_to_score[i,0]
                    change_type = "join"
            if current_proposal is None:
                print("At local maximum!")
                break
            else:
                last_change = possible_add[update_idx,1:3]
                if change_type == "add":
                    possible_add = np.delete(possible_add, update_idx, 0)
                else:
                    action_rm = possible_add[update_idx][0]
                    rm_mask = ~np.any(possible_add == action_rm, axis = 1)
                    possible_add = possible_add[rm_mask]                    
                self._update_actions_graph(**current_proposal)
                current_proposal = None

    def _learn_gpo(self):
        if self._data.size == 0:
            raise ValueError("No data provided")
        self._initialise_learn()
        self._gpo_forward_search()

    def _check_forward_proposals(self, possible_add: np.ndarray, last_change: np.ndarray = None):
        if last_change is not None:
            idx = np.nonzero(np.isin(possible_add[:,1:3], last_change))[0]
        else:
            idx = range(possible_add.shape[0])
        for i in idx:
            possible_add[i,3] = self._can_add_edge(*possible_add[i,1:3])
        for i in idx:
            possible_add[i,4] = self._can_join_actions(*possible_add[i,1:3])
    
    def _check_backward_proposals(self):
        raise NotImplementedError
        action_reps_amat = self._actions_amat[self._areps][:,self._areps]
        np.fill_diagonal(action_reps_amat, 0)
        possible_remove = np.array(action_reps_amat.nonzero()).T
        can_remove = np.zeros(possible_remove.shape[0])
        for i in range(possible_remove.shape[0]):
            can_remove[i] = self._can_remove_edge(*possible_remove[i])

    def _score_current_mLL(self):
        LL = np.log(self.alpha)*np.matmul(self._D1, 1 - self._actions_amat) + \
            np.log(1 - self.alpha)*np.matmul(self._D0, 1 - self._actions_amat) + \
            np.log(1 - self.beta)*np.matmul(self._D1, self._actions_amat) + \
            np.log(self.beta)*np.matmul(self._D0, self._actions_amat)
        self._LLP = LL+np.log(self._attachments_prior)
        self._LLP_sums = logsumexp(self._LLP,axis=1)
        self._score = np.sum(self._LLP_sums)
        if self._structure_prior is not None:
            self._score += self._incorporate_structure_prior(self._actions_amat[:self.nactions, :self.nactions])

    def _score_proposal_mLL(self, actions_amat: np.ndarray, targets: List[int]) -> Dict:
        aamat_col = actions_amat[:,targets]
        LL = np.log(self.alpha)*np.matmul(self._D1, 1 - aamat_col) + \
            np.log(1 - self.alpha)*np.matmul(self._D0, 1 - aamat_col) + \
            np.log(1 - self.beta)*np.matmul(self._D1, aamat_col) + \
            np.log(self.beta)*np.matmul(self._D0, aamat_col)
        b = np.ones((self._LLP_sums.shape[0], 3))
        b[:,2] = -b[:,2]
        new_LLP = LL + np.log(self._attachments_prior[:, targets])
        LLP_sums = logsumexp(np.c_[self._LLP_sums[:,None], new_LLP, self._LLP[:, targets]], 
                             b = b, axis = 1)
        score = np.sum(LLP_sums)
        if self._structure_prior is not None:
            score += self._incorporate_structure_prior(actions_amat[:self.nactions, :self.nactions])
        return {'actions_amat': actions_amat, 'score': score, 'new_LLP': new_LLP, 'LLP_sums': LLP_sums, 'targets': targets}

    def _update_actions_graph(self, actions_amat: np.ndarray, targets: Union[int, List[int]], 
                              score: float, LLP_sums: np.ndarray, new_LLP: np.ndarray):
        self._actions_amat = actions_amat
        self._score = score
        self._LLP_sums = LLP_sums
        self._LLP[:, targets] = new_LLP

    def _gen_d0_d1(self):
        if self._nactions == self._col_data.size:
            self._D1 = (self._data == 1).astype('int64')
            self._D0 = (self._data == 0).astype('int64')
            DNaN = np.isnan(self._data).astype('int64')
        else:
            self._D1 = np.array([np.sum(self._data.T[a == self._col_data] == 1, axis = 0) for a in self.actions()]).T
            self._D0 = np.array([np.sum(self._data.T[a == self._col_data] == 0, axis = 0) for a in self.actions()]).T
            DNaN = np.array([np.sum(np.isnan(self._data.T[a == self._col_data]), axis = 0) for a in self.actions()]).T
        #following two lines means that if there is a NaN, it has likelihood = 1, hence doesn't affect score
        self._D1 += DNaN
        self._D0 += DNaN
        
    def transitive_closure(self):
        raise NotImplementedError

    def transitive_reduction(self):
        raise NotImplementedError