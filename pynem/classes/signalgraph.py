from collections import defaultdict
from typing import Hashable, Set, Union, Tuple, Any, Iterable, Dict, FrozenSet, List

from pynem.utils import core_utils
from pynem.custom_types import *

import numpy as np
import scipy.sparse as sps

class SignalGraph:
    """
    Base class for graphs of perturbed proteins.
    """

    def __init__(self, nodes: Set[Node] = set(), edges: Set[Edge] = set(), signal_graph = None):
        if signal_graph is not None:
            self._nodes = set(signal_graph._nodes)
            self._edges = set(signal_graph._edges)
            self._parents = defaultdict(set)
            for node, par in signal_graph._parents.items():
                self._parents[node] = set(par)
            self._children = defaultdict(set)
            for node, ch in signal_graph._children.items():
                self._children[node] = set(ch)
        else:
            self._nodes = set(nodes)
            self._edges = set()
            self._parents = defaultdict(set, {k: set() for k in self._nodes})
            self._children = defaultdict(set, {k: set() for k in self._nodes})
            self.add_edges_from(edges)

    def __eq__(self, other):
        if not isinstance(other, SignalGraph):
            return False
        return self._nodes == other._nodes and self._edges == other._edges

    def __str__(self):
        return "Signal graph of {nn} nodes and {ne} edges".format(nn = len(self._nodes), ne = len(self._edges))

    def __repr__(self):
        return str(self)

    def copy(self):
        """
        Return a copy of the current SignalGraph.
        """
        return SignalGraph(signal_graph=self)

    def rename_nodes(self, name_map: Dict):
        """
        Rename the nodes in this graph according to ``name_map``.
        Parameters
        ----------
        name_map:
            A dictionary from the current name of each node to the desired name of each node.
        Examples
        --------
        >>> from pynem import SignalGraph
        >>> sg = SignalGraph(edges={('a', 'b'), ('b', 'c')})
        >>> sg2 = sg.rename_nodes({'a': 1, 'b': 2, 'c': 3})
        >>> sg2.edges
        {(2, 3), (1, 2)}
        """
        return SignalGraph(
            nodes={name_map[n] for n in self._nodes},
            edges={(name_map[i], name_map[j]) for i, j in self._edges}
        )

    # === PROPERTIES
    @property
    def nodes(self) -> Set[Node]:
        return set(self._nodes)

    @property
    def nnodes(self) -> int:
        return len(self._nodes)

    @property
    def edges(self) -> Set[Edge]:
        return set(self._edges)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    @property
    def parents(self) -> Dict[Node, Set[Node]]:
        return core_utils.defdict2dict(self._parents, self._nodes)

    @property
    def children(self) -> Dict[Node, Set[Node]]:
        return core_utils.defdict2dict(self._children, self._nodes)
    
    @property
    def amat_tuple(self) -> Tuple[np.ndarray, list]:
        return self._amat_tuple

    # === NODE PROPERTIES
    def parents_of(self, nodes: NodeSet) -> Set[Node]:
        """
        Return all nodes that are parents of the node or set of nodes ``nodes``.
        Parameters
        ----------
        nodes
            A node or set of nodes.
        See Also
        --------
        children_of
        Examples
        --------
        >>> from pynem import SignalGraph
        >>> g = SignalGraph(edges={(1, 2), (2, 3)})
        >>> g.parents_of(2)
        {1}
        >>> g.parents_of({2, 3})
        {1, 2}
        """
        if isinstance(nodes, set):
            return set.union(*(self._parents[n] for n in nodes))
        else:
            return self._parents[nodes].copy()

    def children_of(self, nodes: NodeSet) -> Set[Node]:
        """
        Return all nodes that are children of the node or set of nodes ``nodes``.
        Parameters
        ----------
        nodes
            A node or set of nodes.
        See Also
        --------
        parents_of
        Examples
        --------
        >>> from pynem import SignalGraph
        >>> g = SignalGraph(edges={(1, 2), (2, 3)})
        >>> g.children_of(1)
        {2}
        >>> g.children_of({1, 2})
        {2, 3}
        """
        if isinstance(nodes, set):
            return set.union(*(self._children[n] for n in nodes))
        else:
            return self._children[nodes].copy()
    
    # === GRAPH MODIFICATION
    def add_node(self, node: Node):
        """
        Add ``node`` to the SignalGraph.
        Parameters
        ----------
        node:
            a hashable Python object
        See Also
        --------
        add_nodes_from
        Examples
        --------
        >>> from pynem import SignalGraph
        >>> g = SignalGraph()
        >>> g.add_node(1)
        >>> g.add_node(2)
        >>> len(g.nodes)
        2
        """
        self._nodes.add(node)

    def add_nodes_from(self, nodes: Iterable):
        """
        Add nodes to the graph from the collection ``nodes``.
        Parameters
        ----------
        nodes:
            collection of nodes to be added.
        See Also
        --------
        add_node
        Examples
        --------
        >>> from pynem import SignalGraph
        >>> g = SignalGraph({1, 2})
        >>> g.add_nodes_from({'a', 'b'})
        >>> g.add_nodes_from(range(3, 6))
        >>> g.nodes
        {1, 2, 3, 4, 5, 'a', 'b'}
        """
        for node in nodes:
            self.add_node(node)

    def remove_node(self, node: Node, ignore_error=False):
        """
        Remove the node ``node`` from the graph.
        Parameters
        ----------
        node:
            node to be removed.
        ignore_error:
            if True, ignore the KeyError raised when node is not in the SignalGraph.
        Examples
        --------
        >>> from pynem import SignalGraph
        >>> g = SignalGraph(edges={(1, 2)})
        >>> g.remove_node(2)
        >>> g.nodes
        {1}
        """
        try:
            self._nodes.remove(node)
            for parent in self._parents[node]:
                self._children[parent].remove(node)
            for child in self._children[node]:
                self._parents[child].remove(node)
            self._parents.pop(node, None)
            self._children.pop(node, None)
            self._edges = {(i, j) for i, j in self._edges if i != node and j != node}

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
            source node of the edge
        j:
            target node of the edge
        
        See Also
        --------
        add_edges_from
        Examples
        --------
        >>> from pynem import SignalGraph
        >>> g = SignalGraph({1, 2})
        >>> g.add_edge(1, 2)
        >>> g.edges
        {(1, 2)}
        """
        self._nodes.add(i)
        self._nodes.add(j)
        self._edges.add((i, j))

        self._children[i].add(j)
        self._parents[j].add(i)

    def add_edges_from(self, edges: Union[Set[Edge], Iterable[Edge]]):
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
        >>> from pynem import SignalGraph
        >>> g = SignalGraph(edges={(1, 2)})
        >>> g.add_edges_from({(1, 3), (2, 3)})
        >>> g.edges
        {(2, 3), (1, 2), (1, 3)}
        """
        if not isinstance(edges, set):
            edges = {(i, j) for i, j in edges}
        if len(edges) == 0:
            return

        sources, sinks = zip(*edges)
        self._nodes.update(sources)
        self._nodes.update(sinks)
        self._edges.update(edges)
        for i, j in edges:
            self._children[i].add(j)
            self._parents[j].add(i)
    
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
        >>> from pynem import SignalGraph
        >>> g = SignalGraph(edges={(1, 2)})
        >>> g.remove_edge(1, 2)
        >>> g.edges
        set()
        """
        try:
            self._edges.remove((i, j))
            self._parents[j].remove(i)
            self._children[i].remove(j)
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

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
        >>> from pynem import SignalGraph
        >>> g = SignalGraph(edges={(1, 2), (2, 3), (3, 4)})
        >>> g.remove_edges_from({(1, 2), (2, 3)})
        >>> g.edges
        {(3, 4)}
        """
        for i, j in edges:
            self.remove_edge(i, j, ignore_error=ignore_error)

    def join_nodes(self, nodes_to_join: Set[Hashable]):
        """
        Join the nodes in the set ``nodes_to_join`` into a single multi-node.
        Parameters
        ----------
        nodes_to_join:
            set of nodes to be joined

        See Also
        --------
        split_node
        Examples
        --------
        >>> from pynem import SignalGraph
        >>> g = SignalGraph(edges={(1, 2), (2, 3)})
        >>> g.join_nodes({1, 2})
        >>> g.nodes
        {3, frozenset({1, 2})}
        """
        join_generator = (n if isinstance(n, frozenset) else frozenset({n}) for n in nodes_to_join)
        joined_node = frozenset.union(*join_generator)

        children = self.children_of(nodes_to_join)
        parents = self.parents_of(nodes_to_join)

        from_parents = {(n, joined_node) for n in parents}
        to_children = {(joined_node, n) for n in children}

        self.add_edges_from(from_parents.union(to_children))
        self._children[joined_node].update(children)
        self._parents[joined_node].update(parents)

        for n in nodes_to_join:
            self.remove_node(n)
    
    def split_node(self, node: Node, multinode: FrozenSet[Node], direction: str = 'up'):
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
        join_nodes
        Examples
        --------
        >>> from pynem import SignalGraph
        >>> g = SignalGraph(nodes = {frozenset({1,2}), 3}, edges = {(frozenset({1,2}), 3)})
        >>> g.split_node(node = 1, multinode = frozenset({1,2}), direction = 'up')
        >>> g.nodes, g.edges
        ({1, 2, 3}, {(2, 3), (1, 2), (1, 3)})
        """
        if (multinode not in self._nodes) or (node not in multinode):
            raise KeyError("Either {mn} not in graph or {n} not in {mn}".format(mn = multinode, n = node))

        assert direction in ['up', 'down'], "direction must be either 'up' or 'down'"
        
        new_multinode = multinode.difference({node})
        if len(new_multinode) == 1:
            #This is just to unpack the multinode from the set when it is only length 1
            for new_multinode in new_multinode:
                break
        
        parents = self.parents_of(multinode)
        children = self.children_of(multinode)

        from_parents_to_node = {(n, node) for n in parents}
        from_parents_to_new = {(n, new_multinode) for n in parents}
        to_children_from_node = {(node, n) for n in children}
        to_children_from_new = {(new_multinode, n) for n in children}
        
        self.add_edges_from({*from_parents_to_node, *from_parents_to_new, \
                            *to_children_from_node, *to_children_from_new})

        if direction == 'down':
            self.add_edge(new_multinode, node)
        else:
            self.add_edge(node, new_multinode)
        
        self.remove_node(multinode)

#Some extra methods

    @classmethod
    def from_adjacency(cls, adjacency_matrix: Union[np.ndarray, sps.spmatrix], node_list: List[Node] = list(), save: bool = False):
        """
        Return a SignalGraph with arcs given by ``adjacency_matrix``, i.e. i->j if ``adjacency_matrix[i,j] != 0``.
        Parameters
        ----------
        adjacency_matrix:
            Numpy array or sparse matrix representing edges in the SignalGraph.
        node_list:
            Iterable indexing the column and rows of the adjacency matrix to be used as names of nodes in the SignalGraph.
        save:
            Boolean indicating whether the adjacency matrix and associated node_list should be saved as the ``SignalGraph.amat_tuple`` attribute.
        Examples
        --------
        >>> from pynem import SignalGraph
        >>> import numpy as np
        >>> adjacency_matrix = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
        >>> g = SignalGraph.from_adjacency(adjacency_matrix, node_list = ['a', 'b', 'c'])
        >>> g.edges
        {('a', 'c'), ('b', 'c')}
        """
        adj_mat_copy = adjacency_matrix.copy()
        
        #This removes self loops which are implicit in SignalGraphs
        if isinstance(adj_mat_copy, np.ndarray):
            np.fill_diagonal(adj_mat_copy, 0)
        else:
            adj_mat_copy.setdiag(0)

        node_range = range(adj_mat_copy.shape[0])
        nodes = set(node_range)
        edges = {*zip(*adj_mat_copy.nonzero())}

        out = SignalGraph(nodes=nodes, edges=edges)

        if not node_list:
            if save:
                out._amat_tuple = (adj_mat_copy, node_list)
            return out

        node_rename_map = dict(zip(list(node_range), node_list))
        out = out.rename_nodes(node_rename_map)

        if save:
            out._amat_tuple = (adj_mat_copy, node_list)

        return out
        
    def to_adjacency(self, node_list: List[Node] = list(), save: bool = False) -> Tuple[np.ndarray, list]:
        """
        Return the adjacency matrix for the SignalGraph.
        Parameters
        ----------
        node_list:
            List indexing the rows/columns of the matrix.
        save:
            Boolean indicating whether the adjacency matrix and associated node_list should be saved as the ``SignalGraph.amat_tuple`` attribute
        See Also
        --------
        from_adjacency
        Return
        ------
        (adjacency_matrix, node_list)
        Example
        -------
        >>> from pynem import SignalGraph
        >>> g = SignalGraph(edges={(1, 2), (1, 3), (2, 3)})
        >>> adjacency_matrix, node_list = g.to_adjacency()
        >>> adjacency_matrix
        array([[1, 1, 1],
               [0, 1, 1],
               [0, 0, 1]])
        >>> node_list
        [1, 2, 3]
        """
        if not node_list:
            node_list = list(self._nodes)
            edges = self._edges
        else:
            edges = {(source, target) for source, target in self._edges if source in node_list and target in node_list}

        node2ix = {node: i for i, node in enumerate(node_list)}
        shape = (len(node_list), len(node_list))
        adjacency_matrix = np.zeros(shape, dtype=int)

        for source, target in edges:
            adjacency_matrix[node2ix[source], node2ix[target]] = 1
        
        #SignalGraphs are reflexive by definition
        np.fill_diagonal(adjacency_matrix, 1)

        if save:
            self._amat_tuple = (adjacency_matrix, node_list)

        return adjacency_matrix, node_list