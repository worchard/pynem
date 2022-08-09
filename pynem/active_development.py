from collections import defaultdict
from typing import Set, Union, Tuple, Any, Iterable, Dict, FrozenSet, List

import numpy as np

class SignalGraph:
    """
    Base class for graphs on perturbed proteins.
    """

    def __init__(self, nodes: Set = frozenset(), edges: Set = frozenset(), signal_graph = None):
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
            self._parents = defaultdict(set)
            self._children = defaultdict(set)
            self.add_edges_from(edges) ### COME BACK AND CHANGE!

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
        {(1, 2), (2, 3)}
        """
        return SignalGraph(
            nodes={name_map[n] for n in self._nodes},
            edges={(name_map[i], name_map[j]) for i, j in self._edges}
        )

    def add_edges_from(self, edges: Iterable[Tuple]):
        """
        Add edges to the graph from the collection ``edges``.
        Parameters
        ----------
        Edges:
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
        {(1, 2), (1, 3), (2, 3)}
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