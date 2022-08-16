from pynem import SignalGraph
import pytest

def test_empty_signal_graph():
    g = SignalGraph()
    assert g.edges == set()
    assert g.nodes == set()
    assert g.parents == {} and g.children == {}

def test_node_only_init():
    g = SignalGraph(nodes = {1,2,3})
    assert g.edges == set()
    assert g.nodes == {1,2,3}
    assert g.parents == {1: set(), 2: set(), 3: set()}
    assert g.children == {1: set(), 2: set(), 3: set()}

def test_edge_only_init():
    g = SignalGraph(edges={(1,2), (2,3), (2,4)})
    assert g.edges == {(1,2), (2,3), (2,4)}
    assert g.nodes == {1,2,3,4}
    assert g.parents == {1: set(), 2: {1}, 3: {2}, 4: {2}}
    assert g.children == {1: {2}, 2: {3, 4}, 3: set(), 4: set()}

def test_signalgraph_copy():
    g1 = SignalGraph(edges={(1,2), (2,3), (2,4)})
    g2 = g1.copy()
    assert g1 == g2
    g2.add_edge(1,4)
    assert g1 != g2

def test_parents_of():
    g = SignalGraph(edges={(1,2), (2,3), (2,4)})
    assert g.parents_of(g.nodes) == {1,2}

def test_children_of():
    g = SignalGraph(edges={(1,2), (2,3), (2,4)})
    assert g.children_of(g.nodes) == {2,3,4}

def test_add_node():
    g = SignalGraph(edges={(1,2), (2,3), (2,4)})
    g.add_node(5)
    assert g.nodes == {1,2,3,4,5}

def test_add_nodes_from():
    g = SignalGraph(edges={(1,2), (2,3), (2,4)})
    g.add_nodes_from([5,6,'a'])
    assert g.nodes == {1,2,3,4,5,6,'a'}

def test_remove_node():
    g = SignalGraph(edges={(1,2), (2,3), (2,4)})
    g.remove_node(2)
    assert g.nodes == {1,3,4}

def test_add_edge():
    g = SignalGraph(edges={(1,2), (2,3), (2,4)})
    g.add_edge(1,4)
    assert (1,4) in g.edges

def test_add_edges_from():
    g = SignalGraph(nodes={1,2})
    g.add_edges_from({(1,2), (2,3), (2,4)})
    assert g.nodes == {1,2,3,4} and g.edges == {(1,2), (2,3), (2,4)}

def test_remove_edge():
    g = SignalGraph(edges={(1,2), (2,3)})
    g.remove_edge(1,2)
    assert (1,2) not in g.edges

def test_remove_edges_from():
    g = SignalGraph(edges={(1,2), (2,3), (2,4)})
    g.remove_edges_from({(1,2), (2,3)})
    assert g.edges == {(2,4)}

def test_join_nodes():
    g = SignalGraph(edges={(1,2), (2,3), (2,4)})
    g.join_nodes({1,2})
    assert g.nodes == {frozenset({1,2}), 3, 4}
    assert g.edges == {(frozenset({1,2}), 3), (frozenset({1,2}), 4)}

def test_split_node_1():
    g = SignalGraph(edges={(frozenset({1,2}), 3), (frozenset({1,2}), 4)})
    g.split_node(1, frozenset({1,2}), 'up')
    assert g.nodes == {1,2,3,4}
    assert g.edges == {(1,2), (1,3), (1,4), (2,3), (2,4)}

def test_split_node_2():
    g = SignalGraph(edges={(frozenset({1,2}), 3), (frozenset({1,2}), 4)})
    with pytest.raises(KeyError) as e_info:
        g.split_node(1, frozenset({3}), 'up')