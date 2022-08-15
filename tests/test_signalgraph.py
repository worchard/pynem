from pynem.classes.signalgraph import SignalGraph

def test_remove_edge():
    g = SignalGraph(edges={(1,2), (2,3)})
    g.remove_edge(1,2)
    assert (1,2) not in g.edges