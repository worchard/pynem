from pynem import ExtendedGraph
import pytest
import numpy as np

def test_empty_extended_graph():
    eg = ExtendedGraph()
    assert eg.nactions == 0
    assert eg.neffects == 0
    assert eg.nnodes == 0
    assert eg.action_reps_idx().size == 0
    assert eg.action_reps().size == 0
    assert eg._attachments_amat.size == 0
    assert eg._actions_amat.size == 0
    assert eg._full_amat().size == 0
    assert eg._join_array.size == 0

def test_action_only_init():
    eg = ExtendedGraph(actions=[1,2,3])
    assert np.array_equal(np.array([1,2,3]), eg.actions())
    assert np.array_equal(np.eye(3), eg._actions_amat)
    assert np.array_equal(np.eye(3), eg._join_array)

def test_effect_only_init():
    eg = ExtendedGraph(effects=[1,2,3])
    assert np.array_equal(np.array([1,2,3]), eg.effects())
    assert eg._actions_amat.size == 0
    assert eg._join_array.size == 0
    assert eg._attachments_amat.size == 0
    assert eg._attachments_amat.shape == (0,3)

def test_edges_only_init():
    eg = ExtendedGraph(edges = {(0,1), (1,2), (0,2)})
    assert np.array_equal(np.triu((1,1,1)), eg._actions_amat)
    assert np.array_equal(np.eye(3), eg._join_array)

def test_attachments_only_init():
    eg = ExtendedGraph(attachments = {('S1', 'E1'), ('S1', 'E2'), ('S3', 'E3')})
    assert np.array_equal(np.eye(2), eg._actions_amat)
    assert np.array_equal(np.eye(2), eg._join_array)
    assert set(eg._attachments_amat.sum(axis=1)) == {1,2}

def test_actions_amat_init():
    eg1 = ExtendedGraph(actions_amat=np.triu((1,1,1)))
    eg2 = ExtendedGraph(edges = {(0,1), (1,2), (0,2)})
    assert eg1 == eg2

def test_attachments_amat_init():
    eg1 = ExtendedGraph(attachments_amat=np.array([[1,1,0], [0,0,1]]))
    eg2 = ExtendedGraph(attachments={(0,0), (0,1), (1,2)})
    assert eg1 == eg2

def test_actions_and_edges_init():
    eg = ExtendedGraph(actions=[1,2,3], edges = {(0,1), (1,2), (0,2)})
    assert np.array_equal(eg.actions(), np.array([1, 2, 3, 0]))
    assert np.array_equal(eg._actions_amat, np.array([[1,1,0,0],
                                                      [0,1,0,0],
                                                      [0,0,1,0],
                                                      [1,1,0,1]]))

def test_edges_and_amat_init():
    eg = ExtendedGraph(edges = {(2,0)}, actions_amat=np.triu((1,1,1)))
    arr = np.triu((1,1,1))
    arr[2,0] = 1
    assert np.array_equal(eg._actions_amat, arr)
    with pytest.raises(IndexError) as e:
        eg = ExtendedGraph(edges = {(9,6)}, actions_amat=np.triu((1,1,1)))
    
def test_effects_and_attachments_init():
    eg = ExtendedGraph(effects=[1,2,4], attachments = {(1, 1), (1, 2), (3, 3)})
    assert np.array_equal(eg.effects(), np.array([1,2,4,3]))
    assert np.array_equal(eg._attachments_amat, np.array([[1,1,0,0],
                                                          [0,0,0,1]]))

def test_attachments_and_amat_init():
    with pytest.raises(ValueError) as e:
        eg = ExtendedGraph(attachments = {(1,2)}, attachments_amat=np.array([[1,1,0], [0,0,1]]))
    eg = ExtendedGraph(attachments = {(0,2), (1,0)}, attachments_amat=np.array([[1,1,0], [0,0,1]]))
    assert np.array_equal(eg._attachments_amat, np.array([[0,1,1], [1,0,0]]))

def test_extendedgraph_copy():
    eg1 = ExtendedGraph(actions_amat=np.triu((1,1,1)))
    eg2 = eg1.copy()
    assert eg1 == eg2

def test_actions():
    eg = ExtendedGraph(actions=[1,2,3])
    assert np.array_equal(np.arange(3), eg.actions_idx())
    assert np.array_equal(np.arange(1,4), eg.actions())

def test_action_reps():
    eg = ExtendedGraph(actions=[1,2,3], actions_amat=np.triu((1,1,1)))
    eg._join_actions(0,1)
    reps_idx = eg.action_reps_idx()
    reps = eg.action_reps()
    assert reps_idx.size == 2
    assert 2 in reps_idx
    assert reps.size == 2
    assert 3 in reps

def test_edges():
    eg = ExtendedGraph(actions_amat=np.triu((1,1,1)))
    assert eg.edges() == eg.edges_idx()
    assert eg.edges() == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

def test_attachments():
    eg = ExtendedGraph(attachments_amat=np.array([[1,1,0], [0,0,1]]))
    assert eg.attachments() == [(0,0), (0,1), (1,0)]
    assert eg.attachments_idx() == [(0,0), (0,1), (1,2)]

def test_full_amat():
    with pytest.raises(ValueError) as e:
        eg = ExtendedGraph(actions_amat=np.triu((1,1,1)), attachments_amat=np.array([[1,1,0], [0,0,1]]))
    eg = ExtendedGraph(actions_amat=np.triu((1,1)), attachments_amat=np.array([[1,1,0], [0,0,1]]))
    assert np.array_equal(eg._full_amat(), np.array([[1, 1, 1, 1, 0],
                                                  [0, 1, 0, 0, 1]]))

def test_parents():
    eg = ExtendedGraph(edges={(0,1), (1,2), (0,2)})
    assert {0,1} == eg._parents(2)
    eg._join_actions(1,2)
    assert {0} == eg._parents(2)

def test_children():
    eg = ExtendedGraph(edges={(0,1), (1,2), (0,2)})
    assert {1, 2} == eg._children(0)
    eg._join_actions(0, 1)
    assert {2} == eg._children(0)

def test_parents_of():
    eg = ExtendedGraph(edges={(0,1), (1,2), (0,2), (2,3), (1,3), (0,3)})
    assert {0, 1 ,2} == eg._parents_of({2,3})
    eg._join_actions(2,3)
    assert {0, 1} == eg._parents_of({2,3})

def test_children_of():
    eg = ExtendedGraph(edges={(0,1), (1,2), (0,2), (2,3), (1,3), (0,3)})
    assert {1 ,2, 3} == eg._children_of({0,1})
    eg._join_actions(0,1)
    assert {2, 3} == eg._children_of({0,1})

def test_joined_to():
    eg = ExtendedGraph(actions_amat = np.triu((1,1,1)))
    for a in eg.actions_idx():
        assert eg._joined_to(a) == a
    eg._join_actions(0,1)
    assert np.array_equal(np.array([0,1]), eg._joined_to(0))
    assert np.array_equal(np.array([0,1]), eg._joined_to(1))

def test_add_edge():
    eg = ExtendedGraph(edges={(0,1), (0,2)})
    assert {(0,0), (0,1), (1,1), (0,2), (2,2)} == set(eg.edges())
    aamat = eg.add_edge(1,2, inplace=False)['actions_amat']
    eg.add_edge(1,2)
    assert eg._actions_amat[1,2]
    assert {(0,0), (0,1), (1,1), (1,2), (0,2), (2,2)} == set(eg.edges())
    assert np.array_equal(aamat, eg._actions_amat)

def test_add_edges_from():
    edges = {(0,1), (1,2), (0,2)}
    eg1 = ExtendedGraph(actions=[0,1,2])
    aamat = eg1._add_edges_from(edges, inplace=False)
    eg1._add_edges_from(edges)
    eg2 = ExtendedGraph(edges = edges)
    assert eg1 == eg2
    assert np.array_equal(aamat, eg2._actions_amat)

def test_remove_edge():
    eg = ExtendedGraph(actions_amat=np.triu((1,1,1)))
    assert eg._actions_amat[1,2]
    assert {(0,0), (0,1), (1,1), (1,2), (0,2), (2,2)} == set(eg.edges())
    aamat = eg.remove_edge(1,2, inplace=False)['actions_amat']
    eg.remove_edge(1,2)
    assert not eg._actions_amat[1,2]
    assert {(0,0), (0,1), (1,1), (0,2), (2,2)} == set(eg.edges())
    assert np.array_equal(aamat, eg._actions_amat)

def test_remove_edges_from():
    eg1 = ExtendedGraph(actions_amat=np.triu((1,1,1)))
    eg2 = ExtendedGraph(actions=[0,1,2], edges={(0,1)})
    aamat = eg1.remove_edges_from({(1,2), (0,2)}, inplace=False)
    eg1.remove_edges_from({(1,2), (0,2)})
    assert eg1 == eg2
    assert np.array_equal(aamat, eg2._actions_amat)

def test_attach_effect():
    eg = ExtendedGraph(attachments_amat=np.array([[1,1,0], [0,0,1]]))
    aamat = eg.attach_effect(0, 2, inplace=False)
    eg.attach_effect(0, 2)
    assert np.array_equal(np.array([[1,1,1], [0,0,0]]), eg._attachments_amat)
    assert np.array_equal(eg._attachments_amat, aamat)

def test_attach_effects_from():
    eg = ExtendedGraph(attachments_amat=np.array([[1,1,0], [0,0,1]]))
    aamat = eg.attach_effects_from({(0,2), (1,0), (1,1)}, inplace=False)
    eg.attach_effects_from({(0,2), (1,0), (1,1)})
    assert np.array_equal(np.array([[0,0,1], [1,1,0]]), eg._attachments_amat)
    assert np.array_equal(aamat, eg._attachments_amat)

def test_detach_effect():
    eg = ExtendedGraph(attachments_amat=np.array([[1,1,0], [0,0,1]]))
    aamat = eg.detach_effect(1, inplace=False)
    eg.detach_effect(1)
    assert np.array_equal(np.array([[1,0,0], [0,0,1]]), eg._attachments_amat)
    assert np.array_equal(eg._attachments_amat, aamat)

def test_detach_effects_from():
    eg = ExtendedGraph(attachments_amat=np.array([[1,1,0], [0,0,1]]))
    aamat = eg.detach_effects_from([0,1], inplace=False)
    eg.detach_effects_from([0,1])
    assert np.array_equal(np.array([[0,0,0], [0,0,1]]), eg._attachments_amat)
    assert np.array_equal(eg._attachments_amat, aamat)

def test_remove_effect():
    eg = ExtendedGraph(attachments_amat=np.array([[1,1,0], [0,0,1]]))
    eg.remove_effect(2)
    assert np.array_equal(eg.effects(), [0,1])
    assert np.array_equal(eg._attachments_amat, [[1,1], [0,0]])

def test_join_actions():
    eg = ExtendedGraph(actions=[0,1,2], edges = {(0,1)})
    dd = eg.join_actions(0,1, inplace=False)
    aamat = dd['actions_amat']
    jarr = dd['join_array']
    eg.join_actions(0,1)
    test_arr = np.eye(3)
    test_arr[0,1] = 1
    test_arr[1,0] = 1
    assert np.array_equal(test_arr, jarr)
    assert np.array_equal(test_arr, aamat)
    assert np.array_equal(test_arr, eg._join_array)
    assert np.array_equal(test_arr, eg._actions_amat)
    assert eg._parents(1) == set()
    assert eg._children(0) == set()
    eg.add_edge(0,2)
    assert eg._children(1) == {2}
    assert eg._parents(2) == {0, 1}

def test_splitoff_actions():
    eg = ExtendedGraph(actions_amat=np.triu((1,1,1,1)))
    eg.join_actions(0,1)
    eg.join_actions(1,2)
    action_reps = eg.action_reps_idx()
    assert len(action_reps) == 2
    assert 3 in action_reps
    aamat, jarr = eg.splitoff_actions([0,2], inplace=False)
    eg.splitoff_actions([0,2])
    test_arr = np.ones((4, 4))
    test_arr[1] = [0,1,0,1]
    test_arr[3] = [0,0,0,1]
    assert np.array_equal(aamat, test_arr)
    assert np.array_equal(eg._actions_amat, test_arr)
    test_arr = np.eye(4)
    test_arr[0,2] = 1
    test_arr[2,0] = 1
    assert np.array_equal(jarr, test_arr)
    assert np.array_equal(eg._join_array, test_arr)

def test_splitoff_action():
    eg = ExtendedGraph(actions_amat=np.triu((1,1,1)))
    eg.join_actions(0,1)
    #test split up
    dd = eg.splitoff_action(0, direction='up', inplace=False)
    aamat = dd['actions_amat']
    jarr = dd['join_array']
    eg.splitoff_action(0,direction='up')
    assert np.array_equal(np.triu((1,1,1)), aamat)
    assert np.array_equal(np.eye(3), jarr)
    assert np.array_equal(np.triu((1,1,1)), eg._actions_amat)
    assert np.array_equal(np.eye(3), eg._join_array)
    eg._join_actions(0,1)
    #test split down
    dd = eg.splitoff_action(0, direction='down', inplace=False)
    aamat = dd['actions_amat']
    jarr = dd['join_array']
    eg.splitoff_action(0, direction='down')
    test_arr = np.array([[1,0,1], [1,1,1], [0,0,1]])
    assert np.array_equal(aamat, test_arr)
    assert np.array_equal(eg._actions_amat, test_arr)
    assert np.array_equal(np.eye(3), jarr)
    assert np.array_equal(np.eye(3), eg._join_array)
    #check equivalence between up and down for joined pairs
    eg = ExtendedGraph(actions_amat=np.triu((1,1,1)))
    eg.join_actions(0,1)
    dd = eg.splitoff_action(1, direction = 'up', inplace=False)
    aamat2 = dd['actions_amat']
    jarr2 = dd['join_array']
    np.array_equal(aamat, aamat2)
    np.array_equal(jarr, jarr2)