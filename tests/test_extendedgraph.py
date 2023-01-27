from pynem import ExtendedGraph
import pytest
import numpy as np
import scipy.sparse as sps

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
    assert np.array_equal(np.array([[1,1,0], [0,0,1]]), eg._attachments_amat)

def test_actions_amat_init():
    eg1 = ExtendedGraph(actions_amat=np.triu((1,1,1)))
    eg2 = ExtendedGraph(edges = {(0,1), (1,2), (0,2)})
    assert eg1 == eg2

def test_attachments_amat_init():
    eg1 = ExtendedGraph(attachments_amat=np.array([[1,1,0], [0,0,1]]))
    eg2 = ExtendedGraph(attachments={(0,0), (0,1), (1,2)})
    assert eg1 == eg2