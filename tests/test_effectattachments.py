from pynem import EffectAttachments
import pytest
import numpy as np
import scipy.sparse as sps

def test_empty_effectattachments():
    ea = EffectAttachments()
    assert ea.data == {}
    assert ea.signals == set()

def test_signal_only_init():
    ea = EffectAttachments(signals = {1,2,3})
    assert ea.data == {}
    assert ea.signals == {1,2,3}

def test_fromkeys():
    ea = EffectAttachments.fromkeys({1,2,3})
    assert ea.data == {1: None, 2: None, 3: None}
    assert ea.signals == {None}

def test_fails_when_signal_not_hashable():
    ea = EffectAttachments()
    with pytest.raises(TypeError):
        ea[1] = [3,4]

def test_effectattachments_copy():
    ea1 = EffectAttachments({1:2, 3:4, 5:6}, signals = {7,8,9})
    ea2 = ea1.copy()
    assert ea1 == ea2

def test_rename_nodes():
    ea1 = EffectAttachments({'E1':'S1', 'E2':'S2', 'E3':'S3'}, signals = {'S4'})
    ea2 = ea1.rename_nodes(signal_map = {'S1': 'S1', 'S2':'S3', 'S3':'S4', 'S4':'S2'}, effect_map = {'E1': 'E1', 'E2': 'E3', 'E3':'E2'})
    assert ea2.data == {'E1': 'S1', 'E3': 'S3', 'E2': 'S4'}
    assert ea2.signals == {'S3', 'S2', 'S1', 'S4'}


def test_to_adjacency():
    ea = EffectAttachments({'E1':'S1', 'E2':'S2', 'E3':frozenset({'S3', 'S7'})}, signals = {'S4','S5','S6'})
    adj_test_mat = np.array([[0, 1], [0, 0], [0,0]])
    adjacency_matrix, signal_list, effect_list = ea.to_adjacency(signal_list=['S1', frozenset({'S3', 'S7'}), 'S4'], effect_list = ['E2', 'E1'])
    assert np.all(adjacency_matrix == adj_test_mat)

def test_to_adjacency_save():
    ea = EffectAttachments({'E1':'S1', 'E2':'S2', 'E3':'S3'}, signals = {'S4','S5','S6'})
    adj_test_mat = np.array([[0, 1], [0, 0], [0,0]])
    adjacency_matrix, signal_list, effect_list = ea.to_adjacency(signal_list=['S1', 'S3', 'S4'], effect_list = ['E2', 'E1'], save = True)
    assert np.all(ea.amat_tuple[0] == adj_test_mat)
    assert ea.amat_tuple[1] == signal_list
    assert ea.amat_tuple[2] == effect_list

def test_from_adjacency():
    adjacency_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 0]])
    sps_adjacency_matrix = sps.coo_matrix(adjacency_matrix)
    ea1 = EffectAttachments.from_adjacency(adjacency_matrix, signal_list = ['S1', 'S2', 'S3', 'S4'], effect_list = ['E1', 'E2', 'E3'])
    ea2 = EffectAttachments.from_adjacency(sps_adjacency_matrix, signal_list = ['S1', 'S2', 'S3', 'S4'], effect_list = ['E1', 'E2', 'E3'])
    assert ea1 == ea2
    assert ea1.data == {'E2': 'S1', 'E3': 'S2', 'E1': 'S3'}

def test_from_adjacency_no_signal_list():
    adjacency_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 0]])
    ea = EffectAttachments.from_adjacency(adjacency_matrix, effect_list = ['E1', 'E2', 'E3'])
    assert ea.data == {'E2': 0, 'E3': 1, 'E1': 2}

def test_fromeffects():
    ea = EffectAttachments.fromeffects({'E1', 'E2', 'E3'}, signals = {'S1', 'S2', 'S3'}, value = 'S4')
    assert ea.data == {'E1': 'S4', 'E2': 'S4', 'E3': 'S4'}
    assert ea.signals == {'S1', 'S2', 'S3', 'S4'}