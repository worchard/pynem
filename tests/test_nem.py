from pynem import NestedEffectsModel, SignalGraph, EffectAttachments
import pytest
import numpy as np
import warnings
import anndata as ad
import pandas as pd

### Test data set up
adata_example = ad.AnnData(X = np.ones((10,5)), dtype=np.int0)
var_index = pd.Index(['E'+ str(n) for n in range(5)], name = 'effects')
eff_set = set(var_index)
adata_example.var.set_index(var_index, inplace = True)
signals = ['S' + str(n) for n in range(4)]
sig_set = set(signals)
signals.append('control')
replicates = [s + '_' + str(n) for s in signals for n in range(2)]
signals = [s for s in signals for _ in (0, 1)]
adata_example.obs = pd.DataFrame({'sgRNA': replicates, 'target': signals})
###
sg = SignalGraph(nodes = sig_set, edges = {('S0', 'S1'), ('S1', 'S2')})
ea = EffectAttachments({'E0': 'S0', 'E1':'S1', 'E2':'S2', 'E3':'S3'})

def test_empty_nem_init():
    nem = NestedEffectsModel()
    assert nem.signal_graph == SignalGraph()
    assert nem.effect_attachments == EffectAttachments()
    assert nem.signals == set()
    assert nem.effects == set()
    assert nem.score == None

def test_adata_only_nem():
    nem = NestedEffectsModel(adata = adata_example, signals_column='target')
    assert nem.signal_graph == SignalGraph(nodes = sig_set)
    assert nem.effect_attachments == EffectAttachments.fromeffects(eff_set, signals=sig_set)
    assert nem.signals == sig_set
    assert nem.effects == eff_set
    assert nem.score == None

def test_nem_signal_graph_init():
    nem = NestedEffectsModel(adata = adata_example, signals_column='target', signal_graph=sg)
    assert nem.signals == sg.nodes
    assert nem.effects == eff_set

def test_nem_signal_graph_effect_attachments_init():
    nem = NestedEffectsModel(adata = adata_example, signals_column='target', signal_graph=sg, effect_attachments=ea)
    assert nem.signals == sg.nodes
    assert nem.effects == ea.effects()

def test_nem_copy():
    nem = NestedEffectsModel(adata = adata_example, signals_column='target', signal_graph=sg)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        nem_copy = nem.copy()
        assert nem.signals == nem_copy.signals
        assert nem.effects == nem_copy.effects
        assert nem.score == nem_copy.score
        assert np.all(nem.adata.X == nem_copy.adata.X)
        assert np.all(nem.adata.obs == nem_copy.adata.obs)
        assert np.all(nem.adata.var.index == nem_copy.adata.var.index)

def test_nem_adata_signals():
    nem = NestedEffectsModel(adata = adata_example, signals_column='target', controls={'control', 'S0'})
    nem.signals == sig_set.difference({'S0'})

def test_signal_match():
    ea_copy = ea.copy()
    ea_copy['E1'] = 'S4'
    with pytest.raises(AssertionError) as e_info:
        nem = NestedEffectsModel(adata = adata_example, signals_column='target', signal_graph=sg, effect_attachments=ea_copy)

def test_nem_add_signal():
    nem = NestedEffectsModel()
    nem.add_signal('S1')
    assert nem.signals == {'S1'}

def test_nem_add_signals_from():
    nem = NestedEffectsModel()
    nem.add_signals_from(range(3,10))
    assert nem.signals == {3, 4, 5, 6, 7, 8, 9}

def test_parents_of():
    g = SignalGraph(edges={(1,2), (2,3), (2,4)})
    nem = NestedEffectsModel(signal_graph=g)
    assert nem.parents_of(nem.signals) == {1,2}

def test_children_of():
    g = SignalGraph(edges={(1,2), (2,3), (2,4)})
    nem = NestedEffectsModel(signal_graph=g)
    assert nem.children_of(nem.signals) == {2,3,4}

def test_remove_signal():
    g = SignalGraph(edges={(1,2), (2,3), (2,4)})
    nem = NestedEffectsModel(signal_graph=g)
    nem.remove_signal(2)
    assert nem.signals == {1,3,4}

def test_add_edge():
    g = SignalGraph(edges={(1,2), (2,3), (2,4)})
    nem = NestedEffectsModel(signal_graph=g)
    nem.add_edge(1,4)
    assert (1,4) in nem.signal_graph.edges

def test_add_edges_from():
    g = SignalGraph(nodes={1,2})
    nem = NestedEffectsModel(signal_graph=g)
    nem.add_edges_from({(1,2), (2,3), (2,4)})
    assert nem.signals == {1,2,3,4} and nem.signal_graph.edges == {(1,2), (2,3), (2,4)}

def test_remove_edge():
    g = SignalGraph(edges={(1,2), (2,3)})
    nem = NestedEffectsModel(signal_graph=g)
    nem.remove_edge(1,2)
    assert (1,2) not in nem.signal_graph.edges

def test_remove_edges_from():
    g = SignalGraph(edges={(1,2), (2,3), (2,4)})
    nem = NestedEffectsModel(signal_graph=g)
    nem.remove_edges_from({(1,2), (2,3)})
    assert nem.signal_graph.edges == {(2,4)}

def test_join_signals():
    g = SignalGraph(edges={(1,2), (2,3), (2,4)})
    nem = NestedEffectsModel(signal_graph=g)
    nem.join_signals({1,2})
    assert nem.signals == {frozenset({1,2}), 3, 4}
    assert nem.signal_graph.edges == {(frozenset({1,2}), 3), (frozenset({1,2}), 4)}

def test_join_signals_2():
    g = SignalGraph(nodes={1,2,3})
    nem = NestedEffectsModel(signal_graph=g)
    nem.join_signals({1,2})
    assert nem.signals == {frozenset({1,2}), 3}
    assert nem.signal_graph.edges == set()

def test_split_signal_1():
    g = SignalGraph(edges={(frozenset({1,2}), 3), (frozenset({1,2}), 4)})
    nem = NestedEffectsModel(signal_graph=g)
    nem.split_signal(1, frozenset({1,2}), 'up')
    assert nem.signals == {1,2,3,4}
    assert nem.signal_graph.edges == {(1,2), (1,3), (1,4), (2,3), (2,4)}

def test_split_signal_2():
    g = SignalGraph(edges={(frozenset({1,2}), 3), (frozenset({1,2}), 4)})
    nem = NestedEffectsModel(signal_graph=g)
    with pytest.raises(KeyError) as e_info:
        nem.split_signal(1, frozenset({3}), 'up')

def test_to_adjacency():
    sg = SignalGraph(edges={('S1', 'S2'), ('S1', 'S3'), ('S2', 'S3')})
    ea = EffectAttachments({'E1':'S1', 'E2':'S2', 'E3':'S3'})
    nem = NestedEffectsModel(signal_graph = sg, effect_attachments = ea)
    adj_test_mat = np.array([[1, 1, 1, 1, 0, 0],
               [0, 1, 1, 0, 1, 0],
               [0, 0, 1, 0, 0, 1],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])
    adjacency_matrix, node_list =  nem.to_adjacency(signal_list = ['S1', 'S2', 'S3'], effect_list = ['E1', 'E2', 'E3'])
    assert np.all(adjacency_matrix == adj_test_mat)

def test_to_adjacency_no_node_list():
    sg = SignalGraph(edges={('S1', 'S2'), ('S1', 'S3'), ('S2', 'S3')})
    ea = EffectAttachments({'E1':'S1', 'E2':'S2', 'E3':'S3'})
    nem = NestedEffectsModel(signal_graph = sg, effect_attachments = ea)
    adj_test_mat = np.array([[1, 1, 1, 1, 0, 0],
               [0, 1, 1, 0, 1, 0],
               [0, 0, 1, 0, 0, 1],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])
    adjacency_matrix, node_list = nem.to_adjacency()
    assert np.sum(adjacency_matrix) == np.sum(adj_test_mat)

def test_to_adjacency_save():
    sg = SignalGraph(edges={('S1', 'S2'), ('S1', 'S3'), ('S2', 'S3')})
    ea = EffectAttachments({'E1':'S1', 'E2':'S2', 'E3':'S3'})
    nem = NestedEffectsModel(signal_graph = sg, effect_attachments = ea)
    signal_list = ['S1', 'S2', 'S3']
    effect_list = ['E1', 'E2', 'E3']
    adj_test_mat = np.array([[1, 1, 1, 1, 0, 0],
               [0, 1, 1, 0, 1, 0],
               [0, 0, 1, 0, 0, 1],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])
    adjacency_matrix, node_list =  nem.to_adjacency(signal_list = signal_list, effect_list = effect_list, save=True)
    assert np.all(nem.amat_tuple[0] == adj_test_mat)
    assert nem.amat_tuple[1] == signal_list + effect_list

def test_predict_dict():
    example_dict = {0: {'E0', 'E1', 'E2'}, 1: {'E0', 'E1', 'E2'}, 2: {'E2'}, 3: set()}
    sg = SignalGraph(edges={(0, 1), (0, 2), (1, 2)})
    sg.add_node(3)
    ea = EffectAttachments({'E0': 1, 'E1': 1, 'E2': 2}, signals = {0, 3})
    nem = NestedEffectsModel(signal_graph = sg, effect_attachments = ea)
    predict_dict = nem.predict_dict()
    assert example_dict == predict_dict

def test_item_setter():
    nem = NestedEffectsModel()
    nem['E1'] = 'S1'
    nem['E2'] = None
    assert nem.signals == {None, 'S1'}
    assert nem.effect_attachments.data == {'E1': 'S1', 'E2': None}
    assert nem.signal_graph.nodes == {'S1'}

def test_alpha_beta_setter():
    nem = NestedEffectsModel(alpha = 0.7, beta = 0.2)
    nem.alpha = 0.9
    with pytest.raises(ValueError) as e_info:
        nem.beta = 1.1

def test_predict_array():
    sg = SignalGraph(edges={(0, 1), (0, 2), (1, 2)})
    sg.add_node(3)
    ea = EffectAttachments({'E0': 0, 'E1': 1, 'E2': 2}, signals = {3})
    nem = NestedEffectsModel(signal_graph = sg, effect_attachments = ea)
    F, signal_list, effect_list = nem.predict_array(replicates = 2, signal_list = [2,1,0], effect_list = ['E1', 'E2', 'E0'])
    assert np.all(F == np.array([[0, 1, 0],
                                 [0, 1, 0],
                                 [1, 1, 0],
                                 [1, 1, 0],
                                 [1, 1, 1],
                                 [1, 1, 1]]))
    assert signal_list == [2, 2, 1, 1, 0, 0]