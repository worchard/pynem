# PyNEM
Python implementation of Nested Effects Models

PyNEM consists of two core components: the ExtendedGraph class and the NestedEffectsModel class.

### ExtendedGraph class
ExtendedGraphs are the base class of Nested Effects Models and themselves consistent of two parts.
1. The actions graph: the graph describing the relationships between 'action' nodes, representing perturbed genes/proteins, for which there are no direct activity measurements.
2. The effect attachments: an extension to the actions graph which connects 'effect' nodes to each action in the actions graph, describing how the perturbation of each action influences causally downstream observables.

ExtendedGraphs facilitate the representation and manipulation of the action graph and effect attachments which underlie Nested Effects Models.

### NestedEffectsModel class
NestedEffectsModels are a child class of ExtendedGraphs and serves to connect them to perturbation data. NestedEffectsModels enable:
1. Scoring of given ExtendedGraph structures. 
2. Learning of ExtendedGraph structures. Currently the only supported method is the Greedy Weak Order (GWO) algorithm.
