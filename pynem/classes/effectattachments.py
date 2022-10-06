from collections import defaultdict, UserDict
from typing import Hashable, Set, Union, Tuple, Any, Iterable, Dict, FrozenSet, List, TypedDict

from pynem.utils import core_utils
from pynem.custom_types import *

import numpy as np
import scipy.sparse as sps

class EffectAttachments(UserDict):
    """
    Dictionary-like class giving direct mappings from effects to signals (representing the child-of relation).
    """

    # This is only a global property because we need that it is updated when new keys are added (see __setitem__ below)
    _signals = set()

    #Update so only hashables can be values
    def __setitem__(self, key, value):
        if not isinstance(value, Node):
            raise TypeError(
                f'Invalid type for signal value: '
                f'expected "Hashable", got "{type(value).__name__}"'
            )
        self._signals.add(value)
        super().__setitem__(key, value)

    def __init__(self, *args, signals: NodeSet = None, **kwargs):
        super().__init__(*args, **kwargs)
        if not signals:
            self._signals = set(self.values())
        else:
            self._signals = set(self.values()).union(signals)

    def __eq__(self, other):
        if not isinstance(other, EffectAttachments):
            return False
        return self.data == other.data and self._signals == other._signals

    def __str__(self):
        return str(self.data) + ", signals = " + str(self._signals)

    def __repr__(self):
        return str(self)

    def rename_nodes(self, signal_map: Dict = None, effect_map: Dict = None):
        """
        Rename the signals according to ``signal_map`` and the effects according to ``effect_map``.
        Parameters
        ----------
        signal_map:
            A dictionary from the current name of each signal to the desired name of each signal.
        effect_map:
            A dictionary from the current name of each effect to the desired name of each effect.
        Examples
        --------
        >>> from pynem import EffectAttachments
        >>> ea1 = EffectAttachments({'E1':'S1', 'E2':'S2', 'E3':'S3'}, signals = {'S4'})
        >>> ea2 = ea1.rename_nodes(signal_map = {'S1': 'S1', 'S2':'S3', 'S3':'S4', 'S4':'S2'}, effect_map = {'E1': 'E1', 'E2': 'E3', 'E3':'E2'})
        >>> ea2
        {'E1': 'S1', 'E3': 'S3', 'E2': 'S4'}, signals = {'S3', 'S2', 'S1', 'S4'}
        """
        if not signal_map:
            signal_map = {s: s for s in self._signals}
        if not effect_map:
            effect_map = {e: e for e in self.effects()}
        
        renamed_dict = {effect_map[e]: signal_map[s] for e, s in self.items()}
        return EffectAttachments(renamed_dict, signals = signal_map.keys())


    def to_adjacency(self, signal_list: List = None, effect_list: List = None) -> Tuple[np.ndarray, list, list]:
        """
        Return the adjacency matrix for the effect reporter attachments for signals in ``signal_list`` and effects in ``effect_list``.
        Parameters
        ----------
        signal_list:
            List indexing the rows of the matrix.
        effect_list:
            List indexing the columns of the matrix.
        See Also
        --------
        from_adjacency
        Return
        ------
        (adjacency_matrix, signal_list, effect_list)
        Example
        -------
        >>> from pynem import EffectAttachments
        >>> ea = EffectAttachments({'E1':'S1', 'E2':'S2', 'E3':'S3'}, signals = {'S4','S5','S6'})
        >>> adjacency_matrix, signal_list, effect_list = ea.to_adjacency()
        >>> adjacency_matrix
        array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1],
               [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]])
        >>> signal_list
        ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
        >>> effect_list
        ['E1', 'E2', 'E3']
        """
        if not signal_list:
            signal_list = sorted(self._signals)
        if not effect_list:
            effect_list = sorted(self.effects())
        
        edges = self.items()
        if signal_list or effect_list:
            edges = {(signal, effect) for effect, signal in edges if effect in effect_list and signal in signal_list}

        signal2ix = {signal: i for i, signal in enumerate(signal_list)}
        effect2ix = {effect: i for i, effect in enumerate(effect_list)}
        
        shape = (len(signal_list), len(effect_list))
        adjacency_matrix = np.zeros(shape, dtype=int)
        
        for signal, effect in edges:
            adjacency_matrix[signal2ix[signal], effect2ix[effect]] = 1

        return adjacency_matrix, signal_list, effect_list

    def effects(self):
        return set(self.keys())
    
    @property
    def signals(self):
        return self._signals

#Theta should be a mapping from effects -> signals so of the form theta[e] = s
#as well as a matrix with s rows and e columns where theta_se means s is the parent of e
#mapping must ensure that each e gene has exactly 1 parent - so values must be hashable
#Also need that every value be out of a set of signals or None