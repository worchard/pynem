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
        >>> er = EffectReporters({'E1':'S1', 'E2':'S2', 'E3':'S3'}, signals = {'S4','S5','S6'})
        >>> adjacency_matrix, signal_list, effect_list = er.to_adjacency()
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