"""
PyNEM is the Python implementation of Nested Effects Models (NEMs). For the time being, it implements only the greedy weak order \
algorithm for learning, scoring either using the marginal log-likelihood (mLL), or using an alternating optimisation scheme. \
In time, I may also implement some of the major learning and scoring approaches from the original 'nem' package written in R.
"""

from .classes import *
from .custom_types import *
from . import datasets