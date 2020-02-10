"""
Helper functions...

"""

from typing import List

import numpy as np
from scipy.special import logsumexp

def meansumexp(logps: List[np.ndarray]) -> np.ndarray:
    """
    Mean sum exp of a log p array.
    :param logps: list of log probs [samples x data points]
    :return: avg probability value [1]
    """
    return np.mean(logsumexp(logps, axis=0) - np.log(len(logps)))
