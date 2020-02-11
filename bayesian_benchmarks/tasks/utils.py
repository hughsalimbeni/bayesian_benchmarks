"""
Helper functions...

"""

from typing import List, Union

import numpy as np
from scipy.special import logsumexp

def meanlogsumexp(logps: Union[List[np.ndarray], np.ndarray], axis: int=0) -> np.ndarray:
    """
    Mean log sum exp of a log p list.
    :param logps: list of log probs [samples x data points] or [samples x data points x output dim]
    :param axis: determines reduction
    :return: avg probability value [1]
    """
    return np.mean(logsumexp(logps, axis=axis) - np.log(len(logps)))
