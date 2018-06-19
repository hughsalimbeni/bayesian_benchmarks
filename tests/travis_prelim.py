"""
All the classification come in a single large download. If many tests are run in parallel this can cause 
issues of synchrony. We run this once on travis so the data is there.   
"""

from bayesian_benchmarks.data import ALL_CLASSIFICATION_DATATSETS

ALL_CLASSIFICATION_DATATSETS['iris']()
