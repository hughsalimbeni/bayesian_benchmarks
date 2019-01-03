"""
To add a model, create a new directory in bayesian_benchmarks.models (here) and make file models.py containing at
least one of the classes below.

Model usage is similar to sklearn. For regression:

model = RegressionModel(is_test=False)
model.fit(X, Y)
mean, var = model.predict(X_test)  # first two moments of the posterior
samples = model.sample(X_test, S)  # samples from the posterior

For classification:
model = ClassificationModel(K, is_test=False)  # K is the number of classes
model.fit(X, Y)
p = model.predict(X_test)          # predictive probabilities for each class (i.e. onehot)

It should be feasible to call fit and predict many times (e.g. avoid rebuilding a tensorflow graph on each call).

"""

import numpy as np
import gpflow
from gpflow import params_as_tensors
from scipy.cluster.vq import kmeans2

from .nkn import NeuralKernelNetwork, NKNWrapper, KernelWrapper
from .utils import median_distance_local, get_nkn_config

from bayesian_benchmarks.models.variationally_sparse_gp.models import RegressionModel as _RegressionModel
from bayesian_benchmarks.models.variationally_sparse_gp.models import ClassificationModel as _ClassificationModel


class RegressionModel(_RegressionModel):
    def __init__(self, nkn='default', nkn_config=None, **kwargs):
        _RegressionModel.__init__(self, **kwargs)
        self.nkn_config = nkn_config or get_nkn_config(nkn)

    def _make_kernel(self, X):
        cf = self.nkn_config(X.shape[1], median_distance_local(X))
        return NeuralKernelNetwork(X.shape[1], KernelWrapper(cf['kern']), NKNWrapper(cf['nkn']))


class ClassificationModel(_ClassificationModel):
    def __init__(self, nkn='default', nkn_config=None, **kwargs):
        _ClassificationModel.__init__(self, **kwargs)
        self.nkn_config = nkn_config or get_nkn_config(nkn)

    def _make_kernel(self, X):
        cf = self.nkn_config(X.shape[1], median_distance_local(X))
        return NeuralKernelNetwork(X.shape[1], KernelWrapper(cf['kern']), NKNWrapper(cf['nkn']))
