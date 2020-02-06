# Copyright 2017 Hugh Salimbeni
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np

from bayesian_benchmarks.tasks.regression import run as run_regression
from bayesian_benchmarks.tasks.classification import run as run_classification
from bayesian_benchmarks.tasks.active_learning_continuous import run as run_AL_cont
from bayesian_benchmarks.tasks.active_learning_discrete import run as run_AL_disc
from bayesian_benchmarks.tasks.mmd import run as run_mmd

# only test these
all_regression_models = ['linear']
all_classification_models = ['linear']


class ConvertToNamespace(object):
    def __init__(self, adict):
        adict.update({'seed':0, 'split':0})
        self.__dict__.update(adict)

@pytest.mark.parametrize('model', all_regression_models)
def test_regression(model):
    d = {'dataset':'boston',
         'model' :  model}

    run_regression(ConvertToNamespace(d), is_test=True)


@pytest.mark.parametrize('model', all_regression_models)
def test_active_learning_continuous(model):
    d = {'dataset':'boston',
         'model' :  model,
         'iterations': 2,
         'num_initial_points': 10}

    run_AL_cont(ConvertToNamespace(d), is_test=True)


@pytest.mark.parametrize('model', all_regression_models)
@pytest.mark.parametrize('pca_dim', [0, 2])
def test_mmd(model, pca_dim):
    d = {'dataset':'boston',
         'model' :  model,
         'num_samples' : 2,
         'pca_dim' : pca_dim}

    run_mmd(ConvertToNamespace(d), is_test=True)


@pytest.mark.parametrize('dataset', ['iris', 'planning'])  # binary and multiclass
@pytest.mark.parametrize('model', all_classification_models)
def test_classification(model, dataset):
    d = {'dataset':dataset,
         'model' :  model}

    run_classification(ConvertToNamespace(d), is_test=True)


@pytest.mark.parametrize('dataset', ['iris', 'planning'])  # binary and multiclass
@pytest.mark.parametrize('model', all_regression_models)
def test_active_learning_discrete(model, dataset):
    d = {'dataset':dataset,
         'model' :  model,
         'iterations': 2,
         'num_initial_points': 10}

    run_AL_disc(ConvertToNamespace(d), is_test=True)


# Testing modified regression and classification runs with mocks

class RegressionMock(object):
    """
    Regression mock.
    """
    def fit(self, X: np.ndarray, Y:np.ndarray) -> None:
        pass
    def predict(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        mu = np.array([[1., 2., 3.], [4., 5., 6.]])
        var = np.array([[.1, .2, .3], [.4, .5, .6]])
        return mu, var

class ApproximateRegressionMock(RegressionMock):
    """
    Approximate regression mock.
    """
    def predict(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        mu = np.array([[[1., 2., 3.], [4., 5., 6.]], [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]])
        var = np.array([[[.1, .2, .3], [.4, .5, .6]], [[.1, .2, .3], [.4, .5, .6]]])
        return mu, var

class ClassificationMock(object):
    """
    Classification mock.
    """
    def fit(self, X: np.ndarray, Y:np.ndarray) -> None:
        pass
    def predict(self, X: np.ndarray) -> np.ndarray:
        p = np.array([[.1, .2, .3], [.4, .5, .6]])
        return p

class ApproximateClassificationMock(ClassificationMock):
    """
    Approximate classification mock.
    """
    def predict(self, X: np.ndarray) -> np.ndarray:
        p = np.array([[[.1, .2, .3], [.4, .5, .6]], [[.01, .02, .03], [.04, .05, .06]]])
        return p

class RegressionDataMock(object):
    """
    Regression data mock.
    """
    X_train, Y_train = np.empty(shape=()), np.empty(shape=())
    X_test, Y_test = np.empty(shape=()), np.array([1.])
    Y_std = 2.

class ClassificationDataMock(object):
    """
    Classification data mock.
    """
    X_train, Y_train = np.empty(shape=()), np.empty(shape=())
    X_test, Y_test = np.empty(shape=()), np.array([[0], [1]])
    K = 3  # must be compatible with classification mocks...

regression_results = {}
regression_results['test_loglik'] = -9.8963
regression_results['test_loglik_unnormalized'] = -10.5507
regression_results['test_mae'] = 2.5
regression_results['test_mae_unnormalized'] = 5.0
regression_results['test_rmse'] = 3.0277
regression_results['test_rmse_unnormalized'] = 6.0553

approximate_regression_results = {}
approximate_regression_results['test_loglik'] = -11.8266
approximate_regression_results['test_loglik_unnormalized'] = -12.4904
approximate_regression_results['test_mae'] = 2.75
approximate_regression_results['test_mae_unnormalized'] = 5.5
approximate_regression_results['test_rmse'] = 3.2372
approximate_regression_results['test_rmse_unnormalized'] = 6.4743

classification_results = {}
classification_results['test_loglik'] = -1.4452
classification_results['test_acc'] = 0.0
classification_results['Y_test'] = np.array([0, 1, 2])
classification_results['p_test'] = np.array([[0.1667, 0.3333, 0.5], [0.2667, 0.3333, 0.4]])

approximate_classification_results = {}
approximate_classification_results['test_loglik'] = -2.7520
approximate_classification_results['test_acc'] = 0.0
approximate_classification_results['Y_test'] = np.array([0, 1, 2])
approximate_classification_results['p_test'] = np.array([[0.1333, 0.2667, 0.4], [0.3333, 0.4167, 0.5]])

regression_tuple = (RegressionDataMock(), RegressionMock(), regression_results)
approx_regression_tuple = (RegressionDataMock(), ApproximateRegressionMock(), approximate_regression_results)

@pytest.mark.parametrize('tuple', [regression_tuple, approx_regression_tuple])
def test_regression(tuple):
    data, model, correct_result = tuple
    result = run_regression(None, data=data, model=model, is_test=True)
    assert correct_result['test_loglik'] == pytest.approx(result['test_loglik'], 1e-3)
    assert correct_result['test_loglik_unnormalized'] == pytest.approx(result['test_loglik_unnormalized'], 1e-3)
    assert correct_result['test_mae'] == pytest.approx(result['test_mae'], 1e-3)
    assert correct_result['test_mae_unnormalized'] == pytest.approx(result['test_mae_unnormalized'], 1e-3)
    assert correct_result['test_rmse'] == pytest.approx(result['test_rmse'], 1e-3)
    assert correct_result['test_rmse_unnormalized'] == pytest.approx(result['test_rmse_unnormalized'], 1e-3)

classification_tuple = (ClassificationDataMock(), ClassificationMock(), classification_results)
approx_classification_tuple = (ClassificationDataMock(), ApproximateClassificationMock(), approximate_classification_results)

@pytest.mark.parametrize('tuple', [classification_tuple, approx_classification_tuple])
def test_regression(tuple):
    data, model, correct_result = tuple
    result = run_classification(None, data=data, model=model, is_test=True)
    assert correct_result['test_loglik'] == pytest.approx(result['test_loglik'], 1e-3)
    assert correct_result['test_acc'] == pytest.approx(result['test_acc'], 1e-3)
    assert np.allclose(correct_result['Y_test'], result['Y_test'], rtol=0.0, atol=1e-3)
    assert np.allclose(correct_result['p_test'], result['p_test'], rtol=0.0, atol=1e-3)