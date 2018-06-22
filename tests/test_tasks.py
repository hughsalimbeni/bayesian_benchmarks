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
