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
from numpy.testing import assert_allclose

from bayesian_benchmarks.models.template import RegressionModel as RegressionTemplate
from bayesian_benchmarks.models.template import ClassificationModel as ClassificationTemplate

from bayesian_benchmarks.models.get_model import all_regression_models, all_classification_models
from bayesian_benchmarks.models.get_model import get_regression_model, get_classification_model


@pytest.mark.parametrize('name', all_regression_models)
def test_models_regression(name):
    S, N, Ns, D = 5, 4, 3, 2

    model = get_regression_model(name)(is_test=True)
    model.fit(np.random.randn(N, D), np.random.randn(N, 1))
    model.fit(np.random.randn(N, D), np.random.randn(N, 1))
    m, v = model.predict(np.random.randn(Ns, D))
    assert m.shape == (Ns, 1)
    assert v.shape == (Ns, 1)

    samples = model.sample(np.random.randn(Ns, D), S)
    assert samples.shape == (S, Ns, 1)


@pytest.mark.parametrize('name', all_classification_models)
@pytest.mark.parametrize('K', [2, 3])
def test_models_regression(name, K):
    S, N, Ns, D = 2, 100, 2, 2

    model = get_classification_model(name)(K, is_test=True)
    model.fit(np.random.randn(N, D), np.random.choice(range(K), size=(N, 1)).astype(float))
    model.fit(np.random.randn(N, D), np.random.choice(range(K), size=(N, 1)).astype(float))
    p = model.predict(np.random.randn(Ns, D))
    assert p.shape == (Ns, K)
    assert_allclose(np.sum(p, 1), np.ones(Ns), atol=0.001)


def test_templates():
    regression_model = RegressionTemplate()
    classification_model = ClassificationTemplate(2)

    regression_model.fit(np.ones((1, 1)), np.ones((1, 1)))
    classification_model.fit(np.ones((1, 1)), np.ones((1, 1)))

    with pytest.raises(NotImplementedError):
        regression_model.predict(np.ones((1, 1)))

    with pytest.raises(NotImplementedError):
        classification_model.predict(np.ones((1, 1)))

    with pytest.raises(NotImplementedError):
        regression_model.sample(np.ones((1, 1)), 1)
