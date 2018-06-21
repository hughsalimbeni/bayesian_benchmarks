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

import inspect
import os

from bayesian_benchmarks.models.template import RegressionModel as RegressionTemplate
from bayesian_benchmarks.models.template import ClassificationModel as ClassificationTemplate
import bayesian_benchmarks
from bayesian_benchmarks.models.get_model import all_regression_models, all_classification_models, sklearn_models
from bayesian_benchmarks.models.get_model import get_regression_model, get_classification_model

abs_path = bayesian_benchmarks.models.template.__file__[:-len('/template.py')]

def test_regression():
    """
    Test that all the implemented models do actually implement the methods of the template models
    """

    attributes = inspect.getmembers(RegressionTemplate)
    methods = ([a[0] for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))])

    names = [f for f in os.listdir(abs_path) if
             ((not f.endswith('__')) and os.path.isdir(os.path.join(abs_path, f)))]

    successes = []
    for name in names:
        try:
            M = get_regression_model(name)
            successes.append(name)
        except:
            M = None

        if M:
            for m in methods:
                if not hasattr(M(is_test=True), m):
                    raise NotImplementedError('{} is missing the {} method'.format(M, m))

    assert set(successes) == set(all_regression_models) - set(sklearn_models)

def test_classification():
    attributes = inspect.getmembers(ClassificationTemplate)

    methods = ([a[0] for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))])
    names = [f for f in os.listdir(abs_path) if
             ((not f.endswith('__')) and os.path.isdir(os.path.join(abs_path, f)))]

    successes = []
    for name in names:
        try:
            M = get_classification_model(name)
            successes.append(name)
        except:
            M = None

        if M:
            for m in methods:
                if not hasattr(M(2, is_test=True), m):
                    raise NotImplementedError('{} is missing the {} method'.format(M, m))


    assert set(successes) == set(all_classification_models) - set(sklearn_models)
