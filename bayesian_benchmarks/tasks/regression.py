"""
A conditional Gaussian estimation task: model p(y_n|x_n) = N(a(x_n), b(x_n))

Metrics reported are test log likelihood, mean squared error, and absolute error, all for normalized and unnormalized y.

"""

import argparse
import numpy as np
from scipy.stats import norm
from importlib import import_module

from bayesian_benchmarks.data import ALL_REGRESSION_DATATSETS
from bayesian_benchmarks.database_utils import Database
from bayesian_benchmarks.models.non_bayesian_models import non_bayesian_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='deep_gp_doubly_stochastic', nargs='?', type=str)
    parser.add_argument("--dataset", default='energy', nargs='?', type=str)
    parser.add_argument("--split", default=0, nargs='?', type=int)
    return parser.parse_args()

def run(ARGS, is_test=False):
    data = ALL_REGRESSION_DATATSETS[ARGS.dataset](split=ARGS.split)

    Model = non_bayesian_model(ARGS.model, 'regression') or\
            import_module('bayesian_benchmarks.models.{}.models'.format(ARGS.model)).RegressionModel

    model = Model(is_test=is_test)
    model.fit(data.X_train, data.Y_train)
    m, v = model.predict(data.X_test)

    res = {}

    l = norm.logpdf(data.Y_test, loc=m, scale=v**0.5)
    res['test_loglik'] = np.average(l)

    lu = norm.logpdf(data.Y_test * data.Y_std, loc=m * data.Y_std, scale=(v**0.5) * data.Y_std)
    res['test_loglik_unnormalized'] = np.average(lu)

    d = data.Y_test - m
    du = d * data.Y_std

    res['test_mae'] = np.average(np.abs(d))
    res['test_mae_unnormalized'] = np.average(np.abs(du))

    res['test_rmse'] = np.average(d**2)**0.5
    res['test_rmse_unnormalized'] = np.average(du**2)**0.5

    res.update(ARGS.__dict__)

    with Database() as db:
        db.write('regression', res)


if __name__ == '__main__':
    run(parse_args())