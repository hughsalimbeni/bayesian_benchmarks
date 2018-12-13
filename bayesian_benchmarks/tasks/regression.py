"""
A conditional Gaussian estimation task: model p(y_n|x_n) = N(a(x_n), b(x_n))

Metrics reported are test log likelihood, mean squared error, and absolute error, all for normalized and unnormalized y.

"""

import argparse
import numpy as np
from scipy.stats import norm
import time

from bayesian_benchmarks.data import get_regression_data
from bayesian_benchmarks.database_utils import Database
from bayesian_benchmarks.models.get_model import get_regression_model

def parse_args():  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='linear', nargs='?', type=str)
    parser.add_argument("--dataset", default='energy', nargs='?', type=str)
    parser.add_argument("--split", default=0, nargs='?', type=int)
    parser.add_argument("--seed", default=0, nargs='?', type=int)
    parser.add_argument("--database_path", default='', nargs='?', type=str)
    return parser.parse_args()

def run(ARGS, data=None, model=None, is_test=False):

    data = data or get_regression_data(ARGS.dataset, split=ARGS.split)
    model = model or get_regression_model(ARGS.model)(is_test=is_test, seed=ARGS.seed)

    t = time.time()
    model.fit(data.X_train, data.Y_train)
    m, v = model.predict(data.X_test)
    s = 'Dataset: {:20s} Model: {:30s} N:{:8s} D:{:8s} Time: {:.4f}s'.format(data.name,
                                                                            ARGS.model,
                                                                            str(data.N),
                                                                            str(data.D),
                                                                            time.time() - t)
    print(s)
    with open('res.txt', 'a') as f:
        f.write(s + '\n')

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
    print('{:.4f} {:.4f}'.format(res['test_loglik'], res['test_rmse']))
    # if not is_test:  # pragma: no cover
    #     with Database(ARGS.database_path) as db:
    #         db.write('regression', res)

    return res


if __name__ == '__main__':
    run(parse_args())