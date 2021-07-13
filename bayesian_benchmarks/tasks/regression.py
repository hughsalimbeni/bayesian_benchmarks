"""
A conditional Gaussian estimation task: model p(y_n|x_n) = N(a(x_n), b(x_n))

Metrics reported are test log likelihood, mean squared error, and absolute error, all for normalized and unnormalized y.

"""

import argparse
import numpy as np
from scipy.stats import norm

from bayesian_benchmarks.data import get_regression_data
from bayesian_benchmarks.database_utils import Database
from bayesian_benchmarks.models.get_model import get_regression_model
from bayesian_benchmarks.tasks.utils import meanlogsumexp

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

    model.fit(data.X_train, data.Y_train)
    m, v = model.predict(data.X_test)  # both [data points x output dim] or [samples x data points x output dim]

    assert m.ndim == v.ndim
    assert m.ndim in {2, 3}  # 3-dim in case of approximate predictions (multiple samples per each X)
    assert np.all(v >= 0.0)

    res = {}
    log_eps = np.log(1e-12)  # log probability threshold
    log_1_minus_eps = np.log(1.0 - 1e-12)

    if m.ndim == 2:  # keep analysis as in the original code in case of 2-dim predictions

        l = norm.logpdf(data.Y_test, loc=m, scale=v ** 0.5)  # []
        l = np.clip(l, log_eps, log_1_minus_eps)  # clip
        res['test_loglik'] = np.average(l)

        lu = norm.logpdf(data.Y_test * data.Y_std, loc=m * data.Y_std, scale=(v ** 0.5) * data.Y_std)
        lu = np.clip(lu, log_eps, log_1_minus_eps)  # clip
        res['test_loglik_unnormalized'] = np.average(lu)

        d = data.Y_test - m
        du = d * data.Y_std

        res['test_mae'] = np.average(np.abs(d))
        res['test_mae_unnormalized'] = np.average(np.abs(du))

        res['test_rmse'] = np.average(d ** 2) ** 0.5
        res['test_rmse_unnormalized'] = np.average(du ** 2) ** 0.5

    else:  # compute metrics in case of 3-dim predictions

        res['test_loglik'] = []
        res['test_loglik_unnormalized'] = []

        for n in range(m.shape[0]):  # iterate through samples
            l = norm.logpdf(data.Y_test, loc=m[n], scale=v[n] ** 0.5)
            l = np.clip(l, log_eps, log_1_minus_eps)  # clip
            res['test_loglik'].append(l)

            lu = norm.logpdf(data.Y_test * data.Y_std, loc=m[n] * data.Y_std, scale=(v[n] ** 0.5) * data.Y_std)
            lu = np.clip(lu, log_eps, log_1_minus_eps)  # clip
            res['test_loglik_unnormalized'].append(lu)

        # Mixture test likelihood (mean over per data point evaluations)
        res['test_loglik'] = meanlogsumexp(res['test_loglik'])

        # Mixture test likelihood (mean over per data point evaluations)
        res['test_loglik_unnormalized'] = meanlogsumexp(res['test_loglik_unnormalized'])

        d = data.Y_test - np.mean(m, axis=0)
        du = d * data.Y_std

        res['test_mae'] = np.average(np.abs(d))
        res['test_mae_unnormalized'] = np.average(np.abs(du))

        res['test_rmse'] = np.average(d ** 2) ** 0.5
        res['test_rmse_unnormalized'] = np.average(du ** 2) ** 0.5

    if not is_test:
        res.update(ARGS.__dict__)

    if not is_test:  # pragma: no cover
        with Database(ARGS.database_path) as db:
            db.write('regression', res)

    return res


if __name__ == '__main__':
    run(parse_args())