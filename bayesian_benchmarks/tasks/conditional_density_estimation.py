import argparse
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import norm

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


class KDE_ARGS:
    num_samples = 10000


def run(ARGS, data=None, model=None, is_test=False):

    data = data or get_regression_data(ARGS.dataset, split=ARGS.split)
    model = model or get_regression_model(ARGS.model)(is_test=is_test, seed=ARGS.seed)

    model.fit(data.X_train, data.Y_train)

    res = {}
    logp = np.empty(len(data.X_test))
    for i, (x, y) in enumerate(zip(data.X_test, data.Y_test)):
        samples = model.sample(x.reshape(1, -1), KDE_ARGS.num_samples).reshape(-1, 1)

        bandwidth = 1.06 * np.std(samples) * KDE_ARGS.num_samples**(-1./5)  # Silverman's (1986) rule of thumb.
        kde = KernelDensity(bandwidth=bandwidth)

        l = kde.fit(samples).score(y.reshape(-1, 1))
        logp[i] = float(l)

    res['test_loglik'] = np.average(logp)

    res.update(ARGS.__dict__)

    if not is_test:  # pragma: no cover
        with Database(ARGS.database_path) as db:
            db.write('conditional_density_estimation', res)

    return res


if __name__ == '__main__':
    run(parse_args())