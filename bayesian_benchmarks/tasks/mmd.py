"""
A conditional Gaussian estimation task: model p(y_n|x_n) = N(a(x_n), b(x_n))

Metrics reported are test log likelihood, mean squared error, and absolute error, all for normalized and unnormalized y.

"""

import argparse
import numpy as np
from sklearn.decomposition.pca import PCA

from bayesian_benchmarks.database_utils import Database
from bayesian_benchmarks.models.get_model import get_regression_model
from bayesian_benchmarks.data import get_regression_data

import gpflow

def parse_args():  #prama: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='variationally_sparse_gp', nargs='?', type=str)
    parser.add_argument("--dataset", default='yacht', nargs='?', type=str)
    parser.add_argument("--split", default=0, nargs='?', type=int)
    parser.add_argument("--seed", default=0, nargs='?', type=int)
    parser.add_argument("--num_samples", default=100, nargs='?', type=int)
    parser.add_argument("--pca_dim", default=2, type=int)
    parser.add_argument("--database_path", default='', nargs='?', type=str)
    return parser.parse_args()


def mmd(A, B, kernel):
    KAA = kernel.compute_K_symm(A)
    KAB = kernel.compute_K(A, B)
    KBB = kernel.compute_K_symm(A)

    n = float(A.shape[0])
    m = float(B.shape[0])

    return np.sum(KAA)/m/m - 2*np.sum(KAB)/m/n + np.sum(KBB)/n/n

def run(ARGS, is_test=False):
    data = get_regression_data(ARGS.dataset, split=ARGS.split)

    Model = get_regression_model(ARGS.model)

    model = Model(is_test=is_test, seed=ARGS.seed)
    model.fit(data.X_train, data.Y_train)

    res = {}

    samples = model.sample(data.X_test, ARGS.num_samples)
    data_tiled = np.tile(data.X_test[None, :, :], [ARGS.num_samples, 1, 1])
    shape =  [ARGS.num_samples * data.X_test.shape[0], data.X_test.shape[1] + data.Y_test.shape[1]]
    A = np.reshape(np.concatenate([data_tiled, samples], -1), shape)
    B = np.concatenate([data.X_test, data.Y_test], -1)


    if ARGS.pca_dim > 0:
        AB = np.concatenate([A, B], 0)
        pca = PCA(n_components=ARGS.pca_dim).fit(AB)
        A = pca.transform(A)
        B = pca.transform(B)

    # import matplotlib.pyplot as plt
    # plt.scatter(A[:, 0], A[:, 1], color='b')
    # plt.scatter(B[:, 0], B[:, 1], color='r')
    # plt.show()

    kernel = gpflow.kernels.RBF(A.shape[-1])
    res['mmd'] = mmd(A, B, kernel)

    print(res)

    res.update(ARGS.__dict__)
    if not is_test:  # prgama: no cover
        with Database(ARGS.database_path) as db:
            db.write('mmd', res)


if __name__ == '__main__':
    run(parse_args())