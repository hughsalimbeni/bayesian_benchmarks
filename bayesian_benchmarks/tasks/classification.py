"""
A classification task, which can be either binary or multiclass.

Metrics reported are test loglikelihood, classification accuracy. Also the predictions are stored for 
analysis of calibration etc. 

"""

import sys
sys.path.append('../')

import argparse
import numpy as np

from scipy.stats import multinomial
from scipy.special import logsumexp

from bayesian_benchmarks.data import get_classification_data
from bayesian_benchmarks.models.get_model import get_classification_model
from bayesian_benchmarks.database_utils import Database

def parse_args():  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='variationally_sparse_gp', nargs='?', type=str)
    parser.add_argument("--dataset", default='statlog-german-credit', nargs='?', type=str)
    parser.add_argument("--split", default=0, nargs='?', type=int)
    parser.add_argument("--seed", default=0, nargs='?', type=int)
    parser.add_argument("--database_path", default='', nargs='?', type=str)
    return parser.parse_args()

def run(ARGS, data=None, model=None, is_test=False):

    data = data or get_classification_data(ARGS.dataset, split=ARGS.split)
    model = model or get_classification_model(ARGS.model)(data.K, is_test=is_test, seed=ARGS.seed)

    def onehot(Y, K):
        return np.eye(K)[Y.flatten().astype(int)].reshape(Y.shape[:-1]+(K,))

    Y_oh = onehot(data.Y_test, data.K)[None, :, :]  # [1 x N_test x K]

    model.fit(data.X_train, data.Y_train)
    p = model.predict(data.X_test)  # [N_test x K] or [samples x N_test x K]

    assert len(p.shape) in {2, 3}  # 3-dim in case of approximate predictions (multiple samples per each X)

    # clip very large and small probs
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    p = p / np.expand_dims(np.sum(p, -1), -1)

    assert np.all(p >= 0.0) and np.all(p <= 1.0)

    # evaluation metrics
    res = {}

    if len(p.shape) == 2:  # keep analysis as in the original code in case 2-dim predictions

        logp = multinomial.logpmf(Y_oh, n=1, p=p)

        res['test_loglik'] = np.average(logp)

        pred = np.argmax(p, axis=-1)

        res['test_acc'] = np.average(np.array(pred == data.Y_test.flatten()).astype(float))

        res['Y_test'] = data.Y_test
        res['p_test'] = p

    else:  # compute metrics in case of 3-dim predictions

        res['test_loglik'] = []
        res['Y_test'] = data.Y_test

        for n in range(p.shape[0]):  # iterate through samples
            logp = multinomial.logpmf(Y_oh, n=1, p=p[n])
            res['test_loglik'].append(logp)

        logp = logsumexp(res['test_loglik'], axis=0) - np.log(p.shape[0])
        res['test_loglik'] = np.mean(logp)

        p = np.exp(logp)
        pred = np.argmax(p, axis=-1)

        res['test_acc'] = np.average(np.array(pred == data.Y_test.flatten()).astype(float))
        res['p_test'] = p

    res.update(ARGS.__dict__)

    if not is_test:  # pragma: no cover
        with Database(ARGS.database_path) as db:
            db.write('classification', res)

    return res 


if __name__ == '__main__':
    run(parse_args())