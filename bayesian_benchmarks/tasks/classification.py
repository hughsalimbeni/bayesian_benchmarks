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

from bayesian_benchmarks.data import ALL_CLASSIFICATION_DATATSETS
from bayesian_benchmarks.models.get_model import get_classification_model
from bayesian_benchmarks.database_utils import Database

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='variationally_sparse_gp', nargs='?', type=str)
    parser.add_argument("--dataset", default='statlog-german-credit', nargs='?', type=str)
    parser.add_argument("--split", default=0, nargs='?', type=int)
    return parser.parse_args()

def run(ARGS, is_test=False):
    data = ALL_CLASSIFICATION_DATATSETS[ARGS.dataset]()

    def onehot(Y, K):
        ret = np.zeros((len(Y), K))
        for k in range(K):
            ret[Y.flatten()==k, k] = 1.
        return ret

    Y_oh = onehot(data.Y_test, data.K)[None, :, :]  # 1, N_test, K

    Model = get_classification_model(ARGS.model)
    model = Model(data.K, is_test=is_test)
    model.fit(data.X_train, data.Y_train)
    p = model.predict(data.X_test)  # N_test, K

    # clip very large and small probs
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    p = p / np.expand_dims(np.sum(p, -1), -1)

    # evaluation metrics
    res = {}

    logp = multinomial.logpmf(Y_oh, n=1, p=p)

    res['test_loglik'] = np.average(logp)

    pred = np.argmax(p, axis=-1)

    res['test_acc'] = np.average(np.array(pred == data.Y_test.flatten()).astype(float))

    res['Y_test'] = data.Y_test
    res['p_test'] = p

    res.update(ARGS.__dict__)

    with Database() as db:
        db.write('classification', res)


if __name__ == '__main__':
    run(parse_args())