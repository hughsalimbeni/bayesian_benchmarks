import argparse

from tasks.data import ALL_CLASSIFICATION_DATATSETS

import numpy as np
from tinydb import TinyDB


from scipy.stats import multinomial
from scipy.special import logsumexp

parser = argparse.ArgumentParser()

parser.add_argument("--model", default='linear', nargs='?', type=str)
parser.add_argument("--dataset", default='iris', nargs='?', type=str)
parser.add_argument("--split", default=0, nargs='?', type=int)
parser.add_argument("--num_samples", default=1, nargs='?', type=int)

ARGS = parser.parse_args()

path = '../baseline_models/{}/run.py'.format(ARGS.model)

data = ALL_CLASSIFICATION_DATATSETS[ARGS.dataset]()

exec(open(path).read())  # defines run_classification

predictions = run_classification(data.X_train, data.Y_train, data.X_test, S=ARGS.num_samples)

def onehot(Y, K):
    ret = np.zeros((len(Y), K))
    for k in range(K):
        ret[Y.flatten()==k, k] = 1.
    return ret

Y_oh = onehot(data.Y_test, data.K)

logp_SN = np.array([multinomial.logpmf(Y_oh, n=1, p=p) for p in predictions])
logp_N = logsumexp(logp_SN, b=1./float(ARGS.num_samples), axis=0)

test_lik = np.average(logp_N)

db = TinyDB('db.json')
d = {'test_likelihood':test_lik, 'task':'classification'}
d.update(ARGS.__dict__)
db.insert(d)




