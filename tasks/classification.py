import sys
sys.path.append('../')

import argparse
import numpy as np
from tinydb import TinyDB
from importlib import import_module

from scipy.special import logsumexp
from scipy.stats import multinomial

from tasks.data import ALL_CLASSIFICATION_DATATSETS

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='variationally_sparse_gp', nargs='?', type=str)
parser.add_argument("--dataset", default='acute-nephritis', nargs='?', type=str)
parser.add_argument("--split", default=0, nargs='?', type=int)
ARGS = parser.parse_args()

data = ALL_CLASSIFICATION_DATATSETS[ARGS.dataset]()

def onehot(Y, K):
    ret = np.zeros((len(Y), K))
    for k in range(K):
        ret[Y.flatten()==k, k] = 1.
    return ret

Y_oh = onehot(data.Y_test, data.K)[None, :, :]  # 1, N_test, K

models = import_module('models.{}.models'.format(ARGS.model))

model = models.ClassificationModel(data.K)
model.fit(data.X_train, data.Y_train)
p = model.predict(data.X_test)  # S, N_test, K or N_test, K

if len(p.shape) == 2:
    p = np.expand_dims(p, 0)

# evaluation metrics
res = {}
logp_SN = multinomial.logpmf(Y_oh, n=1, p=p)
logp_N = logsumexp(logp_SN, axis=0, b=1./len(Y_oh))
res['test_loglik'] = np.average(logp_N)

pred_SN = np.argmax(p, -1)
pred_N = np.median(pred_SN, axis=0)
res['test_acc'] = np.average(np.array(pred_N == data.Y_test.flatten()).astype(float))

res.update(ARGS.__dict__)
TinyDB('../results/results_db.json').table('classification').insert(res)
