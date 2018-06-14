"""
A classification task, which can be either binary or multiclass.

Metrics reported are test loglikelihood, classification accuracy (precision, recall, AUC ROC TODO)

"""

import sys
sys.path.append('../')

import argparse
import numpy as np
from database_utils import Database

from importlib import import_module

from scipy.stats import multinomial

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from tasks.data import ALL_CLASSIFICATION_DATATSETS
from models.non_bayesian_models import non_bayesian_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='variationally_sparse_gp', nargs='?', type=str)
parser.add_argument("--dataset", default='statlog-german-credit', nargs='?', type=str)
parser.add_argument("--split", default=0, nargs='?', type=int)
ARGS = parser.parse_args()

data = ALL_CLASSIFICATION_DATATSETS[ARGS.dataset]()

def onehot(Y, K):
    ret = np.zeros((len(Y), K))
    for k in range(K):
        ret[Y.flatten()==k, k] = 1.
    return ret

Y_oh = onehot(data.Y_test, data.K)[None, :, :]  # 1, N_test, K

Model = non_bayesian_model(ARGS.model, 'classification') or\
        import_module('models.{}.models'.format(ARGS.model)).ClassificationModel
model = Model(data.K)

# model = models.ClassificationModel(data.K)
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

res.update(ARGS.__dict__)

with Database('../results/results.db') as db:
    db.write('classification', res)
