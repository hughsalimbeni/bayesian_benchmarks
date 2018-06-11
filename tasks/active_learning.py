import argparse
import numpy as np
from tinydb import TinyDB
from scipy.stats import multinomial
from importlib import import_module
import sys
sys.path.append('../')

from data import ALL_CLASSIFICATION_DATATSETS

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='variationally_sparse_gp', nargs='?', type=str)
parser.add_argument("--dataset", default='statlog-landsat', nargs='?', type=str)
parser.add_argument("--split", default=0, nargs='?', type=int)
parser.add_argument("--iterations", default=10, nargs='?', type=int)
parser.add_argument("--num_initial_points", default=3, nargs='?', type=int)

ARGS = parser.parse_args()

data = ALL_CLASSIFICATION_DATATSETS[ARGS.dataset](split=ARGS.split, prop=1.)

ind = np.zeros(data.X_train.shape[0]).astype(bool)
ind[:ARGS.num_initial_points] = True

X, Y = data.X_train, data.Y_train

def onehot(Y, K):
    ret = np.zeros((len(Y), K))
    for k in range(K):
        ret[Y.flatten()==k, k] = 1.
    return ret

Y_oh = onehot(Y, data.K)

models = import_module('models.{}.models'.format(ARGS.model))
model = models.ClassificationModel(data.K)

test_ll = []
train_ll = []
all_ll = []
test_acc = []
train_acc = []
all_acc = []

for i in range(min(ARGS.iterations, X.shape[0] - ARGS.num_initial_points)):
    model.fit(X[ind], Y[ind])

    p = model.predict(X)  # NK

    # entropy of predictions at the the unseen points
    ent = multinomial.entropy(n=1, p=p[np.invert(ind)])  # N

    # choose the highest entropy point to see
    i = np.argmax(ent)
    ind[i] = True

    logp = multinomial.logpmf(Y_oh, n=1, p=p)  # N
    is_correct = (np.argmax(p, 1) == Y.flatten())  # N

    test_ll.append(np.average(logp[np.invert(ind)]))
    train_ll.append(np.average(logp[ind]))
    all_ll.append(np.average(logp))
    test_acc.append(np.average(is_correct[np.invert(ind)]))
    train_acc.append(np.average(is_correct[ind]))
    all_acc.append(np.average(is_correct))


# save
db = TinyDB('../results/results_db.json').table('active_learning')
d = {'test_loglik':test_ll,
     'total_loglik':train_ll,
     'test_loglik':all_ll,
     'test_acc':test_acc,
     'train_acc':train_acc,
     'total_acc':all_acc,
     }
d.update(ARGS.__dict__)
db.insert(d)
