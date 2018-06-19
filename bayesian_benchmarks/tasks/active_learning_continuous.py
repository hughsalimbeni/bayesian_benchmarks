"""
Active learning for continuous data, using the max variance criterion to select new points

"""

import sys
sys.path.append('../')

import argparse
import numpy as np
from scipy.stats import norm
from importlib import import_module

from data import ALL_REGRESSION_DATATSETS
from database_utils import Database
from bayesian_benchmarks.models.non_bayesian_models import non_bayesian_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='variationally_sparse_gp', nargs='?', type=str)
parser.add_argument("--dataset", default='boston', nargs='?', type=str)
parser.add_argument("--split", default=0, nargs='?', type=int)
parser.add_argument("--iterations", default=10, nargs='?', type=int)
parser.add_argument("--num_initial_points", default=3, nargs='?', type=int)

ARGS = parser.parse_args()

data = ALL_REGRESSION_DATATSETS[ARGS.dataset](split=ARGS.split, prop=1.)

ind = np.zeros(data.X_train.shape[0]).astype(bool)
ind[:ARGS.num_initial_points] = True

X, Y = data.X_train, data.Y_train

Model = non_bayesian_model(ARGS.model, 'regression') or\
        import_module('models.{}.models'.format(ARGS.model)).RegressionModel
model = Model()

test_ll = []
train_ll = []
all_ll = []
test_rmse = []
train_rmse = []
all_rmse = []

for _ in range(min(ARGS.iterations, X.shape[0] - ARGS.num_initial_points)):
    model.fit(X[ind], Y[ind])

    m, v = model.predict(X)  # ND

    vars = v.copy()

    # set the seen ones to -inf so we don't choose them
    vars[ind] = -np.inf

    # choose the highest variance point
    i = np.argmax(vars)
    ind[i] = True

    logp = norm.logpdf(Y, loc=m, scale=v**0.5)  # N
    d2 = (Y - m)**2

    test_ll.append(np.average(logp[np.invert(ind)]))
    train_ll.append(np.average(logp[ind]))
    all_ll.append(np.average(logp))
    test_rmse.append(np.average(d2[np.invert(ind)])**0.5)
    train_rmse.append(np.average(d2[ind])**0.5)
    all_rmse.append(np.average(d2)**0.5)


# save
res = {'test_loglik':np.array(test_ll),
      'train_loglik':np.array(train_ll),
      'total_loglik':np.array(all_ll),
      'test_rmse':np.array(test_rmse),
      'train_rmse':np.array(train_rmse),
      'total_rmse':np.array(all_rmse),
     }
res.update(ARGS.__dict__)

with Database('../results/results.db') as db:
    db.write('active_learning_continuous', res)
