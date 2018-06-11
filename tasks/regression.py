import argparse
import numpy as np
from scipy.stats import norm
from tinydb import TinyDB
from importlib import import_module
from scipy.special import logsumexp

import sys
sys.path.append('../')

from data import ALL_REGRESSION_DATATSETS

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='linear', nargs='?', type=str)
parser.add_argument("--dataset", default='boston', nargs='?', type=str)
parser.add_argument("--split", default=0, nargs='?', type=int)
ARGS = parser.parse_args()

data = ALL_REGRESSION_DATATSETS[ARGS.dataset](split=ARGS.split)

Y_std = data.Y_std.flatten()[None, None, :]  # 1, 1, D_y,
Y_test = data.Y_test[None, :, :]  # 1, N_test, D_y

models = import_module('models.{}.models'.format(ARGS.model))

model = models.RegressionModel()
model.fit(data.X_train, data.Y_train)
m, v = model.predict(data.X_test)

# shape is either (N_test, Dy) or (S, N_test, Dy)
if len(m.shape) == 2:
    m = np.expand_dims(m, 0)
    v = np.expand_dims(v, 0)

# evaluation metrics
# average over samples, assuming equally weighted i.e. simple MC
calculate_lik = lambda y, m, s: np.average(logsumexp(norm.logpdf(y, loc=m, scale=s), axis=0, b=1./m.shape[0]))

res = {}
res['test_loglik'] = calculate_lik(Y_test, m, v**0.5)
res['test_loglik_unnormalized'] = calculate_lik(Y_test * Y_std, m * Y_std, (v**0.5) * Y_std)

d = data.Y_test - m
du = d * Y_std

res['test_mae'] = np.average(np.abs(d))
res['test_mae_unnormalized'] = np.average(np.abs(du))

res['test_rmse'] = np.average(d**2)**0.5
res['test_rmse_unnormalized'] = np.average(du**2)**0.5
res['asdf'] = 0.

# save
db = TinyDB('../results/results_db.json').table('regression')
res.update(ARGS.__dict__)
db.insert(res)
