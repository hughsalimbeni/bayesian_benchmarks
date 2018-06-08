import argparse

import numpy as np
from tinydb import TinyDB

parser = argparse.ArgumentParser()

parser.add_argument("--model", default='linear', nargs='?', type=str)
parser.add_argument("--dataset", default='boston', nargs='?', type=str)
parser.add_argument("--split", default=0, nargs='?', type=int)
parser.add_argument("--levels", default=1000, nargs='?', type=int)

ARGS = parser.parse_args()

from tasks.data import ALL_REGRESSION_DATATSETS
data = ALL_REGRESSION_DATATSETS[ARGS.dataset]()

run_path = '../baseline_models/{}/run.py'.format(ARGS.model)

run_density_estimation = None
exec(open(run_path).read()) # this should redefine run_regression

if not run_density_estimation:
    raise NotImplementedError

levels = np.linspace(np.min(data.Y_test), np.max(data.Y_test), ARGS.levels)

a = np.argmin((levels[:, None] - data.Y_test.flatten()[None, :])**2, 0)

logps = run_density_estimation(data.X_train, data.Y_train, data.X_test, levels)
l = []
for logp, aa in zip(logps.T, a):
    l.append(logp[aa])

test_lik = np.average(l)

db = TinyDB('db.json')
d = {'test_likelihood':test_lik, 'task':'density_estimation'}
d.update(ARGS.__dict__)
db.insert(d)
