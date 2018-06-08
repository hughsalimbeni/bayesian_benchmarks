import argparse

from tinydb import TinyDB

import numpy as np
from scipy.stats import norm

parser = argparse.ArgumentParser()

parser.add_argument("--model", default='linear', nargs='?', type=str)
parser.add_argument("--dataset", default='boston', nargs='?', type=str)
parser.add_argument("--split", default=0, nargs='?', type=int)

ARGS = parser.parse_args()

from data import ALL_REGRESSION_DATATSETS
data = ALL_REGRESSION_DATATSETS[ARGS.dataset](split=ARGS.split)

run_path = '../baseline_models/{}/run.py'.format(ARGS.model)

exec(open(run_path).read())

pred_mean, pred_var = run_regression(data.X_train, data.Y_train, data.X_test)

test_lik = np.average(norm.logpdf(data.Y_test, loc=pred_mean, scale=pred_var**0.5))

db = TinyDB('../results/db.json')
d = {'test_likelihood':test_lik, 'task':'regression'}
d.update(ARGS.__dict__)
db.insert(d)






