import sys
sys.path.append('../')

import argparse
import numpy as np
from importlib import import_module

parser = argparse.ArgumentParser()

parser.add_argument("--model", default='linear', nargs='?', type=str)
parser.add_argument("--dataset", default='boston', nargs='?', type=str)
parser.add_argument("--split", default=0, nargs='?', type=int)
parser.add_argument("--levels", default=1000, nargs='?', type=int)

ARGS = parser.parse_args()

from bayesian_benchmarks.data import ALL_REGRESSION_DATATSETS
data = ALL_REGRESSION_DATATSETS[ARGS.dataset]()

run_path = '../models/{}/models.py'.format(ARGS.model)

Y_test = data.Y_test.flatten()  # N_test,
levels = np.linspace(np.min(Y_test), np.max(Y_test), ARGS.levels)

lower_ind = np.argmin(Y_test[None, :] > levels[:, None], 0) - 1
interp = []
for y, l in zip(Y_test, lower_ind):
    interp.append((y - levels[l])/(levels[l+1] - levels[l]))
interp = np.array(interp)

models = import_module('models.{}.models'.format(ARGS.model))

model = models.DensityEstimationModel()
model.fit(data.X_train, data.Y_train)

logps = model.predict(data.X_test, levels)


logp = interp[:, None] * logps[lower_ind, :, 0] + (1-interp)[:, None] * logps[lower_ind + 1, :, :]

print(logp.shape)
# test_lik = np.average(l)
#
#
# res = {}
# res['test_likelihood'] = test_lik
#
# print(test_lik)
#
# # res.update(ARGS.__dict__)
# # TinyDB('../results/results_db.json').table('density_estimation').insert(res)
