import argparse
import numpy as np
from scipy.stats import norm
from tinydb import TinyDB

from data import ALL_REGRESSION_DATATSETS

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='linear', nargs='?', type=str)
parser.add_argument("--dataset", default='boston', nargs='?', type=str)
parser.add_argument("--split", default=0, nargs='?', type=int)
ARGS = parser.parse_args()


data = ALL_REGRESSION_DATATSETS[ARGS.dataset](split=ARGS.split)

run_path = '../baseline_models/{}/run.py'.format(ARGS.model)

exec(open(run_path).read())

pred_mean, pred_var = run_regression(data.X_train, data.Y_train, data.X_test)

# evaluation metrics
test_lik = np.average(norm.logpdf(data.Y_test, loc=pred_mean, scale=pred_var**0.5))
test_lik_unnormalized = np.average(norm.logpdf(data.Y_test * data.Y_std,
                                               loc=pred_mean * data.Y_std,
                                               scale=pred_var**0.5 * data.Y_std))

test_mae = np.average(np.abs(data.Y_test - pred_mean))
test_rmse = np.average((data.Y_test - pred_mean)**2)**0.5


# save
db = TinyDB('../results/db.json')
d = {'task':'regression',
     'test_loglikelihood':test_lik,
     'test_loglikelihood_unnormalized':test_lik_unnormalized,
     'test_mae':test_mae,
     'test_rmse':test_rmse,
     }
d.update(ARGS.__dict__)
db.insert(d)






