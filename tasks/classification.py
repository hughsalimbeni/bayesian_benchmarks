"""
A classification task, which can be either binary or multiclass.

Metrics reported are test loglikelihood, classification accuracy, precision, recall, AUC ROC

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


# ['heart-va', 200, 13, 5],
#     ['connect-4', 67557, 43, 2],
#     ['wine', 178, 14, 3],
#     ['tic-tac-toe', 958, 10, 2],
#     ['fertility', 100, 10, 2],
#     ['statlog-german-credit', 1000, 25, 2],
#     ['car', 1728, 7, 4],
#     ['libras', 360, 91, 15],
#     ['spambase', 4601, 58, 2],
#     ['pittsburg-bridges-MATERIAL', 106, 8, 3],
#     ['hepatitis', 155, 20, 2],
#     ['acute-inflammation', 120, 7, 2],
#     ['pittsburg-bridges-TYPE', 105, 8, 6],
#     ['arrhythmia', 452, 263, 13],
#     ['musk-2', 6598, 167, 2],
#     ['twonorm', 7400, 21, 2],
#     ['nursery', 12960, 9, 5],
#

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


import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

fraction_of_positives, mean_predicted_value = \
    calibration_curve(data.Y_test[:, 0], p[:, 1], n_bins=10)

ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label='asdf')

ax2.hist(p, range=(0, 1), bins=10, label='asdf', histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.show()


res['test_loglik'] = np.average(logp)

pred = np.argmax(p, axis=-1)

res['test_acc'] = np.average(np.array(pred == data.Y_test.flatten()).astype(float))

res.update(ARGS.__dict__)

print(res)

# with Database('../results/results.db') as db:
#     db.write('classification', res)
