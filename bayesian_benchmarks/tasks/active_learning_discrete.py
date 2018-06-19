"""
Active learning with labels, using the max entropy criterion to select new points

"""

import sys
sys.path.append('../')

import argparse
import numpy as np
from scipy.stats import multinomial
from importlib import import_module

from bayesian_benchmarks.data import ALL_CLASSIFICATION_DATATSETS
from bayesian_benchmarks.database_utils import Database
from bayesian_benchmarks.models.non_bayesian_models import non_bayesian_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='variationally_sparse_gp', nargs='?', type=str)
    parser.add_argument("--dataset", default='statlog-landsat', nargs='?', type=str)
    parser.add_argument("--split", default=0, nargs='?', type=int)
    parser.add_argument("--iterations", default=10, nargs='?', type=int)
    parser.add_argument("--num_initial_points", default=3, nargs='?', type=int)
    return parser.parse_args()

def run(ARGS, is_test=False):

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

    Model = non_bayesian_model(ARGS.model, 'classification') or\
            import_module('bayesian_benchmarks.models.{}.models'.format(ARGS.model)).ClassificationModel
    model = Model(data.K, is_test=is_test)

    test_ll = []
    train_ll = []
    all_ll = []
    test_acc = []
    train_acc = []
    all_acc = []

    for _ in range(min(ARGS.iterations, X.shape[0] - ARGS.num_initial_points)):
        model.fit(X[ind], Y[ind])

        p = model.predict(X)  # NK
        # clip very large and small probs
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        p = p / np.expand_dims(np.sum(p, -1), -1)

        # entropy of predictions at all points
        ent = multinomial.entropy(n=1, p=p)

        # set the seen ones to -inf so we don't choose them
        ent[ind] = -np.inf

        # choose the highest entropy point to see next
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
    res = {'test_loglik':np.array(test_ll),
          'train_loglik':np.array(train_ll),
          'total_loglik':np.array(all_ll),
          'test_acc':np.array(test_acc),
          'train_acc':np.array(train_acc),
          'total_acc':np.array(all_acc),
         }
    res.update(ARGS.__dict__)

    with Database() as db:
        db.write('active_learning_discrete', res)

if __name__ == '__main__':
    run(parse_args())