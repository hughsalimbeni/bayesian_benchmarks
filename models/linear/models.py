import numpy as np
from sklearn import linear_model
from scipy.stats import norm

# from bayesian_benchmarks.model_wrappers import RegressionModel, ClassificationModel, DensityEstimationModel


class RegressionModel(object):
    def __init__(self):
        self.model = linear_model.LinearRegression()
        self.std = None

    def fit(self, X, Y):
        self.model.fit(X, Y)
        self.std = np.std(self.model.predict(X) - Y)

    def predict(self, Xs):
        pred_mean = self.model.predict(Xs)
        return pred_mean, np.ones_like(pred_mean) * self.std ** 2


class ClassificationModel(object):
    def __init__(self, K):
        self.model = linear_model.LogisticRegression()
        self.K = K
        self.Ks_seen = []

    def fit(self, X, Y):
        self.Ks_seen = list(set(Y.flatten().astype(int)))
        self.Ks_seen.sort()
        self.model.fit(X, Y.ravel())

    def predict(self, Xs):
        ps = self.model.predict_proba(Xs)
        if len(self.Ks_seen) == self.K:
            return ps
        else:  # not all classes have been seen, and sklearn can only infer n_classes
            ret = np.zeros((len(ps), self.K))
            for k, p in zip(self.Ks_seen, ps.T):
                ret[:, k] = p
            return ret



class DensityEstimationModel(object):
    def fit(self, X, Y):
        raise NotImplementedError

    def predict(self, Xs, levels):
        raise NotImplementedError



# def run_density_estimation(X, Y, Xs, levels):
#     m, v = run_regression(X, Y, Xs)
#     logp = norm.logpdf(levels[:, None],
#                        loc=m.T,
#                        scale=v.T**0.5)
#     return logp



