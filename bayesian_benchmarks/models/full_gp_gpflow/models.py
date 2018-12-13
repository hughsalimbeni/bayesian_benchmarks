import gpflow
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.stats import norm

class RegressionModel(object):
    def __init__(self, is_test=False, seed=0):
        if is_test:
            class ARGS:
                iterations = 1
                small_iterations = 1
                initial_likelihood_var = 0.01
        else:  # pragma: no cover
            class ARGS:
                iterations = 10000
                small_iterations = 1000
                initial_likelihood_var = 0.01
        self.ARGS = ARGS
        self.model = None

    def fit(self, X, Y):
        # make model if necessary
        if not self.model:
            kern = gpflow.kernels.RBF(X.shape[1], lengthscales=float(X.shape[1])**0.5)
            lik = gpflow.likelihoods.Gaussian()
            lik.variance = self.ARGS.initial_likelihood_var

            self.model = gpflow.models.GPR(X, Y, kern)
            self.model.likelihood.variance = lik.variance.read_value()
            self.sess = self.model.enquire_session()
            self.opt = gpflow.train.ScipyOptimizer()

        # we might have new data
        self.model.X.assign(X, session=self.sess)
        self.model.Y.assign(Y, session=self.sess)

        self.opt.minimize(self.model, session=self.sess, maxiter=self.ARGS.iterations)

    def predict(self, Xs):
        return self.model.predict_y(Xs, session=self.sess)

    def sample(self, Xs, num_samples):
        m, v = self.predict(Xs)
        N, D = np.shape(m)
        m, v = np.expand_dims(m, 0), np.expand_dims(v, 0)
        return m + np.random.randn(num_samples, N, D) * (v ** 0.5)
