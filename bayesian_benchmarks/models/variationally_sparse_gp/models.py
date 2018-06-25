import gpflow
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.stats import norm

class RegressionModel(object):
    def __init__(self, is_test=False, seed=0):
        if is_test:
            class ARGS:
                num_inducing = 2
                iterations = 1
                small_iterations = 1
                initial_likelihood_var = 0.01
        else:  # pragma: no cover
            class ARGS:
                num_inducing = 100
                iterations = 10000
                small_iterations = 1000
                initial_likelihood_var = 0.01
        self.ARGS = ARGS
        self.model = None

    def fit(self, X, Y):
        if X.shape[0] > self.ARGS.num_inducing:
            Z = kmeans2(X, self.ARGS.num_inducing, minit='points')[0]
        else:
            # pad with random values
            Z = np.concatenate([X, np.random.randn(self.ARGS.num_inducing - X.shape[0], X.shape[1])], 0)

        # make model if necessary
        if not self.model:
            kern = gpflow.kernels.RBF(X.shape[1], lengthscales=float(X.shape[1])**0.5)
            lik = gpflow.likelihoods.Gaussian()
            lik.variance = self.ARGS.initial_likelihood_var

            self.model = gpflow.models.SGPR(X, Y, kern, feat=Z)
            self.model.likelihood.variance = lik.variance.read_value()
            self.sess = self.model.enquire_session()
            self.opt = gpflow.train.ScipyOptimizer()

        # we might have new data
        self.model.X.assign(X, session=self.sess)
        self.model.Y.assign(Y, session=self.sess)
        self.model.feature.Z.assign(Z, session=self.sess)

        self.opt.minimize(self.model, session=self.sess, maxiter=self.ARGS.iterations)

    def predict(self, Xs):
        return self.model.predict_y(Xs, session=self.sess)

    def sample(self, Xs, num_samples):
        m, v = self.predict(Xs)
        N, D = np.shape(m)
        m, v = np.expand_dims(m, 0), np.expand_dims(v, 0)
        return m + np.random.randn(num_samples, N, D) * (v ** 0.5)


class ClassificationModel(object):
    def __init__(self, K, is_test=False, seed=0):
        if is_test:
            class ARGS:
                num_inducing = 2
                iterations = 1
                small_iterations = 1
                initial_likelihood_var = 0.01
        else:  # pragma: no cover
            class ARGS:
                num_inducing = 100
                iterations = 5000
                small_iterations = 1000
                initial_likelihood_var = 0.01

        self.ARGS = ARGS
        self.K = K
        self.model = None

    def fit(self, X, Y):
        Z = kmeans2(X, self.ARGS.num_inducing, minit='points')[0] if X.shape[0] > self.ARGS.num_inducing else X.copy()

        if not self.model:
            if self.K == 2:
                lik = gpflow.likelihoods.Bernoulli()
                num_latent = 1
            else:
                lik = gpflow.likelihoods.MultiClass(self.K)
                num_latent = self.K

            kern = gpflow.kernels.RBF(X.shape[1], lengthscales=float(X.shape[1]) ** 0.5)
            self.model = gpflow.models.SVGP(X, Y, kern, lik,
                                            feat=Z,
                                            whiten=False,
                                            num_latent=num_latent,
                                            minibatch_size=None)

            self.sess = self.model.enquire_session()
            self.opt = gpflow.train.ScipyOptimizer()

            iters = self.ARGS.iterations

        else:
            iters = self.ARGS.small_iterations

        # we might have new data
        self.model.X.assign(X, session=self.sess)
        self.model.Y.assign(Y, session=self.sess)
        self.model.feature.Z.assign(Z, session=self.sess)

        num_outputs = self.model.q_sqrt.shape[0]
        self.model.q_mu.assign(np.zeros((self.ARGS.num_inducing, num_outputs)), session=self.sess)
        self.model.q_sqrt.assign(np.tile(np.eye(self.ARGS.num_inducing)[None], [num_outputs, 1, 1]), session=self.sess)

        self.opt.minimize(self.model, maxiter=iters, session=self.sess)

    def predict(self, Xs):
        m, v = self.model.predict_y(Xs, session=self.sess)
        if self.K == 2:
            # convert Bernoulli to onehot
            return np.concatenate([1 - m, m], 1)
        else:
            return m



