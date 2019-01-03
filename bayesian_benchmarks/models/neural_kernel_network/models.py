"""
To add a model, create a new directory in bayesian_benchmarks.models (here) and make file models.py containing at
least one of the classes below.

Model usage is similar to sklearn. For regression:

model = RegressionModel(is_test=False)
model.fit(X, Y)
mean, var = model.predict(X_test)  # first two moments of the posterior
samples = model.sample(X_test, S)  # samples from the posterior

For classification:
model = ClassificationModel(K, is_test=False)  # K is the number of classes
model.fit(X, Y)
p = model.predict(X_test)          # predictive probabilities for each class (i.e. onehot)

It should be feasible to call fit and predict many times (e.g. avoid rebuilding a tensorflow graph on each call).

"""

import numpy as np
import gpflow
from gpflow import params_as_tensors
from scipy.cluster.vq import kmeans2

from .nkn import NeuralKernelNetwork, NKNWrapper, KernelWrapper
from .utils import median_distance_local, get_nkn_config


class RegressionModel:
    def __init__(self, is_test=False, seed=0, nkn='default', nkn_config=None):
        """
        If is_test is True your model should train and predict in a few seconds (i.e. suitable for travis)
        """
        if is_test:
            class ARGS:
                num_inducing = 2
                iterations = 2
                small_iterations = 1
                adam_lr = 0.01
                minibatch_size = 100
                num_posterior_samples = 2
                initial_likelihood_var = 0.01
        else:  # pragma: no cover
            class ARGS:
                num_inducing = 100
                iterations = 20000
                small_iterations = 1000
                adam_lr = 0.01
                minibatch_size = 1000
                num_posterior_samples = 1000
                initial_likelihood_var = 0.01
        self.ARGS = ARGS
        self.model = None
        self.nkn_config = nkn_config or get_nkn_config(nkn)

    def fit(self, X, Y):
        """
        Train the model (and probably create the model, too, since there is no shape information on the __init__)

        :param X: numpy array, of shape N, Dx
        :param Y: numpy array, of shape N, Dy
        :return:
        """
        initial_likelihood_var = self.ARGS.initial_likelihood_var
        class Lik(gpflow.likelihoods.Gaussian):
            def __init__(self):
                gpflow.likelihoods.Gaussian.__init__(self)
                self.variance = initial_likelihood_var
        return self._fit(X, Y, Lik)

    def _fit(self, X, Y, Lik):
        if X.shape[0] > self.ARGS.num_inducing:
            Z = kmeans2(X, self.ARGS.num_inducing, minit='points')[0]
        else:
            # pad with random values
            Z = np.concatenate([X, np.random.randn(self.ARGS.num_inducing - X.shape[0], X.shape[1])], 0)

        if not self.model:
            cf = self.nkn_config(X.shape[1], median_distance_local(X))
            kern = NeuralKernelNetwork(X.shape[1], KernelWrapper(cf['kern']), NKNWrapper(cf['nkn']))
            lik =  gpflow.likelihoods.Gaussian()
            lik.variance = self.ARGS.initial_likelihood_var

            # kern = gpflow.kernels.RBF(X.shape[1], lengthscales=float(X.shape[1])**0.5)

            self.model = gpflow.models.SGPR(X, Y, kern, feat=Z)
            self.model.likelihood.variance = lik.variance.read_value()
            self.adam = gpflow.train.AdamOptimizer(self.ARGS.adam_lr).make_optimize_tensor(self.model)
            self.sess = self.model.enquire_session()
            iters = self.ARGS.iterations
        else:
            iters = self.ARGS.small_iterations

        self.model.X.assign(X, session=self.sess)
        self.model.Y.assign(Y, session=self.sess)
        self.model.feature.Z.assign(Z, session=self.sess)

        try:
            for _ in range(iters):
                if _ % 100 == 0:
                    print('{} {}'.format(_, self.sess.run(self.model.likelihood_tensor)))
                self.sess.run(self.adam)
        except KeyboardInterrupt:  # pragma: no cover
            pass

        self.model.anchor(session=self.sess)

    def predict(self, Xs):
        return self.model.predict_y(Xs, session=self.sess)

    def sample(self, Xs, num_samples):
        m, v = self.predict(Xs)
        N, D = np.shape(m)
        m, v = np.expand_dims(m, 0), np.expand_dims(v, 0)
        return m + np.random.randn(num_samples, N, D) * (v ** 0.5)


class ClassificationModel:
    def __init__(self, is_test=False, seed=0, nkn='default', nkn_config=None):
        """
        If is_test is True your model should train and predict in a few seconds (i.e. suitable for travis)
        """
        if is_test:
            class ARGS:
                num_inducing = 2
                iterations = 2
                small_iterations = 1
                adam_lr = 0.01
                minibatch_size = 100
                num_posterior_samples = 2
                initial_likelihood_var = 0.01
        else:  # pragma: no cover
            class ARGS:
                num_inducing = 100
                iterations = 20000
                small_iterations = 1000
                adam_lr = 0.01
                minibatch_size = 1000
                num_posterior_samples = 1000
                initial_likelihood_var = 0.01
        self.ARGS = ARGS
        self.model = None
        self.nkn_config = nkn_config or get_nkn_config(nkn)

    def fit(self, X, Y):
        """
        Train the model (and probably create the model, too, since there is no shape information on the __init__)

        :param X: numpy array, of shape N, Dx
        :param Y: numpy array, of shape N, Dy
        :return:
        """
        initial_likelihood_var = self.ARGS.initial_likelihood_var
        class Lik(gpflow.likelihoods.Gaussian):
            def __init__(self):
                gpflow.likelihoods.Gaussian.__init__(self)
                self.variance = initial_likelihood_var
        return self._fit(X, Y, Lik)

    def _fit(self, X, Y, Lik):
        if X.shape[0] > self.ARGS.num_inducing:
            Z = kmeans2(X, self.ARGS.num_inducing, minit='points')[0]
        else:
            # pad with random values
            Z = np.concatenate([X, np.random.randn(self.ARGS.num_inducing - X.shape[0], X.shape[1])], 0)

        if not self.model:
            cf = self.nkn_config(X.shape[1], median_distance_local(X))
            kern = NeuralKernelNetwork(X.shape[1], KernelWrapper(cf['kern']), NKNWrapper(cf['nkn']))

            if self.K == 2:
                lik = gpflow.likelihoods.Bernoulli()
                num_latent = 1
            else:
                lik = gpflow.likelihoods.MultiClass(self.K)
                num_latent = self.K

            # kern = gpflow.kernels.RBF(X.shape[1], lengthscales=float(X.shape[1])**0.5)

            self.model = gpflow.models.SVGP(X, Y, kern, lik,
                                            feat=Z,
                                            whiten=False,
                                            num_latent=num_latent,
                                            minibatch_size=None)
            self.adam = gpflow.train.AdamOptimizer(self.ARGS.adam_lr).make_optimize_tensor(self.model)
            self.sess = self.model.enquire_session()
            iters = self.ARGS.iterations
        else:
            iters = self.ARGS.small_iterations

        self.model.X.assign(X, session=self.sess)
        self.model.Y.assign(Y, session=self.sess)
        self.model.feature.Z.assign(Z, session=self.sess)

        try:
            for _ in range(iters):
                if _ % 100 == 0:
                    print('{} {}'.format(_, self.sess.run(self.model.likelihood_tensor)))
                self.sess.run(self.adam)
        except KeyboardInterrupt:  # pragma: no cover
            pass

        self.model.anchor(session=self.sess)

    def predict(self, Xs):
        return self.model.predict_y(Xs, session=self.sess)
