import gpflow
from doubly_stochastic_dgp.dgp import DGP

import numpy as np

from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from scipy.special import logsumexp

num_inducing = 100
iterations = 20000
small_iterations = 1000
adam_lr = 0.01
gamma = 0.1
minibatch_size = 1000
num_posterior_samples = 1000
initial_likelihood_var = 0.01


class RegressionModel(object):
    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        class Lik(gpflow.likelihoods.Gaussian):
            def __init__(self):
                gpflow.likelihoods.Gaussian.__init__(self)
                self.variance = initial_likelihood_var
        return self._fit(X, Y, Lik)

    def _fit(self, X, Y, Lik, **kwargs):
        if not self.model:
            ## build the model
            kerns = []
            for _ in range(2):
                kerns.append(gpflow.kernels.RBF(X.shape[1], lengthscales=float(X.shape[1])**0.5))

            if X.shape[0] > num_inducing:
                Z = kmeans2(X, num_inducing, minit='points')[0]
            else:
                # pad with random values
                Z = np.concatenate([X, np.random.randn(num_inducing - X.shape[0], X.shape[1])], 0)

            mb_size = minibatch_size if X.shape[0] > 5000 else None

            self.model = DGP(X, Y, Z, kerns, Lik(),
                             minibatch_size=mb_size,
                             **kwargs)

            self.model.layers[0].q_sqrt = self.model.layers[0].q_sqrt.read_value() * 1e-5

            if False:#isinstance(self.model.likelihood, gpflow.likelihoods.Gaussian):
                var_list = [[self.model.layers[-1].q_mu, self.model.layers[-1].q_sqrt]]
                self.model.layers[-1].q_mu.set_trainable(False)
                self.model.layers[-1].q_sqrt.set_trainable(False)
                self.ng = gpflow.train.NatGradOptimizer(gamma=gamma).make_optimize_tensor(self.model, var_list=var_list)
            else:
                self.ng = None

            self.adam = gpflow.train.AdamOptimizer(adam_lr).make_optimize_tensor(self.model)

            iters = iterations
        else:
            iters = small_iterations  # after first time use fewer iterations

        # we might have new data
        self.model.X = X
        self.model.Y = Y

        sess = self.model.enquire_session()

        try:
            for _ in range(iters):

                if _ % 100 == 0:
                    print('{} {}'.format(_, sess.run(self.model.likelihood_tensor)))
                if self.ng:
                    sess.run(self.ng)
                sess.run(self.adam)

        except KeyboardInterrupt:
            pass

        self.model.anchor(session=sess)

    def _predict(self, Xs, S):
        ms, vs = [], []
        n = max(len(Xs) / 100, 1)  # predict in small batches
        for xs in np.array_split(Xs, n):
            m, v = self.model.predict_y(xs, S)
            ms.append(m)
            vs.append(v)

        return np.concatenate(ms, 1), np.concatenate(vs, 1)  # num_posterior_samples, N_test, D_y

    def predict(self, Xs):
        ms, vs = self._predict(Xs, num_posterior_samples)

        # the first two moments
        m = np.average(ms, 0)
        v = np.average(vs + ms**2, 0) - m**2
        return m, v

    def sample(self, Xs, S):
        ms, vs = self._predict(Xs, S)
        return ms + vs**0.5 * np.random.randn(*ms.shape)


class ClassificationModel(RegressionModel):
    def __init__(self, K):
        self.K = K
        self.model = None

    def fit(self, X, Y):
        if self.K == 2:
            Lik = gpflow.likelihoods.Bernoulli
            num_latent = 1
        else:
            K = self.K
            class Lik(gpflow.likelihoods.MultiClass):
                def __init__(self):
                    gpflow.likelihoods.MultiClass.__init__(self, K)
            num_latent = K

        return self._fit(X, Y, Lik, num_outputs=num_latent)


    def predict(self, Xs):
        m, v = self.model.predict_y(Xs, num_posterior_samples)  # num_samples, N_test, K
        m = np.average(m, 0)
        if self.K == 2:
            # Bernoulli only gives one output, so append the complement to work with scipy.stats.multinomial
            return np.concatenate([1-m, m], -1)
        else:
            return m
