import gpflow
from doubly_stochastic_dgp.dgp import DGP

import numpy as np
from scipy.cluster.vq import kmeans2

num_inducing = 100
iterations = 5000
adam_lr = 0.001
gamma = 0.01
minibatch_size = 1000
num_posterior_samples = 1000
initial_likelihood_var = 0.1


class RegressionModel(object):
    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        class Lik(gpflow.likelihoods.Gaussian):
            def __init__(self):
                gpflow.likelihoods.Gaussian.__init__(self)
                self.variance = initial_likelihood_var
        return self.fit_with_lik(X, Y, Lik)

    def fit_with_lik(self, X, Y, Lik, **kwargs):
        if not self.model:
            ## build the model
            kerns = []
            for _ in range(2):
                kerns.append(gpflow.kernels.RBF(X.shape[1], lengthscales=float(X.shape[1])**0.5))

            if X.shape[0] > num_inducing:
                Z = kmeans2(X, num_inducing, minit='points')[0]
            else:
                # pad with random values
                Z = np.concatentate([X, np.random.randn(X.shape[0] - num_inducing, X.shape[1])], 0)

            mb_size = minibatch_size if X.shape[0] > 5000 else None

            self.model = DGP(X, Y, Z, kerns, Lik(),
                             minibatch_size=mb_size,
                             **kwargs)

            var_list = [[self.model.layers[-1].q_mu, self.model.layers[-1].q_sqrt]]
            self.model.layers[0].q_sqrt = self.model.layers[0].q_sqrt.read_value() * 1e-5
            self.model.layers[-1].q_mu.set_trainable(False)
            self.model.layers[-1].q_sqrt.set_trainable(False)
            ng = gpflow.train.NatGradOptimizer(gamma=gamma).make_optimize_tensor(self.model, var_list=var_list)
            adam = gpflow.train.AdamOptimizer(adam_lr).make_optimize_tensor(self.model)

        # we might have new data
        self.model.X = X
        self.model.Y = Y

        sess = self.model.enquire_session()

        for _ in range(iterations):
            sess.run(ng)
            sess.run(adam)

        self.model.anchor(session=sess)


    def predict(self, Xs):
        return self.model.predict_y(Xs, num_posterior_samples)


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

        return self.fit_with_lik(X, Y, Lik, num_outputs=num_latent)


    def predict(self, Xs):
        m, v = self.model.predict_y(Xs, num_posterior_samples)
        if self.K == 2:
            # Bernoulli only gives one output, so append the complement to work with scipy.stats.multinomial
            return np.concatenate([m, 1-m], 2)
        else:
            return m









