import gpflow
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.stats import norm

num_inducing = 100
iterations = 20000
small_iterations = 1000
adam_lr = 0.01
gamma = 0.1
minibatch_size = 1000
initial_likelihood_var = 0.01


class RegressionModel(object):
    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        small_data = X.shape[0] < 5000

        if not self.model:
            kern = gpflow.kernels.RBF(X.shape[1], lengthscales=float(X.shape[1])**0.5)
            lik = gpflow.likelihoods.Gaussian()
            lik.variance = initial_likelihood_var

            Z = kmeans2(X, num_inducing, minit='points')[0] if X.shape[0] > num_inducing else X.copy()

            if small_data:
                self.model = gpflow.models.SGPR(X, Y, kern, feat=Z)
                self.model.likelihood.variance = lik.variance.read_value()

            else:
                self.model = gpflow.models.SVGP(X, Y, kern, lik, feat=Z, minibatch_size=minibatch_size)

                var_list = [[self.model.q_mu, self.model.q_sqrt]]
                self.model.q_mu.set_trainable(False)
                self.model.q_sqrt.set_trainable(False)
                ng = gpflow.train.NatGradOptimizer(gamma=gamma).make_optimize_tensor(self.model, var_list=var_list)
                adam = gpflow.train.AdamOptimizer(adam_lr).make_optimize_tensor(self.model)

        if small_data:
            gpflow.train.ScipyOptimizer().minimize(self.model, maxiter=iterations)

        else:
            sess = self.model.enquire_session()
            for _ in range(iterations):
                sess.run(ng)
                sess.run(adam)
            self.model.anchor(session=sess)

    def predict(self, Xs):
        return self.model.predict_y(Xs)


class ClassificationModel(object):
    def __init__(self, K):
        self.K = K
        self.model = None

    def fit(self, X, Y):
        Z = kmeans2(X, num_inducing, minit='points')[0] if X.shape[0] > num_inducing else X.copy()

        if not self.model or Z.shape[0]:
            # NB mb_size does not change once the model is created
            mb_size = minibatch_size if X.shape[0] > 5000 else None

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
                                            minibatch_size=mb_size)

            if mb_size:
                opt = gpflow.train.AdamOptimizer()
            else:
                opt = gpflow.train.ScipyOptimizer()

            iters = iterations
        else:
            iters = small_iterations

        # we might have new data
        self.model.X = X
        self.model.Y = Y
        self.model.feature.Z = Z

        # if Z has changed shape, start with fresh variational distribution
        M_new = Z.shape[0]
        M_old = self.model.q_mu.shape[0]
        if  M_new != M_old:
            q_mu_new = np.zeros((M_new, self.K))
            q_sqrt_new = np.tile(np.eye(M_new)[None], [self.K, 1, 1])
            self.model.q_mu = q_mu_new
            self.model.q_sqrt = q_sqrt_new

        opt.minimize(self.model, maxiter=iters)

    def predict(self, Xs):
        m, v = self.model.predict_y(Xs)
        if self.K == 2:
            # Bernoulli only gives one output, so append the complement to work with scipy.stats.multinomial
            # NB the m probability is for class 1, so 1-m is the probability of the first class
            return np.concatenate([1-m, m], 1)
        else:
            return m


# class DensityEstimationModel(RegressionModel):
#     def predict(self, Xs, levels):
#         m, v = RegressionModel.predict(self, Xs)
#
#         logp = norm.logpdf(levels[:, None, None],
#                            loc=m[None, :, :],
#                            scale=(v** 0.5)[None, :, :])
#         return logp

