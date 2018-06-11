import gpflow
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.stats import norm


class RegressionModel(object):
    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        if not self.model:
            kern = gpflow.kernels.RBF(X.shape[1], lengthscales=float(X.shape[1])**0.5)
            lik = gpflow.likelihoods.Gaussian()
            lik.variance = 0.1

            M = 100  # number of inducing points
            iterations = 2000
            Z = kmeans2(X, M, minit='points')[0] if X.shape[0] > M else X.copy()

            if X.shape[0] < 5000:
                self.model = gpflow.models.SGPR(X, Y, kern, feat=Z)
                self.model.likelihood.variance = lik.variance.read_value()

            else:
                self.model = gpflow.models.SVGP(X, Y, kern, lik, feat=Z, minibatch_size=1000)

                var_list = [[self.model.q_mu, self.model.q_sqrt]]
                ng = gpflow.train.NatGradOptimizer(gamma=0.1).make_optimize_tensor(self.model, var_list=var_list)
                adam = gpflow.train.AdamOptimizer(0.001).make_optimize_tensor(self.model)

        if X.shape[0] < 5000:
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
        M = 100
        Z = kmeans2(X, M, minit='points')[0] if X.shape[0] > M else X.copy()

        if not self.model or Z.shape[0]:
            # NB mb_size does not change once the model is created
            mb_size = 1000 if X.shape[0] > 5000 else None

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
            # q_mu_old = self.model.q_mu.read_value()
            # q_sqrt_old = self.model.q_sqrt.read_value()
            # q_mu_new[:M_old] = q_mu_old
            # q_mu_new[:, :M_old, :M_old] = q_sqrt_old
            self.model.q_mu = q_mu_new
            self.model.q_sqrt = q_sqrt_new

        opt.minimize(self.model, maxiter=2000)

    def predict(self, Xs):
        m, v = self.model.predict_y(Xs)
        if self.K == 2:
            # Bernoulli only gives one output, so append the complement to work with scipy.stats.multinomial
            return np.concatenate([m, 1-m], 1)
        else:
            return m









# def run_density_estimation(X, Y, Xs, levels):
#     m, v = run_regression(X, Y, Xs)
#
#     logp = norm.logpdf(levels[:, None],
#                        loc=m.T,
#                        scale=v.T** 0.5)
#     return logp

