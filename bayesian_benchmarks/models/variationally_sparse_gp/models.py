import gpflow
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.stats import norm


class NatGradsWithAdamOptimizerFixed:
    # Alternating adam and natgrads optimizer, using fixed learning rates for both
    def __init__(self, adam_lr, natgrads_lr):
        self.adam_lr = adam_lr
        self.natgrads_lr = natgrads_lr

    def minimize(self, model, session=None, maxiter=None):
        var_list = [(model.q_mu, model.q_sqrt)]

        # we don't want adam optimizing these
        model.q_mu.set_trainable(False)
        model.q_sqrt.set_trainable(False)

        adam = gpflow.train.AdamOptimizer(self.adam_lr).make_optimize_tensor(model)
        natgrad = gpflow.train.NatGradOptimizer(self.natgrads_lr).make_optimize_tensor(model, var_list=var_list)

        for it in range(maxiter):
            session.run(natgrad)
            session.run(adam)
        session.run(natgrad)

        model.anchor(session)


class RegressionModel(object):
    def __init__(self, is_test=False, seed=0):
        if is_test:

            class ARGS:
                dense_max_N = 2000
                num_inducing = 100
                iterations = 1
                likelihood_var = 0.01
                lengthscale = 1.
                adam_lr = 1e-2
                natgrads_lr = 1e-2

        else:  # pragma: no cover

            class ARGS:
                small_N = 2000  # anything less than this use a full GP
                large_N = 50000  # the point above which we need minibatches
                minibatch_size = 5000
                num_inducing = 500
                iterations = 10000
                likelihood_var = 0.01
                lengthscale = 1.
                adam_lr = 1e-2
                natgrads_lr = 1e-2

        self.ARGS = ARGS
        self.model = None

    def _make_kernel(self, X):
        lengthscale = self.ARGS.lengthscale * float(X.shape[1]) ** 0.5
        return gpflow.kernels.RBF(X.shape[1], lengthscales=lengthscale)

    def fit(self, X, Y):
        N, D = X.shape


        is_sparse = (N > self.ARGS.small_N)
        use_minibatches = (N > self.ARGS.large_N)

        if is_sparse:
            Z = kmeans2(X, self.ARGS.num_inducing, minit='points')[0]

        # make model if necessary
        if not self.model:

            # kernel
            kern = self._make_kernel(X)

            """
            There are three models we can use:
            * Full GP, for N < small_N
            * Sparse GP with 'collapsed' q(u) [Titsias 09], for small_N < N < large_N
            * Sparse GP with explict Gaussian q(u) [Hensman 12], for large_N < N
            
            In the final case we use alternating natural gradient optimization with adam optimization [Salimbeni 2017] 
            """
            if not is_sparse:  # use a full GP
                self.model = gpflow.models.GPR(X, Y, kern)

            else:
                if not use_minibatches:  # use sparse GP, from [Titsias 09]
                    self.model = gpflow.models.SGPR(X, Y, kern, feat=Z)

                else:  # use variational GP [Hensman 12] with natural gradients
                    lik = gpflow.likelihoods.Gaussian()
                    self.model = gpflow.models.SVGP(X, Y, kern, lik,
                                                    feat=Z,
                                                    minibatch_size=self.ARGS.minibatch_size)

            self.model.likelihood.variance = self.ARGS.likelihood_var

            self.sess = self.model.enquire_session()

            if not use_minibatches:
                self.opt = gpflow.train.ScipyOptimizer()
            else:
                self.opt = NatGradsWithAdamOptimizerFixed(self.ARGS.adam_lr, self.ARGS.natgrads_lr)

        # we might have new data
        self.model.X.assign(X, session=self.sess)
        self.model.Y.assign(Y, session=self.sess)

        if is_sparse:
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
                dense_max_N = 2000
                num_inducing = 100
                iterations = 1
                lengthscale = 1.
                adam_lr = 1e-2

        else:  # pragma: no cover

            class ARGS:
                small_N = 2000  # anything less than this use a full GP
                large_N = 50000  # the point above which we need minibatches
                minibatch_size = 5000
                num_inducing = 500
                iterations = 10000
                lengthscale = 1.
                adam_lr = 1e-2

        self.ARGS = ARGS
        self.K = K
        self.model = None

    def _make_kernel(self, X):
        lengthscale = self.ARGS.lengthscale * float(X.shape[1]) ** 0.5
        return gpflow.kernels.RBF(X.shape[1], lengthscales=lengthscale)

    def fit(self, X, Y):
        N, D = X.shape

        is_sparse = (N > self.ARGS.small_N)
        use_minibatches = (N > self.ARGS.large_N)

        if is_sparse:
            Z = kmeans2(X, self.ARGS.num_inducing, minit='points')[0]


        if not self.model:
            """
            There are two models we can use:
            * Dense variational GP, for N < small_N
            * Sparse variational GP, for N > small_N
            
            Additionally, in the sparse case we use minibatches for N > large_N
            
            For minibatch optimization we use the Adam optimizer, otherwise LBFGS
            """

            # likelihood
            if self.K == 2:
                lik = gpflow.likelihoods.Bernoulli()
                num_latent = 1
            else:
                lik = gpflow.likelihoods.MultiClass(self.K)
                num_latent = self.K

            # kernel
            kern =self._make_kernel(X)

            if not is_sparse:
                self.model = gpflow.models.VGP(X, Y, kern, lik, num_latent=num_latent)

            else:
                minibatch_size = self.ARGS.minibatch_size if use_minibatches else None

                self.model = gpflow.models.SVGP(X, Y, kern, lik,
                                                feat=Z,
                                                whiten=True,
                                                num_latent=num_latent,
                                                minibatch_size=minibatch_size)

            self.sess = self.model.enquire_session()
            if use_minibatches:
                self.opt = gpflow.train.AdamOptimizer(self.ARGS.adam_lr)
            else:
                self.opt = gpflow.train.ScipyOptimizer()

            iters = self.ARGS.iterations

        else:
            iters = self.ARGS.small_iterations

        # we might have new data
        self.model.X.assign(X, session=self.sess)
        self.model.Y.assign(Y, session=self.sess)

        if is_sparse:
            self.model.feature.Z.assign(Z, session=self.sess)

        # num_outputs = self.model.q_sqrt.shape[0]
        # self.model.q_mu.assign(np.zeros((self.ARGS.num_inducing, num_outputs)), session=self.sess)
        # self.model.q_sqrt.assign(np.tile(np.eye(self.ARGS.num_inducing)[None], [num_outputs, 1, 1]), session=self.sess)

        self.opt.minimize(self.model, maxiter=iters, session=self.sess)

    def predict(self, Xs):
        m, v = self.model.predict_y(Xs, session=self.sess)
        if self.K == 2:
            # convert Bernoulli to onehot
            return np.concatenate([1 - m, m], 1)
        else:
            return m



