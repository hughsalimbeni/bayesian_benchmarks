import gpflow
from scipy.cluster.vq import kmeans2
from scipy.stats import norm

def run_regression(X, Y, Xs):
    kern = gpflow.kernels.RBF(X.shape[1], lengthscales=float(X.shape[1])**0.5)
    lik = gpflow.likelihoods.Gaussian()
    lik.variance = 0.1

    M = 100  # number of inducing points
    iterations = 1000
    Z = kmeans2(X, M, minit='points')[0] if X.shape[0] > M else X.copy()

    if X.shape[0] < 5000:
        model = gpflow.models.SGPR(X, Y, kern, feat=Z)
        model.likelihood.variance = lik.variance.read_value()
        gpflow.train.ScipyOptimizer().minimize(model, maxiter=iterations)

    else:
        model = gpflow.models.SVGP(X, Y, kern, lik, feat=Z, minibatch_size=1000)

        var_list = [[model.q_mu, model.q_sqrt]]
        ng = gpflow.train.NatGradOptimizer(gamma=0.1).make_optimize_tensor(model, var_list=var_list)
        adam = gpflow.train.AdamOptimizer(0.001).make_optimize_tensor(model)

        sess = model.enquire_session()

        for _ in range(iterations):
            sess.run(ng)
            sess.run(adam)

        model.anchor(session=sess)

    return model.predict_y(Xs)

def run_classification(X, Y, Xs, K):
    lik = gpflow.likelihoods.MultiClass(K)

    kern = gpflow.kernels.RBF(X.shape[1], lengthscales=float(X.shape[1]) ** 0.5)

    M = 100  # number of inducing points
    Z = kmeans2(X, M, minit='points')[0] if X.shape[0] < M else X.copy()

    model = gpflow.models.SVGP(X, Y, kern, lik, feat=Z, whiten=False, num_latent=K)

    gpflow.train.ScipyOptimizer().minimize(model, maxiter=2000)

    return model.predict_y(Xs)

def run_density_estimation(X, Y, Xs, levels):
    m, v = run_regression(X, Y, Xs)

    logp = norm.logpdf(levels[:, None],
                       loc=m.T,
                       scale=v.T** 0.5)
    return logp

