import numpy as np
import math
from scipy.cluster.vq import kmeans2

import torch
import gpytorch


class RegressionModel(object):
    def __init__(self, is_test=False, seed=0):
        if is_test:
            class ARGS:
                iterations = 1
                small_iterations = 1
        else:  # pragma: no cover
            class ARGS:
                iterations = 1000
                initial_likelihood_var = 0.01  # not being set yet
        self.ARGS = ARGS
        self.model = None

    def fit(self, X, Y):
        X = torch.Tensor(X)
        Y = torch.Tensor(Y[:, 0])

        # make model if necessary
        if not self.model:
            class GPRegressionModel(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, likelihood):
                    super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
                    self.mean_module = gpytorch.means.ConstantMean()
                    kern = gpytorch.kernels.RBFKernel()  # lengthscale to sqrt(X.shape[1])?
                    self.covar_module = gpytorch.kernels.ScaleKernel(kern)

                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.model = GPRegressionModel(X, Y, self.likelihood)

            # Find optimal model hyperparameters
            self.model.train()
            self.likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=0.01)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

            def train():
                for i in range(self.ARGS.iterations):
                    optimizer.zero_grad()
                    output = self.model(X)
                    loss = -mll(output, Y)
                    loss.backward()
                    if i % 100 == 0:
                        print('Iter %d/%d - Loss: %.3f' % (i + 1,
                                                           self.ARGS.iterations,
                                                           loss.item()))
                    optimizer.step()

            train()

    def predict(self, Xs):
        # Set into eval mode
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            preds = self.likelihood(self.model(torch.Tensor(Xs)))
            m = preds.mean.numpy().reshape(len(Xs), 1)
            v = preds.variance.numpy().reshape(len(Xs), 1)
        return m, v

