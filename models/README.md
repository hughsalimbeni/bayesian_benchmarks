Regression models should conform to the following


class RegressionModel(object):
    def __init__(self):
        pass

    def fit(self, X, Y):
        pass

    def predict(self, Xs):
        """
        Mean and variance of predictive distribution

        m and v are both shape Ns, Dy

        returns: m, v
        """
        pass

    def sample(self, Xs, num_samples):

        pass

class ClassificationModel(RegressionModel):
    def __init__(self, K):
        pass

    def fit(self, X, Y):
        pass

    def predict(self, Xs):
        pass
