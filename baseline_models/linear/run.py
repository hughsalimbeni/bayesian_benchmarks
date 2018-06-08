import numpy as np
from sklearn import linear_model
from scipy.stats import norm

def run_regression(X, Y, Xs):
    lr = linear_model.LinearRegression()
    lr.fit(X, Y)
    train_std = np.std(lr.predict(X) - Y)
    pred_mean = lr.predict(Xs)
    return pred_mean, np.ones_like(pred_mean) * train_std**2

def run_classification(X, Y, Xs, S=1):
    lr = linear_model.LogisticRegression()
    lr.fit(X, Y.ravel())
    prediction = lr.predict_proba(Xs)
    return np.tile(prediction[None, :, :], [S, 1, 1])

def run_density_estimation(X, Y, Xs, levels):
    m, v = run_regression(X, Y, Xs)
    logp = norm.logpdf(levels[:, None],
                       loc=m.T,
                       scale=v.T**0.5)
    return logp



