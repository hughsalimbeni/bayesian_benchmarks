import numpy as np

from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
from sklearn import naive_bayes
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network

def regression_model(model):
    class SKLWrapperRegression(object):
        def __init__(self):
            self.model = model

        def fit(self, X, Y):
            self.model.fit(X, Y.flatten())
            self.std = np.std(self.model.predict(X) - Y.flatten())

        def predict(self, Xs):
            pred_mean = self.model.predict(Xs)[:, None]
            return pred_mean, np.ones_like(pred_mean) * (self.std + 1e-6) ** 2

        def sample(self, Xs, num_samples):
            m, v = self.predict(Xs)
            N, D = np.shape(m)
            m, v = np.expand_dims(m, 0), np.expand_dims(v, 0)
            return m + np.random.randn(num_samples, N, D) * (v ** 0.5)

    return SKLWrapperRegression


def classification_model(model):
    class SKLWrapperClassification(object):
        def __init__(self, K):
            self.model = model
            self.K = K

        def fit(self, X, Y):
            self.Ks_seen = list(set(Y.flatten().astype(int)))
            self.Ks_seen.sort()
            self.model.fit(X, Y.ravel())

        def predict(self, Xs):
            ps = self.model.predict_proba(Xs)
            if len(self.Ks_seen) == self.K:
                return ps
            else:  # not all classes have been seen, and sklearn doesn't allow setting of n_classes
                ret = np.zeros((len(ps), self.K))
                for k, p in zip(self.Ks_seen, ps.T):
                    ret[:, k] = p
                return ret

    return SKLWrapperClassification


def non_bayesian_model(name, task):
    if name == 'linear' and task == 'regression':
        return regression_model(linear_model.LinearRegression())

    elif name == 'linear' and task == 'classification':
        return classification_model(linear_model.LogisticRegression())

    if name == 'svm' and task == 'regression':
        return regression_model(svm.SVR())

    elif name == 'svm' and task == 'classification':
        return classification_model(svm.SVC(probability=True))

    if name == 'knn' and task == 'regression':
        return regression_model(neighbors.KNeighborsRegressor())  # default is K=5

    elif name == 'knn' and task == 'classification':
        return classification_model(neighbors.KNeighborsClassifier())  # default is K=5

    elif name == 'naive_bayes' and task == 'classification':
        return classification_model(naive_bayes.GaussianNB())

    if name == 'decision_tree' and task == 'regression':
        return regression_model(tree.DecisionTreeRegressor())

    elif name == 'decision_tree' and task == 'classification':
        return classification_model(tree.DecisionTreeClassifier())

    if name == 'random_forest' and task == 'regression':
        return regression_model(ensemble.RandomForestRegressor())

    elif name == 'random_forest' and task == 'classification':
        return classification_model(ensemble.RandomForestClassifier())  # default is 10 estimators

    if name == 'gradient_boosting_machine' and task == 'regression':
        return regression_model(ensemble.GradientBoostingRegressor())

    elif name == 'gradient_boosting_machine' and task == 'classification':
        return classification_model(ensemble.GradientBoostingClassifier()) # default is 100 estimators

    if name == 'adaboost' and task == 'regression':
        return regression_model(ensemble.AdaBoostRegressor())

    elif name == 'adaboost' and task == 'classification':
        return classification_model(ensemble.AdaBoostClassifier()) # default is 100 estimators

    if name == 'mlp' and task == 'regression':
        return regression_model(neural_network.MLPRegressor())

    elif name == 'mlp' and task == 'classification':
        return classification_model(neural_network.MLPClassifier())

    else:
        return None
