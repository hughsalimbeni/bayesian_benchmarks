from importlib import import_module
import os

from bayesian_benchmarks.models.non_bayesian_models import non_bayesian_model

abs_path = os.path.abspath(__file__)[:-len('/get_model.py')]

def get_regression_model(name):
    assert name in all_regression_models
    return non_bayesian_model(name, 'regression') or \
           import_module('bayesian_benchmarks.models.{}.models'.format(name)).RegressionModel

def get_classification_model(name):
    assert name in all_classification_models
    return non_bayesian_model(name, 'classification') or \
           import_module('bayesian_benchmarks.models.{}.models'.format(name)).ClassificationModel

# add new regression models here 
all_regression_models = [
      'linear',
      'variationally_sparse_gp',
      'variationally_sparse_gp_minibatch',
      'deep_gp_doubly_stochastic',
      'svm',
      'knn',
      'decision_tree',
      'random_forest',
      'gradient_boosting_machine',
      'adaboost',
      'mlp',
      ]

# add new classification models here
all_classification_models = [
    'linear',
    'variationally_sparse_gp',
    'variationally_sparse_gp_minibatch',
    'deep_gp_doubly_stochastic',
    'svm',
    'naive_bayes',
    'knn',
    'decision_tree',
    'random_forest',
    'gradient_boosting_machine',
    'adaboost',
    'mlp',
    ]

all_models = list(set(all_regression_models).union(set(all_classification_models)))
