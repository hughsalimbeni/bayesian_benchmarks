"""
If pytest-xdist is installed, pytest can multiple independent jobs in parallel on a single machine. 

Here we use the facility to run experiments.

To run 32 experiments in parallel install xdist (pip install pytest-xdist), then following command can be used

python -m pytest bayesian_benchmarks/scripts/run_all_pytest.py -n 32
 
"""
import pytest

from bayesian_benchmarks.tasks.regression import run as run_regression
from bayesian_benchmarks.tasks.classification import run as run_classification
from bayesian_benchmarks.tasks.active_learning_continuous import run as run_AL_cont
from bayesian_benchmarks.tasks.active_learning_discrete import run as run_AL_disc
from bayesian_benchmarks.tasks.mmd import run as run_mmd

from bayesian_benchmarks.data import regression_datasets, classification_datasets
from bayesian_benchmarks.database_utils import Database


all_regression_models = [
      'linear',
      # 'variationally_sparse_gp',
      # 'variationally_sparse_gp_minibatch',
      # 'deep_gp_doubly_stochastic',
      'svm',
      # 'knn',
      # 'decision_tree',
      # 'random_forest',
      # 'gradient_boosting_machine',
      # 'adaboost',
      # 'mlp',
      ]

all_classification_models = [
    'linear',
    # 'variationally_sparse_gp',
    # 'variationally_sparse_gp_minibatch',
    # 'deep_gp_doubly_stochastic',
    'svm',
    # 'naive_bayes',
    # 'knn',
    # 'decision_tree',
    # 'random_forest',
    # 'gradient_boosting_machine',
    # 'adaboost',
    # 'mlp',
    ]

class ConvertToNamespace(object):
    def __init__(self, adict):
        adict.update({'seed':0,
                      'database_path':''})
        self.__dict__.update(adict)

def check_needs_run(table, d):
    with Database() as db:
        try:
            return (len(db.read(table, ['test_loglik'], d.__dict__)) == 0)
        except:
            return True


@pytest.mark.parametrize('model', all_regression_models)
@pytest.mark.parametrize('dataset', regression_datasets)
@pytest.mark.parametrize('split', range(10))
def test_run_all_regression(model, dataset, split):
    d = ConvertToNamespace({'dataset':dataset,
                            'model' :  model,
                            'split' : split})

    if check_needs_run('regression', d):
        run_regression(d, is_test=False)



@pytest.mark.parametrize('dataset', classification_datasets)
@pytest.mark.parametrize('model', all_classification_models)
@pytest.mark.parametrize('split', range(10))
def test_classification(model, dataset, split):
    d = {'dataset':dataset,
         'model' :  model,
         'split' : split}
    if check_needs_run('classification', d):
        run_classification(ConvertToNamespace(d), is_test=False)



# @pytest.mark.parametrize('dataset', ['iris', 'planning'])  # binary and multiclass
# @pytest.mark.parametrize('model', all_regression_models)
# def test_active_learning_discrete(model, dataset):
#     d = {'dataset':dataset,
#          'model' :  model,
#          'iterations': 2,
#          'num_initial_points': 10}
#
#     run_AL_disc(ConvertToNamespace(d), is_test=True)


# @pytest.mark.parametrize('model', all_regression_models)
# @pytest.mark.parametrize('dataset', regression_datasets)
# @pytest.mark.parametrize('split', range(10))
# def test_active_learning_continuous(model):
#     d = {'dataset':'boston',
#          'model' :  model,
#          'iterations': 2,
#          'num_initial_points': 10}
#
#     run_AL_cont(ConvertToNamespace(d), is_test=True)

#
# @pytest.mark.parametrize('model', all_regression_models)
# @pytest.mark.parametrize('pca_dim', [0, 2])
# def test_mmd(model, pca_dim):
#     d = {'dataset':'boston',
#          'model' :  model,
#          'num_samples' : 2,
#          'pca_dim' : pca_dim}
#
#     run_mmd(ConvertToNamespace(d), is_test=True)
