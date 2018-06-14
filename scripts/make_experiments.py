import sys
sys.path.append('../tasks')

from data import ALL_REGRESSION_DATATSETS, ALL_CLASSIFICATION_DATATSETS
from utils import make_local_jobs, make_condor_jobs, make_experiment_combinations

models = [
          'linear',
          'variationally_sparse_gp',
          'deep_gp_doubly_stochastic',
          'svm',
          'knn',
          'naive_bayes',
          'decision_tree',
          'random_forest',
          'gradient_boosting_machine',
          'adaboost',
          'mlp',
          ]

############# Regression
combinations = []
combinations.append({'dataset' : list(ALL_REGRESSION_DATATSETS.keys())})
combinations.append({'split' : range(1)})
combinations.append({'model' : models})
experiments = make_experiment_combinations(combinations)

make_local_jobs('../tasks/regression', experiments, overwrite=True)
make_condor_jobs('../tasks/regression', experiments, overwrite=True)

# make_local_jobs('../tasks/active_learning_continuous', experiments)
# make_condor_jobs('../tasks/active_learning_continuous', experiments)

# make_local_jobs('../tasks/conditional_density_estimation', experiments)
# make_condor_jobs('../tasks/conditional_density_estimation', experiments)

############# Classification
combinations = []
combinations.append({'dataset' : list(ALL_CLASSIFICATION_DATATSETS.keys())})
combinations.append({'split' : range(1)})
combinations.append({'model' : models})

experiments = make_experiment_combinations(combinations)

make_local_jobs('../tasks/classification', experiments)
make_condor_jobs('../tasks/classification', experiments)

# make_local_jobs('../tasks/active_learning_discrete', experiments)
# make_condor_jobs('../tasks/active_learning_discrete', experiments)
