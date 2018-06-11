import sys
sys.path.append('../tasks')

from data import ALL_REGRESSION_DATATSETS
from utils import make_local_jobs, make_condor_jobs, make_experiment_combinations

combinations = []

combinations.append({'dataset' : list(ALL_REGRESSION_DATATSETS.keys())})
combinations.append({'split' : range(3)})
combinations.append({'model' : ['linear', 'variationally_sparse_gp']})

experiments = make_experiment_combinations(combinations)

make_local_jobs('../tasks/regression', experiments)
make_condor_jobs('../tasks/regression', experiments)