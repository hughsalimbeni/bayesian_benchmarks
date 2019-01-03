import sys
sys.path.append('../../')

import numpy as np
import pandas
from scipy.stats import rankdata

from bayesian_benchmarks.database_utils import Database
from bayesian_benchmarks.data import regression_datasets, classification_datasets
from bayesian_benchmarks.data import _ALL_REGRESSION_DATATSETS, _ALL_CLASSIFICATION_DATATSETS
from bayesian_benchmarks.models.get_model import all_regression_models, all_classification_models

_ALL_DATASETS = {}
_ALL_DATASETS.update(_ALL_REGRESSION_DATATSETS)
_ALL_DATASETS.update(_ALL_CLASSIFICATION_DATATSETS)

def sort_data_by_N(datasets):
    Ns = [_ALL_DATASETS[dataset].N for dataset in datasets]
    order = np.argsort(Ns)
    return list(np.array(datasets)[order])

regression_datasets = sort_data_by_N(regression_datasets)
classification_datasets = sort_data_by_N(classification_datasets)

def read(datasets, models, splits, table, field):
    results = []
    with Database() as db:
        for dataset in datasets:
            for model in models:
                for split in splits:
                    res = db.read(table, [field], {'model': model,
                                                   'dataset': dataset,
                                                   'split' : split})

                    if len(res) > 0:
                        results.append(float(res[0][0]))

                    else:
                        results.append(-1)

    results = np.array(results).reshape(len(datasets), len(models), len(splits))[:, :, 0]
    res = pandas.DataFrame(data=results, index=datasets, columns=models)
    res.insert(0, 'N', [_ALL_DATASETS[dataset].N for dataset in datasets])
    res.insert(1, 'D', [_ALL_DATASETS[dataset].D for dataset in datasets])
    if hasattr(_ALL_DATASETS[datasets[0]], 'K'):
        res.insert(2, 'K', [_ALL_DATASETS[dataset].K for dataset in datasets])

    pandas.DataFrame.to_csv(res, 'results_{}_{}.csv'.format(table, field), float_format='%.6f')

splits = range(1)

read(regression_datasets, all_regression_models, splits, 'regression', 'test_loglik')
read(regression_datasets, all_regression_models, splits, 'regression', 'test_rmse')

read(classification_datasets, all_classification_models, splits, 'classification', 'test_acc')
read(classification_datasets, all_classification_models, splits, 'classification', 'test_loglik')

