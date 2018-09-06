from bayesian_benchmarks.data import regression_datasets, classification_datasets
from bayesian_benchmarks.database_utils import Database

import itertools
import os
from subprocess import call


def make_experiment_combinations(combinations: list):
    """
    The product of all combinations of arguments.
    :param combinations: A list of dictionaries, each with a list of args
    :return: A list of dictionaries for all combinations
    """
    fields = []
    vals = []
    for p in combinations:
        for k in p:
            fields.append(k)
            vals.append(p[k])

    ret = []
    for a in itertools.product(*vals):
        d = {}

        for f, arg in zip(fields, a):
            d[f] = arg

        ret.append(d)

    return ret


def make_local_jobs(script: str, experiments: list, overwrite=False):
    """
    Writes a file of commands to be run in in series on a single machine, e.g. 
    #!/usr/bin/env bash
    python run_regression --split=0
    python run_regression --split=1
    etc. 
    
    If overwrite=True then a new file is written with a shebang, otherwise lines are appended. 

    :param script: name of python script to run
    :param experiments: list of dictionaries of args
    :return: None
    """
    if overwrite:
        with open('local_run', 'w') as f:
            f.write('#!/usr/bin/env bash\n\n')

    with open('local_run', 'a') as f:
        for e in experiments:
            s = 'python {}.py '.format(script)
            for k in e:
                s += '--{}={} '.format(k, e[k])
            s += '\n'
            f.write(s)


def make_condor_jobs(script: str, experiments: list, overwrite=False):
    """
    Writes a condor submission file, and also creates the executable if necessary. Preamble for the 
    exectable (e.g. for setting up the python environment) should go in 'preamble.txt.txt'. Preamble
    for the condor submission should go in condor_preamble.txt.txt.txt.

    If overwrite=True then a new file is written with the condor preamble from condor_preamble.txt.txt,
    otherwise lines are appended. 

    :param script: name of python script to run
    :param experiments: list of dictionaries of args
    :return: None
    """
    condor_run_file = 'condor_run'
    if not os.path.isfile(condor_run_file):
        with open(condor_run_file, 'w') as f:
            f.write("#!/usr/bin/env bash\n")

            preamble = 'preamble.txt'
            if os.path.isfile(preamble):
                for l in open(preamble, 'r'):
                    f.writelines(l)

            t = "python "
            for i in range(1, 10):
                t += '$' + str(i) + ' '
            for i in range(10, 20):
                t += '${' + str(i) + '} '

            f.write(t + '\n')

        call(["chmod", '777', condor_run_file])

    if overwrite:
        with open('condor_jobs', 'w') as f:
            with open('condor_preamble.txt.txt', 'r') as ff:
                f.writelines(ff)

    with open('condor_jobs', 'a') as f:
        for e in experiments:
            t = 'Arguments = {}.py '.format(script)
            for k in e:
                t += '--{}={} '.format(k, e[k])
            t += '\nQueue 1\n'
            f.write(t)

def remove_already_run_experiments(table, experiments):
    res = []

    with Database() as db:
        for e in experiments:
            if len(db.read(table, ['test_loglik'], e)) == 0:
                res.append(e)

    s = 'originally {} experiments, but {} have already been run, so running {} experiments'
    print(s.format(len(experiments), len(experiments) - len(res), len(res)))
    return res

#################################################
models = [
          'linear',
          'variationally_sparse_gp',
          'variationally_sparse_gp_minibatch',
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
combinations.append({'dataset' : regression_datasets})
combinations.append({'split' : range(10)})
combinations.append({'model' : models})
experiments = make_experiment_combinations(combinations)
experiments = remove_already_run_experiments('regression', experiments)

make_local_jobs('../tasks/regression', experiments, overwrite=True)
make_condor_jobs('../tasks/regression', experiments, overwrite=True)

# make_local_jobs('../tasks/active_learning_continuous', experiments)
# make_condor_jobs('../tasks/active_learning_continuous', experiments)

# make_local_jobs('../tasks/conditional_density_estimation', experiments)
# make_condor_jobs('../tasks/conditional_density_estimation', experiments)

############# Classification
combinations = []
combinations.append({'dataset' : classification_datasets})
combinations.append({'split' : range(10)})
combinations.append({'model' : models})

experiments = make_experiment_combinations(combinations)
experiments = remove_already_run_experiments('classification', experiments)

make_local_jobs('../tasks/classification', experiments)
make_condor_jobs('../tasks/classification', experiments)
#
# # make_local_jobs('../tasks/active_learning_discrete', experiments)
# # make_condor_jobs('../tasks/active_learning_discrete', experiments)
