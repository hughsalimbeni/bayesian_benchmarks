import itertools

def make_experiment_combinations(combinations : list):
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

def make_local_jobs(script : str, experiments : list):
    """
    Writes a file of scripts to be run in series on a single machine.

    :param script: name of python script to run
    :param experiments: list of dictionaries of args
    :return: None
    """
    with open('local_run', 'w') as f:
        f.write('#!/usr/bin/env bash\n\n')

        for e in experiments:
            s = 'python {}.py '.format(script)
            for k in e:
                s += '--{}={} '.format(k, e[k])
            s += '\n'
            f.write(s)

def make_condor_jobs(script : str, experiments : list):
    """
    Writes a condor submission file, and also creates the executable if necessary

    :param script: name of python script to run
    :param experiments: list of dictionaries of args
    :return: None
    """
    raise NotImplementedError

def make_kubernetes_jobs(script : str, experiments : list):
    """
    Writes kubernetes submission file

    :param script: name of python script to run
    :param experiments: list of dictionaries of args
    :return: None
    """
    raise NotImplementedError






