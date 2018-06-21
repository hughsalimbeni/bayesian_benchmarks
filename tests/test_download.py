# """
# All the classification data come in a single large download. If many tests are run in parallel this can cause
# synchrony issues. This gets run before all the tests, so the files are there.
#
# We also download the other datasets too, as travis sometimes fails if we don't do this.
#
# """
#
# import pytest
# from bayesian_benchmarks.data import ALL_CLASSIFICATION_DATATSETS, ALL_REGRESSION_DATATSETS
#
# regression_datasets = list(ALL_REGRESSION_DATATSETS.keys())
# regression_datasets.sort()
#
# @pytest.mark.run(order=2)
# @pytest.mark.parametrize('d', regression_datasets)
# def test_regression(d):
#     ALL_REGRESSION_DATATSETS[d]()
#
# @pytest.mark.run(order=1)
# def test_classification():
#     ALL_CLASSIFICATION_DATATSETS['iris']()
