"""
All the classification data come in a single large download. If many tests are run in parallel this can cause 
synchrony issues. This gets run before all the tests, so the files are there.

We also download the other datasets too, as travis sometimes fails if we don't do this.

"""

import unittest
from ddt import ddt, data
from bayesian_benchmarks.data import ALL_CLASSIFICATION_DATATSETS, ALL_REGRESSION_DATATSETS

regression_datasets = list(ALL_REGRESSION_DATATSETS.keys())
regression_datasets.sort()

@ddt
class TestDownloadOnce(unittest.TestCase):
    @data(regression_datasets)
    def test_regression(self, *d):
        ALL_REGRESSION_DATATSETS[d]()

    def test_classification(self):
        ALL_CLASSIFICATION_DATATSETS['iris']()


if __name__ == '__main__':
    unittest.main()

