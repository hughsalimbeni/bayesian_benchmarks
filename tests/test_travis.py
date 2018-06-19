"""
All the classification data come in a single large download. If many tests are run in parallel this can cause 
synchrony issues. This gets run before all the tests, so the file is there.

We also download the other datasets too, as travis sometimes fails if we don't do this.

"""

from bayesian_benchmarks.data import ALL_CLASSIFICATION_DATATSETS, ALL_REGRESSION_DATATSETS
import pytest

@pytest.fixture(scope="session", autouse=True)
def prelim_download(request):
    ALL_CLASSIFICATION_DATATSETS['iris']()
    for d in ALL_REGRESSION_DATATSETS:
        ALL_REGRESSION_DATATSETS[d]()
