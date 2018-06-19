"""
All the classification data come in a single large download. If many tests are run in parallel this can cause 
issues of synchrony. This gets run before all the tests, so the file is there.   
"""

from bayesian_benchmarks.data import ALL_CLASSIFICATION_DATATSETS
import pytest

@pytest.fixture(scope="session", autouse=True)
def prelim_download(request):
    ALL_CLASSIFICATION_DATATSETS['iris']()
