import pytest

from bayesian_benchmarks.data import (
    Wilson_pol,
    Wilson_kin40k,
    Wilson_elevators,
    Wilson_bike,
    Wilson_protein,
    Wilson_keggundirected,
    Wilson_keggdirected,
)


def test_exists():
    """
    TODO: Add all Wilson_* datasets in this test.
    """
    for data in [
        Wilson_keggundirected,
        Wilson_keggdirected,
        Wilson_pol,
        Wilson_protein,
        Wilson_bike,
        Wilson_elevators,
        Wilson_kin40k,
    ]:
        X, y = data().read_data()
        assert X.shape == (data.N, data.D)
        assert y.shape == (data.N, 1)
