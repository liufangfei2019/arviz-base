# pylint: disable=redefined-outer-name
import numpy as np
import pytest
import xarray as xr

from arviz_base import dict_to_dataset, xarray_sel_iter, xarray_to_ndarray


@pytest.fixture(scope="function")
def sample_dataset():
    mu = np.arange(1, 7).reshape(2, 3)
    tau = np.arange(7, 13).reshape(2, 3)

    chain = [0, 1]
    draws = [0, 1, 2]

    data = xr.Dataset(
        {"mu": (["chain", "draw"], mu), "tau": (["chain", "draw"], tau)},
        coords={"draw": draws, "chain": chain},
    )

    return mu, tau, data


def test_dataset_to_numpy_combined(sample_dataset):
    mu, tau, data = sample_dataset
    var_names, data = xarray_to_ndarray(data, combined=True)

    assert len(var_names) == 2
    assert (data[var_names.index("mu")] == mu.reshape(1, 6)).all()
    assert (data[var_names.index("tau")] == tau.reshape(1, 6)).all()


def test_xarray_sel_iter_ordering():
    """Assert that coordinate names stay the provided order"""
    coords = list("dcba")
    data = dict_to_dataset(
        {"x": np.random.randn(1, 100, len(coords))},
        coords={"in_order": coords},
        dims={"x": ["in_order"]},
    )

    coord_names = [sel["in_order"] for _, sel, _ in xarray_sel_iter(data)]
    assert coord_names == coords


def test_xarray_sel_iter_ordering_combined(sample_dataset):
    """Assert that varname order stays consistent when chains are combined"""
    _, _, data = sample_dataset
    var_names = [var for (var, _, _) in xarray_sel_iter(data, var_names=None, combined=True)]
    assert set(var_names) == {"mu", "tau"}


def test_xarray_sel_iter_ordering_uncombined(sample_dataset):
    """Assert that varname order stays consistent when chains are not combined"""
    _, _, data = sample_dataset
    var_names = [(var, selection) for (var, selection, _) in xarray_sel_iter(data, var_names=None)]

    assert len(var_names) == 4
    for var_name in var_names:
        assert var_name in [
            ("mu", {"chain": 0}),
            ("mu", {"chain": 1}),
            ("tau", {"chain": 0}),
            ("tau", {"chain": 1}),
        ]


def test_xarray_sel_data_array(sample_dataset):
    """Assert that varname order stays consistent when chains are combined

    Touches code that is hard to reach.
    """
    _, _, data = sample_dataset
    var_names = [var for (var, _, _) in xarray_sel_iter(data.mu, var_names=None, combined=True)]
    assert set(var_names) == {"mu"}
