# pylint: disable=no-member, no-self-use, invalid-name, redefined-outer-name

import os
from collections import namedtuple
from urllib.parse import urlunsplit

import numpy as np
import pytest
import xarray as xr
from datatree import DataTree

from arviz_base import (
    convert_to_dataset,
    convert_to_datatree,
    dict_to_dataset,
    extract,
    generate_dims_coords,
    list_datasets,
    load_arviz_data,
    make_attrs,
)
from arviz_base.datasets import LOCAL_DATASETS, REMOTE_DATASETS, RemoteFileMetadata, clear_data_home

from .helpers import centered_eight, chains, draws  # pylint: disable=unused-import


@pytest.fixture(autouse=True)
def no_remote_data(monkeypatch, tmpdir):
    """Delete all remote data and replace it with a local dataset."""
    keys = list(REMOTE_DATASETS)
    for key in keys:
        monkeypatch.delitem(REMOTE_DATASETS, key)

    centered = LOCAL_DATASETS["centered_eight"]
    filename = os.path.join(str(tmpdir), os.path.basename(centered.filename))

    url = urlunsplit(("file", "", centered.filename, "", ""))

    monkeypatch.setitem(
        REMOTE_DATASETS,
        "test_remote",
        RemoteFileMetadata(
            name="test_remote",
            filename=filename,
            url=url,
            checksum="8efc3abafe0c796eb9aea7b69490d4e2400a33c57504ef4932e1c7105849176f",
            description=centered.description,
        ),
    )
    monkeypatch.setitem(
        REMOTE_DATASETS,
        "bad_checksum",
        RemoteFileMetadata(
            name="bad_checksum",
            filename=filename,
            url=url,
            checksum="bad!",
            description=centered.description,
        ),
    )
    UnknownFileMetaData = namedtuple(
        "UnknownFileMetaData", ["filename", "url", "checksum", "description"]
    )
    monkeypatch.setitem(
        REMOTE_DATASETS,
        "test_unknown",
        UnknownFileMetaData(
            filename=filename,
            url=url,
            checksum="9ae00c83654b3f061d32c882ec0a270d10838fa36515ecb162b89a290e014849",
            description="Test bad REMOTE_DATASET",
        ),
    )


def test_load_local_arviz_data():
    idata = load_arviz_data("centered_eight")
    assert isinstance(idata, DataTree)
    assert set(idata.observed_data.obs.coords["school"].values) == {
        "Hotchkiss",
        "Mt. Hermon",
        "Choate",
        "Deerfield",
        "Phillips Andover",
        "St. Paul's",
        "Lawrenceville",
        "Phillips Exeter",
    }
    assert idata.posterior["theta"].dims == ("chain", "draw", "school")


def test_clear_data_home():
    resource = REMOTE_DATASETS["test_remote"]
    assert not os.path.exists(resource.filename)
    load_arviz_data("test_remote")
    assert os.path.exists(resource.filename)
    clear_data_home(data_home=os.path.dirname(resource.filename))
    assert not os.path.exists(resource.filename)


def test_load_remote_arviz_data():
    assert load_arviz_data("test_remote")


def test_bad_checksum():
    with pytest.raises(IOError):
        load_arviz_data("bad_checksum")


def test_missing_dataset():
    with pytest.raises(ValueError):
        load_arviz_data("does not exist")


def test_list_datasets():
    dataset_string = list_datasets()
    # make sure all the names of the data sets are in the dataset description
    for key in (
        "centered_eight",
        "non_centered_eight",
        "test_remote",
        "bad_checksum",
        "test_unknown",
    ):
        assert key in dataset_string


def test_dims_coords():
    shape = 4, 20, 5
    var_name = "x"
    dims, coords = generate_dims_coords(shape, var_name)
    assert "x_dim_0" in dims
    assert "x_dim_1" in dims
    assert "x_dim_2" in dims
    assert len(coords["x_dim_0"]) == 4
    assert len(coords["x_dim_1"]) == 20
    assert len(coords["x_dim_2"]) == 5


def test_dims_coords_extra_dims():
    shape = 4, 20
    var_name = "x"
    with pytest.raises(ValueError, match="more dims"):
        generate_dims_coords(shape, var_name, dims=["xx", "xy", "xz"])


@pytest.mark.parametrize("shape", [(4, 20), (4, 20, 1)])
def test_dims_coords_skip_event_dims(shape):
    coords = {"x": np.arange(4), "y": np.arange(20), "z": np.arange(5)}
    dims, coords = generate_dims_coords(
        shape, "name", dims=["x", "y", "z"], coords=coords, skip_event_dims=True
    )
    assert "x" in dims
    assert "y" in dims
    assert "z" not in dims
    assert len(coords["x"]) == 4
    assert len(coords["y"]) == 20
    assert "z" not in coords


def test_make_attrs():
    extra_attrs = {"key": "Value"}
    attrs = make_attrs(attrs=extra_attrs)
    assert "key" in attrs
    assert attrs["key"] == "Value"
    assert "created_at" in attrs
    assert "creation_library" in attrs


class TestNumpyToDataArray:
    def test_1d_dataset(self):
        size = 100
        dataset = convert_to_dataset(np.random.randn(size), sample_dims=["sample"])
        assert len(dataset.data_vars) == 1

        assert set(dataset.coords) == {"sample"}
        assert dataset.dims["sample"] == size

    def test_warns_bad_shape(self):
        ary = np.random.randn(100, 4)
        # Shape should be (chain, draw, *shape)
        with pytest.warns(UserWarning, match="Found chain dimension to be longer than draw"):
            convert_to_dataset(ary, sample_dims=("chain", "draw"))
        # Shape should now be (draw, chain, *shape)
        dataset = convert_to_dataset(ary, sample_dims=("draw", "chain"))
        assert dataset.dims["chain"] == 4
        assert dataset.dims["draw"] == 100

    def test_nd_to_dataset(self):
        shape = (1, 20, 3, 4, 5)
        dataset = convert_to_dataset(
            np.random.randn(*shape), sample_dims=("chain", "draw", "pred_id")
        )
        assert len(dataset.data_vars) == 1
        var_name = list(dataset.data_vars)[0]

        assert len(dataset.coords) == len(shape)
        assert dataset.dims["chain"] == shape[0]
        assert dataset.dims["draw"] == shape[1]
        assert dataset.dims["pred_id"] == shape[2]
        assert dataset[var_name].shape == shape

    def test_nd_to_datatree(self):
        shape = (1, 2, 3, 4, 5)
        data = convert_to_datatree(np.random.randn(*shape), group="prior")
        assert "/prior" in data.groups
        prior = data["prior"]
        assert len(prior.data_vars) == 1
        var_name = list(prior.data_vars)[0]

        assert len(prior.coords) == len(shape)
        assert prior.dims["chain"] == shape[0]
        assert prior.dims["draw"] == shape[1]
        assert prior[var_name].shape == shape

    def test_more_chains_than_draws(self):
        shape = (10, 4)
        with pytest.warns(UserWarning):
            data = convert_to_datatree(np.random.randn(*shape), group="prior")
        assert "/prior" in data.groups
        prior = data["prior"]
        assert len(prior.data_vars) == 1
        var_name = list(prior.data_vars)[0]

        assert len(prior.coords) == len(shape)
        assert prior.dims["chain"] == shape[0]
        assert prior.dims["draw"] == shape[1]
        assert prior[var_name].shape == shape


class TestConvertToDataset:
    @pytest.fixture(scope="class")
    def data(self):
        # pylint: disable=attribute-defined-outside-init
        class Data:
            datadict = {
                "a": np.random.randn(1, 100),
                "b": np.random.randn(1, 100, 10),
                "c": np.random.randn(1, 100, 3, 4),
            }
            coords = {"c1": np.arange(3), "c2": np.arange(4), "b1": np.arange(10)}
            dims = {"b": ["b1"], "c": ["c1", "c2"]}

        return Data

    def test_use_all(self, data):
        dataset = convert_to_dataset(data.datadict, coords=data.coords, dims=data.dims)
        assert set(dataset.data_vars) == {"a", "b", "c"}
        assert set(dataset.coords) == {"chain", "draw", "c1", "c2", "b1"}

        assert set(dataset.a.coords) == {"chain", "draw"}
        assert set(dataset.b.coords) == {"chain", "draw", "b1"}
        assert set(dataset.c.coords) == {"chain", "draw", "c1", "c2"}

    def test_missing_coords(self, data):
        dataset = convert_to_dataset(data.datadict, coords=None, dims=data.dims)
        assert set(dataset.data_vars) == {"a", "b", "c"}
        assert set(dataset.coords) == {"chain", "draw", "c1", "c2", "b1"}

        assert set(dataset.a.coords) == {"chain", "draw"}
        assert set(dataset.b.coords) == {"chain", "draw", "b1"}
        assert set(dataset.c.coords) == {"chain", "draw", "c1", "c2"}

    def test_missing_dims(self, data):
        # missing dims
        coords = {"c_dim_0": np.arange(3), "c_dim_1": np.arange(4), "b_dim_0": np.arange(10)}
        dataset = convert_to_dataset(data.datadict, coords=coords, dims=None)
        assert set(dataset.data_vars) == {"a", "b", "c"}
        assert set(dataset.coords) == {"chain", "draw", "c_dim_0", "c_dim_1", "b_dim_0"}

        assert set(dataset.a.coords) == {"chain", "draw"}
        assert set(dataset.b.coords) == {"chain", "draw", "b_dim_0"}
        assert set(dataset.c.coords) == {"chain", "draw", "c_dim_0", "c_dim_1"}

    def test_skip_dim_0(self, data):
        dims = {"c": [None, "c2"]}
        coords = {"c_dim_0": np.arange(3), "c2": np.arange(4), "b_dim_0": np.arange(10)}
        dataset = convert_to_dataset(data.datadict, coords=coords, dims=dims)
        assert set(dataset.data_vars) == {"a", "b", "c"}
        assert set(dataset.coords) == {"chain", "draw", "c_dim_0", "c2", "b_dim_0"}

        assert set(dataset.a.coords) == {"chain", "draw"}
        assert set(dataset.b.coords) == {"chain", "draw", "b_dim_0"}
        assert set(dataset.c.coords) == {"chain", "draw", "c_dim_0", "c2"}


def test_dict_to_dataset():
    datadict = {"a": np.random.randn(1, 100), "b": np.random.randn(1, 100, 10)}
    dataset = convert_to_dataset(datadict, coords={"c": np.arange(10)}, dims={"b": ["c"]})
    assert set(dataset.data_vars) == {"a", "b"}
    assert set(dataset.coords) == {"chain", "draw", "c"}

    assert set(dataset.a.coords) == {"chain", "draw"}
    assert set(dataset.b.coords) == {"chain", "draw", "c"}


def test_dict_to_dataset_event_dims_error():
    datadict = {"a": np.random.randn(1, 100, 10)}
    coords = {"b": np.arange(10), "c": ["x", "y", "z"]}
    msg = "more dims"
    with pytest.raises(ValueError, match=msg):
        convert_to_dataset(datadict, coords=coords, dims={"a": ["b", "c"]})


def test_dict_to_dataset_with_tuple_coord():
    datadict = {"a": np.random.randn(1, 100), "b": np.random.randn(1, 100, 10)}
    with pytest.raises(TypeError, match="Could not convert tuple"):
        convert_to_dataset(datadict, coords={"c": tuple(range(10))}, dims={"b": ["c"]})


def test_convert_to_dataset_idempotent():
    first = convert_to_dataset(np.random.randn(1, 100))
    second = convert_to_dataset(first)
    assert first.equals(second)


def test_convert_to_datatree_idempotent():
    first = convert_to_datatree(np.random.randn(1, 100), group="prior")
    second = convert_to_datatree(first)
    assert first.prior is second.prior


def test_convert_to_datatree_from_file(tmpdir):
    first = convert_to_datatree(np.random.randn(1, 100), group="prior")
    filename = str(tmpdir.join("test_file.nc"))
    first.to_netcdf(filename)
    second = convert_to_datatree(filename)
    assert first.prior.equals(second.prior)


def test_convert_to_datatree_bad():
    with pytest.raises(ValueError):
        convert_to_datatree(1)


def test_convert_to_dataset_bad(tmpdir):
    first = convert_to_datatree(np.random.randn(1, 100), group="prior")
    filename = str(tmpdir.join("test_file.nc"))
    first.to_netcdf(filename)
    with pytest.raises(ValueError):
        convert_to_dataset(filename, group="bar")


class TestDataConvert:
    @pytest.fixture(scope="class")
    def data(self, draws, chains):
        class Data:
            # fake 8-school output
            obj = {}
            for key, shape in {"mu": [], "tau": [], "eta": [8], "theta": [8]}.items():
                obj[key] = np.random.randn(chains, draws, *shape)

        return Data

    def get_datatree(self, data):
        return convert_to_datatree(
            data.obj,
            group="posterior",
            coords={"school": np.arange(8)},
            dims={"theta": ["school"], "eta": ["school"]},
        )

    def check_var_names_coords_dims(self, dataset):
        assert set(dataset.data_vars) == {"mu", "tau", "eta", "theta"}
        assert set(dataset.coords) == {"chain", "draw", "school"}

    def test_convert_to_datatree(self, data):
        data = self.get_datatree(data)
        assert "/posterior" in data.groups
        self.check_var_names_coords_dims(data.posterior)

    def test_convert_to_dataset(self, draws, chains, data):
        dataset = dict_to_dataset(
            data.obj,
            coords={"school": np.arange(8)},
            dims={"theta": ["school"], "eta": ["school"]},
        )
        assert dataset.draw.shape == (draws,)
        assert dataset.chain.shape == (chains,)
        assert dataset.school.shape == (8,)
        assert dataset.theta.shape == (chains, draws, 8)


class TestExtract:
    def test_default(self, centered_eight):
        post = extract(centered_eight)
        assert isinstance(post, xr.Dataset)
        assert "sample" in post.dims
        assert post.theta.size == (4 * 500 * 8)

    def test_seed(self, centered_eight):
        post = extract(centered_eight, rng=7)
        post_pred = extract(centered_eight, group="posterior_predictive", rng=7)
        assert all(post.sample == post_pred.sample)

    def test_no_combine(self, centered_eight):
        post = extract(centered_eight, combined=False)
        assert "sample" not in post.dims
        assert post.dims["chain"] == 4
        assert post.dims["draw"] == 500

    def test_var_name_group(self, centered_eight):
        prior = extract(centered_eight, group="prior", var_names="the", filter_vars="like")
        assert {} == prior.attrs
        assert "theta" in prior.name

    def test_keep_dataset(self, centered_eight):
        prior = extract(
            centered_eight, group="prior", var_names="the", filter_vars="like", keep_dataset=True
        )
        assert prior.attrs == centered_eight.prior.attrs
        assert "theta" in prior.data_vars
        assert "mu" not in prior.data_vars

    def test_subset_samples(self, centered_eight):
        post = extract(centered_eight, num_samples=10)
        assert post.dims["sample"] == 10
        assert post.attrs == centered_eight.posterior.attrs
