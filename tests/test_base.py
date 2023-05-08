# pylint: disable=no-member, invalid-name, redefined-outer-name

import os
from collections import namedtuple
from urllib.parse import urlunsplit

import numpy as np
import pytest
from datatree import DataTree

from arviz_base import generate_dims_coords, list_datasets, load_arviz_data, make_attrs
from arviz_base.datasets import LOCAL_DATASETS, REMOTE_DATASETS, RemoteFileMetadata, clear_data_home


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
