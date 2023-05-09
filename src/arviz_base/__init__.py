# pylint: disable=wildcard-import,wrong-import-position
"""Base ArviZ features and converters."""

import logging

_log = logging.getLogger("arviz")

from .base import generate_dims_coords, dict_to_dataset, make_attrs
from .datasets import load_arviz_data, list_datasets, get_data_home, clear_data_home
from .rcparams import rcParams, rc_context
from ._version import __version__

__all__ = [
    "dict_to_dataset",
    "generate_dims_coords",
    "make_attrs",
    "load_arviz_data",
    "list_datasets",
    "get_data_home",
    "clear_data_home",
    "rcParams",
    "rc_context",
]
