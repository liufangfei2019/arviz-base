from typing import Dict, List, Tuple, Union

import numpy as np
import pytest
from datatree import DataTree

from arviz_base import load_arviz_data


class ExampleRandomVariable:
    """Example class for random variables."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        """Return argument to constructor as string representation."""
        return self.name


@pytest.fixture(scope="module")
def draws():
    """Share default draw count."""
    return 500


@pytest.fixture(scope="module")
def chains():
    """Share default chain count."""
    return 2


@pytest.fixture(scope="module")
def centered_eight():
    """Share default chain count."""
    return load_arviz_data("centered_eight").load()


@pytest.fixture(scope="module")
def eight_schools_params():
    """Share setup for eight schools."""
    return {
        "J": 8,
        "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }


def check_multiple_attrs(
    test_dict: Dict[str, List[str]], parent: DataTree
) -> List[Union[str, Tuple[str, str]]]:
    """Perform multiple hasattr checks on InferenceData objects.

    It is thought to first check if the parent object contains a given dataset,
    and then (if present) check the attributes of the dataset.

    Given the output of the function, all mismatches between expectation and reality can
    be retrieved: a single string indicates a group mismatch and a tuple of strings
    ``(group, var)`` indicates a mismatch in the variable ``var`` of ``group``.

    Parameters
    ----------
    test_dict: dict of {str : list of str}
        Its structure should be `{dataset1_name: [var1, var2], dataset2_name: [var]}`.
        A ``~`` at the beginning of a dataset or variable name indicates the name NOT
        being present must be asserted.
    parent: InferenceData
        InferenceData object on which to check the attributes.

    Returns
    -------
    list
        List containing the failed checks. It will contain either the dataset_name or a
        tuple (dataset_name, var) for all non present attributes.

    Examples
    --------
    The output below indicates that ``posterior`` group was expected but not found, and
    variables ``a`` and ``b``:

        ["posterior", ("prior", "a"), ("prior", "b")]

    Another example could be the following:

        [("posterior", "a"), "~observed_data", ("sample_stats", "~log_likelihood")]

    In this case, the output indicates that variable ``a`` was not found in ``posterior``
    as it was expected, however, in the other two cases, the preceding ``~`` (kept from the
    input negation notation) indicates that ``observed_data`` group should not be present
    but was found in the InferenceData and that ``log_likelihood`` variable was found
    in ``sample_stats``, also against what was expected.

    """
    failed_attrs: List[Union[str, Tuple[str, str]]] = []
    for dataset_name, attributes in test_dict.items():
        if dataset_name.startswith("~"):
            if hasattr(parent, dataset_name[1:]):
                failed_attrs.append(dataset_name)
        elif hasattr(parent, dataset_name):
            dataset = getattr(parent, dataset_name)
            for attribute in attributes:
                if attribute.startswith("~"):
                    if hasattr(dataset, attribute[1:]):
                        failed_attrs.append((dataset_name, attribute))
                elif not hasattr(dataset, attribute):
                    failed_attrs.append((dataset_name, attribute))
        else:
            failed_attrs.append(dataset_name)
    return failed_attrs
