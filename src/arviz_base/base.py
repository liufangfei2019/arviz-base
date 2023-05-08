"""ArviZ basic functions and converters."""
import datetime
import importlib
from collections.abc import Hashable, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple

import numpy as np
import xarray as xr

from ._version import __version__
from .rcparams import rcParams
from .types import CoordSpec, DictData, DimSpec

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def generate_dims_coords(
    shape: Iterable[int],
    var_name: Hashable,
    dims: Optional[Iterable[Hashable]] = None,
    coords: Optional[CoordSpec] = None,
    index_origin: Optional[int] = None,
    skip_event_dims: bool = False,
) -> Tuple[List[Hashable], CoordSpec]:
    """Generate default dimensions and coordinates for a variable.

    Parameters
    ----------
    shape : iterable of int
        Shape of the variable
    var_name : iterable of hashable
        Name of the variable. If no dimension name(s) is provided, ArviZ
        will generate a default dimension name using ``var_name``, e.g.
        ``"foo_dim_0"`` for the first dimension if ``var_name`` is ``"foo"``.
    dims : iterable of hashable, optional
        Dimension names (or identifiers) for the variable.
        If `skip_event_dims` is ``True`` it can be longer than `shape`.
        In that case, only the first ``len(shape)`` elements in `dims` will be used.
        Moreover, if needed, axis of length 1 in shape will also be given
        different names than the ones provided in `dims`.
    coords : dict of {hashable: array_like}, optional
        Map of dimension names to coordinate values. Dimensions without coordinate
        values mapped to them will be given an integer range as coordinate values.
        It can have keys for dimension names not present in that variable.
    index_origin : int, optional
        Starting value of generated integer coordinate values.
        Defaults to the value in rcParam ``data.index_origin``.
    skip_event_dims : bool, default False
        Whether to allow for different sizes between `shape` and `dims`.
        See description in `dims` for more details.


    Returns
    -------
    list of hashable
        Default dims for that variable
    dict of {hashable: array_like}
        Default coords for that variable
    """
    if index_origin is None:
        index_origin = rcParams["data.index_origin"]
    if dims is None:
        dims = []

    if coords is None:
        coords = {}

    coords = deepcopy(coords)
    dims = deepcopy(dims)

    if len(dims) > len(shape):
        if skip_event_dims:
            dims = dims[: len(shape)]
        else:
            raise ValueError(
                (
                    "In variable {var_name}, there are "
                    + "more dims ({dims_len}) given than existing ones ({shape_len}). "
                    + "dims and shape should match with `skip_event_dims=False`"
                ).format(
                    var_name=var_name,
                    dims_len=len(dims),
                    shape_len=len(shape),
                )
            )
    if skip_event_dims:
        # In some cases, even when there is an event dim, the shape has the
        # right length but the length of the axis doesn't match.
        # For example, the log likelihood of a 3d MvNormal with 20 observations
        # should be (20,) but it can also be (20, 1). The code below ensures
        # the (20, 1) option also works.
        for i, (dim, dim_size) in enumerate(zip(dims, shape)):
            if (dim in coords) and (dim_size != len(coords[dim])):
                dims = dims[:i]
                break

    for idx, dim_len in enumerate(shape):
        if idx + 1 > len(dims) or dims[idx] is None:
            dim_name = f"{var_name}_dim_{idx}"
            dims[idx] = dim_name
        dim_name = dims[idx]
        if dim_name not in coords:
            coords[dim_name] = np.arange(index_origin, dim_len + index_origin)
    coords = {dim_name: coords[dim_name] for dim_name in dims}
    return dims, coords


def dict_to_dataset(
    data: DictData,
    *,
    attrs: Optional[Mapping[Any, Any]] = None,
    inference_library: Optional[str] = None,
    coords: Optional[CoordSpec] = None,
    dims: Optional[DimSpec] = None,
    sample_dims: Optional[Iterable[Hashable]] = None,
    index_origin: Optional[int] = None,
    skip_event_dims: bool = False,
):
    """Convert a dictionary of numpy arrays to an xarray.Dataset.

    The conversion considers some ArviZ conventions and adds extra
    attributes, so it is similar to initializing an :class:`xarray.Dataset`
    but not equivalent.

    Parameters
    ----------
    data : dict of {hashable: array_like}
        Data to convert. Keys are variable names.
    attrs : dict, optional
        JSON-like arbitrary metadata to attach to the dataset, in addition to default
        attributes added by :func:`make_attrs`.

        .. note::

           No serialization checks are done in this function, so you might generate
           :class:`~xarray.Dataset` objects that can't be serialized or that can
           only be serialized to some backends.

    inference_library : module, optional
        Library used for performing inference. Will be included in the
        :class:`xarray.Dataset` attributes.
    coords : dict of {hashable: array_like}, optional
        Coordinates for the dataset
    dims : dict of {hashable: iterable of hashable}, optional
        Dimensions of each variable. The keys are variable names, values are lists of
        coordinates.
    sample_dims : iterable of hashable, optional
        Dimensions that should be assumed to be present in _all_ variables.
        If missing, they will be added as the dimensions corresponding to the
        leading axis.
    index_origin : int, optional
        Passed to :func:`generate_dims_coords`
    skip_event_dims : bool
        Passed to :func:`generate_dims_coords`

    Returns
    -------
    xarray.Dataset

    Examples
    --------
    Generate a :class:`~xarray.Dataset` with two variables
    using ``sample_dims``:

    .. ipython::

        In [1]: from arviz_base import dict_to_dataset
           ...: import numpy as np
           ...:
           ...: rng = np.random.default_rng(2)
           ...: dict_to_dataset(
           ...:     {"a": rng.normal(size=(4, 100)), "b": rng.normal(size=(4, 100))},
           ...:     sample_dims=["chain", "draw"],
           ...: )

    Generate a :class:`~xarray.Dataset` with the ``chain`` and ``draw``
    dimensions in different position. Setting the dimensions for ``a``
    to "group" and "chain", ``sample_dims`` will then be used to prepend
    the "draw" dimension only as "chain" is already there.

    .. ipython::

        In [2]: dict_to_dataset(
           ...:     {"a": rng.normal(size=(10, 5, 4)), "b": rng.normal(size=(10, 4))},
           ...:     dims={"a": ["group", "chain"]}
           ...:     sample_dims=["draw", "chain"]
           ...: )

    """
    if dims is None:
        dims = {}

    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]

    data_vars = {}
    for var_name, values in data.items():
        in_dims = dims.get(var_name, [])
        if sample_dims:
            var_dims = [
                sample_dim for sample_dim in sample_dims if sample_dim not in var_dims
            ].extend(in_dims)
        var_dims, var_coords = generate_dims_coords(
            values.shape,
            var_name=var_name,
            dims=var_dims,
            coords=coords,
            index_origin=index_origin,
            skip_event_dims=skip_event_dims,
        )
        data_vars[var_name] = xr.DataArray(values, coords=var_coords, dims=var_dims)

    return xr.Dataset(
        data_vars=data_vars, attrs=make_attrs(attrs=attrs, inference_library=inference_library)
    )


def make_attrs(attrs=None, inference_library=None):
    """Make standard attributes to attach to xarray datasets.

    Parameters
    ----------
    attrs : dict, optional
        Additional attributes to add or overwrite
    inference_library : module, optional
        Library used to perform inference.

    Returns
    -------
    dict
        attrs
    """
    default_attrs = {
        "created_at": datetime.datetime.utcnow().isoformat(),
        "creation_library": "ArviZ",
        "creation_library_version": __version__,
        "creation_library_language": "Python",
    }
    if inference_library is not None:
        library_name = inference_library.__name__
        default_attrs["inference_library"] = library_name
        try:
            version = importlib.metadata.version(library_name)
            default_attrs["inference_library_version"] = version
        except importlib.metadata.PackageNotFoundError:
            if hasattr(inference_library, "__version__"):
                version = inference_library.__version__
                default_attrs["inference_library_version"] = version
    if attrs is not None:
        default_attrs.update(attrs)
    return default_attrs
