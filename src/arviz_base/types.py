"""ArviZ type definitions."""
from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from xarray.core.types import DaCompatible

CoordSpec = Mapping[Hashable, Any]
DimSpec = Mapping[Hashable, Iterable[Hashable]]

DictData = Mapping[Hashable, "ArrayLike"]
