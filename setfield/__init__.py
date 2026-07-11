"""Provides a framework for expressing a boolean field of sets construction, i.e. an ambient universe set with subsets
that can be combined with boolean setwise operations (intersection, union, complement)."""

from ._base import BaseSubset as BaseSubset
from ._base import DynamicSubset as DynamicSubset
from ._base import FilterSubset as FilterSubset
from ._base import IsoMappedSubset as IsoMappedSubset
from ._base import MappedSubset as MappedSubset
from ._base import Subset as Subset
from ._base import SubsetComplement as SubsetComplement
from ._base import SubsetIntersection as SubsetIntersection
from ._base import SubsetUnion as SubsetUnion
from ._base import get_empty_subset as get_empty_subset
from ._base import get_full_subset as get_full_subset
from .eval import BOOLEAN_SAFE_NODE_TYPES as BOOLEAN_SAFE_NODE_TYPES
from .eval import safe_eval as safe_eval
from .eval import safe_eval_boolean_expr as safe_eval_boolean_expr
from .ranges import Ranges as Ranges
from .ranges import RangeUnionSubset as RangeUnionSubset
from .unicode import AllUnicode as AllUnicode
from .unicode import UnicodeRanges as UnicodeRanges


__version__ = '0.2.4'
