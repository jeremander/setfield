"""Utilities for working with sets of Unicode characters."""

from collections.abc import Iterable, Iterator, Sequence, Set
import sys
from typing import Optional, Self

from setfield._base import IsoMappedSubset
from setfield.ranges import RangeUnionSubset


NUM_UNICODE = sys.maxunicode + 1
UNICODE_RANGE = range(NUM_UNICODE)


class AllUnicode(Set[str]):
    """Singleton class representing the set of all Unicode characters.

    This represents the characters implicitly as a range of code points, instead of concretely in memory."""

    _instance: Optional[Self] = None

    def __new__(cls) -> Self:
        """Returns the singleton instance of the AllUnicode class."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __contains__(self, item: object) -> bool:
        try:
            # check if it is a single character
            _ = ord(item)  # type: ignore[arg-type]
            return True
        except TypeError:
            return False

    def __iter__(self) -> Iterator[str]:
        return map(chr, UNICODE_RANGE)

    def __len__(self) -> int:
        return NUM_UNICODE

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        # fall back on set comparison
        return super().__eq__(other)

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'


# constant representing the set of all Unicode characters
ALL_UNICODE = AllUnicode()


class UnicodeRanges(IsoMappedSubset[int, str]):
    """Represents a set of Unicode characters as a union of code point ranges."""

    def __init__(self, ranges: Iterable[range]) -> None:
        range_seq = ranges if isinstance(ranges, Sequence) else list(ranges)
        range_union = RangeUnionSubset(UNICODE_RANGE, range_seq)
        super().__init__(range_union, chr, ord)

    def _get_universe(self) -> Optional[Set[str]]:
        return AllUnicode()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, UnicodeRanges):
            return self.base_subset == other.base_subset
        return super().__eq__(other)
