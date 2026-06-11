"""Represents an integer set as a union of integer ranges."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property, partial
import itertools
from operator import attrgetter, contains
from typing import Optional, TypeAlias

from ._base import BaseSubset


Ranges: TypeAlias = Sequence[range]


def indices_to_minimal_ranges(indices: Iterable[int]) -> list[range]:
    """Given a set of indices, calculates the minimal set of disjoint ranges equivalent to the indices.
    Returns a list of range objects."""
    ranges = []
    first = None
    for i in sorted(indices):
        if first is None:
            first = last = i
        elif i > last + 1:
            ranges.append(range(first, last + 1))
            first = last = i
        else:
            last = i
    if first is not None:
        ranges.append(range(first, last + 1))
    return ranges

def _ranges_intersection(universe: range, ranges_seq: Sequence[Ranges]) -> list[range]:
    """Given a range corresponding to a universe, and a sequence of range-unions,
    returns a new list of disjoint ranges representing the intersection of the given ranges.
    If the given list of ranges is empty, returns the whole universe (which is implicitly
    the intersection of no ranges)."""
    # NOTE: there might be a more efficient implementation possible,
    # but for simplicity we just take the set intersection and reconstruct the minimal set of ranges.
    if not ranges_seq:  # empty intersection is the full universe
        return [universe]
    in_universe = partial(contains, universe)
    idx_set = set(filter(in_universe, itertools.chain.from_iterable(ranges_seq[0])))
    for ranges in ranges_seq[1:]:
        idx_set &= set(itertools.chain.from_iterable(ranges))
    return indices_to_minimal_ranges(idx_set)

def _ranges_union(ranges_seq: Sequence[Ranges]) -> list[range]:
    """Given a sequence of range-unions, returns a new list of disjoint ranges representing
    the union of the given ranges.
    If the given list of ranges is empty, returns the empty list."""
    ranges: Ranges = sorted([rng for ranges in ranges_seq for rng in ranges], key=attrgetter('start'))
    new_ranges: list[range] = []
    first = None
    for rng in ranges:
        (start, stop) = (rng.start, rng.stop)
        if start >= stop:
            continue
        if first is None:
            (first, last) = (start, stop)
        if start > last:
            new_ranges.append(range(first, last))
            (first, last) = (start, stop)
        else:
            last = max(last, stop)
    if first is not None:
        new_ranges.append(range(first, last))
    return new_ranges

def _ranges_complement(universe: range, ranges: Ranges) -> list[range]:
    """Given a range corresponding to a universe, and a range-union, returns a new list of disjoint ranges
    representing the complement of the given ranges.
    If the given list of ranges is empty, returns the whole universe (which is implicitly
    the complement of the empty set)."""
    if not ranges:
        return [universe]
    new_ranges: list[range] = []
    last = universe.start
    # NOTE: this assumes the ranges are sorted by increasing lower bound
    for rng in ranges:
        (start, stop) = (rng.start, rng.stop)
        if stop <= start:  # skip empty ranges
            continue
        if start > last:
            new_ranges.append(range(last, start))
        last = max(last, stop)
    if (last is not None) and (last < universe.stop):
        new_ranges.append(range(last, universe.stop))
    return new_ranges

def _range_contains(range1: range, range2: range) -> bool:
    """Returns True if range1 completely contains range2."""
    return (range1.start <= range2.start) and (range1.stop >= range2.stop)

def _check_universe_ranges_match(universe_range1: range, universe_range2: range) -> None:
    """Checks whether two universes, represented as ranges, match.
    If not, raises a ValueError."""
    if universe_range1 != universe_range2:
        raise ValueError('universes do not match')


@dataclass(frozen=True, eq=False)
class RangeUnionSubset(BaseSubset[int]):
    """A subset of an integer universe, represented as a disjoint union of sorted ranges.
    This is often a more efficient data structure than a set for enumeration and membership checks,
    as it can be much more compact when there are a lot of contiguous elements in the subset."""

    _universe_range: range
    ranges: Ranges

    def __init__(self, universe_range: range, ranges: Ranges) -> None:
        object.__setattr__(self, '_universe_range', universe_range)
        object.__setattr__(self, 'ranges', ranges)
        # make sure ranges are valid for the universe
        for rng in self.ranges:
            if rng.step not in [1, None]:
                raise ValueError('ranges may not have step != 1')
            if not _range_contains(self._universe_range, rng):
                raise ValueError(
                    f'invalid range [{rng.start}, {rng.stop}), bounds must be contained within universe range'
                )
            if rng.start >= rng.stop:
                raise ValueError(f'invalid range [{rng.start}, {rng.stop}), cannot have start >= stop')
        # make sure ranges are sorted and disjoint
        for (rng1, rng2) in zip(self.ranges, self.ranges[1:]):
            if rng1.stop > rng2.start:
                raise ValueError('ranges must be sorted and not overlap')

    def _universes_match(self, other: BaseSubset[int]) -> bool:
        if isinstance(other, RangeUnionSubset):
            return self._universe_range == other._universe_range
        return super()._universes_match(other)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(universe_range={self._universe_range!r}, ranges={self.ranges!r})'

    @classmethod
    def from_ranges(cls, universe_range: range, ranges: Ranges) -> RangeUnionSubset:
        """Convenience constructor from a universe range and list of ranges, not necessarily sorted or disjoint."""
        return cls(universe_range, _ranges_union([ranges]))

    def __iter__(self) -> Iterator[int]:
        return itertools.chain.from_iterable(self.ranges)

    # TODO: we could implement __contains__ via bisection instead of constructing the set.
    # This is probably more efficient when it's only called a few times, but less so if it is called frequently.

    @cached_property
    def _size(self) -> int:
        """Gets the size of the range union.
        This can be calculated as a simple sum, since the ranges are assumed to be disjoint."""
        return sum(map(len, self.ranges))

    def __len__(self) -> int:
        return self._size

    def _get_universe(self) -> Optional[BaseSubset[int] | frozenset[int]]:
        return frozenset(self._universe_range)

    def _get_elements(self) -> frozenset[int]:
        return frozenset(self.__iter__())

    def __and__(self, other: object) -> BaseSubset[int]:
        if isinstance(other, RangeUnionSubset):
            _check_universe_ranges_match(self._universe_range, other._universe_range)
            intersection_ranges = _ranges_intersection(self._universe_range, [self.ranges, other.ranges])
            return type(self)(self._universe_range, intersection_ranges)
        return super().__and__(other)

    def __or__(self, other: object) -> BaseSubset[int]:
        if isinstance(other, RangeUnionSubset):
            _check_universe_ranges_match(self._universe_range, other._universe_range)
            union_ranges = _ranges_union([self.ranges, other.ranges])
            return type(self)(self._universe_range, union_ranges)
        return super().__or__(other)

    def __invert__(self) -> RangeUnionSubset:
        return type(self)(self._universe_range, _ranges_complement(self._universe_range, self.ranges))
