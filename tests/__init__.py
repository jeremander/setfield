from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from functools import cache
import operator

import hypothesis.strategies as st

from setfield import (
    BaseSubset,
    DynamicSubset,
    RangeUnionSubset,
    Subset,
    SubsetIntersection,
    SubsetUnion,
    get_empty_subset,
    get_full_subset,
)


# max size of a set algebra universe for testing
TEST_UNIVERSE_SIZE = 1_000
TEST_UNIVERSE_MAX = TEST_UNIVERSE_SIZE - 1
TEST_RANGE = range(TEST_UNIVERSE_SIZE)
TEST_UNIVERSE = frozenset(TEST_RANGE)


@cache
def get_universe(universe_size: int) -> frozenset[int]:
    return frozenset(range(universe_size))

def subset_static(elements: Iterable[int], *, universe_size: int = TEST_UNIVERSE_SIZE) -> Subset[int]:
    return Subset(get_universe(universe_size), elements)


# STRATEGIES

@st.composite
def subsets_static(draw: st.DrawFn, *, max_size: int = 25, universe_size: int = TEST_UNIVERSE_SIZE) -> Subset[int]:
    elements = draw(st.lists(st.integers(0, universe_size - 1), max_size=max_size, unique=True))
    return subset_static(elements, universe_size=universe_size)

@st.composite
def subsets_dynamic(
    draw: st.DrawFn,
    *,
    max_size: int = 25,
    universe_size: int = TEST_UNIVERSE_SIZE,
) -> DynamicSubset[int]:
    subset = draw(subsets_static(max_size=max_size, universe_size=universe_size))
    elements: AbstractSet[int] = subset.elements
    return DynamicSubset(get_universe(universe_size), lambda: elements)

def subset_intersections(
    base_strat: st.SearchStrategy[BaseSubset[int]],
    *,
    max_width: int = 5,
    universe_size: int = TEST_UNIVERSE_SIZE,
) -> st.SearchStrategy[BaseSubset[int]]:
    return (
        st.lists(base_strat, max_size=max_width)
        .map(lambda subsets: SubsetIntersection(get_universe(universe_size), subsets))
    )

def subset_unions(
    base_strat: st.SearchStrategy[BaseSubset[int]],
    *,
    max_width: int = 5,
    universe_size: int = TEST_UNIVERSE_SIZE,
) -> st.SearchStrategy[BaseSubset[int]]:
    return (
        st.lists(base_strat, max_size=max_width)
        .map(lambda subsets: SubsetUnion(get_universe(universe_size), subsets))
    )

@cache
def empty_subset(universe_size: int = TEST_UNIVERSE_SIZE) -> Subset[int]:
    return get_empty_subset(get_universe(universe_size))

@cache
def universe_subset(universe_size: int = TEST_UNIVERSE_SIZE) -> Subset[int]:
    return get_full_subset(get_universe(universe_size))


@st.composite
def subsets_range_union(
    draw: st.DrawFn,
    *,
    max_num_ranges: int = 10,
    universe_size: int = TEST_UNIVERSE_SIZE,
) -> RangeUnionSubset:
    """Hypothesis strategy for generating RangeUnionSubsets."""
    def _get_range(upper: int) -> range:
        pair = sorted(draw(st.tuples(st.integers(0, upper), st.integers(0, upper))))
        return range(pair[0], pair[1] + 1)
    def _get_ranges(num_ranges: int, upper: int) -> list[range]:
        if (upper < 0) or (num_ranges == 0):
            return []
        rng = _get_range(upper)
        return _get_ranges(num_ranges - 1, rng.start - 1) + [rng]
    num_ranges = draw(st.integers(0, max_num_ranges))
    return RangeUnionSubset(range(universe_size), _get_ranges(num_ranges, universe_size - 1))

def subsets(
    *,
    max_leaf_size: int = 25,
    max_leaves: int = 25,
    max_width: int = 5,
    universe_size: int = TEST_UNIVERSE_SIZE,
) -> st.SearchStrategy[BaseSubset[int]]:
    """Hypothesis strategy for generating various BaseSubset objects."""
    subsets_leaf = (
        subsets_range_union(universe_size=universe_size)
        | subsets_static(max_size=max_leaf_size, universe_size=universe_size)
        | subsets_dynamic(max_size=max_leaf_size, universe_size=universe_size)
        | st.just(universe_subset(universe_size))
    )
    subsets_rec_without_negation = st.recursive(
        subsets_leaf,
        extend=lambda xs: (
            xs
            | subset_intersections(xs, max_width=max_width, universe_size=universe_size)
            | subset_unions(xs, max_width=max_width, universe_size=universe_size)
        ),
        max_leaves=max_leaves,
    )
    subsets_rec_with_negation = st.recursive(
        subsets_leaf,
        extend=lambda xs: (
            xs
            | xs.map(operator.invert)
            | subset_intersections(xs, max_width=max_width, universe_size=universe_size)
            | subset_unions(xs, max_width=max_width, universe_size=universe_size)
        ),
        max_leaves=max_leaves,
    )
    return st.one_of(subsets_rec_without_negation, subsets_rec_with_negation)
