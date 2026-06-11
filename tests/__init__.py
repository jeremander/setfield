from collections.abc import Iterable
import operator

import hypothesis.strategies as st

from setfield import (
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


def subset_static(elements: Iterable[int]) -> Subset[int]:
    return Subset(TEST_UNIVERSE, elements)


# STRATEGIES

@st.composite
def subsets_static(draw, *, max_size: int = 25):
    elements = draw(st.lists(st.integers(0, TEST_UNIVERSE_MAX), max_size=max_size, unique=True))
    return subset_static(elements)

@st.composite
def subsets_dynamic(draw, *, max_size: int = 25):
    subset = draw(subsets_static(max_size=max_size))
    elements: set[int] = subset.elements
    return DynamicSubset(TEST_UNIVERSE, lambda: elements)

def subset_intersections(base_strat, *, max_width: int = 5):
    return st.lists(base_strat, max_size=max_width).map(lambda subsets: SubsetIntersection(TEST_UNIVERSE, subsets))

def subset_unions(base_strat, *, max_width: int = 5):
    return st.lists(base_strat, max_size=max_width).map(lambda subsets: SubsetUnion(TEST_UNIVERSE, subsets))


empty_subset = get_empty_subset(TEST_UNIVERSE)
universe_subset = get_full_subset(TEST_UNIVERSE)


@st.composite
def subsets_range_union(draw, *, max_num_ranges: int = 10):
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
    return RangeUnionSubset(TEST_RANGE, _get_ranges(num_ranges, TEST_UNIVERSE_MAX))

def subsets(*, max_leaf_size: int = 25, max_leaves: int = 25, max_width: int = 5):
    """Hypothesis strategy for generating various BaseSubset objects."""
    subsets_leaf = (
        subsets_range_union()
        | subsets_static(max_size=max_leaf_size)
        | subsets_dynamic(max_size=max_leaf_size)
        | st.just(universe_subset)
    )
    subsets_rec_without_negation = st.recursive(
        subsets_leaf,
        extend=lambda xs: xs | subset_intersections(xs, max_width=max_width) | subset_unions(xs, max_width=max_width),
        max_leaves=max_leaves,
    )
    subsets_rec_with_negation = st.recursive(
        subsets_leaf,
        extend=lambda xs: (
            xs
            | xs.map(operator.invert)
            | subset_intersections(xs, max_width=max_width)
            | subset_unions(xs, max_width=max_width)
        ),
        max_leaves=max_leaves,
    )
    return st.one_of(subsets_rec_without_negation, subsets_rec_with_negation)
