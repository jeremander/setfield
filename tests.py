import ast
from collections.abc import Callable
from contextlib import suppress
from functools import reduce
import operator
import re
from typing import Optional, TypeVar

from hypothesis import event, given, settings
import hypothesis.errors
import hypothesis.strategies as st
import pytest

import setfield
from setfield import (
    BOOLEAN_SAFE_NODE_TYPES,
    BaseSubset,
    DynamicSubset,
    FilterSubset,
    IsoMappedSubset,
    MappedSubset,
    RangeUnionSubset,
    Subset,
    SubsetComplement,
    SubsetIntersection,
    SubsetUnion,
    indices_to_minimal_ranges,
    safe_eval,
    safe_eval_boolean_expr,
)


T = TypeVar('T')


# max size of a set algebra universe for testing
TEST_UNIVERSE_SIZE = 1_000
TEST_UNIVERSE_MAX = TEST_UNIVERSE_SIZE - 1
TEST_RANGE = range(TEST_UNIVERSE_SIZE)
TEST_UNIVERSE = set(TEST_RANGE)


def subset_static(elements: set[int]) -> Subset[int]:
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

@st.composite
def subsets_range_union(draw, *, max_num_ranges: int = 10):
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

def subset_intersections(base_strat, *, max_width: int = 5):
    return st.lists(base_strat, max_size=max_width).map(lambda subsets: SubsetIntersection(TEST_UNIVERSE, subsets))

def subset_unions(base_strat, *, max_width: int = 5):
    return st.lists(base_strat, max_size=max_width).map(lambda subsets: SubsetUnion(TEST_UNIVERSE, subsets))

empty_subset = Subset.empty(TEST_UNIVERSE)
universe_subset = Subset.full(TEST_UNIVERSE)

def subsets(*, max_leaf_size: int = 25, max_leaves: int = 25, max_width: int = 5):
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


# TESTS

class TestRanges:

    @pytest.mark.parametrize(['indices', 'ranges'], [
        (
            [],
            [],
        ),
        (
            [0],
            [range(0, 1)],
        ),
        (
            [0, 1, 2],
            [range(0, 3)],
        ),
        (
            [10, 11, 12],
            [range(10, 13)],
        ),
        (
            [0, 2, 3],
            [range(0, 1), range(2, 4)],
        ),
        (
            [3, 0, 2],
            [range(0, 1), range(2, 4)],
        ),
    ])
    def test_indices_to_minimal_ranges(self, indices, ranges):
        assert indices_to_minimal_ranges(indices) == ranges

    @pytest.mark.parametrize(['universe', 'ranges_seq', 'intersection'], [
        (
            range(10),
            [],
            [range(10)],
        ),
        (
            range(10),
            [[]],
            [],
        ),
        (
            range(10),
            [[range(0, 0)]],
            [],
        ),
        (
            range(10),
            [[range(10)]],
            [range(10)],
        ),
        (
            range(10),
            [[range(20)]],
            [range(10)],
        ),
        (
            range(10),
            [[range(-20, 20)]],
            [range(10)],
        ),
        (
            range(10),
            [[range(-20, 5)]],
            [range(5)],
        ),
        (
            range(10),
            [[range(0, 3), range(6, 9)]],
            [range(0, 3), range(6, 9)],
        ),
        (
            range(10),
            [[range(0, 3), range(2, 5)]],
            [range(5)],
        ),
        (
            range(10),
            [[range(0, 3)], [range(6, 9)]],
            [],
        ),
        (
            range(10),
            [[range(0, 3)], [range(2, 5)]],
            [range(2, 3)],
        ),
        (
            range(10),
            [[range(2, 5)], [range(0, 3)]],
            [range(2, 3)],
        ),
    ])
    def test_ranges_intersection(self, universe, ranges_seq, intersection):
        assert setfield._ranges_intersection(universe, ranges_seq) == intersection

    @pytest.mark.parametrize(['ranges_seq', 'union'], [
        (
            [],
            [],
        ),
        (
            [[]],
            [],
        ),
        (
            [[range(0, 10)]],
            [range(0, 10)],
        ),
        (
            [[range(0, 10), range(20, 30)]],
            [range(0, 10), range(20, 30)],
        ),
        (
            [[range(0, 10), range(5, 15)]],
            [range(0, 15)],
        ),
        (
            [[range(0, 10)], [range(5, 15)]],
            [range(0, 15)],
        ),
        (
            [[range(0, 10), range(20, 30)], [range(5, 25)]],
            [range(0, 30)],
        ),
        # empty range
        (
            [[range(10, 10)]],
            [],
        ),
        (
            [[range(0, 10), range(5, 5)]],
            [range(0, 10)],
        ),
        (
            [[range(0, 10), range(10, 10)]],
            [range(0, 10)],
        ),
    ])
    def test_ranges_union(self, ranges_seq, union):
        assert setfield._ranges_union(ranges_seq) == union

    @pytest.mark.parametrize(['universe', 'ranges', 'complement'], [
        (
            range(100),
            [],
            [range(0, 100)],
        ),
        (
            range(100),
            [range(0, 0)],
            [range(0, 100)],
        ),
        (
            range(100),
            [range(0, 1)],
            [range(1, 100)],
        ),
        (
            range(100),
            [range(0, 10)],
            [range(10, 100)],
        ),
        (
            range(100),
            [range(10, 20)],
            [range(0, 10), range(20, 100)],
        ),
        (
            range(100),
            [range(0, 10), range(20, 30)],
            [range(10, 20), range(30, 100)],
        ),
        (
            range(100),
            [range(0, 10), range(10, 20)],
            [range(20, 100)],
        ),
        (
            range(100),
            [range(0, 10), range(5, 15)],
            [range(15, 100)],
        ),
        (
            range(100),
            [range(0, 10), range(5, 10)],
            [range(10, 100)],
        ),
        (
            range(100),
            [range(0, 10), range(4, 6)],
            [range(10, 100)],
        ),
    ])
    def test_ranges_complement(self, universe, ranges, complement):
        assert setfield._ranges_complement(universe, ranges) == complement


class TestSubset:

    def _test_base_subset(self, subset):
        """Tests properties that should hold for every BaseSubset."""
        with suppress(hypothesis.errors.InvalidArgument):
            event(f'type: {type(subset).__name__}')
        assert isinstance(subset, BaseSubset)
        assert subset == subset
        assert subset != 123
        elements = subset.elements
        assert set(subset) == elements
        assert len(subset) == len(elements)
        assert bool(subset) == (len(subset) > 0)
        assert all((elt in subset) == (elt in elements) for elt in subset)
        assert subset == elements
        assert subset != list(elements)
        assert elements == subset
        assert sorted(iter(subset)) == sorted(subset.elements)
        # equality holds even if representation is different
        assert subset_static(elements) == subset
        neg_subset = ~subset
        assert subset != neg_subset
        # check necessary boolean properties
        assert subset <= subset
        assert subset <= subset.elements
        assert subset.elements >= subset
        assert empty_subset <= subset
        assert (empty_subset < subset) == (empty_subset != subset)
        assert subset <= universe_subset
        assert (subset < universe_subset) == (subset != universe_subset)
        with pytest.raises(TypeError, match='not supported'):
            _ = subset < 123
        # law of double negation
        assert ~neg_subset == subset
        if isinstance(subset, RangeUnionSubset):
            assert isinstance(neg_subset, RangeUnionSubset)
        elif isinstance(subset, SubsetComplement):
            assert neg_subset is subset.subset
        else:
            assert ~neg_subset is subset
            assert type(neg_subset) is SubsetComplement
        # law of idempotence
        assert subset & subset == subset
        # assert subset & subset is subset
        assert subset | subset == subset
        # assert subset | subset is subset
        # laws of identity
        assert subset & universe_subset == subset
        assert subset | empty_subset == subset
        # laws of annihilation
        assert subset & empty_subset == empty_subset
        assert subset | universe_subset == universe_subset
        # law of excluded middle
        assert subset.isdisjoint(neg_subset)
        assert subset ^ subset == empty_subset
        assert subset ^ neg_subset == universe_subset
        if isinstance(subset, (Subset, RangeUnionSubset)):
            # eval should be the inverse of repr
            subset2 = eval(repr(subset))
            assert type(subset2) is type(subset)
            assert subset2 == subset

    def test_empty_subset(self):
        assert type(empty_subset) is Subset
        assert len(empty_subset) == 0
        assert ~empty_subset == universe_subset
        self._test_base_subset(empty_subset)

    def test_universe_subset(self):
        assert type(universe_subset) is Subset
        assert len(universe_subset) == TEST_UNIVERSE_MAX + 1
        assert 0 in universe_subset
        assert TEST_UNIVERSE_MAX in universe_subset
        assert {0, 1, TEST_UNIVERSE_MAX} < universe_subset
        assert (TEST_UNIVERSE_MAX + 1) not in universe_subset
        assert len(~universe_subset) == 0
        assert set(~universe_subset) == set()
        assert ~universe_subset == empty_subset
        self._test_base_subset(universe_subset)

    def test_infinite_universe(self):
        subset = Subset(None, {0, 1, 2})
        assert set(subset) == {0, 1, 2}
        assert len(subset) == 3
        neg_subset = ~subset
        with pytest.raises(ValueError, match='cannot get length of infinite universe'):
            _ = len(neg_subset)
        with pytest.raises(ValueError, match='cannot enumerate infinite universe'):
            _ = set(neg_subset)
        assert ~neg_subset == subset
        empty_intersection: SubsetIntersection[int] = SubsetIntersection(None, [])
        with pytest.raises(ValueError, match='cannot get length of infinite universe'):
            _ = len(empty_intersection)
        with pytest.raises(ValueError, match='cannot enumerate infinite universe'):
            _ = set(empty_intersection)
        mapped = MappedSubset(subset, lambda i: i % 2)
        assert mapped.universe is None
        assert len(mapped) == 2
        assert set(mapped) == {0, 1}
        subset2 = Subset(None, {1})
        subset3 = subset & (~subset2)
        assert subset3.universe is None
        assert len(subset3) == 2
        assert set(subset3) == {0, 2}

    def test_elements_not_in_universe(self):
        universe = {0, 1, 2}
        with pytest.raises(ValueError, match='3 is not an element of the universe'):
            _ = Subset(universe, {1, 3})
        subset = DynamicSubset(universe, lambda: {1, 3})
        assert subset.universe == universe
        # error is deferred until elements are created
        with pytest.raises(ValueError, match='3 is not an element of the universe'):
            _ = subset.elements

    def test_dynamic_subset(self):
        subset = DynamicSubset(TEST_RANGE, lambda: {0, 1, 2})
        assert subset.universe == TEST_UNIVERSE
        assert subset.elements == {0, 1, 2}
        self._test_base_subset(subset)
        subset = DynamicSubset(TEST_RANGE, lambda: range(3))
        assert subset.universe == TEST_UNIVERSE
        assert subset.elements == {0, 1, 2}

    def test_filter_subset(self):
        subset = FilterSubset(TEST_RANGE, lambda i: i < 5)  # type: ignore[operator]
        assert len(subset) == 5
        assert 0 in subset
        assert 5 not in subset
        assert set(subset) == set(range(5))
        self._test_base_subset(subset)

    def test_range_union(self):
        subset = RangeUnionSubset(TEST_RANGE, [range(5, 10), range(15, 20)])
        assert len(subset) == 10
        assert 5 in subset
        assert 10 not in subset
        assert 15 in subset
        assert 20 not in subset
        assert subset.universe == TEST_UNIVERSE
        assert subset.elements == {5, 6, 7, 8, 9, 15, 16, 17, 18, 19}
        self._test_base_subset(subset)

    @pytest.mark.parametrize(['ranges', 'error'], [
        (
            [range(0, 10, 2)],
            'ranges may not have step != 1',
        ),
        (
            [range(-5, 5)],
            'bounds must be contained within universe range',
        ),
        (
            [range(0, 101)],
            'bounds must be contained within universe range',
        ),
        (
            [range(0, 10), range(-5, 5)],
            'bounds must be contained within universe range',
        ),
        (
            [range(20, 10)],
            'cannot have start >= stop',
        ),
        (
            [range(20, 20)],
            'cannot have start >= stop',
        ),
    ])
    def test_range_union_invalid_bounds(self, ranges, error):
        with pytest.raises(ValueError, match=error):
            _ = RangeUnionSubset(range(100), ranges)

    @pytest.mark.parametrize('ranges', [
        # disjoint but unsorted
        [range(15, 20), range(5, 10)],
        # sorted but not disjoint
        [range(5, 10), range(7, 12)],
        [range(5, 10), range(9, 15)],
        # neither sorted nor disjoint
        [range(5, 10), range(3, 8)],
        # empty range
        [range(10, 10)],
        [range(10), range(10, 10)],
        [range(10), range(20, 20), range(5, 15)],
    ])
    def test_range_union_unsorted_or_overlapping_ranges(self, ranges):
        with pytest.raises(ValueError, match='(ranges must be sorted and not overlap)|(cannot have start >= stop)'):
            _ = RangeUnionSubset(TEST_RANGE, ranges)
        subset = RangeUnionSubset.from_ranges(TEST_RANGE, ranges)
        self._test_base_subset(subset)

    @pytest.mark.parametrize(['subset', 'repr_pattern'], [
        (
            Subset({0, 1, 2}, {0}),
            r'Subset\(universe=\{0, 1, 2\}, elements=\{0\}\)',
        ),
        (
            DynamicSubset({0, 1, 2}, lambda: {0}),
            r'DynamicSubset\(universe=\{0, 1, 2\}, get_elements=.+\)',
        ),
        (
            FilterSubset({0, 1, 2}, lambda i: i % 2 == 0),  # type: ignore[operator]
            r'FilterSubset\(universe=\{0, 1, 2\}, predicate=.+\)',
        ),
        (
            RangeUnionSubset(range(5), [range(2, 4), range(4, 5)]),
            r'RangeUnionSubset\(universe_range=range\(0, 5\), ranges=\[range\(2, 4\), range\(4, 5\)\]\)',
        ),
    ])
    def test_repr(self, subset, repr_pattern):
        subset_repr = repr(subset)
        assert str(subset) == subset_repr
        assert re.match(repr_pattern, subset_repr)

    def test_subset_mapped(self):
        base_subset = subset_static({0, 1, 2, 3, 4})
        subset = MappedSubset(base_subset, lambda i: i % 3)
        assert 1 in subset
        assert 3 not in subset
        assert subset.universe == {0, 1, 2}
        assert set(subset) == {0, 1, 2}
        subset = MappedSubset(base_subset, lambda i: i % 3)

    def test_subset_iso_mapped(self):
        base_subset = subset_static({0, 1, 2})
        # valid one-to-one mapping
        def _safe_int(s: str) -> int:
            if not isinstance(s, str):
                raise TypeError('input must be a string')
            return int(s)
        subset = IsoMappedSubset(base_subset, str, _safe_int)
        assert '1' in subset
        assert 1 not in subset
        assert list(subset) == ['0', '1', '2']
        assert len(subset) == 3
        assert subset.universe == set(map(str, TEST_UNIVERSE))
        assert set(subset) == {'0', '1', '2'}
        neg_subset = (~subset)
        assert neg_subset.universe == subset.universe
        assert len(neg_subset) == TEST_UNIVERSE_SIZE - 3
        assert neg_subset < subset.universe
        # one-to-one on the proper domain, but __contains__ can cause issues if querying an element not in the universe
        subset = IsoMappedSubset(base_subset, str, int)
        assert '1' in subset
        assert 1 in subset  # danger!
        assert list(subset) == ['0', '1', '2']
        assert len(subset) == 3
        assert subset.universe == set(map(str, TEST_UNIVERSE))
        assert set(subset) == {'0', '1', '2'}
        # invalid one-to-one-mapping
        subset = IsoMappedSubset(base_subset, lambda i: i // 2, lambda i: i * 2)  # type: ignore
        assert 0 in subset
        assert 1 in subset
        assert 2 not in subset
        assert list(subset) == [0, 0, 1]  # not unique!
        assert len(subset) == 3  # wrong!
        assert subset.universe == set(range(TEST_UNIVERSE_SIZE // 2))
        assert subset.elements == {0, 1}
        assert set(subset) == {0, 1}

    def test_boolean_operators(self):
        subset1 = subset_static({0, 1, 2})
        subset2 = subset_static({2, 3, 4})
        assert subset1 != subset2
        assert not subset1.isdisjoint(subset2)
        assert subset1 & subset2 == {2}
        assert subset1 ^ subset2 == {0, 1, 3, 4}
        assert empty_subset < subset1 < universe_subset
        union = subset1 | subset2
        assert type(union) is SubsetUnion
        assert union == {0, 1, 2, 3, 4}
        assert subset1 < union
        assert subset1 <= union
        assert union > subset1
        assert union >= subset1
        assert subset1.elements < union
        assert subset1 < union.elements
        assert type(subset1 | subset2.elements) is SubsetUnion
        assert (subset1 | subset2.elements) == union
        assert type(subset1.elements | subset2) is SubsetUnion
        assert (subset1.elements | subset2) == union
        assert type(subset1.elements | subset2.elements) is set
        assert (subset1.elements | subset2.elements) == union
        with pytest.raises(TypeError, match='unsupported operand'):
            _ = subset1 | 123
        with pytest.raises(TypeError, match='unsupported operand'):
            _ = '123' | subset1
        with pytest.raises(TypeError, match='unsupported operand'):
            _ = union | 123
        intersection = subset1 & subset2
        assert type(intersection) is SubsetIntersection
        assert intersection == {2}
        assert intersection < subset1
        assert intersection <= subset1
        assert subset1 > intersection
        assert subset1 >= intersection
        assert subset1.elements > intersection
        assert subset1 > intersection.elements
        assert type(subset1 & subset2.elements) is SubsetIntersection
        assert (subset1 & subset2.elements) == intersection
        assert type(subset1.elements & subset2) is SubsetIntersection
        assert (subset1.elements & subset2) == intersection
        assert type(subset1.elements & subset2.elements) is set
        assert (subset1.elements & subset2.elements) == intersection
        with pytest.raises(TypeError, match='unsupported operand'):
            _ = subset1 & '123'
        with pytest.raises(TypeError, match='unsupported operand'):
            _ = 123 & subset1
        with pytest.raises(TypeError, match='unsupported operand'):
            _ = intersection & 123
        diff = subset1 - subset2
        assert type(diff) is SubsetIntersection
        assert diff == {0, 1}
        assert empty_subset < diff < subset1
        assert type(subset1 - subset2.elements) is SubsetIntersection
        assert (subset1 - subset2.elements) == diff
        assert type(subset1.elements - subset2) is SubsetIntersection
        assert (subset1.elements - subset2) == diff
        assert type(subset1.elements - subset2.elements) is set
        assert (subset1.elements - subset2.elements) == diff
        with pytest.raises(TypeError, match='unsupported operand'):
            _ = subset1 - '123'
        with pytest.raises(TypeError, match='unsupported operand'):
            _ = 123 - subset1

    @pytest.mark.parametrize(['subset1', 'subset2'], [
        (
            Subset(set(range(5)), {0, 1, 2}),
            Subset(set(range(6)), {0, 1, 2}),
        ),
        (
            RangeUnionSubset(range(5), [range(3)]),
            RangeUnionSubset(range(6), [range(3)]),
        ),
    ])
    def test_boolean_operator_universe_mismatch(self, subset1, subset2):
        for op in [
            operator.lt, operator.le, operator.ge, operator.gt,
            operator.and_, operator.or_, operator.xor, operator.sub
        ]:
            with pytest.raises(ValueError, match='universes do not match'):
                op(subset1, subset2)
        # equality does *not* raise an error
        assert not (subset1 == subset2)
        assert subset1 != subset2

    @given(subsets_static())
    def test_subset_static_generic(self, subset):
        self._test_base_subset(subset)

    @given(subsets_dynamic())
    def test_subset_dynamic_generic(self, subset):
        self._test_base_subset(subset)

    @given(subsets_range_union())
    def test_unicode_ranges_generic(self, subset):
        self._test_base_subset(subset)

    @given(subset_intersections(subsets_static()))
    def test_subset_intersection(self, subset):
        self._test_base_subset(subset)
        if not subset.subsets:
            assert subset.elements is TEST_UNIVERSE
        assert all(subset <= component for component in subset.subsets)
        if subset.subsets:
            assert reduce(set.intersection, (component.elements for component in subset.subsets)) == subset.elements

    @given(subset_unions(subsets_static()))
    def test_subset_union(self, subset):
        self._test_base_subset(subset)
        if not subset.subsets:
            assert subset.elements == set()
        assert all(component <= subset for component in subset.subsets)
        assert reduce(set.union, (component.elements for component in subset.subsets), set()) == subset.elements

    @given(subsets())
    @settings(deadline=None)
    def test_subset_generic(self, subset):
        self._test_base_subset(subset)


ARITH_SAFE_NODE_TYPES = BOOLEAN_SAFE_NODE_TYPES | {
    ast.Constant,
    ast.BinOp,
    ast.Add, ast.Div, ast.Mult, ast.USub,
}

def safe_eval_arith_expr(expr: str, eval_name: Optional[Callable[[str], T]] = None) -> T:
    return safe_eval(expr, eval_name=eval_name, safe_node_types=ARITH_SAFE_NODE_TYPES)

def _get_set(name: str) -> set[int]:
    match name:
        case 'A':
            return {1, 2, 3}
        case 'B':
            return {3, 4}
        case 'C':
            return {1, 3, 5}
    raise ValueError(f'invalid name: {name}')

small_universe = {1, 2, 3, 4, 5}

def example_interpret(expr: str) -> BaseSubset[int]:
    def eval_name(name: str) -> Subset[int]:
        return Subset(small_universe, _get_set(name))
    return safe_eval_boolean_expr(expr, eval_name)


class TestInterpretation:

    @pytest.mark.parametrize(['expr', 'eval_names', 'value', 'error'], [
        (
            '',
            [None, _get_set],
            None,
            'invalid expression',
        ),
        (
            '123 +',
            [None, _get_set],
            None,
            'invalid expression',
        ),
        (
            '123',
            [None, _get_set],
            123,
            None,
        ),
        (
            '~123',
            [None, _get_set],
            -124,
            None,
        ),
        (
            '1 + 2.3',
            [None, _get_set],
            3.3,
            None,
        ),
        (
            '-1',
            [None, _get_set],
            -1,
            None,
        ),
        (
            'A',
            None,
            None,
            'disallowed construct: Name',
        ),
        (
            'A',
            _get_set,
            {1, 2, 3},
            None,
        ),
        (
            '(A)',
            None,
            None,
            'disallowed construct: Name',
        ),
        (
            '(A)',
            _get_set,
            {1, 2, 3},
            None,
        ),
        (
            '()',
            [None, _get_set],
            None,
            'disallowed construct: Tuple',
        ),
        (
            'A | B',
            None,
            None,
            'disallowed construct: Name',
        ),
        (
            'A | B',
            _get_set,
            {1, 2, 3, 4},
            None,
        ),
        (
            'A&B',
            None,
            None,
            'disallowed construct: Name',
        ),
        (
            'A&B',
            _get_set,
            {3},
            None,
        ),
        (
            'A & B | C',
            _get_set,
            {1, 3, 5},
            None,
        ),
        (
            'A & (B | C)',
            _get_set,
            {1, 3},
            None,
        ),
        (
            'A - B',
            _get_set,
            {1, 2},
            None,
        ),
        (
            'D',
            None,
            None,
            'disallowed construct: Name',
        ),
        (
            'D',
            _get_set,
            None,
            'invalid name: D',
        ),
        (
            'set()',
            [None, _get_set],
            None,
            'disallowed construct: Call',
        ),
    ])
    def test_interpret_arith_expr(self, expr, eval_names, value, error):
        if not isinstance(eval_names, list):
            eval_names = [eval_names]
        for eval_name in eval_names:
            if value is None:  # expect an error
                with pytest.raises(ValueError, match=error):
                    _ = safe_eval_arith_expr(expr, eval_name)
            else:
                assert error is None
                assert safe_eval_arith_expr(expr, eval_name) == value

    @pytest.mark.parametrize(['expr', 'output_type', 'output_set'], [
        (
            'A',
            Subset,
            {1, 2, 3},
        ),
        (
            '(((A)))',
            Subset,
            {1, 2, 3},
        ),
        (
            '~A',
            SubsetComplement,
            {4, 5},
        ),
        (
            '~~A',
            Subset,
            {1, 2, 3}
        ),
        (
            'A & B',
            SubsetIntersection,
            {3},
        ),
        (
            'A | B',
            SubsetUnion,
            {1, 2, 3, 4},
        ),
        (
            'A | B | C',
            SubsetUnion,
            {1, 2, 3, 4, 5},
        ),
        (
            'A - B',
            SubsetIntersection,
            {1, 2},
        ),
        (
            'A & B | C',
            SubsetUnion,
            {1, 3, 5},
        ),
        (
            'A & (B | C)',
            SubsetIntersection,
            {1, 3},
        ),
    ])
    def test_interpret_bool_expr_valid(self, expr, output_type, output_set):
        value = example_interpret(expr)
        assert type(value) is output_type
        assert set(value) == output_set

    @pytest.mark.parametrize(['expr', 'error'], [
        (
            '',
            'invalid expression',
        ),
        (
            '1',
            'disallowed construct: Constant',
        ),
        (
            '-1',
            'disallowed construct: USub',
        ),
        (
            '1 + 2.3',
            'disallowed construct: Constant',
        ),
        (
            '()',
            'disallowed construct: Tuple',
        ),
        (
            '{{}}',
            'disallowed construct: Set',
        ),
        (
            '{{1}}',
            'disallowed construct: Set',
        ),
        (
            '(A',
            'invalid expression',
        ),
        (
            '(((A))',
            'invalid expression',
        ),
        (
            '-A',
            'disallowed construct: USub',
        ),
        (
            'D',
            'invalid name: D',
        ),
        (
            'A | D',
            'invalid name: D',
        ),
    ])
    def test_interpret_bool_expr_invalid(self, expr, error):
        with pytest.raises(ValueError, match=error):
            _ = example_interpret(expr)
