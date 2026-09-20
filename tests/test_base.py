"""Unit tests for the core functionality in setfield."""

from contextlib import suppress
import dataclasses
from functools import reduce
import itertools
import operator
import re
from typing import Any, Optional, TypeVar

from hypothesis import event, given, settings
import hypothesis.errors
import hypothesis.strategies as st
import pytest

from setfield import (
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
    get_atoms,
    get_full_subset,
)

from . import (
    TEST_RANGE,
    TEST_UNIVERSE,
    TEST_UNIVERSE_MAX,
    TEST_UNIVERSE_SIZE,
    empty_subset,
    subset_intersections,
    subset_static,
    subset_unions,
    subsets,
    subsets_dynamic,
    subsets_range_union,
    subsets_static,
    universe_subset,
)


T = TypeVar('T')


# TESTS

class TestSubset:

    def _test_base_subset(self, subset: BaseSubset[int]) -> None:
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
        empty = empty_subset()
        assert empty <= subset
        assert (empty < subset) == (empty != subset)
        universe = universe_subset()
        assert subset <= universe
        assert (subset < universe) == (subset != universe)
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
        assert subset & universe == subset
        assert subset | empty == subset
        # laws of annihilation
        assert subset & empty == empty
        assert subset | universe == universe
        # law of excluded middle
        assert subset.isdisjoint(neg_subset)
        assert subset ^ subset == empty
        assert subset ^ neg_subset == universe
        if isinstance(subset, (Subset, RangeUnionSubset)):
            # eval should be the inverse of repr
            subset2 = eval(repr(subset))
            assert type(subset2) is type(subset)
            assert subset2 == subset

    def test_empty_subset(self) -> None:
        empty = empty_subset()
        assert type(empty) is Subset
        assert len(empty) == 0
        assert ~empty == universe_subset()
        self._test_base_subset(empty)

    def test_universe_subset(self) -> None:
        universe = universe_subset()
        assert type(universe) is Subset
        assert len(universe) == TEST_UNIVERSE_MAX + 1
        assert 0 in universe
        assert TEST_UNIVERSE_MAX in universe
        assert {0, 1, TEST_UNIVERSE_MAX} < universe
        assert (TEST_UNIVERSE_MAX + 1) not in universe
        assert len(~universe) == 0
        assert set(~universe) == set()
        assert ~universe == empty_subset()
        self._test_base_subset(universe)

    def test_infinite_universe(self) -> None:
        subset = Subset(None, {0, 1, 2})
        assert set(subset) == {0, 1, 2}
        assert len(subset) == 3
        assert subset == subset
        assert subset >= subset
        assert subset <= subset
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

    def test_infinite_subset(self) -> None:
        full_subset: BaseSubset[int] = get_full_subset(None)
        assert isinstance(full_subset, FilterSubset)
        with pytest.raises(ValueError, match='cannot enumerate infinite universe'):
            _ = len(full_subset)
        with pytest.raises(ValueError, match='cannot enumerate infinite universe'):
            _ = set(full_subset)
        assert 0 in full_subset
        assert 'a' in full_subset
        assert full_subset == full_subset
        assert full_subset <= full_subset
        assert full_subset >= full_subset
        assert not (full_subset < full_subset)
        assert not (full_subset > full_subset)
        assert {0, 'a'} <= full_subset
        with pytest.raises(ValueError, match='cannot enumerate infinite universe'):
            assert {0, 'a'} < full_subset
        with pytest.raises(TypeError, match="'>=' not supported"):
            _ = full_subset >= 5

    def test_elements_not_in_universe(self) -> None:
        universe = {0, 1, 2}
        with pytest.raises(ValueError, match='3 is not an element of the universe'):
            _ = Subset(universe, {1, 3})
        subset = DynamicSubset(universe, lambda: {1, 3})
        assert subset.universe == universe
        # error is deferred until elements are created
        with pytest.raises(ValueError, match='3 is not an element of the universe'):
            _ = subset.elements

    def test_dynamic_subset(self) -> None:
        subset = DynamicSubset(TEST_RANGE, lambda: {0, 1, 2})
        assert subset.universe == TEST_UNIVERSE
        assert subset.elements == {0, 1, 2}
        self._test_base_subset(subset)
        subset = DynamicSubset(TEST_RANGE, lambda: range(3))
        assert subset.universe == TEST_UNIVERSE
        assert subset.elements == {0, 1, 2}

    def test_subset_immutable(self) -> None:
        xs = {0, 1}
        subset = subset_static(xs)
        assert type(subset.elements) is frozenset
        assert subset == xs
        assert subset.elements is not xs
        with pytest.raises(AttributeError, match="'Subset' object has no attribute 'add'"):
            subset.add(2)  # type: ignore[attr-defined]
        with pytest.raises(AttributeError, match="'frozenset' object has no attribute 'add'"):
            subset.universe.add(2)  # type: ignore[union-attr]
        with pytest.raises(AttributeError, match="'frozenset' object has no attribute 'add'"):
            subset.elements.add(2)  # type: ignore[attr-defined]
        with pytest.raises(dataclasses.FrozenInstanceError):
            subset.universe = frozenset(xs)
        with pytest.raises(dataclasses.FrozenInstanceError):
            subset.elements = frozenset(xs)

    def test_filter_subset(self) -> None:
        subset = FilterSubset(TEST_RANGE, lambda i: i < 5)
        assert len(subset) == 5
        assert 0 in subset
        assert 5 not in subset
        assert set(subset) == set(range(5))
        assert type(subset.universe) is frozenset
        assert subset.universe == frozenset(TEST_RANGE)
        self._test_base_subset(subset)
        subset = FilterSubset(subset_static({0, 10}), lambda i: i < 5)
        assert len(subset) == 1
        assert 0 in subset
        assert 10 not in subset
        # NOTE: the universe type is not a frozenset, but a Subset
        assert type(subset.universe) is Subset
        assert subset.universe == {0, 10}

    def test_range_union(self) -> None:
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
    def test_range_union_invalid_bounds(self, ranges: list[range], error: str) -> None:
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
    def test_range_union_unsorted_or_overlapping_ranges(self, ranges: list[range]) -> None:
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
            FilterSubset({0, 1, 2}, lambda i: i % 2 == 0),
            r'FilterSubset\(universe=\{0, 1, 2\}, predicate=.+\)',
        ),
        (
            RangeUnionSubset(range(5), [range(2, 4), range(4, 5)]),
            r'RangeUnionSubset\(universe_range=range\(0, 5\), ranges=\[range\(2, 4\), range\(4, 5\)\]\)',
        ),
    ])
    def test_repr(self, subset: BaseSubset[int], repr_pattern: str) -> None:
        subset_repr = repr(subset)
        assert str(subset) == subset_repr
        assert re.match(repr_pattern, subset_repr)

    def test_subset_mapped(self) -> None:
        base_subset = subset_static({0, 1, 2, 3, 4})
        subset = MappedSubset(base_subset, lambda i: i % 3)
        assert 1 in subset
        assert 3 not in subset
        assert subset.universe == {0, 1, 2}
        assert set(subset) == {0, 1, 2}
        subset = MappedSubset(base_subset, lambda i: i % 3)

    def test_subset_iso_mapped(self) -> None:
        base_subset = subset_static({0, 1, 2})
        # valid one-to-one mapping
        def _safe_int(s: str) -> int:
            if not isinstance(s, str):
                raise TypeError('input must be a string')
            return int(s)
        subset: BaseSubset[str] = IsoMappedSubset(base_subset, str, _safe_int)
        assert '1' in subset
        assert 1 not in subset  # type: ignore[comparison-overlap]
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
        assert 1 in subset  # type: ignore[comparison-overlap]  # danger!
        assert list(subset) == ['0', '1', '2']
        assert len(subset) == 3
        assert subset.universe == set(map(str, TEST_UNIVERSE))
        assert set(subset) == {'0', '1', '2'}
        # invalid one-to-one-mapping
        int_subset = IsoMappedSubset(base_subset, lambda i: i // 2, lambda i: i * 2)
        assert 0 in int_subset
        assert 1 in int_subset
        assert 2 not in int_subset
        assert list(int_subset) == [0, 0, 1]  # not unique!
        assert len(int_subset) == 3  # wrong!
        assert int_subset.universe == set(range(TEST_UNIVERSE_SIZE // 2))
        assert int_subset.elements == {0, 1}
        assert set(int_subset) == {0, 1}

    def test_boolean_operators(self) -> None:
        subset1 = subset_static({0, 1, 2})
        subset2 = subset_static({2, 3, 4})
        empty = empty_subset()
        universe = universe_subset()
        assert subset1 != subset2
        assert not subset1.isdisjoint(subset2)
        assert subset1 & subset2 == {2}
        assert subset1 ^ subset2 == {0, 1, 3, 4}
        assert empty < subset1 < universe
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
        assert type(subset1.elements | subset2.elements) is frozenset
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
        assert type(subset1.elements & subset2.elements) is frozenset
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
        assert empty < diff < subset1
        assert type(subset1 - subset2.elements) is SubsetIntersection
        assert (subset1 - subset2.elements) == diff
        assert type(subset1.elements - subset2) is SubsetIntersection
        assert (subset1.elements - subset2) == diff
        assert type(subset1.elements - subset2.elements) is frozenset
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
    def test_boolean_operator_universe_mismatch(self, subset1: BaseSubset[int], subset2: BaseSubset[int]) -> None:
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
    def test_subset_static_generic(self, subset: Subset[int]) -> None:
        self._test_base_subset(subset)

    @given(subsets_dynamic())
    def test_subset_dynamic_generic(self, subset: DynamicSubset[int]) -> None:
        self._test_base_subset(subset)

    @given(subsets_range_union())
    def test_unicode_ranges_generic(self, subset: SubsetUnion[int]) -> None:
        self._test_base_subset(subset)

    @given(subset_intersections(subsets_static()))
    def test_subset_intersection(self, subset: SubsetIntersection[int]) -> None:
        self._test_base_subset(subset)
        if not subset.subsets:
            assert subset.elements == TEST_UNIVERSE
        assert all(subset <= component for component in subset.subsets)
        if subset.subsets:
            assert reduce(
                frozenset.intersection, (component.elements for component in subset.subsets)  # type: ignore[arg-type]
            ) == subset.elements

    @given(subset_unions(subsets_static()))
    def test_subset_union(self, subset: SubsetUnion[int]) -> None:
        self._test_base_subset(subset)
        if not subset.subsets:
            assert subset.elements == frozenset()
        assert all(component <= subset for component in subset.subsets)
        assert reduce(
            frozenset.union, (component.elements for component in subset.subsets), frozenset()  # type: ignore[arg-type]
        ) == subset.elements

    @given(subsets())
    @settings(deadline=None)
    def test_subset_generic(self, subset: BaseSubset[int]) -> None:
        self._test_base_subset(subset)

    def test_subset_repr(self) -> None:
        subset1 = Subset({1, 2}, {1})
        subset2 = Subset({1, 2}, {2})
        assert repr(subset1) == 'Subset(universe={1, 2}, elements={1})'
        assert repr(~subset1) == 'SubsetComplement(Subset(universe={1, 2}, elements={1}))'
        assert repr(subset1 & subset2) == (
            'SubsetIntersection(subsets=[Subset(universe={1, 2}, elements={1}), '
            'Subset(universe={1, 2}, elements={2})])'
        )
        assert repr(subset1 | subset2) == (
            'SubsetUnion(subsets=[Subset(universe={1, 2}, elements={1}), '
            'Subset(universe={1, 2}, elements={2})])'
        )


def _test_get_atoms(
    subsets: list[BaseSubset[Any]],
    atoms: Optional[list[tuple[tuple[bool, ...], BaseSubset[Any]]]] = None,
) -> None:
    true_atoms = get_atoms(subsets)
    if atoms is not None:
        # atoms match what we expect
        assert true_atoms == atoms
    atom_sets = [set(atom) for (_, atom) in true_atoms]
    # sign vectors are all distinct
    assert len({signs for (signs, _) in true_atoms}) == len(true_atoms)
    # all atoms are non-empty
    for atom in atom_sets:
        assert len(atom) > 0
    if subsets:
        universe = subsets[0].universe
        assert all(subset.universe == universe for subset in subsets[1:])
        # union of atoms is the universe
        assert set.union(*atom_sets) == universe
    else:
        assert not atom_sets
    # atoms are pairwise disjoint
    for (atom1, atom2) in itertools.combinations(atom_sets, 2):
        assert not (atom1 & atom2)
    # for each atom A and subset S, either (S & A) = A or (S & A) = {}
    for (atom, subset) in itertools.product(atom_sets, subsets):
        conj = atom & subset
        assert (conj == atom) or (not conj)

@pytest.mark.parametrize(['subsets', 'atoms'], [
    (
        [],
        [],
    ),
    (
        [Subset('a', 'a')],
        [((True,), Subset('a', 'a'))],
    ),
    (
        [Subset('a', '')],
        [((False,), Subset('a', 'a'))],
    ),
    (
        [Subset('a', 'a'), Subset('a', 'a')],
        [((True, True), Subset('a', 'a'))],
    ),
    (
        [Subset('a', ''), Subset('a', '')],
        [((False, False), Subset('a', 'a'))],
    ),
    (
        [Subset('a', ''), Subset('a', 'a')],
        [((False, True), Subset('a', 'a'))],
    ),
    (
        [Subset('a', 'a'), Subset('a', '')],
        [((True, False), Subset('a', 'a'))],
    ),
    (
        [Subset('abc', 'abc')],
        [((True,), Subset('abc', 'abc'))],
    ),
    (
        [Subset('abc', '')],
        [((False,), Subset('abc', 'abc'))],
    ),
    (
        [Subset('abc', 'ab')],
        [((True,), Subset('abc', 'ab')), ((False,), Subset('abc', 'c'))],
    ),
    (
        [Subset('abc', 'ab'), Subset('abc', 'bc')],
        [
            ((True, True), Subset('abc', 'b')),
            ((True, False), Subset('abc', 'a')),
            ((False, True), Subset('abc', 'c')),
        ],
    ),
    (
        [Subset('abc', 'a'), Subset('abc', 'b'), Subset('abc', 'c')],
        [
            ((True, False, False), Subset('abc', 'a')),
            ((False, True, False), Subset('abc', 'b')),
            ((False, False, True), Subset('abc', 'c')),
        ],
    ),
])
def test_get_atoms_specific(
    subsets: list[BaseSubset[Any]],
    atoms: list[tuple[tuple[bool, ...], BaseSubset[Any]]],
) -> None:
    _test_get_atoms(subsets, atoms)

@given(
    st.lists(
        subsets(max_leaf_size=10, max_leaves=10, max_width=3, universe_size=10),
        min_size=1,
        max_size=3,
    )
)
def test_get_atoms_generic(subsets: list[BaseSubset[int]]) -> None:
    _test_get_atoms(subsets)

def test_get_atoms_invalid() -> None:
    subset1 = Subset(None, {1})
    subset2 = Subset({0}, {0})
    subset3 = Subset({0, 1}, {0})
    with pytest.raises(ValueError, match='universe must be finite'):
        _ = get_atoms([subset1])
    with pytest.raises(ValueError, match='universe must be finite'):
        _ = get_atoms([subset1, subset2])
    with pytest.raises(ValueError, match='all subsets must have the same universe'):
        _ = get_atoms([subset2, subset3])
