from __future__ import annotations

from abc import abstractmethod
import ast
from collections.abc import Callable, Iterable, Iterator, Sequence, Set
from dataclasses import dataclass, field
from functools import cached_property, partial, reduce, total_ordering
import itertools
import operator
from operator import attrgetter, contains
from typing import Generic, Optional, TypeAlias, TypeVar


__version__ = '0.1.0'


S = TypeVar('S')
T = TypeVar('T')

Ranges: TypeAlias = Sequence[range]


###############
# SET ALGEBRA #
###############

@total_ordering
class Subset(Set[T]):

    @abstractmethod
    def _get_universe(self) -> set[T]:
        """Gets the set of elements representing the ambient set (universe) of an algebra of sets."""

    @cached_property
    def universe(self) -> set[T]:
        return self._get_universe()

    @abstractmethod
    def _get_elements(self) -> set[T]:
        """Gets the set of elements for this subset."""

    @cached_property
    def elements(self) -> set[T]:
        return self._get_elements()

    def __contains__(self, item: object) -> bool:
        return item in self.elements

    def __iter__(self) -> Iterator[T]:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Subset):
            # check universe and setwise equality, even if the representation is different
            return (self.universe == other.universe) and (self.elements == other.elements)
        if isinstance(other, set):
            # assume the same universe for other
            return self.elements == other
        return False

    def _compare(self, cmp: Callable[[set[T], set[T]], bool], other: object) -> bool:
        if isinstance(other, Subset):
            _check_universes_match(self, other)
            return cmp(self.elements, other.elements)
        if isinstance(other, set):
            # assume the same universe for other
            return cmp(self.elements, other)
        return NotImplemented  # type: ignore[no-any-return]

    def __lt__(self, other: object) -> bool:
        return self._compare(operator.lt, other)

    def __le__(self, other: object) -> bool:
        return self._compare(operator.le, other)

    def __gt__(self, other: object) -> bool:
        return self._compare(operator.gt, other)

    def __ge__(self, other: object) -> bool:
        return self._compare(operator.ge, other)

    def __and__(self, other: object) -> Subset[T]:
        if isinstance(other, Subset):
            _check_universes_match(self, other)
            return SubsetIntersection(self.universe, [self, other])
        if isinstance(other, set):
            return SubsetIntersection(self.universe, [self, SubsetStatic(self.universe, other)])
        return NotImplemented

    def __rand__(self, other: object) -> Subset[T]:
        if isinstance(other, set):
            return SubsetStatic(self.universe, other) & self
        return NotImplemented

    def __or__(self, other: object) -> Subset[T]:
        if isinstance(other, Subset):
            _check_universes_match(self, other)
            return SubsetUnion(self.universe, [self, other])
        if isinstance(other, set):
            return SubsetUnion(self.universe, [self, SubsetStatic(self.universe, other)])
        return NotImplemented

    def __ror__(self, other: object) -> Subset[T]:
        if isinstance(other, set):
            return SubsetStatic(self.universe, other) | self
        return NotImplemented

    def __invert__(self) -> Subset[T]:
        return SubsetComplement(self)

    def __sub__(self, other: object) -> Subset[T]:
        if isinstance(other, Subset):
            _check_universes_match(self, other)
            return self & ~other
        if isinstance(other, set):
            return self & ~(SubsetStatic(self.universe, other))
        return NotImplemented

    def __rsub__(self, other: object) -> Subset[T]:
        if isinstance(other, set):
            return SubsetStatic(self.universe, other) - self
        return NotImplemented

    def __xor__(self, other: object) -> Subset[T]:
        return (self | other) - (self & other)


def _check_universes_match(subset1: Subset[T], subset2: Subset[T]) -> None:
    if subset1.universe != subset2.universe:
        raise ValueError('universes do not match')


@dataclass(eq=False)
class ConcreteSubset(Subset[T]):
    _universe: set[T] = field(repr=False)

    def _get_universe(self) -> set[T]:
        return self._universe


@dataclass(eq=False)
class SubsetStatic(ConcreteSubset[T]):
    _elements: set[T]

    def __init__(self, universe: set[T], elements: Iterable[T]) -> None:
        self._universe = universe
        self._elements = elements if isinstance(elements, set) else set(elements)

    def _get_elements(self) -> set[T]:
        return self._elements

    @classmethod
    def empty(cls, universe: set[T]) -> SubsetStatic[T]:
        return cls(universe, set())

    @classmethod
    def full(cls, universe: set[T]) -> SubsetStatic[T]:
        return cls(universe, universe)


@dataclass(eq=False)
class SubsetDynamic(ConcreteSubset[T]):
    get_elements: Callable[[], Iterable[T]]

    def _get_elements(self) -> set[T]:
        elements = self.get_elements()
        return elements if isinstance(elements, set) else set(elements)


@dataclass(eq=False)
class SubsetFilter(ConcreteSubset[T]):

    pred: Callable[[object], bool]

    def _get_elements(self) -> set[T]:
        return set(filter(self.pred, self._universe))

    def __contains__(self, item: object) -> bool:
        # NOTE: pred may have to evaluate things that are not of type T
        return self.pred(item)


#######################
# BOOLEAN COMBINATORS #
#######################

@dataclass(eq=False)
class SubsetComplement(SubsetFilter[T]):
    subset: Subset[T]

    def __init__(self, subset: Subset[T]) -> None:
        pred = lambda elt: elt not in subset
        super().__init__(subset.universe, pred)
        self.subset = subset

    def __len__(self) -> int:
        return len(self.universe) - len(self.subset)

    def __invert__(self) -> Subset[T]:
        # apply law of double negation
        return self.subset


@dataclass(eq=False)
class SubsetIntersection(ConcreteSubset[T]):
    subsets: list[Subset[T]]

    @cached_property
    def _length_sort_indices(self) -> list[int]:
        """Gets the list of indices which would sort the constituent subsets by increasing length."""
        return [i for (i, _) in sorted(enumerate(self.subsets), key=lambda pair: len(pair[1]))]

    def _get_elements(self) -> set[T]:
        match len(self.subsets):
            case 0:
                return self.universe
            case 1:
                return self.subsets[0].elements
        # NOTE: naive implementation of taking full setwise intersection can be expensive if subsets are large.
        # Instead, we start with the smallest set and filter by membership in the others.
        indices = self._length_sort_indices
        smallest_subset = self.subsets[indices[0]]
        bigger_subsets = [self.subsets[i] for i in indices[1:]]
        pred = lambda c: all(c in subset for subset in bigger_subsets)
        return set(filter(pred, smallest_subset.elements))

    def __contains__(self, item: object) -> bool:
        return all(item in subset for subset in self.subsets)

    def __and__(self, other: object) -> SubsetIntersection[T]:
        if isinstance(other, Subset):
            _check_universes_match(self, other)
            if isinstance(other, SubsetIntersection):
                return type(self)(self.universe, self.subsets + other.subsets)
            return type(self)(self.universe, self.subsets + [other])
        return NotImplemented


@dataclass(eq=False)
class SubsetUnion(ConcreteSubset[T]):
    subsets: list[Subset[T]]

    def _get_elements(self) -> set[T]:
        if not self.subsets:
            return set()
        return reduce(set.union, (subset.elements for subset in self.subsets))

    def __contains__(self, item: object) -> bool:
        return any(item in cs for cs in self.subsets)

    def __or__(self, other: object) -> SubsetUnion[T]:
        if isinstance(other, Subset):
            _check_universes_match(self, other)
            if isinstance(other, SubsetUnion):
                return type(self)(self.universe, self.subsets + other.subsets)
            return type(self)(self.universe, self.subsets + [other])
        return NotImplemented


###################
# UNION OF RANGES #
###################

def indices_to_minimal_ranges(indices: Iterable[int]) -> list[range]:
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
    # NOTE: there might be a more efficient implementation possible, but for simplicity we just take the set intersection and reconstruct the minimal set of ranges.
    if not ranges_seq:  # empty intersection is the full universe
        return [universe]
    in_universe = partial(contains, universe)
    idx_set = set(filter(in_universe, itertools.chain.from_iterable(ranges_seq[0])))
    for ranges in ranges_seq[1:]:
        idx_set &= set(itertools.chain.from_iterable(ranges))
    return indices_to_minimal_ranges(idx_set)

def _ranges_union(ranges_seq: Sequence[Ranges]) -> list[range]:
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
    if universe_range1 != universe_range2:
        raise ValueError('universes do not match')


@dataclass(eq=False)
class SubsetRangeUnion(Subset[int]):
    _universe_range: range
    ranges: Ranges

    def __post_init__(self) -> None:
        # make sure ranges are valid for the universe
        for rng in self.ranges:
            if rng.step not in [1, None]:
                raise ValueError('ranges may not have step != 1')
            if not _range_contains(self._universe_range, rng):
                raise ValueError(f'invalid range [{rng.start}, {rng.stop}), bounds must be contained within universe range')
            if rng.start >= rng.stop:
                raise ValueError(f'invalid range [{rng.start}, {rng.stop}), cannot have start >= stop')
        # make sure ranges are sorted and disjoint
        for (rng1, rng2) in zip(self.ranges, self.ranges[1:]):
            if rng1.stop > rng2.start:
                raise ValueError('ranges must be sorted and not overlap')

    @classmethod
    def from_ranges(cls, universe_range: range, ranges: Ranges) -> SubsetRangeUnion:
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

    def _get_universe(self) -> set[int]:
        return set(self._universe_range)

    def _get_elements(self) -> set[int]:
        return set(self.__iter__())

    def __and__(self, other: object) -> Subset[int]:
        if isinstance(other, SubsetRangeUnion):
            _check_universe_ranges_match(self._universe_range, other._universe_range)
            intersection_ranges = _ranges_intersection(self._universe_range, [self.ranges, other.ranges])
            return type(self)(self._universe_range, intersection_ranges)
        return super().__and__(other)

    def __or__(self, other: object) -> Subset[int]:
        if isinstance(other, SubsetRangeUnion):
            _check_universe_ranges_match(self._universe_range, other._universe_range)
            union_ranges = _ranges_union([self.ranges, other.ranges])
            return type(self)(self._universe_range, union_ranges)
        return super().__or__(other)

    def __invert__(self) -> SubsetRangeUnion:
        return type(self)(self._universe_range, _ranges_complement(self._universe_range, self.ranges))


####################
# FUNCTION MAPPING #
####################

@dataclass(eq=False)
class SubsetMapped(Subset[T], Generic[S, T]):
    base_subset: Subset[S]
    map_func: Callable[[S], T]

    def _get_universe(self) -> set[T]:
        return set(map(self.map_func, self.base_subset.universe))

    def _get_elements(self) -> set[T]:
        return set(map(self.map_func, self.base_subset))


@dataclass(eq=False)
class SubsetIsoMapped(SubsetMapped[S, T]):
    map_func_inv: Callable[[T], S]

    # NOTE: we avoid calling self.elements as much as possible to defer full enumeration

    def __contains__(self, item: object) -> bool:
        try:
            return self.map_func_inv(item) in self.base_subset  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def __iter__(self) -> Iterator[T]:
        return map(self.map_func, iter(self.base_subset))

    def __len__(self) -> int:
        return len(self.base_subset)


#######################
# BOOLEAN EXPRESSIONS #
#######################

# ast node types safe for boolean expressions
BOOLEAN_SAFE_NODE_TYPES = {
    # identifiers
    ast.Load, ast.Name,
    # expression heads
    ast.BinOp, ast.Expression, ast.UnaryOp,
    # unary complement
    ast.Invert,
    # bitwise operators
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.Sub,
}

def safe_eval(
    expr: str,
    eval_name: Optional[Callable[[str], T]] = None,
    *,
    safe_node_types: set[type],
) -> T:
    if eval_name is None:
        safe_node_types = safe_node_types - {ast.Name}
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError('invalid expression') from e
    _locals = {}
    for node in ast.walk(tree):
        if (tp := type(node)) not in safe_node_types:
            raise ValueError(f'disallowed construct: {tp.__name__}')
        if isinstance(node, ast.Name):
            # NOTE: eval_identifier should raise an error if identifier is invalid
            _locals[node.id] = eval_name(node.id)  # type: ignore[misc]
    # evaluate directly from code object (avoids re-parsing from a string)
    return eval(compile(tree, '<string>', 'eval'), {'__builtins__': {}}, _locals)  # type: ignore[no-any-return]

def safe_eval_boolean_expr(expr: str, eval_name: Optional[Callable[[str], T]] = None) -> T:
    return safe_eval(expr, eval_name=eval_name, safe_node_types=BOOLEAN_SAFE_NODE_TYPES)
