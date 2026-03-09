"""Provides a framework for expressing a boolean field of sets construction, i.e. an ambient universe set with subsets that can be combined with boolean setwise operations (intersection, union, complement)."""

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


#################
# FIELD OF SETS #
#################

@total_ordering
class BaseSubset(Set[T]):
    """Abstract base class representing a subset of some ambient universe set."""

    @abstractmethod
    def _get_universe(self) -> set[T]:
        """Gets the set of elements representing the ambient set (universe) of the field of sets."""

    @cached_property
    def universe(self) -> set[T]:
        """Cached property returning the set of elements representing the ambient set (universe) of the field of sets."""
        return self._get_universe()

    @abstractmethod
    def _get_elements(self) -> set[T]:
        """Gets the set of elements in this subset."""

    @cached_property
    def elements(self) -> set[T]:
        """Cached property returning the set of elements in this subset."""
        return self._get_elements()

    def __contains__(self, item: object) -> bool:
        return item in self.elements

    def __iter__(self) -> Iterator[T]:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BaseSubset):
            # check universe and setwise equality, even if the representation is different
            return (self.universe == other.universe) and (self.elements == other.elements)
        if isinstance(other, set):
            # assume the same universe for other
            return self.elements == other
        return False

    def _compare(self, cmp: Callable[[set[T], set[T]], bool], other: object) -> bool:
        if isinstance(other, BaseSubset):
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

    def __and__(self, other: object) -> BaseSubset[T]:
        if isinstance(other, BaseSubset):
            _check_universes_match(self, other)
            return SubsetIntersection(self.universe, [self, other])
        if isinstance(other, set):
            return SubsetIntersection(self.universe, [self, SubsetStatic(self.universe, other)])
        return NotImplemented

    def __rand__(self, other: object) -> BaseSubset[T]:
        if isinstance(other, set):
            return SubsetStatic(self.universe, other) & self
        return NotImplemented

    def __or__(self, other: object) -> BaseSubset[T]:
        if isinstance(other, BaseSubset):
            _check_universes_match(self, other)
            return SubsetUnion(self.universe, [self, other])
        if isinstance(other, set):
            return SubsetUnion(self.universe, [self, SubsetStatic(self.universe, other)])
        return NotImplemented

    def __ror__(self, other: object) -> BaseSubset[T]:
        if isinstance(other, set):
            return SubsetStatic(self.universe, other) | self
        return NotImplemented

    def __invert__(self) -> BaseSubset[T]:
        return SubsetComplement(self)

    def __sub__(self, other: object) -> BaseSubset[T]:
        if isinstance(other, BaseSubset):
            _check_universes_match(self, other)
            return self & ~other
        if isinstance(other, set):
            return self & ~(SubsetStatic(self.universe, other))
        return NotImplemented

    def __rsub__(self, other: object) -> BaseSubset[T]:
        if isinstance(other, set):
            return SubsetStatic(self.universe, other) - self
        return NotImplemented

    def __xor__(self, other: object) -> BaseSubset[T]:
        return (self | other) - (self & other)


def _check_universes_match(subset1: BaseSubset[T], subset2: BaseSubset[T]) -> None:
    """Checks whether the universes of two Subsets match.
    If not, raises a ValueError."""
    if subset1.universe != subset2.universe:
        raise ValueError('universes do not match')


@dataclass(eq=False)
class ConcreteSubset(BaseSubset[T]):
    """Base class for a concrete subset where the universe is stored explicitly as a set."""

    _universe: set[T] = field(repr=False)

    def _get_universe(self) -> set[T]:
        return self._universe


@dataclass(eq=False)
class SubsetStatic(ConcreteSubset[T]):
    """A concrete subset which stores the universe and the subset explicitly as sets."""

    _elements: set[T]

    def __init__(self, universe: set[T], elements: Iterable[T]) -> None:
        self._universe = universe
        self._elements = elements if isinstance(elements, set) else set(elements)

    def _get_elements(self) -> set[T]:
        return self._elements

    @classmethod
    def empty(cls, universe: set[T]) -> SubsetStatic[T]:
        """Constructor which, given the universe, returns a SubsetStatic representing the empty subset.
        This is also called the "bottom" element of the field of sets."""
        return cls(universe, set())

    @classmethod
    def full(cls, universe: set[T]) -> SubsetStatic[T]:
        """Constructor which, given the universe, returns a SubsetStatic representing the whole universe.
        This is also called the "top" element of the field of sets."""
        return cls(universe, universe)


@dataclass(eq=False)
class SubsetDynamic(ConcreteSubset[T]):
    """A subset which stores the universe concretely but computes the subset lazily via a callable.
    The first time the subset is computed, it is stored on the object and then reused."""

    get_elements: Callable[[], Iterable[T]]

    def _get_elements(self) -> set[T]:
        elements = self.get_elements()
        return elements if isinstance(elements, set) else set(elements)


@dataclass(eq=False)
class SubsetFilter(ConcreteSubset[T]):
    """A subset which stores the universe concretely but uses a (callable) predicate to determine if an element is in the subset.
    This can sometimes be more efficient than computing the full set, especially when there are a large number of different subsets to deal with."""

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
    """A subset which is a complement of another subset.
    This stores the original subset as a `subset` field.
    Set membership is computed as the negation of membership in the inner subset."""

    subset: BaseSubset[T]

    def __init__(self, subset: BaseSubset[T]) -> None:
        pred = lambda elt: elt not in subset
        super().__init__(subset.universe, pred)
        self.subset = subset

    def __len__(self) -> int:
        return len(self.universe) - len(self.subset)

    def __invert__(self) -> BaseSubset[T]:
        # apply law of double negation
        return self.subset


@dataclass(eq=False)
class SubsetIntersection(ConcreteSubset[T]):
    """A subset which is the intersection of other subsets.
    This stores a list of subsets to intersect as a `subsets` field.
    Set membership is computed as the logical conjuction (AND) of membership in all of the inner subsets."""

    subsets: list[BaseSubset[T]]

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
        if isinstance(other, BaseSubset):
            _check_universes_match(self, other)
            if isinstance(other, SubsetIntersection):
                return type(self)(self.universe, self.subsets + other.subsets)
            return type(self)(self.universe, self.subsets + [other])
        return NotImplemented


@dataclass(eq=False)
class SubsetUnion(ConcreteSubset[T]):
    """A subset which is the union of other subsets.
    This stores a list of subsets to union as a `subsets` field.
    Set membership is computed as the logical disjunction (OR) of membership in all of the inner subsets."""

    subsets: list[BaseSubset[T]]

    def _get_elements(self) -> set[T]:
        if not self.subsets:
            return set()
        return reduce(set.union, (subset.elements for subset in self.subsets))

    def __contains__(self, item: object) -> bool:
        return any(item in cs for cs in self.subsets)

    def __or__(self, other: object) -> SubsetUnion[T]:
        if isinstance(other, BaseSubset):
            _check_universes_match(self, other)
            if isinstance(other, SubsetUnion):
                return type(self)(self.universe, self.subsets + other.subsets)
            return type(self)(self.universe, self.subsets + [other])
        return NotImplemented


###################
# UNION OF RANGES #
###################

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
    """Given a range corresponding to a universe, and a sequence of range-unions, returns a new list of disjoint ranges representing the intersection of the given ranges.
    If the given list of ranges is empty, returns the whole universe (which is implicitly the intersection of no ranges)."""
    # NOTE: there might be a more efficient implementation possible, but for simplicity we just take the set intersection and reconstruct the minimal set of ranges.
    if not ranges_seq:  # empty intersection is the full universe
        return [universe]
    in_universe = partial(contains, universe)
    idx_set = set(filter(in_universe, itertools.chain.from_iterable(ranges_seq[0])))
    for ranges in ranges_seq[1:]:
        idx_set &= set(itertools.chain.from_iterable(ranges))
    return indices_to_minimal_ranges(idx_set)

def _ranges_union(ranges_seq: Sequence[Ranges]) -> list[range]:
    """Given a sequence of range-unions, returns a new list of disjoint ranges representing the union of the given ranges.
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
    """Given a range corresponding to a universe, and a range-union, returns a new list of disjoint ranges representing the complement of the given ranges.
    If the given list of ranges is empty, returns the whole universe (which is implicitly the complement of the empty set)."""
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


@dataclass(eq=False)
class SubsetRangeUnion(BaseSubset[int]):
    """A subset of an integer universe, represented as a disjoint union of sorted ranges.
    This is often a more efficient data structure than a set for enumeration and membership checks, as it can be much more compact when there are a lot of contiguous elements in the subset."""

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

    def __and__(self, other: object) -> BaseSubset[int]:
        if isinstance(other, SubsetRangeUnion):
            _check_universe_ranges_match(self._universe_range, other._universe_range)
            intersection_ranges = _ranges_intersection(self._universe_range, [self.ranges, other.ranges])
            return type(self)(self._universe_range, intersection_ranges)
        return super().__and__(other)

    def __or__(self, other: object) -> BaseSubset[int]:
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
class SubsetMapped(BaseSubset[T], Generic[S, T]):
    """A subset formed by mapping a function, `map_func`, onto a base subset.
    This may transform the type of the base subset depending on the output type of the function.
    The function need not be one-to-one, and the function will need to be applied to all elements to determine the new set."""

    base_subset: BaseSubset[S]
    map_func: Callable[[S], T]

    def _get_universe(self) -> set[T]:
        return set(map(self.map_func, self.base_subset.universe))

    def _get_elements(self) -> set[T]:
        return set(map(self.map_func, self.base_subset))


@dataclass(eq=False)
class SubsetIsoMapped(SubsetMapped[S, T]):
    """A subset formed by mapping a one-to-one function, `map_func`, onto a base subset.
    This may transform the type of the base subset depending on the output type of the function.
    Additionally, the *inverse* of `map_func`, `map_func_inv` should be provided, since it will be used to check set membership without having to map all the base set elements themselves.
    In order for things to work properly, the assumed properties must hold that `map_func` and `map_func_inv` are one-to-one and inverses of each other."""

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
    """Calls Python's `eval` function in a more "safe" context, in that the caller must provide:
        1. `eval_name`: a callable which maps names (identifiers) to Python objects, and errors if the name is invalid.
        2. `safe_node_types`: a set of `ast.Node` objects indicating which elements of Python syntax are permitted in the expression.
    This makes it easy to create miniature Embedded Domain Specific Languages (EDSLs) using only a fragment of Python syntax.
    Most notably, it can support expressions that only consist of names and boolean connectives.
    If `eval_name` is None, then no identifiers will be permitted."""
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
    """Given an expression and a callable `eval_name`, evaluates the expression to a Python object using a safe version of `eval` which only allows specific identifiers and boolean connectives.
    `eval_name` should be a function that maps names to Python objects, and it should raise an exception if the name is not valid."""
    return safe_eval(expr, eval_name=eval_name, safe_node_types=BOOLEAN_SAFE_NODE_TYPES)
