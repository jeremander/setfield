from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Set
from dataclasses import dataclass, field
from functools import cached_property, reduce, total_ordering
from math import inf
import operator
from typing import Generic, Literal, Optional, TypeVar, overload


S = TypeVar('S')
T = TypeVar('T')


@total_ordering
class BaseSubset(Set[T]):
    """Abstract base class representing a subset of some ambient universe set."""

    @abstractmethod
    def _get_universe(self) -> Optional[Set[T]]:
        """Gets the set of elements representing the ambient set (universe) of the field of sets."""

    @cached_property
    def universe(self) -> Optional[Set[T]]:
        """Cached property returning the set of elements representing the ambient set (universe)
        of the field of sets."""
        return self._get_universe()

    def _validate_elements(self, elements: Iterable[T]) -> None:
        """Checks whether each of the given elements is in the universe, raising a ValueError otherwise."""
        if (universe := self.universe) is not None:
            for elt in elements:
                if elt not in universe:
                    raise ValueError(f'{elt} is not an element of the universe')

    def _universes_match(self, other: BaseSubset[T]) -> bool:
        """Returns True if the universe of this BaseSubset matches that of another."""
        return self.universe == other.universe

    def _check_universes_match(self, other: BaseSubset[T]) -> None:
        if not self._universes_match(other):
            raise ValueError('universes do not match')

    @abstractmethod
    def _get_elements(self) -> Set[T]:
        """Gets the set of elements in this subset."""

    @cached_property
    def elements(self) -> Set[T]:
        """Cached property returning the set of elements in this subset."""
        return self._get_elements()

    def __contains__(self, item: object) -> bool:
        return item in self.elements

    def __iter__(self) -> Iterator[T]:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, BaseSubset):
            # check universe and setwise equality, even if the representation is different
            return self._universes_match(other) and (self.elements == other.elements)
        if isinstance(other, Set):
            # assume the same universe for other
            return self.elements == other
        return False

    def _compare(self, cmp: Callable[[set[T], set[T]], bool], other: object) -> bool:
        if isinstance(other, BaseSubset):
            self._check_universes_match(other)
            return cmp(self.elements, other.elements)  # type: ignore[arg-type]
        if isinstance(other, Set):
            # assume the same universe for other
            return cmp(self.elements, other)  # type: ignore[arg-type]
        return NotImplemented  # type: ignore[no-any-return]

    def __lt__(self, other: object) -> bool:
        if self is other:
            return False
        return self._compare(operator.lt, other)

    def __le__(self, other: object) -> bool:
        if self is other:
            return True
        return self._compare(operator.le, other)

    def __gt__(self, other: object) -> bool:
        if self is other:
            return False
        return self._compare(operator.gt, other)

    def __ge__(self, other: object) -> bool:
        if self is other:
            return True
        return self._compare(operator.ge, other)

    def __and__(self, other: object) -> BaseSubset[T]:
        if isinstance(other, BaseSubset):
            self._check_universes_match(other)
            return SubsetIntersection(self.universe, [self, other])
        if isinstance(other, Set):
            return SubsetIntersection(self.universe, [self, Subset(self.universe, other)])
        return NotImplemented

    def __rand__(self, other: object) -> BaseSubset[T]:
        if isinstance(other, Set):
            return Subset(self.universe, other) & self
        return NotImplemented

    def __or__(self, other: object) -> BaseSubset[T]:
        if isinstance(other, BaseSubset):
            self._check_universes_match(other)
            return SubsetUnion(self.universe, [self, other])
        if isinstance(other, Set):
            return SubsetUnion(self.universe, [self, Subset(self.universe, other)])
        return NotImplemented

    def __ror__(self, other: object) -> BaseSubset[T]:
        if isinstance(other, Set):
            return Subset(self.universe, other) | self
        return NotImplemented

    def __invert__(self) -> BaseSubset[T]:
        return SubsetComplement(self)

    def __sub__(self, other: object) -> BaseSubset[T]:
        if isinstance(other, BaseSubset):
            self._check_universes_match(other)
            return self & ~other
        if isinstance(other, Set):
            return self & ~(Subset(self.universe, other))
        return NotImplemented

    def __rsub__(self, other: object) -> BaseSubset[T]:
        if isinstance(other, Set):
            return Subset(self.universe, other) - self
        return NotImplemented

    def __xor__(self, other: object) -> BaseSubset[T]:
        return (self | other) - (self & other)


def _get_set_repr(obj: Optional[Set[T]]) -> str:
    if isinstance(obj, frozenset) and obj:
        return str(obj).removeprefix('frozenset(').removesuffix(')')
    return repr(obj)


@dataclass(frozen=True, eq=False)
class _Subset(BaseSubset[T]):

    _universe: Optional[Set[T]] = field(repr=False)

    def __init__(self, universe: Optional[Iterable[T]]) -> None:
        if (universe is not None) and (not isinstance(universe, Set)):
            universe = frozenset(universe)
        object.__setattr__(self, '_universe', universe)

    def _get_universe(self) -> Optional[Set[T]]:
        return self._universe


@dataclass(frozen=True, eq=False)
class Subset(_Subset[T]):
    """A subset which stores both the universe and the subset elements as concrete objects."""

    _elements: Set[T]

    def __init__(self, universe: Optional[Iterable[T]], elements: Iterable[T]) -> None:
        super().__init__(universe)
        if (elements is not None) and (not isinstance(elements, Set)):
            elements = frozenset(elements)
        object.__setattr__(self, '_elements', elements)
        self._validate_elements(self._elements)

    def __repr__(self) -> str:
        universe_str = _get_set_repr(self._universe)
        elements_str = _get_set_repr(self._elements)
        return f'{self.__class__.__name__}(universe={universe_str}, elements={elements_str})'

    def _get_elements(self) -> Set[T]:
        return self._elements if isinstance(self._elements, frozenset) else frozenset(self._elements)


@dataclass(frozen=True, eq=False)
class DynamicSubset(_Subset[T]):
    """A subset which stores the universe concretely but computes the subset lazily via a callable.
    The first time the subset is computed, it is stored on the object and then reused."""

    get_elements: Callable[[], Iterable[T]]

    def __init__(self, universe: Iterable[T], get_elements: Callable[[], Iterable[T]]) -> None:
        super().__init__(universe)
        object.__setattr__(self, 'get_elements', get_elements)

    def __repr__(self) -> str:
        universe_str = _get_set_repr(self._universe)
        return f'{self.__class__.__name__}(universe={universe_str}, get_elements={self.get_elements!r})'

    def _get_elements(self) -> Set[T]:
        elements = self.get_elements()
        if not isinstance(elements, frozenset):
            elements = frozenset(elements)
        self._validate_elements(elements)
        return elements


@dataclass(frozen=True, eq=False)
class FilterSubset(_Subset[T]):
    """A subset which stores the universe concretely but uses a (callable) predicate to determine if
    an element is in the subset.
    This can sometimes be more efficient than computing the full set, especially when there are a large number
    of different subsets to deal with."""

    predicate: Callable[[T], bool]

    def __init__(self, universe: Optional[Iterable[T]], predicate: Callable[[T], bool]) -> None:
        super().__init__(universe)
        object.__setattr__(self, 'predicate', predicate)

    def __repr__(self) -> str:
        universe_str = _get_set_repr(self._universe)
        return f'{self.__class__.__name__}(universe={universe_str}, predicate={self.predicate!r})'

    def _get_elements(self) -> Set[T]:
        if self._universe is None:
            raise ValueError('cannot enumerate infinite universe')
        return frozenset(filter(self.predicate, self._universe))

    def __contains__(self, item: object) -> bool:
        # NOTE: in practice, predicate may have to evaluate things that are not of type T
        return self.predicate(item)  # type: ignore[arg-type]

    def __ge__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, Set):
            return all(map(self.predicate, other))
        return super().__ge__(other)


######################
# EMPTY/FULL SUBSETS #
######################

def get_empty_subset(universe: Optional[Iterable[T]]) -> Subset[T]:
    """Given a universe, returns a Subset representing the empty subset.
    This is also called the "bottom" element of the field of sets."""
    return Subset(universe, frozenset())

@overload
def get_full_subset(universe: Literal[None]) -> FilterSubset[T]:
    ...

@overload
def get_full_subset(universe: Iterable[T]) -> Subset[T]:
    ...

def get_full_subset(universe: Optional[Iterable[T]]) -> Subset[T] | FilterSubset[T]:
    """Given a universe, returns a Subset representing the entire universe.
    This is also called the "top" element of the field of sets."""
    if universe is None:
        return FilterSubset(universe, lambda _: True)
    return Subset(universe, universe)


#######################
# BOOLEAN COMBINATORS #
#######################

@dataclass(frozen=True, eq=False)
class SubsetComplement(FilterSubset[T]):
    """A subset which is a complement of another subset.
    This stores the original subset as a `subset` field.
    Set membership is computed as the negation of membership in the inner subset."""

    subset: BaseSubset[T]

    def __init__(self, subset: BaseSubset[T]) -> None:
        pred = lambda elt: elt not in subset
        super().__init__(subset.universe, pred)
        object.__setattr__(self, 'subset', subset)

    def __len__(self) -> int:
        if self.universe is None:
            raise ValueError('cannot get length of infinite universe')
        return len(self.universe) - len(self.subset)

    def __invert__(self) -> BaseSubset[T]:
        # apply law of double negation
        return self.subset


@dataclass(frozen=True, eq=False)
class SubsetIntersection(_Subset[T]):
    """A subset which is the intersection of other subsets.
    This stores a list of subsets to intersect as a `subsets` field.
    Set membership is computed as the logical conjuction (AND) of membership in all of the inner subsets."""

    subsets: list[BaseSubset[T]]

    @cached_property
    def _length_sort_indices(self) -> list[int]:
        """Gets the list of indices which would sort the constituent subsets by increasing length."""
        def _get_length(pair: tuple[int, BaseSubset[T]]) -> float:
            (_, subset) = pair
            try:
                return len(subset)
            except ValueError:  # infinite universe
                return inf
        return [i for (i, _) in sorted(enumerate(self.subsets), key=_get_length)]

    def _get_elements(self) -> Set[T]:
        match len(self.subsets):
            case 0:
                if self.universe is None:
                    raise ValueError('cannot enumerate infinite universe')
                return self.universe if isinstance(self.universe, frozenset) else frozenset(self.universe)
            case 1:
                return self.subsets[0].elements
        # NOTE: naive implementation of taking full setwise intersection can be expensive if subsets are large.
        # Instead, we start with the smallest set and filter by membership in the others.
        indices = self._length_sort_indices
        smallest_subset = self.subsets[indices[0]]
        bigger_subsets = [self.subsets[i] for i in indices[1:]]
        pred = lambda c: all(c in subset for subset in bigger_subsets)
        return frozenset(filter(pred, smallest_subset.elements))

    def __len__(self) -> int:
        if (self.universe is None) and (len(self.subsets) == 0):
            raise ValueError('cannot get length of infinite universe')
        return super().__len__()

    def __contains__(self, item: object) -> bool:
        return all(item in subset for subset in self.subsets)

    def __and__(self, other: object) -> SubsetIntersection[T]:
        if isinstance(other, BaseSubset):
            self._check_universes_match(other)
            if isinstance(other, SubsetIntersection):
                return type(self)(self.universe, self.subsets + other.subsets)
            return type(self)(self.universe, self.subsets + [other])
        return NotImplemented


@dataclass(frozen=True, eq=False)
class SubsetUnion(_Subset[T]):
    """A subset which is the union of other subsets.
    This stores a list of subsets to union as a `subsets` field.
    Set membership is computed as the logical disjunction (OR) of membership in all of the inner subsets."""

    subsets: list[BaseSubset[T]]

    def _get_elements(self) -> Set[T]:
        if not self.subsets:
            return frozenset()
        return reduce(frozenset.union, (subset.elements for subset in self.subsets))  # type: ignore[arg-type]

    def __contains__(self, item: object) -> bool:
        return any(item in cs for cs in self.subsets)

    def __or__(self, other: object) -> SubsetUnion[T]:
        if isinstance(other, BaseSubset):
            self._check_universes_match(other)
            if isinstance(other, SubsetUnion):
                return type(self)(self.universe, self.subsets + other.subsets)
            return type(self)(self.universe, self.subsets + [other])
        return NotImplemented


####################
# FUNCTION MAPPING #
####################

@dataclass(frozen=True, eq=False)
class MappedSubset(BaseSubset[T], Generic[S, T]):
    """A subset formed by mapping a function, `map_func`, onto a base subset.
    This may transform the type of the base subset depending on the output type of the function.
    The function need not be one-to-one, and the function will need to be applied to all elements
    to determine the new set."""

    base_subset: BaseSubset[S]
    map_func: Callable[[S], T]

    def _get_universe(self) -> Optional[Set[T]]:
        if (universe := self.base_subset.universe) is None:
            return None
        return frozenset(map(self.map_func, universe))

    def _get_elements(self) -> Set[T]:
        return frozenset(map(self.map_func, self.base_subset))


@dataclass(frozen=True, eq=False)
class IsoMappedSubset(MappedSubset[S, T]):
    """A subset formed by mapping a one-to-one function (isomorphism), `map_func`, onto a base subset.
    This may transform the type of the base subset depending on the output type of the function.
    Additionally, the *inverse* of `map_func`, `map_func_inv` should be provided, since it will be used to check
    set membership without having to map all the base set elements themselves.
    In order for things to work properly, the assumed properties must hold that `map_func` and `map_func_inv`
    are one-to-one and inverses of each other."""

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
