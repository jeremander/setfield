# Setfield

[![PyPI - Version](https://img.shields.io/pypi/v/setfield)](https://pypi.org/project/setfield/)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/jeremander/setfield/workflow.yml)
![Coverage Status](https://github.com/jeremander/setfield/raw/coverage-badge/coverage-badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://raw.githubusercontent.com/jeremander/setfield/refs/heads/main/LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

<img src="logo.svg" width="350" alt="Setfield logo (a baseball field with bases numbered in von Neumann ordinal notation"/>

**Setfield** is a small Python library providing a framework for expressing a [field of sets](https://en.wikipedia.org/wiki/Field_of_sets) construction. Also known as a (Boolean) *set algebra*, this represents the common notion of "overlapping subsets," with the ability to perform setwise operations like union, intersection, and complement.

Applications include:

- Instantiating *ontologies* like a hierarchy of topics.
- Writing domain-specific languages involving boolean operations.
- Pedagogical applications to aid in learning predicate logic.

Two advantages of `setfield` over ordinary Python sets are:

1. The presence of an ambient *universe* set makes the complement well-defined.
2. Compositional constructs like union-of-ranges and boolean operators can be used to make set construction and membership querying more efficient in both time and memory.

## Installation

The library is available on PyPI. To install:

`pip install setfield`

## Usage

Suppose we have a collection (universe) of movies and want to organize them by genre, allowing some movies to belong to multiple genres. We can create a **field of sets** like so:

```python
from setfield import Subset

movies = {
    'Alien',
    'Blade Runner',
    'Casablanca',
    'Dunkirk',
    'Frankenstein',
    'Her',
    'Interstellar',
    'The Shining',
}

# construct subsets
sci_fi = Subset(movies, {'Alien', 'Blade Runner', 'Her', 'Interstellar'})
horror = Subset(movies, {'Alien', 'Frankenstein', 'The Shining'})
romance = Subset(movies, {'Casablanca', 'Her', 'Interstellar'})

# check setwise relationships
assert sci_fi < movies
assert 'Her' in horror | romance
assert 'Frankenstein' in horror - sci_fi
assert sci_fi & horror == {'Alien'}
assert 'Dunkirk' in ~(sci_fi | horror | romance)
```

### Parsing Boolean Expressions

We can also use `setfield` to create a miniature *Domain-Specific Language* (DSL) for boolean expressions with our custom field of sets. This is useful if we want to provide a way for users to express set combinations with a simple, natural syntax. Here's an example, continuing with movie genres:

```python
from setfield import safe_eval_boolean_expr

genres = {
    'sci_fi': sci_fi,
    'horror': horror,
    'romance': romance,
}

def movies_for_genre(genre: str) -> Subset[str]:
    """Get the movies for a given genre, raising a ValueError if the genre is unknown."""
    if genre in genres:
        return genres[genre]
    raise ValueError(f'unknown genre: {genre}')

def interpret_genres(expr: str) -> Subset[str]:
    """Interpret a boolean expression involving genres."""
    return safe_eval_boolean_expr(expr, eval_name=movies_for_genre)


assert interpret_genres('sci_fi & horror') == {'Alien'}
assert interpret_genres('horror - sci_fi') == {'Frankenstein', 'The Shining'}
assert interpret_genres('~(sci_fi | horror | romance)') == {'Dunkirk'}
```

As the name suggests, `safe_eval_boolean_expr` is "safe" in that it will not execute arbitrary Python code—it evaluates names exclusively with the provided `eval_name` function and then applies boolean operations to the results.

### Union of Ranges

If we're concerned with integer subsets, there is a `RangeUnionSubset` data structure which is often more efficient than a typical `Subset` (which stores all of its elements in a set). This consists of an ordered sequence of non-overlapping *ranges*, or [start, stop) pairs.

As an example:

```python
from setfield import RangeUnionSubset

subset = RangeUnionSubset(range(100), [range(0, 10), range(50, 75)])

assert len(subset) == 35
assert 9 in subset
assert 20 not in subset
assert 55 in subset

# complement is calculated efficiently
print(~subset)
# RangeUnionSubset(universe_range=range(0, 100), ranges=[range(10, 50), range(75, 100)])

# likewise with intersections, unions, and differences
subset2 = RangeUnionSubset(range(100), [range(40, 60)])

print(subset & subset2)
# RangeUnionSubset(universe_range=range(0, 100), ranges=[range(50, 60)])

print(subset | subset2)
# RangeUnionSubset(universe_range=range(0, 100), ranges=[range(0, 10), range(40, 75)])

print(subset - subset2)
# RangeUnionSubset(universe_range=range(0, 100), ranges=[range(0, 10), range(60, 75)])
```

## License

This library is open-source and licensed under the [MIT License](LICENSE).

Contributions are welcome!
