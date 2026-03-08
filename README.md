# Setfield

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/setfield)](https://pypi.org/project/setfield/)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/jeremander/setfield/workflow.yml)
![Coverage Status](https://github.com/jeremander/setfield/raw/coverage-badge/coverage-badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://raw.githubusercontent.com/jeremander/setfield/refs/heads/main/LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) -->

**Setfield** is a small Python library providing a framework for expressing a [field of sets](https://en.wikipedia.org/wiki/Field_of_sets) construction. Also known as a (Boolean) algebra of sets, this represents the common notion of "overlapping subsets," with the ability to perform setwise operations like union, intersection, and complement.

Applications include:

- Instantiating *ontologies* like a hierarchy of topics.
- Writing domain-specific languages involving boolean operations.
- Pedagogical applications to aid in learning predicate logic.

Two advantages of `setfield` over ordinary Python sets are:

1. The presence of an ambient *universe* set makes the complement well-defined.
2. Compositional constructs like "union of ranges" and boolean operators can be used to make set construction and membership querying more efficient in both time and memory.

**Undercat** is a small Python library implementing a functional programming construct called the *Reader functor*. This pattern is particularly useful for dependency injection, composing functions, and operating on immutable, context-aware computations.

<!--
TODO: real-world example of mini-ontology (e.g. "tags")

Example:

fruit: apple, lime, orange, ...
food: apple, lime, orange, potato, tomato, pie
color: red, lime, mauve, orange, purple, ...
-->