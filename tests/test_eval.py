"""Unit tests for interpretation of boolean expressions."""

import ast
from collections.abc import Callable
from typing import Optional, TypeVar

import pytest

from setfield import (
    BOOLEAN_SAFE_NODE_TYPES,
    BaseSubset,
    Subset,
    SubsetComplement,
    SubsetIntersection,
    SubsetUnion,
    get_empty_subset,
    safe_eval,
    safe_eval_boolean_expr,
)


T = TypeVar('T')


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
        case 'A Z':  # requires quotes to express as a name
            return {1, 5}
    raise ValueError(f'invalid name: {name}')

small_universe = {1, 2, 3, 4, 5}

def example_eval_name(name: str) -> Subset[int]:
    """Example function which evaluates a name to an integer Subset."""
    return Subset(small_universe, _get_set(name))

def example_eval_callable(name: str) -> Callable[..., Subset[int]]:
    """Example function which evaluates a name to a callable producing the empty set."""
    empty = get_empty_subset(small_universe)
    match name:
        case 'empty0':
            return lambda: empty
        case 'empty1':
            return lambda _set1: empty
        case 'empty2':
            return lambda _set1, _set2: empty
    raise ValueError(f'invalid callable: {name}')

def example_interpret(expr: str, *, allow_quotes: bool = False, allow_callable: bool = False) -> BaseSubset[int]:
    """Example interpretation function for evaluating a boolean expression combining named sets.
    If allow_quotes=True, allows quoted names.
    If allow_callable, allows example callables (empty0, empty1, empty2)."""
    eval_callable = example_eval_callable if allow_callable else None
    return safe_eval_boolean_expr(expr, example_eval_name, allow_quotes=allow_quotes, eval_callable=eval_callable)


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
        """Tests an example evaluation function, for valid expressions."""
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
        (
            'A Z',
            'invalid expression',
        ),
        # quoted literals not permitted
        (
            '"A Z"',
            'disallowed construct: Constant',
        ),
    ])
    def test_interpret_bool_expr_invalid(self, expr, error):
        """Tests an example evaluation function, for invalid expressions."""
        with pytest.raises(ValueError, match=error):
            _ = example_interpret(expr)

    @pytest.mark.parametrize(['expr', 'output_set'], [
        (
            '"A"',
            {1, 2, 3},
        ),
        (
            "'A'",
            {1, 2, 3},
        ),
        (
            '"A" & B',
            {3},
        ),
        # name with a space
        (
            "'A Z'",
            {1, 5},
        ),
        (
            '"A Z"',
            {1, 5},
        ),
        (
            '"A Z"&\'A Z\'',
            {1, 5},
        ),
    ])
    def test_interpret_bool_expr_with_quotes_valid(self, expr, output_set):
        """Tests an example evaluation function when allowing quoted names, for valid expressions."""
        value = example_interpret(expr, allow_quotes=True)
        assert set(value) == output_set

    @pytest.mark.parametrize(['expr', 'error'], [
        (
            '"D"',
            'invalid name: D',
        ),
        (
            '1',
            'disallowed literal type: int',
        ),
        (
            'A & 1',
            'disallowed literal type: int',
        ),
    ])
    def test_interpret_bool_expr_with_quotes_invalid(self, expr, error):
        """Tests an example evaluation function when allowing quoted names, for invalid expressions."""
        with pytest.raises(ValueError, match=error):
            _ = example_interpret(expr, allow_quotes=True)

    @pytest.mark.parametrize(['expr', 'output_set'], [
        (
            'empty0()',
            set(),
        ),
        (
            'A',
            {1, 2, 3},
        ),
        (
            'A & empty0()',
            set(),
        ),
        (
            'empty1(A)',
            set(),
        ),
        (
            'empty1(A | B)',
            set(),
        ),
        (
            'empty2(A, B)',
            set(),
        ),
    ])
    def test_interpret_bool_expr_with_callable_valid(self, expr, output_set):
        """Tests an example evaluation function which permits callables, for valid expressions."""
        value = example_interpret(expr, allow_callable=True)
        assert set(value) == output_set

    @pytest.mark.parametrize(['expr', 'error'], [
        (
            'D',
            'invalid name: D',
        ),
        (
            'empty0',
            'invalid name: empty0',
        ),
        (
            'empty0(',
            'invalid expression',
        ),
        (
            'empty0(A)',
            'takes 0 positional arguments but 1 was given',
        ),
        (
            'empty1()',
            'missing 1 required positional argument',
        ),
        (
            'empty2(A)',
            'missing 1 required positional argument',
        ),
        (
            'A()',
            'invalid callable: A',
        ),
    ])
    def test_interpret_bool_expr_with_callable_invalid(self, expr, error):
        """Tests an example evaluation function which permits callables, for invalid expressions."""
        with pytest.raises((ValueError, TypeError), match=error):
            _ = example_interpret(expr, allow_callable=True)
