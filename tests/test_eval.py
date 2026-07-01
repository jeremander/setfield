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
    raise ValueError(f'invalid name: {name}')

small_universe = {1, 2, 3, 4, 5}

def example_interpret(expr: str, *, allow_quotes: bool = False) -> BaseSubset[int]:
    def eval_name(name: str) -> Subset[int]:
        return Subset(small_universe, _get_set(name))
    return safe_eval_boolean_expr(expr, eval_name, allow_quotes=allow_quotes)


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

    @pytest.mark.parametrize(['expr', 'output_set', 'error'], [
        (
            '"A"',
            {1, 2, 3},
            None,
        ),
        (
            "'A'",
            {1, 2, 3},
            None,
        ),
        (
            '"A" & B',
            {3},
            None,
        ),
        (
            '"D"',
            None,
            'invalid name: D',
        ),
        (
            '1',
            None,
            'disallowed literal type: int',
        ),
        (
            'A & 1',
            None,
            'disallowed literal type: int',
        ),
    ])
    def test_interpret_bool_expr_with_quotes(self, expr, output_set, error):
        if error is None:
            value = example_interpret(expr, allow_quotes=True)
            assert set(value) == output_set
        else:
            with pytest.raises(ValueError, match=error):
                _ = example_interpret(expr, allow_quotes=True)
