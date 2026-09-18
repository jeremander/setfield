"""Unit tests for interpretation of boolean expressions."""

import ast
from collections.abc import Callable
import re
from typing import Optional, TypeAlias, TypeVar

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

EvalNameFunc: TypeAlias = Callable[[str], set[int]]


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
        case 'func':
            return lambda _set1: Subset(small_universe, {1, 2})
    raise ValueError(f'invalid callable: {name}')

def example_eval_callable_with_lit_args(name: str) -> Optional[Callable[..., Subset[int]]]:
    match name:
        case 'func':  # NOTE: this shadows and takes priority over the eval_callable 'func'
            return lambda _set1: Subset(small_universe, {3})
        case 'str2ints':
            return lambda s: Subset(small_universe, {int(c) for c in s})
        case _:
            return None

def example_interpret(
    expr: str,
    *,
    allow_quotes: bool = False,
    allow_callable: bool = False,
    allow_callable_with_lit_args: bool = False,
) -> BaseSubset[int]:
    """Example interpretation function for evaluating a boolean expression combining named sets.
    If allow_quotes=True, allows quoted names.
    If allow_callable, allows example callables with evaluated args.
    If allow_callable, allows example callables with literal args."""
    eval_callable = example_eval_callable if allow_callable else None
    eval_callable_with_lit_args = example_eval_callable_with_lit_args if allow_callable_with_lit_args else None
    return safe_eval_boolean_expr(
        expr,
        example_eval_name,
        allow_quotes=allow_quotes,
        eval_callable=eval_callable,
        eval_callable_with_lit_args=eval_callable_with_lit_args,
    )


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
    def test_interpret_arith_expr(
        self,
        expr: str,
        eval_names: EvalNameFunc | list[EvalNameFunc],
        value: Optional[set[int]],
        error: Optional[str],
    ) -> None:
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
    def test_interpret_bool_expr_valid(
        self,
        expr: str,
        output_type: type[BaseSubset[int]],
        output_set: set[int],
    ) -> None:
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
    def test_interpret_bool_expr_invalid(self, expr: str, error: str) -> None:
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
    def test_interpret_bool_expr_with_quotes_valid(self, expr: str, output_set: set[int]) -> None:
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
    def test_interpret_bool_expr_with_quotes_invalid(self, expr: str, error: str) -> None:
        """Tests an example evaluation function when allowing quoted names, for invalid expressions."""
        with pytest.raises(ValueError, match=error):
            _ = example_interpret(expr, allow_quotes=True)

    @pytest.mark.parametrize(['expr', 'allow_quotes', 'allow_callable_with_lit_args', 'output_set_or_err'], [
        (
            'D',
            None,
            None,
            ValueError('invalid name: D'),
        ),
        (
            'A',
            None,
            None,
            {1, 2, 3},
        ),
        (
            'A()',
            None,
            None,
            ValueError('invalid callable: A'),
        ),
        (
            'empty0',
            None,
            None,
            ValueError('invalid name: empty0'),
        ),
        (
            'empty0()',
            None,
            None,
            set(),
        ),
        (
            'empty0(',
            None,
            None,
            ValueError('invalid expression'),
        ),
        (
            'A & empty0()',
            None,
            None,
            set(),
        ),
        (
            'empty0(A)',
            None,
            None,
            TypeError('takes 0 positional arguments but 1 was given'),
        ),
        (
            'empty1()',
            None,
            None,
            TypeError('missing 1 required positional argument'),
        ),
        (
            'empty1(123)',
            False,
            None,
            ValueError('disallowed construct: Constant'),
        ),
        (
            'empty1(123)',
            True,
            None,
            ValueError('disallowed literal type: int'),
        ),
        (
            'empty1("A")',
            False,
            None,
            ValueError('disallowed construct: Constant'),
        ),
        (
            "empty1('A')",
            False,
            None,
            ValueError('disallowed construct: Constant'),
        ),
        (
            'empty2(A)',
            None,
            None,
            TypeError('missing 1 required positional argument'),
        ),
        (
            'empty1(A)',
            None,
            None,
            set(),
        ),
        (
            'empty1("A")',
            True,
            None,
            set(),
        ),
        (
            "empty1('A')",
            True,
            None,
            set(),
        ),
        (
            'empty1(A | B)',
            None,
            None,
            set(),
        ),
        (
            'empty2(A, B)',
            None,
            None,
            set(),
        ),
        (
            'str2ints()',
            None,
            False,
            ValueError('invalid callable: str2ints'),
        ),
        (
            'str2ints()',
            None,
            True,
            TypeError('missing 1 required positional argument'),
        ),
        (
            'str2ints(A)',
            None,
            False,
            ValueError('invalid callable: str2ints'),
        ),
        (
            'str2ints(A)',
            None,
            True,
            ValueError('all arguments to str2ints must be literals'),
        ),
        (
            'str2ints("A")',
            False,
            False,
            ValueError('disallowed construct: Constant'),
        ),
        (
            'str2ints("A")',
            True,
            False,
            ValueError('invalid callable: str2ints')
        ),
        (
            'str2ints("A")',
            False,
            True,
            ValueError('disallowed construct: Constant'),
        ),
        (
            'str2ints("A")',
            True,
            True,
            ValueError(r'invalid literal for int\(\) with base 10'),
        ),
        (
            'str2ints(123)',
            False,
            True,
            ValueError('disallowed construct: Constant'),
        ),
        (
            'str2ints(123)',
            True,
            True,
            TypeError("'int' object is not iterable"),
        ),
        (
            'str2ints("123")',
            False,
            True,
            ValueError('disallowed construct: Constant'),
        ),
        (
            'str2ints("123")',
            True,
            True,
            {1, 2, 3},
        ),
        (
            'str2ints("0123")',
            True,
            True,
            ValueError('0 is not an element of the universe'),
        ),
        (
            'str2ints("12") | str2ints("3") | empty0()',
            True,
            True,
            {1, 2, 3},
        ),
        (
            'func(A)',
            None,
            False,
            {1, 2},
        ),
        (
            'func(A)',
            None,
            True,
            ValueError('all arguments to func must be literals'),
        ),
        (
            'func("A")',
            True,
            False,
            {1, 2},
        ),
        (
            'func("A")',
            True,
            True,
            {3},
        ),
        (
            "func('A')",
            True,
            False,
            {1, 2},
        ),
        (
            "func('A')",
            True,
            True,
            {3},
        ),
        (
            'empty1(empty0())',
            None,
            None,
            set(),
        ),
        (
            'func(empty1(empty0()))',
            None,
            False,
            {1, 2},
        ),
        (
            'func(empty1(empty0()))',
            None,
            True,
            ValueError('all arguments to func must be literals'),
        ),
        (
            'empty2(empty0(), empty1(A))',
            None,
            None,
            set(),
        ),
        (
            'empty0() | empty0() | empty1(A) | empty1(B) | func(A) | empty1(A) | func(A)',
            None,
            False,
            {1, 2},
        ),
        (
            'empty0() | empty0() | empty1(A) | empty1(B) | func(A) | empty1(A) | func(A)',
            None,
            True,
            ValueError('all arguments to func must be literals'),
        ),
    ])
    def test_interpret_bool_expr_with_callable_valid(
        self,
        expr: str,
        allow_quotes: Optional[bool],
        allow_callable_with_lit_args: Optional[bool],
        output_set_or_err: set[int] | Exception,
    ) -> None:
        """Tests an example evaluation function which permits callables, for valid expressions."""
        allow_quotes_flags = [False, True] if (allow_quotes is None) else [allow_quotes]
        allow_callable_with_lit_args_flags = (
            [False, True]
            if (allow_callable_with_lit_args is None)
            else [allow_callable_with_lit_args]
        )
        for allow_quotes_flag in allow_quotes_flags:
            for allow_callable_with_lit_args_flag in allow_callable_with_lit_args_flags:
                try:
                    value = example_interpret(
                        expr,
                        allow_quotes=allow_quotes_flag,
                        allow_callable=True,
                        allow_callable_with_lit_args=allow_callable_with_lit_args_flag,
                    )
                except Exception as e:
                    assert type(e) is type(output_set_or_err)  # noqa: PT017
                    assert re.search(str(output_set_or_err), str(e))  # noqa: PT017
                else:
                    assert set(value) == output_set_or_err
