"""Provides a miniature domain-specific language (DSL) for combining sets via boolean algebra expressions."""

import ast
from collections.abc import Callable
from typing import Any, Optional, TypeAlias, TypeVar


T = TypeVar('T')


# function taking variadic args as input and returning T
EvalCallable: TypeAlias = Callable[..., T]


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


class _LiteralWrapper(ast.NodeTransformer):
    """Helper class for wrapping ast.Constant nodes into ast.Call nodes with the constant as the argument."""

    def visit_Constant(self, node: ast.expr) -> ast.Call:
        return ast.Call(
            func=ast.Name(id='__lit__', ctx=ast.Load()),
            args=[node],
            keywords=[],
        )

class _EvalCallableOnLiterals(ast.NodeTransformer):
    """Helper class for modifying a parsed AST to pre-apply a callable with literal arguments."""

    def __init__(self, eval_callable_with_lit_args: Callable[[str], Optional[EvalCallable[Any]]]) -> None:
        self.eval_callable_with_lit_args = eval_callable_with_lit_args
        self._precompute_ctr = 0
        self._precompute_funcs: dict[str, Callable[[], Any]] = {}

    def _add_precompute_func(self, val: Any) -> str:
        name = f'__precomputed{self._precompute_ctr}'
        self._precompute_ctr += 1
        self._precompute_funcs[name] = lambda: val
        return name

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if (func := self.eval_callable_with_lit_args(func_name)) is not None:
                args = []
                for arg in node.args:
                    if not isinstance(arg, ast.Constant):
                        raise ValueError(f'all arguments to {func_name} must be literals')
                    args.append(arg.value)
                # create a new placeholder function which takes no arguments and returns the precomputed value
                placeholder_func_name = self._add_precompute_func(func(*args))
                call_node = ast.Call(func=ast.Name(id=placeholder_func_name, ctx=ast.Load()), args=[], keywords=[])
                # mark the node to distinguish it from a regular Call node
                call_node._precomputed = True  # type: ignore[attr-defined]
                return call_node
        return self.generic_visit(node)  # pragma: no cover


def safe_eval(
    expr: str,
    eval_name: Optional[Callable[[str], T]] = None,
    *,
    safe_node_types: set[type],
    allow_quotes: bool = False,
    eval_callable: Optional[Callable[[str], EvalCallable[T]]] = None,
    eval_callable_with_lit_args: Optional[Callable[[str], Optional[EvalCallable[T]]]] = None,
) -> T:
    """Calls Python's `eval` function in a more "safe" context, in that the caller must provide:
        1. `eval_name`: a callable which maps names (identifiers) to Python objects of type T, and errors if the name
        is invalid.
        2. `safe_node_types`: a set of `ast.Node` objects indicating which elements of Python syntax are permitted
        in the expression.
    This makes it easy to create miniature Embedded Domain Specific Languages (EDSLs) using only a fragment
    of Python syntax.
    Most notably, it can support expressions that only consist of names and boolean connectives.
    If `eval_name` is None, then no identifiers will be permitted.
    If `allow_quotes` is True, additionally allows the use of quoted literals as names as well.
        - This is useful when names may contain symbols not permitted in Python identifiers.
    If `eval_callable` is provided, it should be a function which evaluates names to callables which take any number of
    arguments of type T as input and return a T as output. The arguments are assumed to already be recursively
    evaluated.
    If `eval_callable_with_lit_args` is provided, it should be a function which evaluates names to callables which take
    any number of raw (unevaluated) literals as input and return a T as output."""
    if eval_name is None:
        safe_node_types = safe_node_types - {ast.Name}
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError('invalid expression') from e
    # first, validate the AST nodes
    for node in ast.walk(tree):
        if (
            ((tp := type(node)) not in safe_node_types)
            and not (
                (isinstance(node, ast.Call) and (eval_callable or eval_callable_with_lit_args))
                or (isinstance(node, ast.Constant) and allow_quotes)
            )
        ):
            raise ValueError(f'disallowed construct: {tp.__name__}')
    _locals: dict[str, T | Callable[..., T]] = {}
    if eval_callable_with_lit_args:
        # process the AST to pre-evaluate callable nodes with all-literal children
        transformer = _EvalCallableOnLiterals(eval_callable_with_lit_args)
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        # retrieve the names of the placeholder functions which will return the precomputed values
        _locals.update(transformer._precompute_funcs)
    eval_lit = None
    if allow_quotes and (eval_name is not None):
        # since string names are permitted, we will evaluate literal strings with the `eval_name` function
        def eval_lit(s: str) -> T:
            if isinstance(s, str):
                return eval_name(s)
            raise ValueError(f'disallowed literal type: {type(s).__name__}')
        # wrap Constant nodes into Call nodes with the name as the argument, to be evaluated later by `eval_lit`
        tree = _LiteralWrapper().visit(tree)
        ast.fix_missing_locations(tree)
    for node in ast.walk(tree):
        if (
            eval_callable
            and isinstance(node, ast.Call)
            and (not getattr(node, '_precomputed', False))
            and (node.func.id != '__lit__')  # type: ignore[attr-defined]
        ):
            _locals[node.func.id] = eval_callable(node.func.id)  # type: ignore[attr-defined]
        elif isinstance(node, ast.Name) and (node.id != '__lit__') and (node.id not in _locals):
            # NOTE: eval_identifier should raise an error if identifier is invalid
            _locals[node.id] = eval_name(node.id)  # type: ignore[misc]
    # evaluate directly from code object (avoids re-parsing from a string)
    compiled = compile(tree, '<string>', 'eval')
    return eval(compiled, {'__builtins__': {}, '__lit__': eval_lit}, _locals)  # type: ignore[no-any-return]

def safe_eval_boolean_expr(
    expr: str,
    eval_name: Optional[Callable[[str], T]] = None,
    *,
    allow_quotes: bool = False,
    eval_callable: Optional[Callable[[str], EvalCallable[T]]] = None,
    eval_callable_with_lit_args: Optional[Callable[[str], Optional[EvalCallable[T]]]] = None,
) -> T:
    """Given an expression and a callable `eval_name`, evaluates the expression to a Python object using
    a safe version of `eval` which only allows specific identifiers and boolean connectives.
    `eval_name` should be a function that maps names to Python objects, and it should raise an exception if
    the name is not valid.
    If `allow_quotes` is True, additionally allows the use of quoted literals as names as well.
        - This is useful when names may contain symbols not permitted in Python identifiers.
    If `eval_callable` is provided, it should be a function which evaluates names to callables which take any number of
    arguments of type T as input and return a T as output. The arguments are assumed to already be recursively
    evaluated.
    If `eval_callable_with_lit_args` is provided, it should be a function which evaluates names to callables which take
    any number of raw (unevaluated) literals as input and return a T as output."""
    return safe_eval(
        expr,
        eval_name=eval_name,
        safe_node_types=BOOLEAN_SAFE_NODE_TYPES,
        allow_quotes=allow_quotes,
        eval_callable=eval_callable,
        eval_callable_with_lit_args=eval_callable_with_lit_args,
    )
