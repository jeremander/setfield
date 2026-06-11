"""Provides a miniature domain-specific language (DSL) for combining sets via boolean algebra expressions."""

import ast
from collections.abc import Callable
from typing import Optional, TypeVar


T = TypeVar('T')


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
        2. `safe_node_types`: a set of `ast.Node` objects indicating which elements of Python syntax are permitted
        in the expression.
    This makes it easy to create miniature Embedded Domain Specific Languages (EDSLs) using only a fragment
    of Python syntax.
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
    """Given an expression and a callable `eval_name`, evaluates the expression to a Python object using
    a safe version of `eval` which only allows specific identifiers and boolean connectives.
    `eval_name` should be a function that maps names to Python objects, and it should raise an exception if
    the name is not valid."""
    return safe_eval(expr, eval_name=eval_name, safe_node_types=BOOLEAN_SAFE_NODE_TYPES)
