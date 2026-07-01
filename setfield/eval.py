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


class _LiteralWrapper(ast.NodeTransformer):
    """Helper class for wrapping ast.Constant nodes into ast.Call nodes with the constant as the argument."""

    def visit_Constant(self, node: ast.expr) -> ast.Call:
        return ast.Call(
            func=ast.Name(id='__lit__', ctx=ast.Load()),
            args=[node],
            keywords=[],
        )


def safe_eval(
    expr: str,
    eval_name: Optional[Callable[[str], T]] = None,
    *,
    safe_node_types: set[type],
    allow_quotes: bool = False,
) -> T:
    """Calls Python's `eval` function in a more "safe" context, in that the caller must provide:
        1. `eval_name`: a callable which maps names (identifiers) to Python objects, and errors if the name is invalid.
        2. `safe_node_types`: a set of `ast.Node` objects indicating which elements of Python syntax are permitted
        in the expression.
    This makes it easy to create miniature Embedded Domain Specific Languages (EDSLs) using only a fragment
    of Python syntax.
    Most notably, it can support expressions that only consist of names and boolean connectives.
    If `eval_name` is None, then no identifiers will be permitted.
    If `allow_quotes` is True, additionally allows the use of quoted literals as names as well.
    This is useful when names may contain symbols not permitted in Python identifiers."""
    if eval_name is None:
        safe_node_types = safe_node_types - {ast.Name}
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError('invalid expression') from e
    eval_lit = None
    if allow_quotes and (eval_name is not None):
        # wrap constants in
        tree = _LiteralWrapper().visit(tree)
        ast.fix_missing_locations(tree)
        def eval_lit(s: str) -> T:
            if isinstance(s, str):
                return eval_name(s)
            raise ValueError(f'disallowed literal type: {type(s).__name__}')
    _locals = {}
    for node in ast.walk(tree):
        if allow_quotes and isinstance(node, (ast.Call, ast.Constant)):
            continue
        if (tp := type(node)) not in safe_node_types:
            raise ValueError(f'disallowed construct: {tp.__name__}')
        if isinstance(node, ast.Name) and (node.id != '__lit__'):
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
) -> T:
    """Given an expression and a callable `eval_name`, evaluates the expression to a Python object using
    a safe version of `eval` which only allows specific identifiers and boolean connectives.
    `eval_name` should be a function that maps names to Python objects, and it should raise an exception if
    the name is not valid.
    If `allow_quotes` is True, additionally allows the use of quoted literals as names as well.
    This is useful when names may contain symbols not permitted in Python identifiers."""
    return safe_eval(expr, eval_name=eval_name, safe_node_types=BOOLEAN_SAFE_NODE_TYPES, allow_quotes=allow_quotes)
