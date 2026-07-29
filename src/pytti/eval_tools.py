"""
Evaluation of the config's parametric expressions (e.g. translate_x:
'10*sin(t/2)'), plus small parsing/IO helpers shared by the prompt DSL.

Expressions are evaluated over an AST allowlist — arithmetic, comparisons,
math-module functions, `t`, audio band variables, and caller-supplied
values — rather than raw eval(), so loading someone else's config is not
code execution.
"""

import ast
import io
import math
import re
from pathlib import Path

import requests

_MATH_ENV = {
    name: getattr(math, name) for name in dir(math) if not name.startswith("_")
}
_MATH_ENV.update({"abs": abs, "max": max, "min": min, "pow": pow, "round": round})

# expression state: advanced by the render loop via set_t / set_bands
global_t = 0.0
global_bands: dict[str, float] = {}
global_bands_prev: dict[str, float] = {}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.BoolOp,
    ast.IfExp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.List,
    ast.Tuple,
    # operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Load,
)


def parametric_eval(string, **vals):
    """
    Evaluate a parametric expression string. Non-strings pass through
    unchanged. The environment is rebuilt on every call from explicit state,
    so per-frame values (t, audio bands, depth stats) can never go stale.
    """
    if not isinstance(string, str):
        return string
    env = dict(_MATH_ENV)
    env["t"] = global_t
    env.update(global_bands)
    env.update({f"{k}_prev": v for k, v in global_bands_prev.items()})
    env.update(vals)
    try:
        tree = ast.parse(string, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise ValueError(f"disallowed syntax: {type(node).__name__}")
            if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
                raise ValueError("only plain function calls are allowed")
            if isinstance(node, ast.Name) and node.id not in env:
                raise NameError(f"unknown name {node.id!r}")
        return eval(compile(tree, "<parametric>", "eval"), {"__builtins__": {}}, env)
    except Exception as e:
        raise RuntimeError(
            f"Could not evaluate parametric expression {string!r}: {e}"
        ) from e


def set_t(t, band_dict=None):
    """
    Update the expression time. If band_dict is provided, also update the
    audio band variables (rolling the previous frame's values into *_prev) —
    only pass band_dict at frame boundaries.
    """
    global global_t
    global_t = t
    if band_dict is not None:
        set_bands(band_dict)


def set_bands(band_dict):
    """
    Update audio band variables. The previous frame's values become the
    `<band>_prev` variables. When the audio track has ended (empty dict),
    known bands decay to 0.0 instead of freezing at their last value.
    """
    global global_bands, global_bands_prev
    if not band_dict and global_bands:
        band_dict = {k: 0.0 for k in global_bands}
    global_bands_prev = dict(global_bands) if global_bands else dict(band_dict)
    global_bands = dict(band_dict)


def is_zero_weight(weight) -> bool:
    """A weight that disables its loss: 0, 0.0, '', '0', '0.0'."""
    if isinstance(weight, (int, float)):
        return weight == 0
    return str(weight).strip() in ("", "0", "0.0")


def fetch(url_or_path):
    """
    Return a binary file-like object for a local path or http(s) URL.
    Always fully buffered — no file handles are left open.
    """
    text = str(url_or_path)
    if text.startswith(("http://", "https://")):
        r = requests.get(text, timeout=60)
        r.raise_for_status()
        return io.BytesIO(r.content)
    return io.BytesIO(Path(text).read_bytes())


def parse(string, split, defaults):
    """
    Given a string, a regex pattern, and a list of defaults,
    split the string using the regex pattern,
    and return the split string + the defaults

    :param string: The string to be parsed
    :param split: The regex that defines where to split the string
    :param defaults: A list of default values for the tokens
    :return: A list of the tokens.
    """
    tokens = re.split(split, string, maxsplit=len(defaults) - 1)
    tokens = tokens + defaults[len(tokens) :]
    return tokens
