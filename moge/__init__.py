import types as _types
import typing as _typing

# Python <3.10 compatibility for external deps expecting types.EllipsisType
if not hasattr(_types, "EllipsisType"):
    _types.EllipsisType = type(Ellipsis)

# Provide typing.ParamSpec for Python versions that lack it
try:
    from typing import ParamSpec as _ParamSpec  # type: ignore
except ImportError:
    from typing_extensions import ParamSpec as _ParamSpec  # type: ignore
    _typing.ParamSpec = _ParamSpec  # type: ignore[attr-defined]
else:
    if not hasattr(_typing, "ParamSpec"):
        _typing.ParamSpec = _ParamSpec  # type: ignore[attr-defined]

del _types
del _typing
del _ParamSpec
