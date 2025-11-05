"""Runtime compatibility shims for older Python versions.

This module centralises all runtime patches needed when executing MoGE under
Python versions (< 3.10) that lack newer typing helpers. The public helper
``ensure_runtime_compatibility`` is idempotent so it can be imported safely by
entry points as well as ``sitecustomize`` when available.
"""

from __future__ import annotations

import builtins
import types
import typing

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - numpy not available yet
    _np = None  # type: ignore

try:  # typing_extensions may be absent; handled later if so
    import typing_extensions
except ImportError:  # pragma: no cover - optional dependency
    typing_extensions = None  # type: ignore

_ORIGINAL_TYPING_GETATTR = getattr(typing, "__getattr__", None)

_COMPAT_APPLIED = False


def _ensure_ellipsis_type() -> None:
    if not hasattr(types, "EllipsisType"):
        types.EllipsisType = type(Ellipsis)


def _ensure_param_spec() -> None:
    if hasattr(typing, "ParamSpec"):
        return
    try:
        from typing_extensions import ParamSpec  # type: ignore
    except Exception:  # pragma: no cover - typing_extensions missing
        class _SyntheticParamSpec:
            __slots__ = ("__name__", "args", "kwargs")

            def __init__(self, name: str, *args: object, **kwargs: object) -> None:
                self.__name__ = name
                self.args = args
                self.kwargs = kwargs

            def __repr__(self) -> str:  # pragma: no cover - debugging aid
                return f"ParamSpec({self.__name__!r})"

        def ParamSpec(name: str, *args: object, **kwargs: object) -> _SyntheticParamSpec:  # type: ignore[override]
            return _SyntheticParamSpec(name, *args, **kwargs)

    typing.ParamSpec = ParamSpec  # type: ignore[attr-defined]
    if typing_extensions is not None and not hasattr(typing_extensions, "ParamSpec"):
        typing_extensions.ParamSpec = ParamSpec  # type: ignore[attr-defined]
    if not hasattr(builtins, "ParamSpec"):
        builtins.ParamSpec = ParamSpec  # type: ignore[attr-defined]


def _ensure_typing_symbol(name: str) -> None:
    if hasattr(typing, name):
        return
    symbol = None
    if typing_extensions is not None:
        symbol = getattr(typing_extensions, name, None)
    if symbol is None:
        return
    setattr(typing, name, symbol)
    if typing_extensions is not None and not hasattr(typing_extensions, name):
        setattr(typing_extensions, name, symbol)
    if not hasattr(builtins, name):
        setattr(builtins, name, symbol)


def _install_typing_getattr_fallback() -> None:
    current_getattr = getattr(typing, "__getattr__", None)
    if getattr(current_getattr, "__moge_patch__", False):  # type: ignore[attr-defined]
        return

    def _patched_typing_getattr(name: str):  # type: ignore[override]
        if name == "ParamSpec":
            _ensure_param_spec()
            if hasattr(typing, "ParamSpec"):
                return typing.__dict__["ParamSpec"]
        if name in {"TypeAlias", "TypeGuard", "Concatenate"}:
            _ensure_typing_symbol(name)
            if name in typing.__dict__:
                return typing.__dict__[name]
        if current_getattr is not None:
            return current_getattr(name)
        raise AttributeError(f"module 'typing' has no attribute {name!r}")

    _patched_typing_getattr.__moge_patch__ = True  # type: ignore[attr-defined]
    typing.__getattr__ = _patched_typing_getattr  # type: ignore[assignment]


def _ensure_numpy_matrix_transpose() -> None:
    if _np is None:
        return
    if hasattr(_np.ndarray, "mT"):
        return

    def _matrix_transpose(self):  # type: ignore[override]
        return self.T

    try:
        setattr(_np.ndarray, "mT", property(_matrix_transpose))
    except Exception:  # pragma: no cover - unable to patch ndarray type
        pass


def ensure_runtime_compatibility() -> None:
    """Apply one-off runtime patches required for third-party libs."""

    global _COMPAT_APPLIED
    if _COMPAT_APPLIED:
        return

    _ensure_ellipsis_type()
    _ensure_param_spec()
    for helper in ("TypeAlias", "TypeGuard", "Concatenate"):
        _ensure_typing_symbol(helper)
    _install_typing_getattr_fallback()
    _ensure_numpy_matrix_transpose()

    _COMPAT_APPLIED = True


__all__ = ["ensure_runtime_compatibility"]

