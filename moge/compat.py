"""Runtime compatibility shims for older Python versions.

This module centralises all runtime patches needed when executing MoGE under
Python versions (< 3.10) that lack newer typing helpers. The public helper
``ensure_runtime_compatibility`` is idempotent so it can be imported safely by
entry points as well as ``sitecustomize`` when available.
"""

from __future__ import annotations

import types
import typing

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


def _ensure_typing_symbol(name: str) -> None:
    if hasattr(typing, name):
        return
    try:
        from typing_extensions import __dict__ as typing_ext_dict  # type: ignore
    except Exception:  # pragma: no cover - typing_extensions missing
        return
    symbol = typing_ext_dict.get(name)
    if symbol is None:
        return
    setattr(typing, name, symbol)


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

    _COMPAT_APPLIED = True


__all__ = ["ensure_runtime_compatibility"]
