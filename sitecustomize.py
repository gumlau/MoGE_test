"""Ensure MoGE runtime compatibility patches are applied eagerly.

When Python's site initialisation imports ``sitecustomize``, we trigger the
project's compatibility shims before any heavy modules load.
"""

try:
    from moge.compat import ensure_runtime_compatibility
except Exception:  # pragma: no cover - best-effort during interpreter startup
    pass
else:
    ensure_runtime_compatibility()
