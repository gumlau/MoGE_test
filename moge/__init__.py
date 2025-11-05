import types as _types

# Python <3.10 compatibility for external deps expecting types.EllipsisType
if not hasattr(_types, "EllipsisType"):
    _types.EllipsisType = type(Ellipsis)

del _types
