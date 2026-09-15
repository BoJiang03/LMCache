"""Import the real LMCache connector/IPC types without a built native extension.

``lmcache.lmcache_native`` (and the ``lmcache.device_ops`` it enables) only
exist once the native extension is compiled.  Nothing on the token-id
transmission path measured here calls into them, but several modules read
their attributes at import time, so recursive import-time stubs are enough to
reach the real dataclasses and the real msgspec key type.
"""

# Standard
import importlib.util
import itertools
import sys
import types
from typing import Any


_stub_counter = itertools.count(1)


class _StubMeta(type):
    """Metaclass answering any class-level attribute with another stub class.

    Stubs are also int-able (each gets a unique value) because registry
    modules enumerate native enum members into ``int``-keyed tables at import
    time.
    """

    def __getattr__(cls, name: str) -> Any:
        value = _StubMeta(name, (), {})
        setattr(cls, name, value)
        return value

    def __int__(cls) -> int:
        # ``cls.__dict__`` (not ``hasattr``) because ``__getattr__`` above
        # fabricates a stub for any missing name, including this one.
        if "_stub_int" not in cls.__dict__:
            cls._stub_int = next(_stub_counter)
        return cls.__dict__["_stub_int"]

    def __hash__(cls) -> int:
        return id(cls)


class _StubModule(types.ModuleType):
    """Module stub answering any attribute with a recursive stub class."""

    # Real strings so ``inspect`` can walk ``sys.modules`` without tripping
    # over a fabricated ``__file__``.
    __file__ = "<lmcache native stub>"
    __spec__ = None

    def __getattr__(self, name: str) -> Any:
        value = _StubMeta(name, (), {})
        setattr(self, name, value)
        return value


def install_native_stubs() -> None:
    """Register no-op ``lmcache_native``/``device_ops`` modules.

    A checkout that has been built already exposes the real extensions, and
    shadowing them would measure the stubs instead; each one is stubbed only
    when it is genuinely absent.
    """
    # First Party
    import lmcache

    if "lmcache.lmcache_native" not in sys.modules and not _real_module_exists(
        "lmcache.lmcache_native"
    ):
        native = _StubModule("lmcache.lmcache_native")
        lmcache.lmcache_native = native  # type: ignore[attr-defined]
        sys.modules["lmcache.lmcache_native"] = native
    if getattr(lmcache, "device_ops", None) is None:
        ops = _StubModule("lmcache.device_ops")
        lmcache.device_ops = ops  # type: ignore[attr-defined]
        sys.modules["lmcache.device_ops"] = ops


def _real_module_exists(name: str) -> bool:
    """Whether ``name`` can be imported for real from this checkout."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False
