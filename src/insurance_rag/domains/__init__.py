"""Insurance domain registry.

Domains register themselves via the :func:`register_domain` decorator.
Use :func:`get_domain` to instantiate a registered domain by name.
"""
from __future__ import annotations

from insurance_rag.domains.base import InsuranceDomain

_REGISTRY: dict[str, type[InsuranceDomain]] = {}


def register_domain(cls: type[InsuranceDomain]) -> type[InsuranceDomain]:
    """Class decorator that registers a domain plugin."""
    instance = cls()
    _REGISTRY[instance.name] = cls
    return cls


def get_domain(name: str) -> InsuranceDomain:
    """Return an instance of the named domain. Raises KeyError if unknown."""
    if name not in _REGISTRY:
        _discover_domains()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown domain {name!r}. Available: {', '.join(sorted(_REGISTRY))}."
        )
    return _REGISTRY[name]()


def list_domains() -> list[str]:
    """Return sorted list of registered domain names."""
    _discover_domains()
    return sorted(_REGISTRY)


def _discover_domains() -> None:
    """Import built-in domain packages so they self-register."""
    import importlib

    for mod_name in ("insurance_rag.domains.medicare", "insurance_rag.domains.auto"):
        try:
            importlib.import_module(mod_name)
        except ImportError:
            pass
