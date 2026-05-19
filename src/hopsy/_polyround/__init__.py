"""Public facade for hopsy's vendored PolyRound integration."""

from . import polytope
from .api import (
    PolyRoundApi,
    active_backend,
    backend_name,
)
from .polytope import Polytope
from .settings import DEFAULT_BACKEND, PolyRoundSettings

__version__ = "0.4.0"
__author__ = "Axel Theorell, Johann Fredrik Jadebeck"

__all__ = [
    "DEFAULT_BACKEND",
    "PolyRoundApi",
    "PolyRoundSettings",
    "Polytope",
    "__author__",
    "__version__",
    "active_backend",
    "backend_name",
    "polytope",
]
