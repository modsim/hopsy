"""Public facade for hopsy's vendored PolyRound integration."""

from . import polytope
from .api import (
    PolyRoundApi,
    active_backend,
    backend_name,
)
from .polytope import Polytope
from .settings import DEFAULT_BACKEND, PolyRoundSettings

__all__ = [
    "DEFAULT_BACKEND",
    "PolyRoundApi",
    "PolyRoundSettings",
    "Polytope",
    "active_backend",
    "backend_name",
    "polytope",
]
