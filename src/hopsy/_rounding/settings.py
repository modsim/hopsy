from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_BACKEND = "highs"
DEFAULT_HP_FLAGS = {
    "FeasibilityTol": 1e-9,
    "OptimalityTol": 1e-8,
}


def fix_backend_name(backend: Any | None) -> str:
    if backend in (None, ""):
        return DEFAULT_BACKEND
    return str(backend).strip().lower()


@dataclass
class RoundingSettings:
    """
    Public rounding settings object used by hopsy.LP().
    """

    backend: str = DEFAULT_BACKEND
    hp_flags: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_HP_FLAGS))
    thresh: float = 1e-7
    verbose: bool = False
    sgp: bool = False
    reduce: bool = True
    regularize: bool = False
    check_lps: bool = False
    simplify_only: bool = False
    presolve: bool = False
    numerics_threshold: float = 1e-12
    accepted_tol_violation: float = 1e2

    _initialized: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.backend = fix_backend_name(self.backend)
