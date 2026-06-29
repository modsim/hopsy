"""Facade for hopsy's vendored PolyRound backends.

The public API mirrors the external PolyRound package: PolyRound calls accept
and return :class:`hopsy._polyround.polytope.Polytope` objects. Backend choice is
handled here through ``PolyRoundSettings.backend``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .polytope import Polytope
from .settings import PolyRoundSettings, fix_backend_name

_SUPPORTED_BACKENDS = frozenset({"gurobi", "gurobi-cupy", "highs"})


def backend_name(settings: Any | None = None) -> str:
    """Return the configured PolyRound backend name."""

    name = getattr(settings, "backend", None) if settings is not None else None
    return fix_backend_name(name)


def active_backend(settings: Any | None = None):
    """Return the backend adapter selected by ``settings.backend``."""

    return _backend_for(settings)


class PolyRoundApi:
    """Main PolyRound API. Route the calls to the right backend."""

    @staticmethod
    def backend_name(settings: Any | None = None) -> str:
        return backend_name(settings)

    @staticmethod
    def simplify_polytope(
        polytope: Polytope,
        settings: Any | None = None,
        normalize: bool = True,
    ) -> Polytope:
        settings = _settings_or_default(settings)
        polytope = _require_polytope(polytope)
        return _backend_for(settings).simplify_polytope(
            polytope,
            settings=settings,
            normalize=normalize,
        )

    @staticmethod
    def transform_polytope(
        polytope: Polytope,
        settings: Any | None = None,
    ) -> Polytope:
        settings = _settings_or_default(settings)
        polytope = _require_polytope(polytope)
        return _backend_for(settings).transform_polytope(
            polytope,
            settings=settings,
        )

    @staticmethod
    def round_polytope(
        polytope: Polytope,
        settings: Any | None = None,
    ) -> Polytope:
        settings = _settings_or_default(settings)
        polytope = _require_polytope(polytope)
        return _backend_for(settings).round_polytope(
            polytope,
            settings=settings,
        )

    @staticmethod
    def simplify_transform_and_round(
        polytope: Polytope,
        settings: Any | None = None,
    ) -> Polytope:
        settings = _settings_or_default(settings)
        polytope = _require_polytope(polytope)
        return _backend_for(settings).simplify_transform_and_round(
            polytope,
            settings=settings,
        )

    @staticmethod
    def cobra_model_to_polytope(model, settings: Any | None = None) -> Polytope:
        settings = _settings_or_default(settings)
        return _backend_for(settings).cobra_model_to_polytope(model)

    @staticmethod
    def polytope_to_csvs(
        polytope: Polytope,
        dirname: str,
        settings: Any | None = None,
    ) -> None:
        settings = _settings_or_default(settings)
        polytope = _require_polytope(polytope)
        return _backend_for(settings).polytope_to_csvs(polytope, dirname)

    @staticmethod
    def sbml_to_polytope(
        file,
        settings: Any | None = None,
        inf_bound=1e5,
        prescale=False,
    ) -> Polytope:
        settings = _settings_or_default(settings)
        backend = _backend_for(settings)
        if hasattr(backend, "sbml_to_polytope"):
            return backend.sbml_to_polytope(
                file,
                settings=settings,
                inf_bound=inf_bound,
                prescale=prescale,
            )
        return backend.parse(
            file,
            settings,
            inf_bound=inf_bound,
            prescale=prescale,
        )

    @staticmethod
    def chebyshev_center(polytope: Polytope, settings: Any | None = None):
        settings = _settings_or_default(settings)
        polytope = _require_polytope(polytope)
        center, radius = _backend_for(settings).chebyshev_center(
            polytope,
            settings,
        )
        center = np.asarray(center).reshape((-1, 1))
        radius = float(np.asarray(radius).reshape(-1)[0])
        if not np.isfinite(center).all() or not np.isfinite(radius):
            raise ValueError(
                "Chebyshev center computation returned non-finite values. "
                "Check polytope feasibility or LP solver settings."
            )
        return center, np.array([radius])

    @staticmethod
    def iterative_solve(polytope: Polytope, settings: Any | None = None):
        settings = _settings_or_default(settings)
        polytope = _require_polytope(polytope)
        return _backend_for(settings).iterative_solve(polytope, settings)

    @staticmethod
    def polytope_to_model(polytope: Polytope, settings: Any | None = None):
        settings = _settings_or_default(settings)
        polytope = _require_polytope(polytope)
        model = _backend_for(settings).polytope_to_model(polytope, settings)
        return _StatusReturningModel(model)


class _StatusReturningModel:
    def __init__(self, model):
        self._model = model

    # See src/hopsy/misc.py:719
    def optimize(self):
        self._model.optimize()
        return self._model.status

    @property
    def status(self):
        return self._model.status

    def __getattr__(self, name):
        return getattr(self._model, name)


def _settings_or_default(settings: Any | None):
    if settings is None:
        return PolyRoundSettings()
    return settings


def _require_polytope(polytope: Polytope) -> Polytope:
    if not isinstance(polytope, Polytope):
        raise TypeError("PolyRound API expects a Polytope instance.")
    return polytope


def _checked_backend_name(settings):
    name = backend_name(settings)
    if name not in _SUPPORTED_BACKENDS:
        _raise_unknown_backend(name)
    return name


def _backend_for(settings):
    name = _checked_backend_name(settings)
    if name == "gurobi":
        return _gurobi_backend()
    if name == "highs":
        return _highs_backend()
    if name == "gurobi-cupy":
        return _gurobi_cupy_backend()
    if name == "exp2":
        return _exp2_backend()
    if name == "glpk":
        return _glpk_backend()
    _raise_unknown_backend(name)


def _gurobi_backend():
    try:
        from .gurobi.backend import GurobiBackend

        return GurobiBackend()
    except Exception as error:
        raise RuntimeError("Could not initialize PolyRound backend 'gurobi'") from error


def _gurobi_cupy_backend():
    try:
        from .gurobi_cupy.backend import GurobiCuPyBackend

        return GurobiCuPyBackend()
    except Exception as error:
        raise RuntimeError(
            "Could not initialize PolyRound backend 'gurobi-cupy'"
        ) from error


def _exp2_backend():
    try:
        from .exp2.backend import Exp2Backend

        return Exp2Backend()
    except Exception as error:
        raise RuntimeError("Could not initialize PolyRound backend 'exp2'") from error


def _glpk_backend():
    try:
        from .glpk.backend import GlpkBackend

        return GlpkBackend()
    except Exception as error:
        raise RuntimeError("Could not initialize PolyRound backend 'glpk'") from error


def _highs_backend():
    try:
        from .highs.backend import HiGHSBackend

        return HiGHSBackend()
    except Exception as error:
        raise RuntimeError("Could not initialize PolyRound backend 'highs'") from error


def _raise_unknown_backend(name: str):
    valid = ", ".join(sorted(_SUPPORTED_BACKENDS))
    raise ValueError(
        f"Unknown PolyRound backend {name!r}. Available backends: {valid}."
    )
