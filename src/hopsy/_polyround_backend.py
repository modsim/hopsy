"""Thin hopsy adapter around the vendored Gurobi rounding implementation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._polyround.api import PolyRoundApi
from ._polyround.mutable_classes.polytope import Polytope
from ._polyround.settings import PolyRoundSettings
from ._polyround.static_classes.lp_interfacing import GurobiInterfacer
from ._polyround.static_classes.lp_utils import ChebyshevFinder
from ._polyround.static_classes.rounding.maximum_volume_ellipsoid import (
    MaximumVolumeEllipsoidFinder,
)


@dataclass(frozen=True)
class PolyRoundResult:
    A: np.ndarray
    b: np.ndarray
    transformation: np.ndarray
    shift: np.ndarray
    transformed: bool = True
    S: np.ndarray | None = None
    h: np.ndarray | None = None


def default_settings() -> PolyRoundSettings:
    return PolyRoundSettings()


def make_polytope(A, b, S=None, h=None) -> Polytope:
    return Polytope(A=A, b=b, S=S, h=h)


def _ensure_identity_transform(polytope: Polytope) -> None:
    dim = polytope.A.shape[1]
    polytope.transformation = pd.DataFrame(np.eye(dim))
    polytope.transformation.index = [str(i) for i in range(dim)]
    polytope.transformation.columns = [str(i) for i in range(dim)]
    polytope.shift = pd.Series(np.zeros(dim))


def _as_result(polytope: Polytope, transformed: bool = True) -> PolyRoundResult:
    return PolyRoundResult(
        A=polytope.A.values,
        b=polytope.b.values,
        transformation=polytope.transformation.values,
        shift=polytope.shift.values,
        transformed=transformed,
        S=None if polytope.S is None else polytope.S.values,
        h=None if polytope.h is None else polytope.h.values,
    )


def simplify(A, b, settings: PolyRoundSettings, S=None, h=None) -> PolyRoundResult:
    polytope = make_polytope(A=A, b=b, S=S, h=h)
    polytope = PolyRoundApi.simplify_polytope(polytope, settings=settings)
    return _as_result(polytope, transformed=False)


def transform(A, b, settings: PolyRoundSettings, S=None, h=None) -> PolyRoundResult:
    polytope = make_polytope(A=A, b=b, S=S, h=h)
    if polytope.S is not None:
        polytope = PolyRoundApi.transform_polytope(polytope, settings=settings)
        return _as_result(polytope, transformed=True)

    _ensure_identity_transform(polytope)
    return _as_result(polytope, transformed=False)


def round_polytope(
    A,
    b,
    settings: PolyRoundSettings,
    simplify_first: bool = True,
) -> PolyRoundResult:
    polytope = make_polytope(A, b)

    if simplify_first:
        simplified = simplify(polytope.A.values, polytope.b.values, settings)
        polytope = make_polytope(
            A=simplified.A,
            b=simplified.b,
            S=simplified.S,
            h=simplified.h,
        )

    transformed = transform(
        polytope.A.values,
        polytope.b.values,
        settings,
        S=None if polytope.S is None else polytope.S.values,
        h=None if polytope.h is None else polytope.h.values,
    )
    polytope = make_polytope(transformed.A, transformed.b)
    polytope.transformation = pd.DataFrame(transformed.transformation)
    polytope.shift = pd.Series(transformed.shift)

    polytope = PolyRoundApi.round_polytope(polytope, settings=settings)
    return _as_result(polytope)


def chebyshev_center(A, b, settings: PolyRoundSettings):
    polytope = make_polytope(A, b)
    center, radius = ChebyshevFinder.chebyshev_center(polytope, settings)
    return center.flatten(), float(radius[0])


def is_empty(A, b, settings: PolyRoundSettings, S=None, h=None) -> bool:
    polytope = make_polytope(A=A, b=b, S=S, h=h)
    model = GurobiInterfacer.polytope_to_model(polytope, settings)
    model.optimize()
    return model.status != "optimal"


def sqrt_maximum_volume_ellipsoid(A, b, settings: PolyRoundSettings) -> np.ndarray:
    simplified = simplify(A, b, settings)
    transformed = transform(
        simplified.A,
        simplified.b,
        settings,
        S=simplified.S,
        h=simplified.h,
    )
    polytope = make_polytope(transformed.A, transformed.b)
    polytope.transformation = pd.DataFrame(transformed.transformation)
    polytope.shift = pd.Series(transformed.shift)

    MaximumVolumeEllipsoidFinder.iterative_solve(polytope, settings)
    return polytope.transformation.values
