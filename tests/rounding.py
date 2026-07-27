import importlib
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

import hopsy
import hopsy._rounding as rounding_module
from hopsy._rounding import (
    DEFAULT_BACKEND,
    Polytope,
    RoundingApi,
    RoundingSettings,
    active_backend,
    backend_name,
)
from hopsy._rounding.default_settings import default_numerics_threshold
from hopsy._rounding.gurobi.constraint_removal_reduction import (
    null_space as gurobi_null_space,
)
from hopsy._rounding.gurobi.geometric_mean_scaling import (
    geometric_mean_scaling as gurobi_geometric_mean_scaling,
)
from hopsy._rounding.gurobi.lp_interfacing import Interfacer as GurobiInterfacer
from hopsy._rounding.gurobi.maximum_volume_ellipsoid import run_mve as gurobi_run_mve
from hopsy._rounding.gurobi.nearest_symmetric_positive_definite import (
    get_NSPD as gurobi_get_nspd,
)
from hopsy._rounding.gurobi_cupy.constraint_removal_reduction import (
    null_space as gurobi_cupy_null_space,
)
from hopsy._rounding.highs.constraint_removal_reduction import (
    null_space as highs_null_space,
)
from hopsy._rounding.highs.geometric_mean_scaling import (
    geometric_mean_scaling as highs_geometric_mean_scaling,
)
from hopsy._rounding.highs.lp_interfacing import Interfacer as HiGHSInterfacer
from hopsy._rounding.highs.maximum_volume_ellipsoid import run_mve as highs_run_mve
from hopsy._rounding.highs.nearest_symmetric_positive_definite import (
    get_NSPD as highs_get_nspd,
)
from hopsy._rounding.settings import DEFAULT_HP_FLAGS, fix_backend_name

try:
    GurobiInterfacer.require_package()
except ImportError:
    HAS_GUROBI = False
else:
    HAS_GUROBI = True

CPU_MATH_BACKENDS = (
    (
        "gurobi",
        gurobi_geometric_mean_scaling,
        gurobi_get_nspd,
        gurobi_run_mve,
    ),
    (
        "highs",
        highs_geometric_mean_scaling,
        highs_get_nspd,
        highs_run_mve,
    ),
)

NULL_SPACE_BACKENDS = (
    ("gurobi", gurobi_null_space),
    ("highs", highs_null_space),
    ("gurobi-cupy", gurobi_cupy_null_space),
)


def solver_backends():
    """Return installed CPU LP backends, keeping required HiGHS first."""
    backends = [("highs", HiGHSInterfacer)]
    if HAS_GUROBI:
        backends.append(("gurobi", GurobiInterfacer))
    return backends


def load_gpu_math_backend(test_case):
    """Load CuPy rounding internals only when this test runs on a GPU node."""
    try:
        cupy = importlib.import_module("cupy")
    except ImportError as error:
        test_case.skipTest(f"requires the optional CuPy dependency: {error}")

    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except Exception as error:
        test_case.skipTest(f"requires an available CUDA device: {error}")
    if device_count < 1:
        test_case.skipTest("requires an available CUDA device")

    geometric_mean_scaling_module = importlib.import_module(
        "hopsy._rounding.gurobi_cupy.geometric_mean_scaling"
    )
    maximum_volume_ellipsoid_module = importlib.import_module(
        "hopsy._rounding.gurobi_cupy.maximum_volume_ellipsoid"
    )
    nspd_module = importlib.import_module(
        "hopsy._rounding.gurobi_cupy.nearest_symmetric_positive_definite"
    )
    return (
        cupy,
        geometric_mean_scaling_module.geometric_mean_scaling,
        nspd_module.get_NSPD,
        maximum_volume_ellipsoid_module.run_mve,
    )


# Helpers: polytope builders


def box_polytope(n, lo=-1.0, hi=1.0):
    """Standard 2n-row hypercube box [lo, hi]^n."""
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.concatenate([np.full(n, hi), np.full(n, -lo)])
    return Polytope(A, b)


def add_scaled_duplicate_rows(polytope, n_extra, scale=2.0, seed=0):
    """
    Append n_extra rows that are provably redundant: copies of existing rows
    with b multiplied by scale (scale > 1 > looser bound > redundant).
    Returns (new_polytope, n_original_rows).
    """
    rng = np.random.default_rng(seed)
    A = polytope.A.values
    b = polytope.b.values
    idx = rng.integers(0, A.shape[0], size=n_extra)
    A_extra = A[idx]
    b_extra = b[idx] * scale
    return Polytope(np.vstack([A, A_extra]), np.concatenate([b, b_extra])), A.shape[0]


def add_interior_halfspace_rows(polytope, n_extra, margin=1.0, seed=0):
    """
    Append n_extra rows that are provably redundant: random halfspaces whose
    bound is strictly larger than the support function of the polytope's box hull.
    Works when the polytope contains the origin (e.g. a centred box).
    """
    rng = np.random.default_rng(seed)
    n = polytope.A.shape[1]
    A = polytope.A.values
    b = polytope.b.values
    # Random unit-norm directions
    a_extra = rng.standard_normal((n_extra, n))
    a_extra /= np.linalg.norm(a_extra, axis=1, keepdims=True)
    # Upper bound on support function over the box [-1,1]^n: L1 norm of row
    b_extra = np.abs(a_extra).sum(axis=1) + margin
    return Polytope(np.vstack([A, a_extra]), np.concatenate([b, b_extra])), A.shape[0]


# Helpers: invariants


def canonical_rows(A, b, decimals=8):
    """
    Normalize each row by its L2 norm and sort lexicographically.
    Used to compare constraint sets as unordered collections.
    """
    norms = np.linalg.norm(A, axis=1)
    norms[norms == 0] = 1.0
    A_n = A / norms[:, None]
    b_n = b / norms
    rows = np.column_stack([A_n, b_n])
    order = np.lexsort(np.round(rows, decimals).T[::-1])
    return rows[order]


def all_rows_subset(A_sub, b_sub, A_full, b_full, atol=1e-6):
    """Return True if every row of (A_sub, b_sub) appears in (A_full, b_full)."""
    can_sub = canonical_rows(A_sub, b_sub)
    can_full = canonical_rows(A_full, b_full)
    for row in can_sub:
        diffs = np.abs(can_full - row).max(axis=1)
        if not np.any(diffs < atol):
            return False
    return True


def is_feasible(A, b, x, atol=1e-8):
    """Check Ax <= b + atol for all rows."""
    return bool(np.all(A @ x <= b + atol))


def origin_is_interior(polytope, atol=0.0):
    """Return True if the origin is strictly inside the polytope."""
    x = np.zeros(polytope.A.shape[1])
    return bool(polytope.border_distance(x) > atol)


def absolute_mve_center(center, initial_point, message):
    """Convert the low-level MVE solver's iteration-relative center."""
    center = np.asarray(center).reshape(-1)
    if message == 0:
        return center + np.asarray(initial_point).reshape(-1)
    return center


class RoundingTests(unittest.TestCase):
    """Tests for internal rounding behavior and public hopsy wiring."""

    def setUp(self):
        self._saved_lp_settings = hopsy.LP().settings
        hopsy.LP().reset()

    def tearDown(self):
        hopsy.LP().settings = self._saved_lp_settings

    def _settings(self, simplify_only=False):
        return RoundingSettings(simplify_only=simplify_only)

    # Facade and settings contracts

    def test_rounding_facade_exports_public_api(self):
        expected_exports = {
            "DEFAULT_BACKEND",
            "Polytope",
            "RoundingApi",
            "RoundingSettings",
            "active_backend",
            "backend_name",
            "polytope",
        }

        self.assertEqual(set(rounding_module.__all__), expected_exports)
        self.assertIs(rounding_module.RoundingApi, RoundingApi)
        self.assertIs(rounding_module.RoundingSettings, RoundingSettings)
        self.assertIs(rounding_module.Polytope, Polytope)

    def test_backend_names_are_normalized_and_default_to_highs(self):
        self.assertEqual(DEFAULT_BACKEND, "highs")
        self.assertEqual(fix_backend_name(None), DEFAULT_BACKEND)
        self.assertEqual(fix_backend_name(""), DEFAULT_BACKEND)
        self.assertEqual(fix_backend_name("  GuRoBi-CuPy  "), "gurobi-cupy")
        self.assertEqual(backend_name(), DEFAULT_BACKEND)
        self.assertEqual(
            backend_name(SimpleNamespace(backend="  GuRoBi  ")),
            "gurobi",
        )
        self.assertEqual(
            RoundingApi.backend_name(RoundingSettings(backend="HiGHS")),
            "highs",
        )

    def test_settings_have_independent_mutable_defaults(self):
        first = RoundingSettings()
        second = RoundingSettings()

        self.assertEqual(first.hp_flags, DEFAULT_HP_FLAGS)
        self.assertEqual(second.hp_flags, DEFAULT_HP_FLAGS)
        self.assertIsNot(first.hp_flags, second.hp_flags)

        first.hp_flags["custom_option"] = 42
        self.assertNotIn("custom_option", second.hp_flags)
        self.assertNotIn("custom_option", DEFAULT_HP_FLAGS)

    def test_lp_reset_restores_fresh_default_settings(self):
        old_settings = hopsy.LP().settings
        old_settings.backend = "gurobi"
        old_settings.hp_flags["custom_option"] = 42

        hopsy.LP().reset()

        self.assertIsNot(hopsy.LP().settings, old_settings)
        self.assertEqual(hopsy.LP().settings, RoundingSettings())
        self.assertNotIn("custom_option", hopsy.LP().settings.hp_flags)

    def test_active_backend_returns_requested_adapter(self):
        for name in ("highs", "gurobi"):
            with self.subTest(backend=name):
                backend = active_backend(RoundingSettings(backend=name))
                self.assertEqual(backend.name, name)

    def test_unknown_backend_reports_all_supported_choices(self):
        settings = RoundingSettings(backend="not-a-solver")

        with self.assertRaisesRegex(
            ValueError,
            (
                r"Unknown rounding backend 'not-a-solver'.*"
                r"gurobi, gurobi-cupy, highs"
            ),
        ):
            RoundingApi.round_polytope(box_polytope(2), settings)

    def test_core_api_rejects_non_polytope_before_backend_selection(self):
        calls = {
            "simplify": lambda: RoundingApi.simplify_polytope(object()),
            "transform": lambda: RoundingApi.transform_polytope(object()),
            "round": lambda: RoundingApi.round_polytope(object()),
            "pipeline": lambda: RoundingApi.simplify_transform_and_round(object()),
            "chebyshev": lambda: RoundingApi.chebyshev_center(object()),
            "iterative_solve": lambda: RoundingApi.iterative_solve(object()),
            "polytope_to_model": lambda: RoundingApi.polytope_to_model(object()),
        }

        with mock.patch("hopsy._rounding.api._backend_for") as select_backend:
            for name, call in calls.items():
                with self.subTest(method=name):
                    with self.assertRaisesRegex(
                        TypeError,
                        "Rounding API expects a Polytope instance",
                    ):
                        call()
            select_backend.assert_not_called()

    def test_api_dispatch_forwards_settings_and_keyword_arguments(self):
        backend = mock.Mock()
        polytope = box_polytope(2)
        settings = RoundingSettings(backend="highs")
        expected = object()
        backend.simplify_polytope.return_value = expected

        with mock.patch(
            "hopsy._rounding.api._backend_for",
            return_value=backend,
        ) as select_backend:
            actual = RoundingApi.simplify_polytope(
                polytope,
                settings,
                normalize=False,
            )

        self.assertIs(actual, expected)
        select_backend.assert_called_once_with(settings)
        backend.simplify_polytope.assert_called_once_with(
            polytope,
            settings=settings,
            normalize=False,
        )

    def test_api_creates_a_fresh_default_settings_object_per_call(self):
        backend = mock.Mock()
        backend.round_polytope.side_effect = lambda polytope, settings: polytope
        selected_settings = []

        def select_backend(settings):
            selected_settings.append(settings)
            return backend

        with mock.patch(
            "hopsy._rounding.api._backend_for",
            side_effect=select_backend,
        ):
            RoundingApi.round_polytope(box_polytope(1))
            RoundingApi.round_polytope(box_polytope(1))

        self.assertEqual(len(selected_settings), 2)
        self.assertTrue(
            all(
                isinstance(settings, RoundingSettings) for settings in selected_settings
            )
        )
        self.assertIsNot(selected_settings[0], selected_settings[1])
        self.assertIsNot(
            selected_settings[0].hp_flags,
            selected_settings[1].hp_flags,
        )

    def test_chebyshev_api_canonicalizes_backend_output_shapes(self):
        backend = mock.Mock()
        backend.chebyshev_center.return_value = (
            np.array([1.0, -2.0]),
            np.array([[3.0]]),
        )

        with mock.patch(
            "hopsy._rounding.api._backend_for",
            return_value=backend,
        ):
            center, radius = RoundingApi.chebyshev_center(box_polytope(2))

        self.assertEqual(center.shape, (2, 1))
        self.assertEqual(radius.shape, (1,))
        np.testing.assert_array_equal(center[:, 0], np.array([1.0, -2.0]))
        np.testing.assert_array_equal(radius, np.array([3.0]))

    def test_chebyshev_api_rejects_nonfinite_backend_results(self):
        backend = mock.Mock()
        invalid_results = (
            (np.array([np.nan, 0.0]), 1.0),
            (np.array([0.0, np.inf]), 1.0),
            (np.array([0.0, 0.0]), np.inf),
            (np.array([0.0, 0.0]), np.nan),
        )

        with mock.patch(
            "hopsy._rounding.api._backend_for",
            return_value=backend,
        ):
            for center, radius in invalid_results:
                with self.subTest(center=center, radius=radius):
                    backend.chebyshev_center.return_value = (center, radius)
                    with self.assertRaisesRegex(
                        ValueError,
                        "returned non-finite values",
                    ):
                        RoundingApi.chebyshev_center(box_polytope(2))

    def test_polytope_to_model_wraps_optimize_status_and_delegates_attributes(self):
        class NativeModel:
            marker = "native attribute"

            def __init__(self):
                self.status = "loaded"

            def optimize(self):
                self.status = "optimal"

        backend = mock.Mock()
        native_model = NativeModel()
        backend.polytope_to_model.return_value = native_model

        with mock.patch(
            "hopsy._rounding.api._backend_for",
            return_value=backend,
        ):
            model = RoundingApi.polytope_to_model(box_polytope(2))

        self.assertEqual(model.status, "loaded")
        self.assertEqual(model.marker, "native attribute")
        self.assertEqual(model.optimize(), "optimal")
        self.assertEqual(model.status, "optimal")

    # Polytope algebra and defensive assertions

    def test_polytope_removes_consistent_zero_rows(self):
        polytope = Polytope(
            A=np.array([[0.0, 0.0], [1.0, -1.0]]),
            b=np.array([0.0, 2.0]),
            S=np.array([[0.0, 0.0], [1.0, 1.0]]),
            h=np.array([0.0, 0.5]),
        )

        self.assertEqual(polytope.A.shape, (1, 2))
        self.assertEqual(polytope.b.shape, (1,))
        self.assertEqual(polytope.S.shape, (1, 2))
        self.assertEqual(polytope.h.shape, (1,))
        np.testing.assert_array_equal(polytope.A.values, [[1.0, -1.0]])
        np.testing.assert_array_equal(polytope.S.values, [[1.0, 1.0]])

    def test_polytope_rejects_inconsistent_zero_rows_and_orphan_rhs(self):
        with self.assertRaises(AssertionError):
            Polytope(np.zeros((1, 2)), np.array([1.0]))
        with self.assertRaises(ValueError):
            Polytope(np.eye(2), np.ones(2), h=np.zeros(1))

    def test_polytope_normalization_preserves_halfspaces(self):
        A = np.array([[3.0, 4.0], [-12.0, 5.0], [0.0, -2.0]])
        b = np.array([10.0, 26.0, 4.0])
        row_norms = np.linalg.norm(A, axis=1)
        polytope = Polytope(A, b)

        polytope.normalize()

        np.testing.assert_allclose(
            np.linalg.norm(polytope.A.values, axis=1),
            np.ones(A.shape[0]),
        )
        np.testing.assert_allclose(polytope.A.values, A / row_norms[:, None])
        np.testing.assert_allclose(polytope.b.values, b / row_norms)

        points = np.array([[0.0, 0.0], [2.0, -1.0], [-3.0, 4.0]])
        original_feasibility = points @ A.T <= b
        normalized_feasibility = points @ polytope.A.values.T <= polytope.b.values
        np.testing.assert_array_equal(
            normalized_feasibility,
            original_feasibility,
        )

    def test_polytope_copy_is_deep(self):
        original = box_polytope(2)
        copied = original.copy()

        copied.A.iloc[0, 0] = 99.0
        copied.b.iloc[0] = 99.0
        copied.shift.iloc[0] = 99.0
        copied.transformation.iloc[0, 0] = 99.0

        np.testing.assert_array_equal(
            original.A.values,
            np.vstack([np.eye(2), -np.eye(2)]),
        )
        np.testing.assert_array_equal(original.b.values, np.ones(4))
        np.testing.assert_array_equal(original.shift.values, np.zeros(2))
        np.testing.assert_array_equal(original.transformation.values, np.eye(2))

    # Redundancy: scaled duplicates

    def test_simplify_box_no_redundant_rows_unchanged(self):
        """Simplifying a plain box should not remove any constraints."""
        polytope = box_polytope(3)
        n_original = polytope.A.shape[0]
        result = RoundingApi.simplify_polytope(polytope, self._settings())
        self.assertLessEqual(result.A.shape[0], n_original)
        # All 6 box rows should survive (none are redundant in a tight box)
        self.assertEqual(result.A.shape[0], n_original)

    def test_simplify_removes_scaled_duplicates_3d(self):
        """Box 3D + 10 redundant rows: simplified result should have exactly 6 rows."""
        polytope, n_base = add_scaled_duplicate_rows(box_polytope(3), n_extra=10)
        result = RoundingApi.simplify_polytope(polytope, self._settings())
        self.assertEqual(
            result.A.shape[0],
            n_base,
            f"Expected {n_base} rows, got {result.A.shape[0]}",
        )

    def test_simplify_removes_scaled_duplicates_10d(self):
        """Box 10D + 60 redundant rows."""
        polytope, n_base = add_scaled_duplicate_rows(box_polytope(10), n_extra=60)
        result = RoundingApi.simplify_polytope(polytope, self._settings())
        self.assertEqual(result.A.shape[0], n_base)

    def test_simplify_result_rows_are_subset_of_original(self):
        """No new rows should appear — simplified rows must be a subset of input rows."""
        polytope, _ = add_scaled_duplicate_rows(box_polytope(5), n_extra=20)
        result = RoundingApi.simplify_polytope(polytope, self._settings())
        self.assertTrue(
            all_rows_subset(
                result.A.values, result.b.values, polytope.A.values, polytope.b.values
            ),
            "Simplified rows are not a subset of original rows",
        )

    # Redundancy: interior halfspaces

    def test_simplify_removes_interior_halfspaces(self):
        """Halfspaces that don't touch the box are all redundant."""
        polytope, n_base = add_interior_halfspace_rows(box_polytope(5), n_extra=30)
        result = RoundingApi.simplify_polytope(polytope, self._settings())
        self.assertLessEqual(result.A.shape[0], n_base)

    # Invariant: feasibility of a known interior point

    def test_simplify_preserves_interior_point(self):
        """The origin (interior of box) must remain feasible after simplify."""
        polytope, _ = add_scaled_duplicate_rows(box_polytope(5), n_extra=30)
        result = RoundingApi.simplify_polytope(polytope, self._settings())
        x0 = np.zeros(result.A.shape[1])
        self.assertTrue(
            is_feasible(result.A.values, result.b.values, x0),
            "Origin is no longer feasible after simplify",
        )

    # Idempotency

    def test_simplify_is_idempotent(self):
        """Simplifying twice should not change the row count."""
        polytope, _ = add_scaled_duplicate_rows(box_polytope(5), n_extra=20)
        once = RoundingApi.simplify_polytope(polytope, self._settings())
        twice = RoundingApi.simplify_polytope(once.copy(), self._settings())
        self.assertEqual(once.A.shape[0], twice.A.shape[0])

    # simplify_only flag

    def test_simplify_only_does_not_refunction_narrow_constraints(self):
        """With simplify_only=True, thin directions are not turned into equalities."""
        # 2D box, then collapse one dimension to near-zero width
        A = np.vstack([np.eye(2), -np.eye(2)])
        b = np.array([1.0, 1e-8, 1.0, 1e-8])  # x1 in [-1, 1], x2 in [-1e-8, 1e-8]
        polytope = Polytope(A, b)
        result = RoundingApi.simplify_polytope(
            polytope, RoundingSettings(simplify_only=True)
        )
        self.assertIsNone(
            result.S, "simplify_only should not produce equality constraints"
        )

    def _equality_box(self):
        """Box in R^3 with one equality constraint x2 = 0.5, so dim reduces to 2."""
        A = np.vstack([np.eye(3), -np.eye(3)])
        b = np.ones(6)
        S = np.array([[0.0, 1.0, 0.0]])
        h = np.array([0.5])
        return Polytope(A, b, S, h)

    def test_transform_eliminates_equality_constraints(self):
        """After transform, the polytope should be inequality-only (S is None)."""
        polytope = self._equality_box()
        result = RoundingApi.transform_polytope(polytope)
        self.assertTrue(result.inequality_only)
        self.assertIsNone(result.S)

    def test_transform_reduces_dimension(self):
        """With one equality constraint, ambient dimension drops by 1."""
        polytope = self._equality_box()
        original_dim = polytope.A.shape[1]
        result = RoundingApi.transform_polytope(polytope)
        self.assertEqual(result.A.shape[1], original_dim - 1)

    def test_transform_backtransform_roundtrip(self):
        """Back-transforming a point in the reduced space should give a point
        feasible in the original polytope."""
        polytope = self._equality_box()
        result = RoundingApi.transform_polytope(polytope)
        reduced_dim = result.A.shape[1]
        x_reduced = np.zeros(reduced_dim)
        x_original = result.back_transform(x_reduced)
        self.assertTrue(
            is_feasible(polytope.A.values, polytope.b.values, x_original),
            "Back-transformed origin is not feasible in the original polytope",
        )

    def _roundable_polytope(self, n=4):
        """A box already transformed to be inequality-only (no equalities)."""
        return box_polytope(n)

    def test_round_origin_in_interior(self):
        """After rounding, the origin must be strictly inside: b.min() > 0."""
        polytope = self._roundable_polytope()
        result = RoundingApi.round_polytope(polytope)
        self.assertGreater(float(result.b.min()), 0.0)

    def test_round_transformation_full_rank(self):
        """The rounding transformation must be square and full-rank."""
        polytope = self._roundable_polytope(n=4)
        result = RoundingApi.round_polytope(polytope)
        T = result.transformation.values
        self.assertEqual(T.shape[0], T.shape[1])
        rank = np.linalg.matrix_rank(T)
        self.assertEqual(rank, T.shape[0])

    def test_round_backtransform_feasible(self):
        """The origin in rounded space should back-transform to a feasible point
        in the original polytope."""
        polytope = self._roundable_polytope(n=4)
        original_A = polytope.A.values.copy()
        original_b = polytope.b.values.copy()
        result = RoundingApi.round_polytope(polytope)
        x_original = result.back_transform(np.zeros(result.A.shape[1]))
        self.assertTrue(is_feasible(original_A, original_b, x_original))

    def test_rounding_constraints_match_affine_back_transform(self):
        A = np.vstack([np.eye(2), -np.eye(2)])
        b = np.array([4.0, 6.0, 2.0, 0.0])
        polytope = Polytope(A, b)

        rounded = RoundingApi.round_polytope(polytope)
        rng = np.random.default_rng(123)
        rounded_points = rng.normal(size=(16, 2))

        for rounded_point in rounded_points:
            original_point = rounded.back_transform(rounded_point)
            original_slacks = b - A @ original_point
            rounded_slacks = rounded.b.values - rounded.A.values @ rounded_point
            np.testing.assert_allclose(
                rounded_slacks,
                original_slacks,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_rounding_rejects_infinite_constraint_data_for_cpu_backends(self):
        finite_A = np.vstack([np.eye(2), -np.eye(2)])
        finite_b = np.ones(4)
        invalid_inputs = (
            (finite_A.copy(), np.array([np.inf, 1.0, 1.0, 1.0])),
            (
                np.array([[np.inf, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]),
                finite_b.copy(),
            ),
        )

        for backend_name_, _ in solver_backends():
            for A, b in invalid_inputs:
                with self.subTest(backend=backend_name_, invalid=A):
                    with self.assertRaisesRegex(
                        ValueError,
                        "contains inf",
                    ):
                        RoundingApi.round_polytope(
                            Polytope(A, b),
                            RoundingSettings(backend=backend_name_),
                        )

    def test_transform_rejects_inequality_only_polytope(self):
        for backend_name_, _ in solver_backends():
            with self.subTest(backend=backend_name_):
                with self.assertRaisesRegex(
                    ValueError,
                    "only contains inequality constraints",
                ):
                    RoundingApi.transform_polytope(
                        box_polytope(2),
                        RoundingSettings(backend=backend_name_),
                    )

    def test_full_pipeline_box_with_equality(self):
        """End-to-end simplify_transform_and_round on a box with one equality."""
        A = np.vstack([np.eye(4), -np.eye(4)])
        b = np.ones(8)
        S = np.array([[1.0, 1.0, 0.0, 0.0]])
        h = np.array([0.0])
        polytope = Polytope(A, b, S, h)
        result = RoundingApi.simplify_transform_and_round(polytope)
        self.assertTrue(result.inequality_only)
        self.assertGreater(float(result.b.min()), 0.0)

    def test_full_pipeline_box_with_redundant_rows(self):
        """Pipeline should handle redundant rows without crashing."""
        polytope, _ = add_scaled_duplicate_rows(box_polytope(5), n_extra=30)
        result = RoundingApi.simplify_transform_and_round(polytope)
        self.assertTrue(result.inequality_only)
        self.assertGreater(float(result.b.min()), 0.0)

    def test_exact_small_simplex_rounding(self):
        dim = 2
        A = np.vstack([-np.eye(dim), np.ones(shape=(1, dim))])
        b = np.zeros(dim + 1)
        b[-1] = 1
        expected_center = np.full(dim, 1.0 / (dim + 1))
        expected_shape = (np.eye(dim) - np.ones((dim, dim)) / (dim + 1)) / (
            dim * (dim + 1)
        )

        for backend_name_, _ in solver_backends():
            with self.subTest(backend=backend_name_):
                rounded = RoundingApi.round_polytope(
                    Polytope(A, b),
                    RoundingSettings(backend=backend_name_),
                )
                transformation = rounded.transformation.values

                np.testing.assert_allclose(
                    rounded.shift.values,
                    expected_center,
                    atol=1e-7,
                )
                np.testing.assert_allclose(
                    transformation @ transformation.T,
                    expected_shape,
                    atol=1e-7,
                )
                np.testing.assert_allclose(
                    rounded.A.values,
                    A @ transformation,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    rounded.b.values,
                    b - A @ rounded.shift.values,
                    atol=1e-12,
                )

    def test_gurobi_cupy_rounding_matches_known_simplex_geometry(self):
        load_gpu_math_backend(self)
        if not HAS_GUROBI:
            self.skipTest("requires the optional gurobipy dependency")

        dimension = 2
        A = np.vstack([-np.eye(dimension), np.ones((1, dimension))])
        b = np.array([0.0, 0.0, 1.0])
        expected_center = np.full(dimension, 1.0 / (dimension + 1))
        expected_shape = (
            np.eye(dimension) - np.ones((dimension, dimension)) / (dimension + 1)
        ) / (dimension * (dimension + 1))

        rounded = RoundingApi.round_polytope(
            Polytope(A, b),
            RoundingSettings(backend="gurobi-cupy"),
        )

        np.testing.assert_allclose(
            rounded.shift.values,
            expected_center,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            rounded.transformation.values @ rounded.transformation.values.T,
            expected_shape,
            atol=1e-7,
        )

    def test_exact_larger_example_rounding(self):
        A = np.array(
            [
                -0.890496033275099,
                -1.00806441730899,
                0.139061858656017,
                0.944284824573101,
                -0.236144297158048,
                -2.42395713384503,
                -0.0754591290328577,
                -0.223831428498817,
                -0.358571912766115,
                0.0580698827354712,
                -2.07763485529806,
                -0.424614015056491,
                -0.143545710236981,
                -0.202917945340724,
                1.39334147492104,
                -1.51307697899823,
                0.651804091657409,
                -1.12635186101317,
                -0.377133557739639,
                -0.815002157728395,
                -0.661443059471046,
                0.366614269701525,
                0.248957976189754,
                -0.586106758460856,
                -0.383516157216677,
                1.53740902604256,
                -0.528479803889375,
                0.140071528525743,
                0.0553883642703117,
                -1.86276666587731,
                1.25376857106666,
                -0.454193096983248,
                -2.52000363943994,
                -0.652074105236213,
                0.584856120354184,
                0.103317876922552,
            ]
        ).reshape((18, 2))
        b = np.array(
            [
                -0.755972280243298,
                1.27585691710246,
                -0.181010860594784,
                0.237445950423737,
                0.0217277772435122,
                -1.46201477997428,
                0.236818223531106,
                1.50419911335932,
                0.473911340657419,
                -0.421851787336940,
                -0.0358193558740663,
                0.978031282093556,
                0.877954743133157,
                -0.157160347511024,
                -0.116894695624955,
                1.85188802037506,
                -2.00206974955625,
                1.15734049563925,
            ]
        )

        rounded = RoundingApi.round_polytope(Polytope(A, b), RoundingSettings())

        expected_transform = np.array(
            [0.526199405698215, 0.0, 0.201452165863056, 0.280713912893698]
        )
        np.testing.assert_allclose(
            rounded.transformation.values.flatten(),
            expected_transform,
            atol=1e-6,
        )

    def test_mve_solve_simplex(self):
        dimension = 3
        A = np.vstack([-np.eye(dimension), np.ones((1, dimension))])
        b = np.array([[0], [0], [0], [1]], dtype=float)
        x = np.array([[0.1], [0.1], [0.1]])
        expected_center = np.full(dimension, 1.0 / (dimension + 1))
        expected_shape = (
            np.eye(dimension) - np.ones((dimension, dimension)) / (dimension + 1)
        ) / (dimension * (dimension + 1))

        for backend_name_, _, _, run_mve in CPU_MATH_BACKENDS:
            with self.subTest(backend=backend_name_):
                center, transform, message, delta_b, delta_s = run_mve(
                    A,
                    b,
                    x,
                    1e-3,
                )

                self.assertIn(message, (0, 1, 2))
                self.assertTrue(np.all(np.isfinite(center)))
                self.assertTrue(np.all(np.isfinite(transform)))
                self.assertGreater(
                    np.min(np.linalg.eigvalsh(transform @ transform.T)),
                    0.0,
                )
                self.assertLess(
                    np.max(np.abs(delta_b)),
                    default_numerics_threshold,
                )
                self.assertLess(
                    np.max(np.abs(delta_s)),
                    default_numerics_threshold,
                )
                np.testing.assert_allclose(
                    absolute_mve_center(center, x, message),
                    expected_center,
                    atol=1e-3,
                )
                np.testing.assert_allclose(
                    transform @ transform.T,
                    expected_shape,
                    atol=1e-3,
                )

    def test_mve_is_equivariant_under_diagonal_rescaling(self):
        dimension = 3
        simplex_A = np.vstack([-np.eye(dimension), np.ones((1, dimension))])
        b = np.array([[0.0], [0.0], [0.0], [1.0]])
        initial = np.full((dimension, 1), 0.1)
        scaling = np.diag([0.2, 3.0, 10.0])
        scaled_A = simplex_A @ np.linalg.inv(scaling)
        expected_center = np.full(dimension, 1.0 / (dimension + 1))

        for backend_name_, _, _, run_mve in CPU_MATH_BACKENDS:
            with self.subTest(backend=backend_name_):
                center, transform, message, _, _ = run_mve(
                    simplex_A,
                    b,
                    initial,
                    1e-3,
                )
                scaled_center, scaled_transform, scaled_message, _, _ = run_mve(
                    scaled_A,
                    b,
                    scaling @ initial,
                    1e-3,
                )

                np.testing.assert_allclose(
                    absolute_mve_center(center, initial, message),
                    expected_center,
                    atol=1e-3,
                )
                np.testing.assert_allclose(
                    absolute_mve_center(
                        scaled_center,
                        scaling @ initial,
                        scaled_message,
                    ),
                    scaling @ expected_center,
                    rtol=5e-3,
                    atol=1e-3,
                )
                np.testing.assert_allclose(
                    scaled_transform @ scaled_transform.T,
                    scaling @ (transform @ transform.T) @ scaling,
                    rtol=5e-3,
                    atol=1e-3,
                )

    def test_mve_reports_numerical_correction_for_boundary_start(self):
        dimension = 2
        A = np.vstack([-np.eye(dimension), np.ones((1, dimension))])
        b = np.array([[0.0], [0.0], [1.0]])
        boundary_start = np.zeros((dimension, 1))

        for backend_name_, _, _, run_mve in CPU_MATH_BACKENDS:
            with self.subTest(backend=backend_name_):
                center, transform, _, delta_b, delta_s = run_mve(
                    A,
                    b,
                    boundary_start,
                    1e-3,
                )

                self.assertTrue(np.all(np.isfinite(center)))
                self.assertTrue(np.all(np.isfinite(transform)))
                self.assertTrue(np.all(np.asarray(delta_b) >= 0.0))
                self.assertTrue(np.all(np.asarray(delta_s) >= 0.0))
                self.assertGreater(np.max(delta_b), 0.0)

    def test_geometric_mean_scaling_has_expected_mathematical_contract(self):
        A = np.array(
            [
                [1.0, -2.0, 3.0, -4.0],
                [1e1, -1e2, 1e3, -1e4],
                [1e-1, -1e-2, 1e-3, -1e-4],
            ]
        )
        original = A.copy()
        backend_results = {}

        for backend_name_, geometric_mean_scaling, _, _ in CPU_MATH_BACKENDS:
            with self.subTest(backend=backend_name_):
                with np.errstate(divide="ignore", invalid="ignore"):
                    cscale, rscale = geometric_mean_scaling(A, 0, 0.99)
                cscale = np.asarray(cscale).reshape(-1)
                rscale = np.asarray(rscale).reshape(-1)
                scaled = np.abs(A) / rscale[:, None] / cscale[None, :]

                self.assertEqual(cscale.shape, (A.shape[1],))
                self.assertEqual(rscale.shape, (A.shape[0],))
                self.assertTrue(np.all(np.isfinite(cscale)))
                self.assertTrue(np.all(np.isfinite(rscale)))
                self.assertTrue(np.all(cscale > 0.0))
                self.assertTrue(np.all(rscale > 0.0))
                np.testing.assert_allclose(
                    np.max(scaled, axis=0),
                    np.ones(A.shape[1]),
                    rtol=1e-12,
                    atol=1e-12,
                )
                backend_results[backend_name_] = (cscale, rscale)

        np.testing.assert_array_equal(A, original)
        np.testing.assert_allclose(
            backend_results["highs"][0],
            backend_results["gurobi"][0],
        )
        np.testing.assert_allclose(
            backend_results["highs"][1],
            backend_results["gurobi"][1],
        )

    def test_geometric_mean_scaling_handles_empty_rows_and_columns(self):
        A = np.array(
            [
                [0.0, 1e-8, 1e4, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1e-4, 1.0, 0.0],
            ]
        )
        nonempty_columns = np.any(A != 0.0, axis=0)

        for backend_name_, geometric_mean_scaling, _, _ in CPU_MATH_BACKENDS:
            with self.subTest(backend=backend_name_):
                with np.errstate(divide="ignore", invalid="ignore"):
                    cscale, rscale = geometric_mean_scaling(A, 0, 0.99)
                cscale = np.asarray(cscale).reshape(-1)
                rscale = np.asarray(rscale).reshape(-1)
                scaled = np.abs(A) / rscale[:, None] / cscale[None, :]

                self.assertTrue(np.all(np.isfinite(cscale)))
                self.assertTrue(np.all(np.isfinite(rscale)))
                self.assertTrue(np.all(cscale > 0.0))
                self.assertTrue(np.all(rscale > 0.0))
                self.assertEqual(rscale[1], 1.0)
                np.testing.assert_array_equal(
                    cscale[~nonempty_columns],
                    np.ones(np.count_nonzero(~nonempty_columns)),
                )
                np.testing.assert_allclose(
                    np.max(scaled[:, nonempty_columns], axis=0),
                    np.ones(np.count_nonzero(nonempty_columns)),
                )

    def test_nearest_spd_backends_project_symmetric_part_to_positive_definite(self):
        A = np.array(
            [
                [1.0, 4.0, -2.0],
                [-3.0, -5.0, 7.0],
                [6.0, 2.0, 0.5],
            ]
        )
        symmetric_part = (A + A.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric_part)
        nearest_psd = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T

        for backend_name_, _, get_nspd, _ in CPU_MATH_BACKENDS:
            with self.subTest(backend=backend_name_):
                corrected = get_nspd(A)

                self.assertTrue(np.all(np.isfinite(corrected)))
                np.testing.assert_allclose(corrected, corrected.T, atol=1e-13)
                self.assertGreater(np.min(np.linalg.eigvalsh(corrected)), 0.0)
                np.testing.assert_allclose(
                    corrected,
                    nearest_psd,
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_nearest_spd_leaves_positive_definite_input_unchanged(self):
        A = np.array([[4.0, 1.0, -0.5], [1.0, 3.0, 0.25], [-0.5, 0.25, 2.0]])

        for backend_name_, _, get_nspd, _ in CPU_MATH_BACKENDS:
            with self.subTest(backend=backend_name_):
                np.testing.assert_allclose(get_nspd(A), A, atol=1e-13)

    def test_nearest_spd_rejects_nonsquare_input(self):
        for backend_name_, _, get_nspd, _ in CPU_MATH_BACKENDS:
            with self.subTest(backend=backend_name_):
                with self.assertRaises(AssertionError):
                    get_nspd(np.ones((2, 3)))

    def test_gpu_math_backend_matches_cpu_reference(self):
        cupy, geometric_mean_scaling, get_nspd, run_mve = load_gpu_math_backend(self)

        scaling_matrix = np.array(
            [[1e-4, -2.0, 3e2], [4e2, 5e-3, -6.0], [7.0, -8e1, 9e-2]]
        )
        expected_cscale, expected_rscale = highs_geometric_mean_scaling(
            scaling_matrix,
            0,
            0.99,
        )
        cscale, rscale = geometric_mean_scaling(scaling_matrix, 0, 0.99)
        np.testing.assert_allclose(cscale, expected_cscale, rtol=1e-12)
        np.testing.assert_allclose(rscale, np.squeeze(expected_rscale), rtol=1e-12)

        matrix = np.array([[1.0, 4.0, -2.0], [-3.0, -5.0, 7.0], [6.0, 2.0, 0.5]])
        gpu_nspd = cupy.asnumpy(get_nspd(cupy.asarray(matrix)))
        np.testing.assert_allclose(
            gpu_nspd,
            highs_get_nspd(matrix),
            rtol=1e-11,
            atol=1e-11,
        )

        dimension = 3
        A = np.vstack([-np.eye(dimension), np.ones((1, dimension))])
        b = np.array([[0.0], [0.0], [0.0], [1.0]])
        initial = np.full((dimension, 1), 0.1)
        cpu_center, cpu_transform, _, cpu_delta_b, cpu_delta_s = highs_run_mve(
            A,
            b,
            initial,
            1e-3,
        )
        gpu_center, gpu_transform, _, gpu_delta_b, gpu_delta_s = run_mve(
            A,
            b,
            initial,
            1e-3,
        )
        np.testing.assert_allclose(gpu_center, cpu_center, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(
            gpu_transform @ gpu_transform.T,
            cpu_transform @ cpu_transform.T,
            rtol=1e-8,
            atol=1e-9,
        )
        np.testing.assert_allclose(gpu_delta_b, cpu_delta_b)
        np.testing.assert_allclose(gpu_delta_s, cpu_delta_s)

    def test_shift_transform_backtransform_composition(self):
        S = np.ones((1, 3))
        A = np.vstack((np.eye(3), -np.eye(3), S))
        b = np.array([1, 1, 1, 1, 1, 1, 0], dtype=float)
        x = np.array([4, 9, 83])
        polytope = Polytope(A, b, S=S)

        shift1 = np.array([1, 2, 3])
        transform1 = np.array([[43, 6, 3], [11, 6, 4], [-54, 5, 5431]])
        shift2 = np.array([-5, 3, 0.41])
        transform2 = np.matmul(shift1[:, None], shift2[:, None].T) + np.eye(3)
        shift3 = np.power(shift1, shift2)

        x_mod = x.copy()
        x_mod -= shift1
        polytope.apply_shift(shift1)
        x_mod = np.matmul(np.linalg.inv(transform1), x_mod)
        polytope.apply_transformation(transform1)
        np.testing.assert_allclose(polytope.back_transform(x_mod.copy()), x)

        x_mod -= shift2
        polytope.apply_shift(shift2)
        np.testing.assert_allclose(polytope.back_transform(x_mod.copy()), x)

        x_mod = np.matmul(np.linalg.inv(transform2), x_mod)
        polytope.apply_transformation(transform2)
        np.testing.assert_allclose(polytope.back_transform(x_mod.copy()), x)

        x_mod -= shift3
        polytope.apply_shift(shift3)
        np.testing.assert_allclose(polytope.back_transform(x_mod.copy()), x)

    def test_degenerate_polytope_raises(self):
        S = np.array([[1, 0], [0, 1]])
        A = np.eye(2)
        b = np.ones(2)

        with self.assertRaises(ValueError):
            RoundingApi.simplify_polytope(Polytope(A, b, S=S))

    def test_null_space_backends_return_complete_orthonormal_basis(self):
        matrices = (
            np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]),
            np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        )

        for backend_name_, compute_null_space in NULL_SPACE_BACKENDS:
            for matrix in matrices:
                with self.subTest(backend=backend_name_, shape=matrix.shape):
                    null = compute_null_space(matrix)
                    expected_nullity = matrix.shape[1] - np.linalg.matrix_rank(matrix)

                    self.assertEqual(
                        null.shape,
                        (matrix.shape[1], expected_nullity),
                    )
                    self.assertTrue(np.all(np.isfinite(null)))
                    np.testing.assert_allclose(
                        matrix @ null,
                        np.zeros((matrix.shape[0], expected_nullity)),
                        atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        null.T @ null,
                        np.eye(expected_nullity),
                        atol=1e-12,
                    )

    def test_null_space_threshold_controls_near_singular_direction(self):
        matrix = np.diag([1.0, 1e-12])

        for backend_name_, compute_null_space in NULL_SPACE_BACKENDS:
            with self.subTest(backend=backend_name_):
                coarse = compute_null_space(matrix, eps=1e-10)
                fine = compute_null_space(matrix, eps=1e-14)

                self.assertEqual(coarse.shape, (2, 1))
                self.assertEqual(fine.shape, (2, 0))
                np.testing.assert_allclose(matrix @ coarse, 0.0, atol=1e-12)

    def test_keep_equalities(self):
        S = np.array([[1, -1, 0], [0, 1, -1]])
        A = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        b = np.array([5.25, 1.75])

        result = RoundingApi.simplify_transform_and_round(Polytope(A, b, S=S))

        self.assertEqual(result.transformation.shape, (3, 1))

    def test_chebyshev_center_matches_translated_square_for_lp_backends(self):
        A = np.vstack([np.eye(2), -np.eye(2)])
        b = np.array([4.0, 4.0, 2.0, 2.0])

        for backend_name_, _ in solver_backends():
            with self.subTest(backend=backend_name_):
                center, radius = RoundingApi.chebyshev_center(
                    Polytope(A, b),
                    RoundingSettings(backend=backend_name_),
                )

                self.assertEqual(center.shape, (2, 1))
                self.assertEqual(radius.shape, (1,))
                np.testing.assert_allclose(center[:, 0], np.ones(2), atol=1e-8)
                np.testing.assert_allclose(radius, np.array([3.0]), atol=1e-8)

    def test_lp_backends_solve_equality_constrained_problem(self):
        A = np.vstack([np.eye(2), -np.eye(2)])
        b = np.array([1.0, 1.0, 0.0, 0.0])
        S = np.array([[1.0, 1.0]])
        h = np.array([1.0])
        objective = np.array([1.0, 2.0])

        for backend_name_, interfacer in solver_backends():
            with self.subTest(backend=backend_name_):
                settings = RoundingSettings(backend=backend_name_)
                solution, model = interfacer.solve(
                    objective,
                    A,
                    b,
                    settings,
                    S=S,
                    h=h,
                )

                np.testing.assert_allclose(solution, np.array([1.0, 0.0]), atol=1e-8)
                self.assertEqual(model.status, "optimal")
                self.assertEqual(model.objective.direction, "min")
                self.assertAlmostEqual(model.objective.value, 1.0, places=8)
                self.assertEqual(set(model.primal_values), {"0", "1"})
                np.testing.assert_allclose(
                    list(model.primal_values.values()),
                    solution,
                    atol=1e-8,
                )

    def test_lp_backend_model_roundtrip_preserves_named_constraints(self):
        A = pd.DataFrame(
            [[1.0, 2.0], [-1.0, 0.0], [0.0, -1.0]],
            index=["diagonal", "lower_x", "lower_y"],
            columns=["x", "y"],
        )
        b = pd.Series([3.0, 1.0, 2.0], index=A.index)
        S = pd.DataFrame([[1.0, -1.0]], index=["balance"], columns=A.columns)
        h = pd.Series([0.25], index=S.index)
        polytope = Polytope(A, b, S=S, h=h)

        for backend_name_, interfacer in solver_backends():
            with self.subTest(backend=backend_name_):
                settings = RoundingSettings(backend=backend_name_)
                model = interfacer.polytope_to_model(polytope, settings)
                restored = interfacer.model_to_polytope(model)

                self.assertEqual(
                    [str(variable) for variable in restored.A.columns],
                    ["x", "y"],
                )
                np.testing.assert_allclose(
                    canonical_rows(restored.A.values, restored.b.values),
                    canonical_rows(A.values, b.values),
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    canonical_rows(restored.S.values, restored.h.values),
                    canonical_rows(S.values, h.values),
                    atol=1e-12,
                )

    def test_check_lps_validates_optimal_solutions_for_lp_backends(self):
        A = np.vstack([np.eye(2), -np.eye(2)])
        b = np.ones(4)
        objective = np.array([-1.0, -1.0])

        for backend_name_, interfacer in solver_backends():
            with self.subTest(backend=backend_name_):
                settings = RoundingSettings(
                    backend=backend_name_,
                    check_lps=True,
                )
                solution, model = interfacer.solve(objective, A, b, settings)
                optimum = interfacer.get_opt(model, settings)

                np.testing.assert_allclose(solution, np.ones(2), atol=1e-8)
                self.assertAlmostEqual(optimum, -2.0, places=8)

    def test_simplification_agrees_across_lp_backends(self):
        source, expected_rows = add_scaled_duplicate_rows(
            box_polytope(2),
            n_extra=4,
        )
        backend_results = {}

        for backend_name_, _ in solver_backends():
            with self.subTest(backend=backend_name_):
                simplified = RoundingApi.simplify_polytope(
                    source,
                    RoundingSettings(
                        backend=backend_name_,
                        simplify_only=True,
                    ),
                )
                self.assertEqual(simplified.A.shape[0], expected_rows)
                backend_results[backend_name_] = canonical_rows(
                    simplified.A.values,
                    simplified.b.values,
                )

        if HAS_GUROBI:
            np.testing.assert_allclose(
                backend_results["highs"],
                backend_results["gurobi"],
                atol=1e-8,
            )

    def test_constraint_simplification_non_bounded(self):
        A = np.vstack((-np.eye(3), -np.ones((1, 3))))
        b = np.array([1, 1, 1, 3], dtype=float)

        reduced_polytope = RoundingApi.simplify_polytope(
            Polytope(A, b),
            settings=RoundingSettings(simplify_only=True),
        )

        A_true = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        b_true = np.array([1, 1, 1], dtype=float)
        np.testing.assert_allclose(reduced_polytope.A.values, A_true)
        np.testing.assert_allclose(reduced_polytope.b.values, b_true)

    def _reset_lp(self):
        hopsy.LP().reset()

    def _box_problem(self, n=2, lo=-1.0, hi=1.0, starting_point=None):
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.concatenate([np.full(n, hi), np.full(n, -lo)])
        return hopsy.Problem(A, b, starting_point=starting_point)

    def _facet_distance_ratio(self, A, b):
        row_norms = np.linalg.norm(A, axis=1)
        self.assertTrue(np.all(row_norms > 0.0))
        distances = b / row_norms
        self.assertTrue(np.all(distances > 0.0))
        return float(np.max(distances) / np.min(distances))

    def test_public_round_returns_new_problem_without_mutating_input(self):
        starting_point = np.array([0.25, -0.5])
        problem = self._box_problem(
            n=2,
            lo=-2.0,
            hi=3.0,
            starting_point=starting_point,
        )
        original_A = problem.A.copy()
        original_b = problem.b.copy()

        rounded = hopsy.round(problem, simplify=False)

        self.assertIsNot(rounded, problem)
        np.testing.assert_array_equal(problem.A, original_A)
        np.testing.assert_array_equal(problem.b, original_b)
        np.testing.assert_array_equal(problem.starting_point, starting_point)
        self.assertIsNone(problem.transformation)
        self.assertIsNone(problem.shift)
        np.testing.assert_array_equal(rounded.original_A, original_A)
        np.testing.assert_array_equal(rounded.original_b, original_b)
        self.assertIsNotNone(rounded.transformation)
        self.assertIsNotNone(rounded.shift)

    def test_public_round_composes_an_existing_rectangular_transformation(self):
        current_A = np.vstack([np.eye(2), -np.eye(2)])
        current_b = np.ones(4)
        previous_transformation = np.array([[2.0, 0.5], [0.0, 3.0], [1.0, -1.0]])
        previous_shift = np.array([1.0, -2.0, 0.5])
        current_starting_point = np.array([0.2, -0.3])
        original_starting_point = (
            previous_transformation @ current_starting_point + previous_shift
        )
        problem = hopsy.Problem(
            current_A,
            current_b,
            starting_point=current_starting_point,
            transformation=previous_transformation,
            shift=previous_shift,
        )

        rounded = hopsy.round(problem, simplify=False)

        self.assertEqual(rounded.transformation.shape, (3, 2))
        self.assertEqual(rounded.shift.shape, (3,))
        self.assertEqual(rounded.starting_point.shape, (2,))
        self.assertTrue(np.all(rounded.b - rounded.A @ rounded.starting_point > 0.0))
        np.testing.assert_allclose(
            hopsy.back_transform(rounded, [rounded.starting_point])[0],
            original_starting_point,
            atol=1e-8,
        )

        rng = np.random.default_rng(9)
        current_points = rng.uniform(-0.8, 0.8, size=(12, 2))
        original_points = current_points @ previous_transformation.T + previous_shift
        rounded_points = hopsy.transform(rounded, original_points)
        np.testing.assert_allclose(
            hopsy.back_transform(rounded, rounded_points),
            original_points,
            atol=1e-8,
        )
        self.assertTrue(
            np.all(rounded.b[None, :] - rounded_points @ rounded.A.T >= -1e-8)
        )

    def test_simplify_full_dimensional_problem_does_not_store_identity_transform(self):
        self._reset_lp()
        problem = self._box_problem(n=2)

        simplified = hopsy.simplify(problem)

        self.assertIsNone(simplified.transformation)
        self.assertIsNone(simplified.shift)
        self.assertEqual(simplified.A.shape[1], 2)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hopsy.add_box_constraints(simplified, -2.0, 2.0, simplify=False)

        user_warnings = [
            warning for warning in caught if issubclass(warning.category, UserWarning)
        ]
        self.assertEqual(user_warnings, [])

    def test_simplified_full_dimensional_problem_accepts_equality_constraints(self):
        self._reset_lp()
        problem = self._box_problem(n=2)
        simplified = hopsy.simplify(problem)

        constrained = hopsy.add_equality_constraints(
            simplified,
            np.array([[1.0, 0.0]]),
            np.array([0.0]),
        )

        self.assertEqual(constrained.A.shape[1], 1)
        self.assertIsNotNone(constrained.transformation)
        self.assertIsNotNone(constrained.shift)

    def test_add_box_constraints_simplify_does_not_mark_plain_box_transformed(self):
        self._reset_lp()
        problem = hopsy.Problem(np.zeros((0, 2)), np.zeros((0,)))

        constrained = hopsy.add_box_constraints(problem, -1.0, 1.0, simplify=True)

        self.assertEqual(constrained.A.shape[1], 2)
        self.assertIsNone(constrained.transformation)
        self.assertIsNone(constrained.shift)

    def test_simplify_dimension_reduction_transforms_starting_point(self):
        self._reset_lp()
        A = np.vstack([np.eye(3), -np.eye(3)])
        b = np.array([1.0, 1.0, 1e-8, 1.0, 1.0, 1e-8])
        original_starting_point = np.array([0.0, 0.0, 0.0])
        problem = hopsy.Problem(A, b, starting_point=original_starting_point)

        simplified = hopsy.simplify(problem)

        self.assertEqual(simplified.A.shape[1], 2)
        self.assertEqual(simplified.starting_point.shape[0], 2)
        self.assertIsNotNone(simplified.transformation)
        self.assertIsNotNone(simplified.shift)
        self.assertTrue(
            np.all(simplified.b - simplified.A @ simplified.starting_point >= -1e-8)
        )

        roundtrip = hopsy.back_transform(simplified, [simplified.starting_point])[0]
        np.testing.assert_allclose(roundtrip, original_starting_point, atol=1e-7)

    def test_round_after_dimension_reducing_simplify_composes_starting_point_mapping(
        self,
    ):
        self._reset_lp()
        A = np.vstack([np.eye(3), -np.eye(3)])
        b = np.array([1.0, 1.0, 1e-8, 1.0, 1.0, 1e-8])
        original_starting_point = np.array([0.0, 0.0, 0.0])
        problem = hopsy.Problem(A, b, starting_point=original_starting_point)

        simplified = hopsy.simplify(problem)
        rounded = hopsy.round(simplified, simplify=False)

        self.assertEqual(rounded.A.shape[1], rounded.starting_point.shape[0])
        self.assertTrue(np.all(rounded.b - rounded.A @ rounded.starting_point > 0))

        roundtrip = hopsy.back_transform(rounded, [rounded.starting_point])[0]
        np.testing.assert_allclose(roundtrip, original_starting_point, atol=1e-7)

    def test_markov_chain_after_noop_simplify_uses_untransformed_problem_contract(self):
        self._reset_lp()
        problem = hopsy.simplify(self._box_problem(n=2))

        self.assertIsNone(problem.transformation)
        self.assertIsNone(problem.shift)

        markov_chain = hopsy.MarkovChain(problem)
        self.assertEqual(markov_chain.state.shape[0], 2)
        self.assertTrue(np.all(problem.b - problem.A @ markov_chain.state > 0))

    def test_round_improves_axis_aligned_box_facet_distance_isotropy(self):
        self._reset_lp()
        widths = np.array([1.0e-2, 1.0, 1.0e2])
        A = np.vstack([np.eye(3), -np.eye(3)])
        b = np.concatenate([widths, widths])
        original_ratio = self._facet_distance_ratio(A, b)

        problem = hopsy.Problem(A, b, starting_point=np.zeros(3))
        rounded = hopsy.round(problem, simplify=False)

        self.assertIsNotNone(rounded.transformation)
        self.assertIsNotNone(rounded.shift)
        self.assertTrue(np.all(np.isfinite(rounded.transformation)))
        self.assertTrue(np.all(np.isfinite(rounded.shift)))
        self.assertEqual(
            np.linalg.matrix_rank(rounded.transformation),
            rounded.transformation.shape[1],
        )
        self.assertLessEqual(
            np.linalg.cond(rounded.transformation), original_ratio * 1.01
        )
        self.assertGreater(float(np.min(rounded.b)), 0.0)

        rounded_ratio = self._facet_distance_ratio(rounded.A, rounded.b)
        self.assertLess(rounded_ratio, original_ratio / 100.0)
        self.assertLess(rounded_ratio, 10.0)

    def test_reverse_mapping_for_sampled_original_points_after_equality_and_round(self):
        self._reset_lp()
        A = np.vstack([np.eye(3), -np.eye(3)])
        b = np.full(6, 2.0)
        A_eq = np.array([[1.0, 1.0, 0.0]])
        b_eq = np.array([0.5])

        rng = np.random.default_rng(7)
        first_coordinate = rng.uniform(-0.75, 0.75, size=8)
        third_coordinate = rng.uniform(-1.0, 1.0, size=8)
        original_points = np.column_stack(
            [
                first_coordinate,
                b_eq[0] - first_coordinate,
                third_coordinate,
            ]
        )
        self.assertTrue(np.all(b[None, :] - original_points @ A.T > 0.0))
        np.testing.assert_allclose(
            original_points @ A_eq.T,
            np.full((original_points.shape[0], 1), b_eq[0]),
        )

        problem = hopsy.Problem(A, b, starting_point=original_points[0])
        constrained = hopsy.add_equality_constraints(problem, A_eq, b_eq)
        rounded = hopsy.round(constrained, simplify=False)

        for transformed_problem in (constrained, rounded):
            transformed_points = hopsy.transform(transformed_problem, original_points)
            self.assertEqual(
                transformed_points.shape,
                (original_points.shape[0], transformed_problem.A.shape[1]),
            )
            slacks = transformed_problem.b[None, :] - (
                transformed_points @ transformed_problem.A.T
            )
            self.assertTrue(np.all(slacks >= -1e-7))

            roundtrip_points = hopsy.back_transform(
                transformed_problem, transformed_points
            )
            np.testing.assert_allclose(
                roundtrip_points,
                original_points,
                atol=1e-7,
            )
