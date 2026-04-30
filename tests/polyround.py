import unittest
import warnings

import numpy as np

import hopsy

try:
    from hopsy._polyround.api import PolyRoundApi
    from hopsy._polyround.default_settings import default_numerics_threshold
    from hopsy._polyround.mutable_classes.polytope import Polytope
    from hopsy._polyround.settings import PolyRoundSettings
    from hopsy._polyround.static_classes.constraint_removal_reduction import (
        PolytopeReducer,
    )
    from hopsy._polyround.static_classes.lp_interfacing import GurobiInterfacer
    from hopsy._polyround.static_classes.lp_utils import ChebyshevFinder
    from hopsy._polyround.static_classes.rounding.geometric_mean_scaling import (
        geometric_mean_scaling,
    )
    from hopsy._polyround.static_classes.rounding.maximum_volume_ellipsoid import (
        MaximumVolumeEllipsoidFinder,
    )
    from hopsy._polyround.static_classes.rounding.nearest_symmetric_positive_definite import (
        NSPD,
    )
except ImportError:
    from PolyRound.api import PolyRoundApi
    from PolyRound.default_settings import default_numerics_threshold
    from PolyRound.mutable_classes.polytope import Polytope
    from PolyRound.settings import PolyRoundSettings
    from PolyRound.static_classes.constraint_removal_reduction import PolytopeReducer
    from PolyRound.static_classes.lp_interfacing import GurobiInterfacer
    from PolyRound.static_classes.lp_utils import ChebyshevFinder
    from PolyRound.static_classes.rounding.geometric_mean_scaling import (
        geometric_mean_scaling,
    )
    from PolyRound.static_classes.rounding.maximum_volume_ellipsoid import (
        MaximumVolumeEllipsoidFinder,
    )
    from PolyRound.static_classes.rounding.nearest_symmetric_positive_definite import (
        NSPD,
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


# Simplify


class PolyRoundSimplifyTests(unittest.TestCase):

    def _settings(self, simplify_only=False):
        return PolyRoundSettings(simplify_only=simplify_only)

    # Redundancy: scaled duplicates

    def test_simplify_box_no_redundant_rows_unchanged(self):
        """Simplifying a plain box should not remove any constraints."""
        polytope = box_polytope(3)
        n_original = polytope.A.shape[0]
        result = PolyRoundApi.simplify_polytope(polytope, self._settings())
        self.assertLessEqual(result.A.shape[0], n_original)
        # All 6 box rows should survive (none are redundant in a tight box)
        self.assertEqual(result.A.shape[0], n_original)

    def test_simplify_removes_scaled_duplicates_3d(self):
        """Box 3D + 10 redundant rows: simplified result should have exactly 6 rows."""
        polytope, n_base = add_scaled_duplicate_rows(box_polytope(3), n_extra=10)
        result = PolyRoundApi.simplify_polytope(polytope, self._settings())
        self.assertEqual(
            result.A.shape[0],
            n_base,
            f"Expected {n_base} rows, got {result.A.shape[0]}",
        )

    def test_simplify_removes_scaled_duplicates_10d(self):
        """Box 10D + 60 redundant rows."""
        polytope, n_base = add_scaled_duplicate_rows(box_polytope(10), n_extra=60)
        result = PolyRoundApi.simplify_polytope(polytope, self._settings())
        self.assertEqual(result.A.shape[0], n_base)

    def test_simplify_result_rows_are_subset_of_original(self):
        """No new rows should appear — simplified rows must be a subset of input rows."""
        polytope, n_base = add_scaled_duplicate_rows(box_polytope(5), n_extra=20)
        result = PolyRoundApi.simplify_polytope(polytope, self._settings())
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
        result = PolyRoundApi.simplify_polytope(polytope, self._settings())
        self.assertLessEqual(result.A.shape[0], n_base)

    # Invariant: feasibility of a known interior point

    def test_simplify_preserves_interior_point(self):
        """The origin (interior of box) must remain feasible after simplify."""
        polytope, _ = add_scaled_duplicate_rows(box_polytope(5), n_extra=30)
        result = PolyRoundApi.simplify_polytope(polytope, self._settings())
        x0 = np.zeros(result.A.shape[1])
        self.assertTrue(
            is_feasible(result.A.values, result.b.values, x0),
            "Origin is no longer feasible after simplify",
        )

    # Idempotency

    def test_simplify_is_idempotent(self):
        """Simplifying twice should not change the row count."""
        polytope, _ = add_scaled_duplicate_rows(box_polytope(5), n_extra=20)
        once = PolyRoundApi.simplify_polytope(polytope, self._settings())
        twice = PolyRoundApi.simplify_polytope(once.copy(), self._settings())
        self.assertEqual(once.A.shape[0], twice.A.shape[0])

    # simplify_only flag

    def test_simplify_only_does_not_refuntion_narrow_constraints(self):
        """With simplify_only=True, thin directions are not turned into equalities."""
        # 2D box, then collapse one dimension to near-zero width
        A = np.vstack([np.eye(2), -np.eye(2)])
        b = np.array([1.0, 1e-8, 1.0, 1e-8])  # x1 in [-1, 1], x2 in [-1e-8, 1e-8]
        polytope = Polytope(A, b)
        result = PolyRoundApi.simplify_polytope(
            polytope, PolyRoundSettings(simplify_only=True)
        )
        self.assertIsNone(
            result.S, "simplify_only should not produce equality constraints"
        )


# Transform


class PolyRoundTransformTests(unittest.TestCase):

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
        result = PolyRoundApi.transform_polytope(polytope)
        self.assertTrue(result.inequality_only)
        self.assertIsNone(result.S)

    def test_transform_reduces_dimension(self):
        """With one equality constraint, ambient dimension drops by 1."""
        polytope = self._equality_box()
        original_dim = polytope.A.shape[1]
        result = PolyRoundApi.transform_polytope(polytope)
        self.assertEqual(result.A.shape[1], original_dim - 1)

    def test_transform_backtransform_roundtrip(self):
        """Back-transforming a point in the reduced space should give a point
        feasible in the original polytope."""
        polytope = self._equality_box()
        result = PolyRoundApi.transform_polytope(polytope)
        reduced_dim = result.A.shape[1]
        x_reduced = np.zeros(reduced_dim)
        x_original = result.back_transform(x_reduced)
        self.assertTrue(
            is_feasible(polytope.A.values, polytope.b.values, x_original),
            "Back-transformed origin is not feasible in the original polytope",
        )


# Round


class PolyRoundRoundTests(unittest.TestCase):

    def _roundable_polytope(self, n=4):
        """A box already transformed to be inequality-only (no equalities)."""
        return box_polytope(n)

    def test_round_origin_in_interior(self):
        """After rounding, the origin must be strictly inside: b.min() > 0."""
        polytope = self._roundable_polytope()
        result = PolyRoundApi.round_polytope(polytope)
        self.assertGreater(float(result.b.min()), 0.0)

    def test_round_transformation_full_rank(self):
        """The rounding transformation must be square and full-rank."""
        polytope = self._roundable_polytope(n=4)
        result = PolyRoundApi.round_polytope(polytope)
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
        result = PolyRoundApi.round_polytope(polytope)
        x_original = result.back_transform(np.zeros(result.A.shape[1]))
        self.assertTrue(is_feasible(original_A, original_b, x_original))


# Full pipeline (simplify > transform > round)


class PolyRoundPipelineTests(unittest.TestCase):

    def test_full_pipeline_box_with_equality(self):
        """End-to-end simplify_transform_and_round on a box with one equality."""
        A = np.vstack([np.eye(4), -np.eye(4)])
        b = np.ones(8)
        S = np.array([[1.0, 1.0, 0.0, 0.0]])
        h = np.array([0.0])
        polytope = Polytope(A, b, S, h)
        result = PolyRoundApi.simplify_transform_and_round(polytope)
        self.assertTrue(result.inequality_only)
        self.assertGreater(float(result.b.min()), 0.0)

    def test_full_pipeline_box_with_redundant_rows(self):
        """Pipeline should handle redundant rows without crashing."""
        polytope, n_base = add_scaled_duplicate_rows(box_polytope(5), n_extra=30)
        result = PolyRoundApi.simplify_transform_and_round(polytope)
        self.assertTrue(result.inequality_only)
        self.assertGreater(float(result.b.min()), 0.0)


# OG tests


class PolyRoundOldReferenceTests(unittest.TestCase):
    """Small, dependency-free tests ported from old PolyRound/unit_tests.py."""

    def test_exact_small_simplex_rounding(self):
        dim = 2
        A = np.vstack([-np.eye(dim), np.ones(shape=(1, dim))])
        b = np.zeros(dim + 1)
        b[-1] = 1

        rounded = PolyRoundApi.round_polytope(Polytope(A, b), PolyRoundSettings())

        expected_transform = np.array(
            [0.333333302249001, 0.0, -0.166666635206467, 0.288675116865272]
        )
        expected_shift = np.array([0.333333335613954, 0.333333335613954])
        np.testing.assert_allclose(
            rounded.transformation.values.flatten(),
            expected_transform,
            atol=1e-7,
        )
        np.testing.assert_allclose(rounded.shift.values, expected_shift, atol=1e-7)

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

        rounded = PolyRoundApi.round_polytope(Polytope(A, b), PolyRoundSettings())

        expected_transform = np.array(
            [0.526199405698215, 0.0, 0.201452165863056, 0.280713912893698]
        )
        np.testing.assert_allclose(
            rounded.transformation.values.flatten(),
            expected_transform,
            atol=1e-6,
        )

    def test_mve_solve_simplex(self):
        A = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 1, 1]], dtype=float)
        b = np.array([[0], [0], [0], [1]], dtype=float)
        x = np.array([[0.1], [0.1], [0.1]])

        _, E, _, delta_b, delta_s = MaximumVolumeEllipsoidFinder.run_mve(A, b, x, 1e-3)

        E_sol = np.array(
            [[0.25, 0, 0], [-0.0833, 0.2357, 0], [-0.0833, -0.1178, 0.2042]]
        )
        self.assertLess(np.max(np.abs(delta_b)), default_numerics_threshold)
        self.assertLess(np.max(np.abs(delta_s)), default_numerics_threshold)
        np.testing.assert_allclose(E, E_sol, atol=1e-3)

    def test_geometric_mean_scaling(self):
        A = np.array([[1, 2, 3, 4], [1e1, 1e2, 1e3, 1e4], [1e-1, 1e-2, 1e-3, 1e-4]])

        cscale, rscale = geometric_mean_scaling(A, 0, 0.99)

        cscale_sol = np.array(
            [31.6227766016838, 3.16227766016838, 3.16227766016838, 31.6227766016838]
        )
        rscale_sol = np.array([2.0, 316.227766016837, 0.00316227766016838])
        np.testing.assert_allclose(cscale, cscale_sol, atol=1e-10)
        np.testing.assert_allclose(np.squeeze(rscale), rscale_sol, atol=1e-10)

    def test_nearest_symmetric_positive_definite(self):
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        A_sol = np.array([[2, 3, 4], [3, 5, 6], [4, 6, 9]])

        A_hat = NSPD.get_NSPD(A)

        self.assertLess(np.max(np.abs(A_hat - A_sol)), 1)

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
            PolyRoundApi.simplify_polytope(Polytope(A, b, S=S))

    def test_simple_null_space(self):
        S = np.array([[1, -1, 0], [0, 1, -1]])

        null = PolytopeReducer.null_space(S)

        self.assertEqual(null.shape[1], 1)
        self.assertLessEqual(np.linalg.norm(S @ null), 1e-9)

    def test_keep_equalities(self):
        S = np.array([[1, -1, 0], [0, 1, -1]])
        A = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        b = np.array([5.25, 1.75])

        result = PolyRoundApi.simplify_transform_and_round(Polytope(A, b, S=S))

        self.assertEqual(result.transformation.shape, (3, 1))

    def test_chebyshev_center(self):
        settings = PolyRoundSettings()
        S = np.ones((1, 3))
        A = np.vstack((np.eye(3), -np.eye(3), S))
        b = np.array([1, 1, 1, 1, 1, 1, 0], dtype=float)

        x, dist0 = ChebyshevFinder.chebyshev_center(Polytope(A, b), settings)
        dist0 = float(np.squeeze(dist0))
        self.assertTrue(np.all(np.abs(x - x[0]) < 1e-10))
        self.assertGreater(dist0, 0.0)

        b[3] = 0
        x, dist1 = ChebyshevFinder.chebyshev_center(Polytope(A, b), settings)
        dist1 = float(np.squeeze(dist1))
        self.assertTrue(np.all(x[0] - x[1:] > 0))
        self.assertGreater(dist0, dist1)

    def test_minimal_lp(self):
        b = np.array([1, 1, 1, 1], dtype=float)
        A_ext = np.array([[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]], dtype=float)
        obj = np.array([0, 0, -1], dtype=float)

        val, _ = GurobiInterfacer.gurobi_solve(obj, A_ext, b, PolyRoundSettings())

        np.testing.assert_allclose(val, np.array([0, 0, 1]), atol=1e-10)

    def test_check_lps_validates_native_lp_without_warning(self):
        b = np.array([1, 1, 1, 1], dtype=float)
        A_ext = np.array([[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]], dtype=float)
        obj = np.array([0, 0, -1], dtype=float)
        settings = PolyRoundSettings(check_lps=True)

        _, model = GurobiInterfacer.gurobi_solve(obj, A_ext, b, settings)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            GurobiInterfacer.get_opt(model, settings)

        messages = [str(warning.message) for warning in caught]
        self.assertFalse(
            any("check_lps is not implemented" in message for message in messages)
        )

    def test_constraint_simplification_non_bounded(self):
        A = np.vstack((-np.eye(3), -np.ones((1, 3))))
        b = np.array([1, 1, 1, 3], dtype=float)

        reduced_polytope = PolyRoundApi.simplify_polytope(
            Polytope(A, b),
            settings=PolyRoundSettings(simplify_only=True),
        )

        A_true = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        b_true = np.array([1, 1, 1], dtype=float)
        np.testing.assert_allclose(reduced_polytope.A.values, A_true)
        np.testing.assert_allclose(reduced_polytope.b.values, b_true)


# Public calls


class PolyRoundWiringTests(unittest.TestCase):
    """Regression tests for the public hopsy API that wraps native PolyRound."""

    def setUp(self):
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

    def test_simplify_full_dimensional_problem_does_not_store_identity_transform(self):
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
        problem = hopsy.Problem(np.zeros((0, 2)), np.zeros((0,)))

        constrained = hopsy.add_box_constraints(problem, -1.0, 1.0, simplify=True)

        self.assertEqual(constrained.A.shape[1], 2)
        self.assertIsNone(constrained.transformation)
        self.assertIsNone(constrained.shift)

    def test_simplify_dimension_reduction_transforms_starting_point(self):
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
        problem = hopsy.simplify(self._box_problem(n=2))

        self.assertIsNone(problem.transformation)
        self.assertIsNone(problem.shift)

        markov_chain = hopsy.MarkovChain(problem)
        self.assertEqual(markov_chain.state.shape[0], 2)
        self.assertTrue(np.all(problem.b - problem.A @ markov_chain.state > 0))

    def test_round_improves_axis_aligned_box_facet_distance_isotropy(self):
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
