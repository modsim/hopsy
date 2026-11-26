"""Independent slice and distribution checks for the explicitly partitioned model."""

import math
from concurrent.futures import ThreadPoolExecutor

import _hopsy_core as core
import numpy as np
import pytest


def geometry(c=0.2, lower=0.0, upper=1.0, sign=1.0):
    A = np.vstack((np.eye(2), -np.eye(2)))
    b = np.array([upper, upper, -lower, -lower])
    return core.BilinearPolytope(
        A, b, split=1, cross_terms=[np.array([[sign]])], bilinear_bounds=np.array([c])
    )


def chain(
    g,
    name="UniformHitAndRunKernel",
    seed=42,
    target=None,
    initial=None,
    coordinates=None,
):
    initial = np.array([0.1, 0.1]) if initial is None else initial
    target = core.UniformTarget() if target is None else target
    problem = core.Problem(g, target, coordinates=coordinates)
    kernel = getattr(core, name)(initial)
    assert core.supports_geometry(kernel, g)
    return core.bind_chain(problem, kernel, initial), core.RandomNumberGenerator(seed)


@pytest.mark.parametrize(
    "name",
    [
        "GaussianRandomWalkKernel",
        "DikinWalkKernel",
        "CSmMALAKernel",
        "OverRelaxedHitAndRunKernel",
        "OverRelaxedCoordinateHitAndRunKernel",
        "OverRelaxedRandomCoordinateHitAndRunKernel",
    ],
)
def test_unsupported_combinations_have_an_explicit_error(name):
    g = geometry()
    initial = np.array([0.1, 0.1])
    k = getattr(core, name)(initial)
    assert not core.supports_geometry(k, g)
    for coordinates in (None, core.AffineCoordinates(np.eye(2), np.zeros(2))):
        for target in (
            core.UniformTarget(),
            core.GaussianTarget(np.zeros(2), np.eye(2)),
        ):
            with pytest.raises(TypeError, match="does not support BilinearPolytope"):
                core.bind_chain(core.Problem(g, target, coordinates), k, initial)


@pytest.mark.parametrize(
    "name",
    [
        f"{line}{direction}HitAndRunKernel"
        for line in ("Uniform", "Gaussian")
        for direction in ("", "Coordinate", "RandomCoordinate")
    ],
)
@pytest.mark.parametrize("mapped", [False, True])
def test_supported_bindings_and_native_targets(name, mapped):
    g = geometry()
    target = core.GaussianTarget(np.zeros(2), np.eye(2))
    coordinates = (
        core.AffineCoordinates(
            np.array([[1.0, 0.2], [0.0, 1.0]]), np.array([0.2, -0.1])
        )
        if mapped
        else None
    )
    a, ra = chain(g, name=name, target=target, coordinates=coordinates)
    b, rb = chain(g, name=name, target=target, coordinates=coordinates)
    sampling = a.draw_sampling(ra, 200)
    physical = b.draw(rb, 200)
    expected = (
        coordinates.matrix @ sampling + coordinates.shift[:, None]
        if mapped
        else sampling
    )
    np.testing.assert_allclose(physical, expected)
    assert np.max(sampling[0] * sampling[1]) <= 0.2 + 1e-12
    assert np.min(sampling) >= -1e-12 and np.max(sampling) <= 1 + 1e-12


def test_exact_disconnected_slice_and_numerical_validation():
    g = geometry()
    roots = [(1 - math.sqrt(0.2)) / 2, (1 + math.sqrt(0.2)) / 2]
    np.testing.assert_allclose(
        g.line_domain(np.array([0.0, 1.0]), np.array([1.0, -1.0])),
        [[0.0, roots[0]], [roots[1], 1.0]],
    )
    assert g.line_domain(np.zeros(2), np.zeros(2)).shape == (1, 2)
    with pytest.raises(ValueError):
        geometry().line_domain(np.array([np.nan, 0.0]), np.ones(2))
    with pytest.raises(ValueError):
        core.BilinearPolytope(np.eye(2), np.ones(2), 0, [np.ones((1, 1))], np.ones(1))
    with pytest.raises(ValueError):
        core.BilinearPolytope(np.eye(2), np.ones(2), 1, [np.ones((2, 2))], np.ones(1))
    with pytest.raises(ValueError):
        core.BilinearPolytope(
            np.eye(2),
            np.ones(2),
            1,
            [np.ones((1, 1))],
            np.ones(1),
            affine=np.empty((1, 0)),
        )
    with pytest.raises(ValueError):
        core.BilinearPolytope(
            np.eye(2),
            np.ones(2),
            1,
            [np.ones((1, 1))],
            np.ones(1),
            affine=np.ones((2, 2)),
        )
    terms = g.cross_terms
    terms[0][:] = 100.0
    assert g.is_feasible(np.array([0.1, 0.1]))


def independent_intervals(A, b, cross, affine, rhs, z, d, split):
    slopes = A @ d
    slacks = b - A @ z
    lo = max((s / v for s, v in zip(slacks, slopes) if v < 0), default=-np.inf)
    hi = min((s / v for s, v in zip(slacks, slopes) if v > 0), default=np.inf)
    if lo > hi:
        return np.empty((0, 2))
    points = [lo, hi]
    for B, row, bound in zip(cross, affine, rhs):
        a = d[:split] @ B @ d[split:]
        bb = d[:split] @ B @ z[split:] + z[:split] @ B @ d[split:] + row @ d
        c = z[:split] @ B @ z[split:] + row @ z - bound
        roots = np.roots([a, bb, c] if a != 0 else [bb, c])
        points.extend(
            float(r.real) for r in roots if abs(r.imag) < 1e-10 and lo < r.real < hi
        )
    points = sorted(set(points))
    out = []
    for a, bb in zip(points[:-1], points[1:]):
        x = z + (a + bb) / 2 * d
        feasible = all(
            x[:split] @ B @ x[split:] + row @ x <= bound
            for B, row, bound in zip(cross, affine, rhs)
        )
        if feasible:
            if out and out[-1][1] == a:
                out[-1][1] = bb
            else:
                out.append([a, bb])
    return np.array(out).reshape((-1, 2))


def test_independent_numpy_root_solver_and_dense_grid():
    rng = np.random.default_rng(273)
    A, b = np.vstack((np.eye(4), -np.eye(4))), np.ones(8)
    for _ in range(80):
        B = rng.normal(size=(2, 2))
        cross = [B, -B]
        affine, rhs = rng.normal(size=(2, 4)), np.full(2, 0.2)
        z, d = rng.uniform(-0.8, 0.8, 4), rng.normal(size=4)
        g = core.BilinearPolytope(A, b, 2, cross, rhs, affine=affine)
        expected = independent_intervals(A, b, cross, affine, rhs, z, d, 2)
        actual = g.line_domain(z, d)
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=1e-10)
        for t in np.linspace(-2, 2, 71):
            point = z + t * d
            lhs = [
                point[:2] @ B @ point[2:] + row @ point - bound
                for B, row, bound in zip(cross, affine, rhs)
            ]
            if (
                min(abs(v) for v in lhs) < 1e-10
                or np.min(np.abs(np.abs(point) - 1)) < 1e-10
            ):
                continue
            feasible = np.all(A @ point <= b) and max(lhs) <= 0
            assert any(a <= t <= bb for a, bb in actual) == feasible


@pytest.mark.parametrize("name", ["UniformHitAndRunKernel", "GaussianHitAndRunKernel"])
def test_nonuniform_target_on_disconnected_region(name):
    g = geometry(0.0, -1.0, 1.0)
    c, rng = chain(
        g,
        name=name,
        target=core.GaussianTarget(np.zeros(2), np.eye(2)),
        initial=np.array([-0.5, 0.5]),
    )
    c.draw(rng, 4000)
    samples = c.draw(rng, 60000, thinning=2)
    normalizer = math.erf(1 / math.sqrt(2))
    second = 1 - 2 * math.exp(-0.5) / math.sqrt(2 * math.pi) / normalizer
    absolute_mean = math.sqrt(2 / math.pi) * (1 - math.exp(-0.5)) / normalizer
    assert np.max(np.abs(samples.mean(axis=1))) < 0.03
    np.testing.assert_allclose((samples**2).mean(axis=1), second, atol=0.015)
    assert abs(np.mean(samples[0] * samples[1]) + absolute_mean**2) < 0.015
    assert np.max(samples[0] * samples[1]) <= 1e-12


def test_parallel_reproducibility_and_coordinate_reducibility():
    g = geometry(0.0, -1.0, 1.0)

    def make(seed):
        return chain(g, seed=seed, initial=np.array([-0.5, 0.5]))

    serial = [make(i) for i in range(4)]
    parallel = [make(i) for i in range(4)]
    expected = [c.draw(r, 2000) for c, r in serial]
    with ThreadPoolExecutor(4) as pool:
        actual = list(pool.map(lambda pair: pair[0].draw(pair[1], 2000), parallel))
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
        assert np.any(a[0] > 0) and np.any(a[0] < 0)
    coordinate, rng = chain(
        g, name="UniformCoordinateHitAndRunKernel", initial=np.array([-0.5, 0.5])
    )
    samples = coordinate.draw(rng, 2000)
    assert np.all(samples[0] <= 0) and np.all(samples[1] >= 0)


def test_unrepresentable_components_raise_instead_of_being_dropped():
    A = np.vstack((np.eye(2), -np.eye(2)))
    tiny = core.BilinearPolytope(A, np.ones(4), 1, [np.array([[1e300]])], np.ones(1))
    with pytest.raises(
        ValueError, match="positive-width slice component cannot be represented"
    ):
        tiny.line_domain(np.zeros(2), np.full(2, 1e300))
    huge = core.BilinearPolytope(
        np.empty((0, 2)), np.empty(0), 1, [np.array([[1e-300]])], np.ones(1)
    )
    with pytest.raises(
        OverflowError, match="finite slice endpoint exceeds scalar range"
    ):
        huge.line_domain(np.zeros(2), np.full(2, 1e-300))
