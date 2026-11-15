from concurrent.futures import ThreadPoolExecutor

import _hopsy_core as core
import numpy as np
import pytest

NAMES = [
    f"{line}{direction}Kernel"
    for line in ("Uniform", "Gaussian", "OverRelaxed")
    for direction in ("HitAndRun", "CoordinateHitAndRun", "RandomCoordinateHitAndRun")
]
NAMES += ["GaussianRandomWalkKernel", "DikinWalkKernel", "CSmMALAKernel"]


def make(name, matrix=None, shift=None, seed=42):
    matrix = np.array([[-1.3, 0.2], [0.4, 0.8]]) if matrix is None else matrix
    shift = np.full(matrix.shape[0], 0.3) if shift is None else shift
    geometry = core.LinearPolytope(np.vstack((np.eye(2), -np.eye(2))), np.full(4, 3.0))
    coordinates = core.AffineCoordinates(matrix, shift)
    kwargs = {"step_size": 0.3} if name in NAMES[-3:] else {}
    chain = getattr(core, name)(
        geometry, np.zeros(2), seed=seed, coordinates=coordinates, **kwargs
    )
    target = core.GaussianTarget(np.zeros(matrix.shape[0]), np.eye(matrix.shape[0]))
    return chain, target, coordinates


@pytest.mark.parametrize("name", NAMES)
def test_mapped_draw_and_explicit_sampling_space(name):
    a, target, mapping = make(name)
    b, _, _ = make(name)
    physical = a.draw(300, thinning=2, target=target)
    sampling = b.draw_sampling(300, thinning=2, target=target)
    np.testing.assert_allclose(
        physical,
        mapping.matrix @ sampling + mapping.shift[:, None],
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_array_equal(a.sampling_state, b.state)
    np.testing.assert_allclose(a.target_state, physical[:, -1])
    assert a.dimension == 2 and a.target_dimension == 2
    saved = a.target_state
    a.draw(10, target=target)
    np.testing.assert_allclose(saved, physical[:, -1])


@pytest.mark.parametrize("name", ["UniformCoordinateHitAndRunKernel"] + NAMES[-3:])
@pytest.mark.parametrize("layout", ["C", "F", "slice"])
def test_rectangular_affine_map_and_strided_output(name, layout):
    T = np.array([[1.0, 0.0], [0.5, 2.0], [1.0, -1.0]])
    a, target, mapping = make(name, T)
    b, _, _ = make(name, T)
    if layout == "slice":
        storage = np.full((6, 200), 99.0)
        output = storage[::2, ::2]
    else:
        output = np.empty((3, 100), order=layout)
    a.draw_into(output, target=target)
    np.testing.assert_array_equal(output, b.draw(100, target=target))
    assert a.dimension == 2 and a.target_dimension == 3
    np.testing.assert_allclose(a.target_state, mapping.to_target(a.state))
    if layout == "slice":
        assert np.all(storage[1::2] == 99.0)
        assert np.all(storage[:, 1::2] == 99.0)


def test_invalid_maps_and_target_dimensions():
    with pytest.raises(ValueError):
        core.AffineCoordinates(np.zeros((2, 2)), np.zeros(2))
    with pytest.raises(ValueError):
        core.AffineCoordinates(np.eye(2), np.zeros(3))
    g = core.LinearPolytope(np.eye(2), np.ones(2))
    with pytest.raises(ValueError):
        core.CSmMALAKernel(
            g, np.zeros(2), coordinates=core.AffineCoordinates(np.eye(3), np.zeros(3))
        )
    chain, _, _ = make("CSmMALAKernel", np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    before = chain.state
    with pytest.raises(ValueError):
        chain.draw(10, target=core.GaussianTarget(np.zeros(2), np.eye(2)))
    np.testing.assert_array_equal(chain.state, before)


def test_shared_mapping_and_native_target_are_thread_safe():
    chains = [make("CSmMALAKernel", seed=i)[0] for i in range(4)]
    reference = [make("CSmMALAKernel", seed=i)[0] for i in range(4)]
    target = core.GaussianTarget(np.zeros(2), np.eye(2))
    expected = [c.draw(1000, target=target) for c in reference]
    with ThreadPoolExecutor(4) as pool:
        actual = list(pool.map(lambda c: c.draw(1000, target=target), chains))
    for x, y in zip(actual, expected):
        np.testing.assert_array_equal(x, y)


@pytest.fixture(scope="module")
def polyround_problem():
    pytest.importorskip("PolyRound")
    from PolyRound.api import PolyRoundApi
    from PolyRound.mutable_classes.polytope import Polytope
    from PolyRound.settings import PolyRoundSettings

    A = np.vstack((np.eye(3), -np.eye(3)))
    b = np.array([2.0, 1.0, 1.0, 2.0, 1.0, 1.0])
    S, h = np.ones((1, 3)), np.zeros(1)
    settings = PolyRoundSettings(backend="glpk", verbose=False)
    polytope = PolyRoundApi.simplify_polytope(Polytope(A, b, S, h), settings)
    polytope = PolyRoundApi.transform_polytope(polytope, settings)
    polytope = PolyRoundApi.round_polytope(polytope, settings)
    T, shift = polytope.transformation.to_numpy(), polytope.shift.to_numpy()
    geometry = core.LinearPolytope(polytope.A.to_numpy(), polytope.b.to_numpy())
    coordinates = core.AffineCoordinates(T, shift)
    initial = np.linalg.lstsq(T, -shift, rcond=None)[0]
    assert T.shape == (3, 2) and geometry.contains(initial)
    return geometry, coordinates, initial, A, b, S


@pytest.mark.parametrize(
    "name",
    ["UniformCoordinateHitAndRunKernel", "GaussianRandomWalkKernel", "CSmMALAKernel"],
)
def test_actual_polyround_output_with_equality_reduction(polyround_problem, name):
    geometry, coordinates, initial, A, b, S = polyround_problem
    options = {"step_size": 0.3} if name != "UniformCoordinateHitAndRunKernel" else {}
    chain = getattr(core, name)(
        geometry, initial, coordinates=coordinates, seed=741, **options
    )
    target = core.GaussianTarget(np.zeros(3), np.eye(3))
    samples = chain.draw(5000, thinning=2, target=target)
    assert samples.shape == (3, 5000)
    assert np.max(A @ samples - b[:, None]) < 1e-8
    np.testing.assert_allclose(S @ samples, 0.0, atol=1e-8)
    np.testing.assert_allclose(samples[:, -1], chain.target_state)
