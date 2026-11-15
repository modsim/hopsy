from concurrent.futures import ThreadPoolExecutor

import _hopsy_core as core
import numpy as np
import pytest
from test_bindings import GAUSSIAN, NAMES, box, make


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("rectangular", [False, True])
def test_mapped_outputs_match_sampling_outputs(name, rectangular):
    T = (
        np.array([[1.0, 0.3], [0.1, 1.5], [1.0, -1.0]])
        if rectangular
        else np.array([[1.0, 0.3], [0.1, 1.5]])
    )
    shift = np.full(T.shape[0], 0.2)
    coordinates = core.AffineCoordinates(T, shift)
    target = core.GaussianTarget(np.zeros(T.shape[0]), np.eye(T.shape[0]))
    a, ra = make(name, geometry=box(2), target=target, coordinates=coordinates)
    b, rb = make(name, geometry=box(2), target=target, coordinates=coordinates)
    x = a.draw(ra, 100, thinning=2)
    z = b.draw_sampling(rb, 100, thinning=2)
    np.testing.assert_allclose(x, T @ z + shift[:, None])
    assert a.sampling_dimension == 2 and a.target_dimension == T.shape[0]
    np.testing.assert_allclose(a.target_state, coordinates.to_target(a.sampling_state))


@pytest.mark.parametrize("name", ["UniformCoordinateHitAndRunKernel"] + GAUSSIAN)
@pytest.mark.parametrize("layout", ["C", "F", "slice"])
def test_both_mapped_output_spaces_support_strides(name, layout):
    T = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    coordinates = core.AffineCoordinates(T, np.zeros(3))
    for sampling in (False, True):
        a, ra = make(name, geometry=box(2), coordinates=coordinates)
        b, rb = make(name, geometry=box(2), coordinates=coordinates)
        rows = 2 if sampling else 3
        if layout == "slice":
            storage = np.full((2 * rows, 100), 99.0)
            out = storage[::2, ::2]
        else:
            out = np.empty((rows, 50), order=layout)
        method = a.draw_sampling_into if sampling else a.draw_into
        method(ra, out, thinning=2)
        expected = b.draw_sampling(rb, 50, 2) if sampling else b.draw(rb, 50, 2)
        np.testing.assert_array_equal(out, expected)
        if layout == "slice":
            assert np.all(storage[1::2] == 99.0) and np.all(storage[:, 1::2] == 99.0)


def test_mapping_validation_and_rebind_identity():
    with pytest.raises(ValueError):
        core.AffineCoordinates(np.zeros((2, 2)), np.zeros(2))
    with pytest.raises(ValueError):
        core.AffineCoordinates(np.eye(2), np.zeros(3))
    g = box(2)
    target = core.GaussianTarget(np.zeros(3), np.eye(3))
    with pytest.raises(ValueError):
        core.Problem(g, target)
    with pytest.raises(ValueError):
        core.Problem(g, target, core.AffineCoordinates(np.eye(3), np.zeros(3)))
    coordinates = core.AffineCoordinates(np.eye(2), np.zeros(2))
    p = core.Problem(g, core.UniformTarget(), coordinates)
    c = core.bind_chain(p, core.GaussianRandomWalkKernel(np.zeros(2)), np.zeros(2))
    with pytest.raises(ValueError):
        c.rebind_problem(
            core.Problem(
                g, core.UniformTarget(), core.AffineCoordinates(np.eye(2), np.zeros(2))
            )
        )
    c.rebind_problem(core.Problem(g, core.UniformTarget(), coordinates))


def test_shared_mapped_problem_is_thread_safe():
    coordinates = core.AffineCoordinates(
        np.array([[1.0, 0.2], [0.1, 2.0]]), np.zeros(2)
    )
    p = core.Problem(box(2), core.GaussianTarget(np.zeros(2), np.eye(2)), coordinates)

    def pair(seed):
        return core.bind_chain(
            p, core.CSmMALAKernel(np.zeros(2), step_size=0.3), np.zeros(2)
        ), core.RandomNumberGenerator(seed)

    serial = [pair(i) for i in range(4)]
    parallel = [pair(i) for i in range(4)]
    expected = [c.draw(r, 1000) for c, r in serial]
    with ThreadPoolExecutor(4) as pool:
        actual = list(pool.map(lambda pair: pair[0].draw(pair[1], 1000), parallel))
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
    assert T.shape == (3, 2) and geometry.is_feasible(initial)
    return geometry, coordinates, initial, A, b, S


@pytest.mark.parametrize("name", ["UniformCoordinateHitAndRunKernel"] + GAUSSIAN)
def test_actual_polyround_equality_reduction(polyround_problem, name):
    geometry, coordinates, initial, A, b, S = polyround_problem
    target = core.GaussianTarget(np.zeros(3), np.eye(3))
    chain, rng = make(
        name,
        seed=741,
        geometry=geometry,
        coordinates=coordinates,
        initial=initial,
        target=target,
    )
    samples = chain.draw(rng, 5000, thinning=2)
    assert samples.shape == (3, 5000)
    assert np.max(A @ samples - b[:, None]) < 1e-8
    np.testing.assert_allclose(S @ samples, 0.0, atol=1e-8)
    np.testing.assert_allclose(samples[:, -1], chain.target_state)
