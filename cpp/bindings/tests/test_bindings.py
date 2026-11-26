"""The public API binds concrete C++ objects; no compatibility facade is used."""

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import _hopsy_core as core
import numpy as np
import pytest

HR = [
    f"{line}{direction}HitAndRunKernel"
    for line in ("Uniform", "Gaussian", "OverRelaxed")
    for direction in ("", "Coordinate", "RandomCoordinate")
]
GAUSSIAN = ["GaussianRandomWalkKernel", "DikinWalkKernel", "CSmMALAKernel"]
NAMES = HR + GAUSSIAN


def box(n=3, radius=3.0):
    return core.LinearPolytope(
        np.vstack((np.eye(n), -np.eye(n))), np.full(2 * n, radius)
    )


def make(name, seed=42, target=None, geometry=None, initial=None, coordinates=None):
    geometry = box() if geometry is None else geometry
    initial = np.zeros(geometry.num_dimension()) if initial is None else initial
    target = core.UniformTarget() if target is None else target
    problem = core.Problem(geometry, target, coordinates=coordinates)
    options = {"step_size": 0.3} if name in GAUSSIAN else {}
    kernel = getattr(core, name)(initial, **options)
    return core.bind_chain(problem, kernel, initial), core.RandomNumberGenerator(seed)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("native", [False, True])
def test_batching_matches_single_transitions(name, native):
    target = (
        core.GaussianTarget(np.zeros(3), np.eye(3)) if native else core.UniformTarget()
    )
    a, ra = make(name, target=target)
    b, rb = make(name, target=target)
    expected = np.column_stack([a.draw(ra, 1)[:, 0] for _ in range(100)])
    actual = b.draw(rb, 100)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(a.sampling_state, b.sampling_state)
    assert actual.shape == (3, 100)
    assert np.max(np.abs(actual)) <= 3.0
    assert not hasattr(a, "seed") and not hasattr(a, "reset")
    with pytest.raises(TypeError):
        b.draw(rb, 1, target=target)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("layout", ["C", "F", "slice"])
def test_output_layouts_and_thinning(name, layout):
    a, ra = make(name)
    b, rb = make(name)
    if layout == "slice":
        storage = np.full((6, 80), 99.0)
        out = storage[::2, ::2]
    else:
        out = np.empty((3, 40), order=layout)
    a.draw_into(ra, out, thinning=3)
    np.testing.assert_array_equal(out, b.draw(rb, 40, thinning=3))
    if layout == "slice":
        assert np.all(storage[1::2] == 99.0) and np.all(storage[:, 1::2] == 99.0)


@pytest.mark.parametrize("name", NAMES)
def test_parallel_native_chains_are_reproducible(name):
    target = core.GaussianTarget(np.zeros(3), np.eye(3))
    expected = [make(name, seed=i, target=target) for i in range(4)]
    actual = [make(name, seed=i, target=target) for i in range(4)]
    serial = [chain.draw(rng, 1000) for chain, rng in expected]
    with ThreadPoolExecutor(4) as pool:
        parallel = list(pool.map(lambda pair: pair[0].draw(pair[1], 1000), actual))
    for x, y in zip(serial, parallel):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("name", NAMES)
def test_explicit_object_lifetimes_and_detached_state(name):
    initial = np.zeros(3)
    g = box()
    p = core.Problem(g, core.UniformTarget())
    k = getattr(core, name)(initial)
    chain = core.bind_chain(p, k, initial)
    with pytest.raises(RuntimeError):
        core.bind_chain(p, k, initial)
    initial[:] = 2.0
    del g, p, k
    gc.collect()
    np.testing.assert_array_equal(chain.sampling_state, np.zeros(3))
    detached = chain.sampling_state
    detached[:] = 100.0
    assert np.max(np.abs(chain.draw(core.RandomNumberGenerator(2), 100))) <= 3.0


def test_kernel_and_rng_are_direct_separate_objects():
    s = np.zeros(2)
    g = box(2)
    k = core.UniformCoordinateHitAndRunKernel(
        s, core.CyclingCoordinateDirectionGenerator(2), core.UniformLineKernel()
    )
    p = core.Problem(g, core.UniformTarget())
    c = core.bind_chain(p, k, s)
    assert c.step(core.RandomNumberGenerator(1))
    assert g.num_dimension() == 2 and g.num_constraints() == 4
    assert g.is_feasible(s)
    assert not hasattr(k, "draw") and not hasattr(k, "seed")
    last = c.sampling_state
    del c
    # The lease ends when the C++ owner is destroyed.
    core.bind_chain(p, k, last)


@pytest.mark.parametrize("name", GAUSSIAN)
def test_step_size_contract(name):
    s = np.zeros(3)
    p = core.Problem(box(), core.UniformTarget())
    k = getattr(core, name)(s, step_size=0.2)
    assert k.step_size == 0.2
    k.step_size = 0.4
    chain = core.bind_chain(p, k, s)
    assert chain.step_size == 0.4
    with pytest.raises(RuntimeError):
        k.step_size = 0.7
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            chain.step_size = bad
        assert chain.step_size == 0.4
    chain.step(core.RandomNumberGenerator(4))
    chain.step_size = 0.3
    assert chain.step_size == 0.3


def test_validation_does_not_consume_rng_or_change_state():
    chain, rng = make("CSmMALAKernel")
    other, other_rng = make("CSmMALAKernel")
    for count, thin in [(-1, 1), (1, 0)]:
        with pytest.raises(ValueError):
            chain.draw(rng, count, thinning=thin)
    with pytest.raises(ValueError):
        chain.draw_into(rng, np.empty((2, 3)))
    with pytest.raises((TypeError, ValueError)):
        chain.draw_into(rng, np.empty((3, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        chain.draw_into(rng, np.empty((3, 3))[:, ::-1])
    out = np.empty((3, 3))
    out.flags.writeable = False
    with pytest.raises((TypeError, ValueError)):
        chain.draw_into(rng, out)
    assert chain.draw(rng, 0).shape == (3, 0)
    np.testing.assert_array_equal(chain.draw(rng, 100), other.draw(other_rng, 100))
    p = core.Problem(box(), core.UniformTarget())
    with pytest.raises(ValueError):
        core.bind_chain(p, core.GaussianRandomWalkKernel(np.zeros(3)), np.ones(3))
    with pytest.raises(ValueError):
        core.bind_chain(p, core.GaussianRandomWalkKernel(np.zeros(2)), np.zeros(2))
    with pytest.raises(ValueError):
        core.bind_chain(
            p, core.GaussianRandomWalkKernel(np.full(3, 4.0)), np.full(3, 4.0)
        )


def test_gaussian_moments():
    target = core.GaussianTarget(np.zeros(2), np.eye(2))
    chain, rng = make("GaussianRandomWalkKernel", target=target, geometry=box(2, 6.0))
    chain.draw(rng, 3000)
    samples = chain.draw(rng, 40000, thinning=3)
    assert np.max(np.abs(samples.mean(axis=1))) < 0.06
    np.testing.assert_allclose(np.cov(samples), np.eye(2), atol=0.08)
