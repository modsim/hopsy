import gc
from concurrent.futures import ThreadPoolExecutor

import _hopsy_core as core
import numpy as np
import pytest

NAMES = [
    f"{line}{direction}Kernel"
    for line in ("Uniform", "Gaussian", "OverRelaxed")
    for direction in ("HitAndRun", "CoordinateHitAndRun", "RandomCoordinateHitAndRun")
]


def box(n=3):
    return core.LinearPolytope(np.vstack((np.eye(n), -np.eye(n))), np.ones(2 * n))


def chain(name="UniformCoordinateHitAndRunKernel", n=3, seed=42):
    return getattr(core, name)(box(n), np.zeros(n), seed=seed)


@pytest.mark.parametrize("name", NAMES)
def test_all_kernels_and_sampling_entry_points(name):
    a, b = chain(name), chain(name)
    assert isinstance(core.step(a), bool)
    assert b.step(core.UniformTarget()) is True or "Gaussian" in name
    np.testing.assert_array_equal(a.state, b.state)
    x = core.draw(a, 300, thinning=3)
    y = b.draw(300, thinning=3, target=core.UniformTarget())
    assert x.shape == (3, 300) and x.dtype == np.float64
    np.testing.assert_array_equal(x, y)
    assert np.max(np.abs(x)) <= 1 + 1e-12
    np.testing.assert_array_equal(x[:, -1], a.state)


@pytest.mark.parametrize("name", NAMES)
def test_reset_reseed_and_ownership(name):
    initial = np.zeros(3)
    geometry = box()
    k = getattr(core, name)(geometry, initial, seed=91)
    initial[:] = 10
    del geometry
    gc.collect()
    expected = k.draw(20)
    saved = k.state
    # Returned arrays may be read-only, but must never alias live chain state.
    k.draw(20)
    np.testing.assert_array_equal(saved, expected[:, -1])
    k.reset(np.zeros(3))
    k.seed(91)
    np.testing.assert_array_equal(k.draw(20), expected)


@pytest.mark.parametrize("layout", ["C", "F", "slice"])
def test_draw_into_zero_copy(layout):
    a, b = chain(), chain()
    if layout == "slice":
        backing = np.full((6, 40), 99.0)
        out = backing[::2, ::2]
    else:
        out = np.empty((3, 20), order=layout)
    address = out.ctypes.data
    core.draw_into(a, out, thinning=2)
    assert out.ctypes.data == address
    np.testing.assert_array_equal(out, b.draw(20, thinning=2))
    if layout == "slice":
        assert np.all(backing[1::2] == 99)
        assert np.all(backing[:, 1::2] == 99)


def test_invalid_output_and_counts_do_not_advance_chain():
    k = chain()
    for count, thinning in [(-1, 1), (2, 0), (2, -1)]:
        with pytest.raises(ValueError):
            k.draw(count, thinning)
    for out in [np.zeros((2, 10)), np.zeros((3, 10))[:, ::-1]]:
        with pytest.raises(ValueError):
            k.draw_into(out)
    for out in [np.zeros((3, 10), dtype=np.float32), np.zeros((3, 10), dtype=int)]:
        with pytest.raises(TypeError):
            k.draw_into(out)
    readonly = np.zeros((3, 10))
    readonly.flags.writeable = False
    with pytest.raises(TypeError):
        k.draw_into(readonly)
    assert k.draw(0).shape == (3, 0)
    k.draw_into(np.empty((3, 0)))
    np.testing.assert_array_equal(k.state, np.zeros(3))
    np.testing.assert_array_equal(k.draw(10), chain().draw(10))


@pytest.mark.parametrize(
    "initial", [np.zeros(2), np.array([0.0, 0.0, np.nan]), np.array([0.0, 0.0, 2.0])]
)
def test_invalid_initial_state(initial):
    with pytest.raises(ValueError):
        core.UniformHitAndRunKernel(box(), initial)


def test_geometry_and_gaussian_validation():
    with pytest.raises(ValueError):
        core.LinearPolytope(np.ones((2, 3)), np.ones(3))
    with pytest.raises(ValueError):
        core.LinearPolytope(np.array([[np.nan]]), np.ones(1))
    for sigma in [0, -1, np.nan, np.inf]:
        with pytest.raises(ValueError):
            core.GaussianHitAndRunKernel(box(), np.zeros(3), standard_deviation=sigma)
    g = box()
    assert g.dimension == 3 and g.num_constraints == 6
    assert g.contains(np.zeros(3)) and not g.contains(np.ones(3) * 2)
    a = g.A
    g2 = core.LinearPolytope(a, g.b)
    del g
    assert g2.contains(np.zeros(3))


@pytest.mark.parametrize("name", NAMES)
def test_python_nonuniform_target(name):
    class Tilted:
        def log_density(self, x):
            return float(x[0])

    k = chain(name, n=2)
    x = k.draw(15000, thinning=2, target=Tilted())
    assert abs(x[0].mean() - (1 / np.tanh(1.0) - 1)) < 0.045
    assert abs(x[1].mean()) < 0.045
    assert np.max(np.abs(x)) <= 1 + 1e-11


@pytest.mark.parametrize("ratio", [float("nan"), -float("inf")])
def test_target_rejection_and_callback_snapshots(ratio):
    saved = []

    class Reject:
        def log_density(self, x):
            saved.append(x)
            return 0.0 if np.all(x == 0) else ratio

    k = chain()
    assert not k.step(Reject())
    x = k.draw(10, target=Reject())
    assert np.all(x == 0)
    copies = [a.copy() for a in saved]
    k.draw(30)
    for a, b in zip(saved, copies):
        np.testing.assert_array_equal(a, b)


def test_callback_exception_recovery_and_reentrancy():
    k = chain()

    class Failing:
        def log_density(self, x):
            raise LookupError("target failed")

    with pytest.raises(LookupError, match="target failed"):
        k.step(Failing())
    assert np.all(k.state == 0)
    assert k.step()

    class Reentrant:
        def log_density(self, x):
            k.step()
            return 0.0

    with pytest.raises(RuntimeError, match="already in use"):
        k.step(Reentrant())
    assert k.step()


def test_independent_native_chains_in_threads():
    seeds = [14, 15, 16, 17]

    def run(seed):
        return chain(n=8, seed=seed).draw(10000, thinning=3)

    expected = [run(seed) for seed in seeds]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(run, seeds))
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_concurrent_access_to_same_chain_is_rejected():
    from threading import Event

    entered, resume = Event(), Event()
    k = chain()

    class BlockingTarget:
        def log_density(self, x):
            entered.set()
            if not resume.wait(5):
                raise TimeoutError("test synchronization timed out")
            return 0.0

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(k.step, BlockingTarget())
        try:
            assert entered.wait(5)
            with pytest.raises(RuntimeError, match="already in use"):
                k.step()
            with pytest.raises(RuntimeError, match="already in use"):
                k.reset(np.zeros(3))
        finally:
            resume.set()
        assert future.result(timeout=5)
    assert k.step()


def test_python_target_cannot_mutate_live_state():
    class MutatingTarget:
        def log_density(self, x):
            # Either a writable detached copy or a read-only detached copy is safe.
            try:
                x[:] = 1000.0
            except ValueError:
                pass
            return 0.0

    k = chain()
    x = k.draw(100, target=MutatingTarget())
    assert np.max(np.abs(x)) <= 1 + 1e-12


def test_invalid_reset_preserves_state_and_sampling_sequence():
    a, b = chain(), chain()
    with pytest.raises(ValueError):
        a.reset(np.full(3, 2.0))
    np.testing.assert_array_equal(a.draw(30), b.draw(30))


def test_uniform_subclass_override_is_respected():
    class Reject(core.UniformTarget):
        def log_density(self, x):
            return 0.0 if np.all(x == 0) else -float("inf")

    k = chain()
    assert not k.step(Reject())
    assert np.all(k.state == 0)
