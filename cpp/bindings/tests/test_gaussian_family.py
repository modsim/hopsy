from concurrent.futures import ThreadPoolExecutor

import _hopsy_core as core
import numpy as np
import pytest

NAMES = ["GaussianRandomWalkKernel", "DikinWalkKernel", "CSmMALAKernel"]


def make(name, seed=42):
    n = 3
    geometry = core.LinearPolytope(
        np.vstack((np.eye(n), -np.eye(n))), np.full(2 * n, 3.0)
    )
    return getattr(core, name)(geometry, np.zeros(n), seed=seed, step_size=0.3)


def target():
    return core.GaussianTarget(np.zeros(3), np.eye(3))


@pytest.mark.parametrize("name", NAMES)
def test_scale_reset_reseed_and_draw_into(name):
    a, b = make(name), make(name)
    density = target()
    a.step_size = 0.4
    b.step_size = 0.4
    for bad in [0.0, -1.0, np.nan, np.inf]:
        with pytest.raises(ValueError):
            a.step_size = bad
        assert a.step_size == 0.4
    out = np.empty((3, 100), order="F")
    a.draw_into(out, target=density)
    np.testing.assert_array_equal(out, b.draw(100, target=density))
    saved = a.state
    a.draw(10, target=density)
    np.testing.assert_array_equal(saved, out[:, -1])
    a.reset(np.zeros(3))
    a.seed(42)
    np.testing.assert_array_equal(a.draw(100, target=density), out)
    before = a.state
    with pytest.raises(ValueError):
        a.reset(np.ones(2))
    np.testing.assert_array_equal(a.state, before)


@pytest.mark.parametrize("name", NAMES)
def test_native_target_parallel_reproducibility(name):
    density = target()
    serial = [make(name, i).draw(3000, target=density) for i in range(4)]
    chains = [make(name, i) for i in range(4)]
    with ThreadPoolExecutor(4) as pool:
        parallel = list(pool.map(lambda c: c.draw(3000, target=density), chains))
    for x, y in zip(serial, parallel):
        np.testing.assert_array_equal(x, y)
        assert np.max(np.abs(y)) <= 3.0


@pytest.mark.parametrize("name", NAMES)
def test_python_gradient_target_and_recovery(name):
    class PythonTarget:
        def log_density(self, x):
            return -0.5 * x.dot(x)

        def log_gradient(self, x):
            return -x

    class Broken(PythonTarget):
        def log_density(self, x):
            raise RuntimeError("test target failure")

    chain = make(name)
    with pytest.raises(RuntimeError, match="test target failure"):
        chain.draw(10, target=Broken())
    samples = chain.draw(100, target=PythonTarget())
    assert samples.shape == (3, 100)
    assert np.isfinite(samples).all()


def test_native_gaussian_density_and_input_validation():
    density = target()
    x = np.array([0.25, -0.5, 1.0])
    np.testing.assert_allclose(density.log_gradient(x), -x)
    assert density.log_density(x) == pytest.approx(
        -1.5 * np.log(2 * np.pi) - 0.5 * x.dot(x)
    )
    with pytest.raises(ValueError):
        core.GaussianTarget(np.zeros(3), np.zeros((3, 3)))
    g = core.LinearPolytope(np.vstack((np.eye(3), -np.eye(3))), np.ones(6))
    for name in ["DikinWalkKernel", "CSmMALAKernel"]:
        with pytest.raises(ValueError):
            getattr(core, name)(g, np.ones(3))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("entry", ["draw", "step"])
def test_changed_python_targets_start_with_fresh_cache(name, entry):
    class MutableTarget:
        mean = 0.0

        def log_density(self, x):
            return -0.5 * np.sum((x - self.mean) ** 2)

        def log_gradient(self, x):
            return self.mean - x

    density = MutableTarget()
    a, b = make(name), make(name)
    np.testing.assert_array_equal(
        a.draw(50, target=density), b.draw(50, target=density)
    )
    density.mean = 0.8
    # Rebuild only the reference context while preserving its RNG position.
    b.reset(b.state)
    if entry == "draw":
        np.testing.assert_array_equal(
            a.draw(100, target=density), b.draw(100, target=density)
        )
    else:
        for _ in range(20):
            density.mean += 0.01
            b.reset(b.state)
            assert a.step(density) == b.step(density)
            np.testing.assert_array_equal(a.state, b.state)
