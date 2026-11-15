import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import _hopsy_core as core
import numpy as np
import pytest
from test_bindings import GAUSSIAN, box, make


class Counted:
    def __init__(self, mean=0.0):
        self.mean = mean
        self.densities = self.gradients = 0

    def log_density(self, x):
        self.densities += 1
        return -0.5 * np.sum((x - self.mean) ** 2)

    def log_gradient(self, x):
        self.gradients += 1
        return self.mean - x


@pytest.mark.parametrize("mapped", [False, True])
def test_csmala_caches_survive_python_batches_and_rebinding_releases_target(mapped):
    target = Counted()
    old = weakref.ref(target)
    g = box(2, 100.0)
    coordinates = core.AffineCoordinates(np.eye(2), np.zeros(2)) if mapped else None
    p = core.Problem(g, target, coordinates)
    assert p.geometry is g and p.target is target
    if mapped:
        assert p.coordinates is coordinates
    chain = core.bind_chain(
        p, core.CSmMALAKernel(np.zeros(2), step_size=0.001), np.zeros(2)
    )
    rng = core.RandomNumberGenerator(4)
    chain.draw(rng, 4)
    assert (target.densities, target.gradients) == (5, 5)
    chain.draw(rng, 4)
    assert (target.densities, target.gradients) == (9, 9)
    replacement = Counted(0.5)
    p2 = core.Problem(g, replacement, coordinates)
    chain.rebind_problem(p2)
    del p, target, p2
    gc.collect()
    assert old() is None
    chain.step(rng)
    assert (replacement.densities, replacement.gradients) == (2, 2)
    # The currently bound problem is retained by C++ shared ownership.
    chain.draw(rng, 2)
    assert replacement.densities == 4


@pytest.mark.parametrize("name", GAUSSIAN + ["UniformCoordinateHitAndRunKernel"])
def test_rebinding_changes_distribution_without_changing_geometry(name):
    g = box(2, 6.0)
    p = core.Problem(g, core.GaussianTarget(np.zeros(2), np.eye(2)))
    initial = np.zeros(2)
    chain = core.bind_chain(p, getattr(core, name)(initial), initial)
    rng = core.RandomNumberGenerator(8)
    chain.draw(rng, 10)
    before = chain.sampling_state
    new_target = core.GaussianTarget(np.ones(2), np.eye(2))
    with pytest.raises(ValueError):
        chain.rebind_problem(core.Problem(box(2, 6.0), new_target))
    np.testing.assert_array_equal(chain.sampling_state, before)
    chain.rebind_problem(core.Problem(g, new_target))
    chain.draw(rng, 3000)
    samples = chain.draw(rng, 25000, thinning=3)
    assert np.max(np.abs(samples.mean(axis=1) - 1.0)) < 0.1


@pytest.mark.parametrize("name", GAUSSIAN + ["UniformCoordinateHitAndRunKernel"])
def test_callback_failure_cancels_pending_transition(name):
    class Target(Counted):
        fail = True

        def log_density(self, x):
            if self.fail:
                raise RuntimeError("deliberate callback failure")
            return super().log_density(x)

    t = Target()
    chain, rng = make(name, target=t)
    with pytest.raises(RuntimeError, match="deliberate"):
        chain.step(rng)
    np.testing.assert_array_equal(chain.sampling_state, np.zeros(3))
    t.fail = False  # Restore the same mathematical density, not a target update.
    assert np.all(np.isfinite(chain.draw(rng, 100)))


def test_callbacks_cannot_mutate_or_retain_core_state():
    class Target(Counted):
        retained = []

        def log_gradient(self, x):
            value = -x.copy()
            self.retained.append(x)
            x[:] = 100.0
            return value

        def log_density(self, x):
            value = -0.5 * x.dot(x)
            self.retained.append(x)
            x[:] = 100.0
            return value

    t = Target()
    a, ra = make("CSmMALAKernel", target=t)
    b, rb = make("CSmMALAKernel", target=Counted())
    np.testing.assert_array_equal(a.draw(ra, 100), b.draw(rb, 100))
    for x in t.retained:
        x[:] = -100.0
    assert np.max(np.abs(a.sampling_state)) <= 3.0


def test_reentrant_and_shared_rng_access_fail_without_deadlock():
    entered, resume = threading.Event(), threading.Event()

    class Blocking(Counted):
        def log_density(self, x):
            entered.set()
            assert resume.wait(5.0)
            return super().log_density(x)

    a, rng = make("GaussianRandomWalkKernel", target=Blocking())
    b, _ = make("GaussianRandomWalkKernel")
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(a.step, rng)
        try:
            assert entered.wait(5.0)
            with pytest.raises(RuntimeError):
                a.step(core.RandomNumberGenerator(1))
            with pytest.raises(RuntimeError):
                b.step(rng)
            with pytest.raises(RuntimeError):
                rng.seed(2)
        finally:
            resume.set()
        future.result()
    rng.seed(2)


def test_bad_gradients_fail_and_lease_is_released():
    class Bad(Counted):
        def log_gradient(self, x):
            return np.zeros(x.size + 1)

    chain, rng = make("CSmMALAKernel", target=Bad())
    with pytest.raises(ValueError, match="gradient dimension"):
        chain.step(rng)
    # Recovery is allowed with the same geometry; use a fresh chain here.
    other, _ = make("GaussianRandomWalkKernel")
    other.step(rng)


def test_native_target_subclass_overrides_use_python_adapter():
    class Overridden(core.GaussianTarget):
        def log_density(self, x):
            raise RuntimeError("Python override was evaluated")

    target = Overridden(np.zeros(3), np.eye(3))
    chain, rng = make("GaussianRandomWalkKernel", target=target)
    with pytest.raises(RuntimeError, match="Python override"):
        chain.step(rng)


def test_callback_can_reference_chain_weakly_without_an_ownership_cycle():
    class Observed(Counted):
        chain = None

        def log_density(self, x):
            assert self.chain() is not None
            return super().log_density(x)

    target = Observed()
    chain, rng = make("GaussianRandomWalkKernel", target=target)
    target.chain = weakref.ref(chain)
    chain.step(rng)
    del chain
    gc.collect()
    assert target.chain() is None


def test_exception_partway_through_batch_keeps_completed_transitions():
    class Interrupted(Counted):
        def log_density(self, x):
            if self.densities == 4:
                raise RuntimeError("interrupted batch")
            return super().log_density(x)

    target = Interrupted()
    g = box(2, 100.0)
    initial = np.zeros(2)
    p = core.Problem(g, target)
    c = core.bind_chain(p, core.CSmMALAKernel(initial, step_size=0.001), initial)
    reference = core.bind_chain(
        core.Problem(g, Counted()),
        core.CSmMALAKernel(initial, step_size=0.001),
        initial,
    )
    rng, expected_rng = core.RandomNumberGenerator(6), core.RandomNumberGenerator(6)
    reference.draw(expected_rng, 3)
    with pytest.raises(RuntimeError, match="interrupted batch"):
        c.draw(rng, 20)
    np.testing.assert_array_equal(c.sampling_state, reference.sampling_state)
    c.rebind_problem(core.Problem(g, Counted()))
    assert np.isfinite(c.draw(rng, 20)).all()
