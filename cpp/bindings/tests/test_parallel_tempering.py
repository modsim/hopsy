"""Exercise the native NRPT sampler through the nanobind ownership boundary."""

import gc
import time
from concurrent.futures import ThreadPoolExecutor

import _capsule_gaussian as native
import _hopsy_core as core
import numpy as np
import pytest

KERNELS = [
    f"{line}{direction}HitAndRunKernel"
    for line in ("Uniform", "Gaussian", "OverRelaxed")
    for direction in ("", "Coordinate", "RandomCoordinate")
] + ["GaussianRandomWalkKernel", "DikinWalkKernel", "CSmMALAKernel"]


class CountedDensity(core.PythonTarget):
    def __init__(self, offset=0.0):
        super().__init__()
        self.evaluated = []
        self.offset = offset

    def log_density(self, x):
        value = self.offset - x @ x
        self.evaluated.append(value)
        return value

    def log_gradient(self, x):
        return -2 * x


def box():
    return core.LinearPolytope(np.vstack((np.eye(2), -np.eye(2))), np.ones(4) * 3)


def bind(
    problem,
    name="UniformHitAndRunKernel",
    states=None,
    ladder=None,
    replica_targets=(),
    reference_basis=None,
    reference_max_beta=0.0,
):
    if ladder is None:
        ladder = core.TemperatureLadder.linear(4)
    if states is None:
        states = [np.array([slot / 4, 0.0]) for slot in range(len(ladder))]
    return core.bind_parallel_tempering(
        problem=problem,
        ladder=ladder,
        kernels=[getattr(core, name)(x) for x in states],
        initial_states=states,
        rngs=[
            core.RandomNumberGenerator(123, stream=slot) for slot in range(len(ladder))
        ],
        exchange_rng=core.RandomNumberGenerator(134, stream=len(ladder)),
        replica_targets=replica_targets,
        reference_basis=reference_basis,
        reference_max_beta=reference_max_beta,
    )


@pytest.mark.parametrize("name", KERNELS)
def test_seeded_draw_matches_split_exploration(name):
    problem = core.Problem(box(), core.GaussianTarget(np.zeros(2), np.eye(2)))
    batched = bind(problem, name)
    split = bind(problem, name)
    samples = batched.draw(32, local_steps=2, reference_steps=3)

    expected = []
    for _ in range(samples.shape[1]):
        for slot in range(len(split.ladder)):
            split.explore(slot, 3 if slot == 0 else 2)
        expected.append(split.target_state(3))
        split.communicate()

    np.testing.assert_array_equal(samples, np.column_stack(expected))
    assert samples.shape == (2, 32)
    assert np.max(np.abs(samples)) < 3
    assert batched.replica_indices == split.replica_indices
    assert batched.mean_exchange_rejections() == split.mean_exchange_rejections()
    assert batched.round_trips == split.round_trips
    assert [(e.attempts, e.accepts) for e in batched.exchange_statistics] == [
        (e.attempts, e.accepts) for e in split.exchange_statistics
    ]


def test_parity_and_schedule_adaptation():
    sampler = bind(core.Problem(box(), core.UniformTarget()))
    sampler.communicate()
    assert sampler.replica_indices == [0, 2, 1, 3]
    sampler.communicate()
    assert sampler.replica_indices == [2, 0, 3, 1]
    assert [(e.attempts, e.accepts) for e in sampler.exchange_statistics] == [
        (1, 1)
    ] * 3

    ladder = sampler.ladder
    tuned = ladder.adapted([0.2, 0.05, 0.4])
    assert tuned.inverse_temperatures != ladder.inverse_temperatures
    sampler.set_ladder(tuned)
    assert sampler.ladder.inverse_temperatures == tuned.inverse_temperatures
    assert sampler.replica_indices == [2, 0, 3, 1]
    assert sampler.round_trips == [0] * 4
    assert [e.attempts for e in sampler.exchange_statistics] == [0] * 3
    sampler.scan(1, 1)
    sampler.reset_statistics()
    with pytest.raises(RuntimeError, match="Mean exchange rejection"):
        sampler.mean_exchange_rejections()


def test_native_capsule_ownership_and_affine_outputs():
    gc.collect()
    before = native.destroyed()
    target = core.CapsuleTarget(native.make(1))
    offsets = np.array([0.2, 0.3])
    coords = core.AffineCoordinates(np.eye(2) * 2, offsets)
    problem = core.Problem(box(), target, coords)
    sampler = bind(problem, "CSmMALAKernel")
    del target, coords, problem
    gc.collect()
    assert native.destroyed() == before
    state = sampler.sampling_state(3)
    np.testing.assert_allclose(sampler.target_state(3), state * 2 + offsets)
    state[:] = 10
    assert np.all(sampler.sampling_state(3) < 3)
    assert sampler.draw(4, 1, 1).shape == (2, 4)
    del sampler
    gc.collect()
    assert native.destroyed() == before + 1


def test_python_target_and_missing_gradient():
    class Density(core.PythonTarget):
        def log_density(self, x):
            return -x @ x

    problem = core.Problem(box(), Density())
    assert bind(problem).draw(10, 1, 1).shape == (2, 10)
    with pytest.raises(TypeError, match="log_gradient"):
        bind(problem, "CSmMALAKernel")


@pytest.mark.parametrize("name", KERNELS)
def test_standalone_caches_current_and_proposed_log_densities(name):
    target = CountedDensity()
    geometry = box()
    state = np.zeros(2)
    chain = core.bind_chain(
        core.Problem(geometry, target), getattr(core, name)(state), state
    )
    rng = core.RandomNumberGenerator(188)

    assert chain.state_log_density is chain.proposal_log_density is None
    for _ in range(64):
        old_calls = len(target.evaluated)
        old_density = chain.state_log_density
        accepted = chain.step(rng)
        calls = target.evaluated[old_calls:]
        proposal = chain.proposal_log_density
        if proposal is None:  # Geometrically infeasible.
            assert calls == []
            assert not accepted
        else:
            assert len(calls) == (2 if old_density is None else 1)
            assert proposal == calls[-1]
        if accepted:
            assert chain.state_log_density == proposal
        elif old_density is not None:
            assert chain.state_log_density == old_density
        if chain.state_log_density is not None:
            x = chain.target_state
            assert chain.state_log_density == pytest.approx(-x @ x)
    assert chain.state_log_density is not None

    old_calls = len(target.evaluated)
    draws = chain.draw(rng, 10, thinning=2)
    new_calls = len(target.evaluated) - old_calls
    assert 0 < new_calls <= 20
    x = draws[:, -1]
    assert chain.state_log_density == pytest.approx(-x @ x)


def test_density_cache_starts_fresh_after_rebind():
    geometry = box()
    target = CountedDensity()
    state = np.zeros(2)
    chain = core.bind_chain(
        core.Problem(geometry, target), core.UniformHitAndRunKernel(state), state
    )
    rng = core.RandomNumberGenerator(42)
    chain.step(rng)
    assert chain.state_log_density is not None
    replacement = CountedDensity(offset=10.0)
    chain.rebind_problem(core.Problem(geometry, replacement))
    assert chain.state_log_density is chain.proposal_log_density is None
    assert replacement.evaluated == []
    chain.step(rng)
    assert len(replacement.evaluated) == 2
    x = chain.sampling_state
    assert chain.state_log_density == pytest.approx(10.0 - x @ x)


def test_density_cache_preserves_state_after_rejection_and_callback_failure():
    class RestrictedDensity(CountedDensity):
        fail = False

        def log_density(self, x):
            if self.fail:
                raise RuntimeError("density evaluation failed")
            return -np.inf if x.any() else super().log_density(x)

    target = RestrictedDensity()
    state = np.zeros(2)
    chain = core.bind_chain(
        core.Problem(box(), target), core.UniformHitAndRunKernel(state), state
    )
    rng = core.RandomNumberGenerator(42)

    assert not chain.step(rng)
    assert chain.state_log_density == 0.0
    assert chain.proposal_log_density == -np.inf
    target.fail = True
    with pytest.raises(RuntimeError, match="density evaluation failed"):
        chain.step(rng)
    assert chain.state_log_density == 0.0
    assert chain.proposal_log_density is None
    target.fail = False
    assert not chain.step(rng)
    assert chain.proposal_log_density == -np.inf
    assert target.evaluated == [0.0]


@pytest.mark.parametrize("name", KERNELS)
def test_nrpt_reuses_untempered_density_for_exchanges(name):
    target = CountedDensity()
    sampler = bind(core.Problem(box(), target), name)
    assert len(target.evaluated) == 4
    for _ in range(4):
        assert np.isfinite(sampler.state_log_density(0))
        for slot in range(1, 4):
            old_calls = len(target.evaluated)
            sampler.explore(slot, 5)
            assert len(target.evaluated) - old_calls <= 5
            if name.endswith("HitAndRunKernel"):
                assert len(target.evaluated) - old_calls == 5
            if sampler.proposal_log_density(slot) is not None:
                assert sampler.proposal_log_density(slot) == target.evaluated[-1]
        old_calls = len(target.evaluated)
        sampler.explore(0, 5)
        assert len(target.evaluated) - old_calls <= 1
        assert sampler.proposal_log_density(0) is None
        old_calls = len(target.evaluated)
        sampler.communicate()
        for slot in range(4):
            state = sampler.sampling_state(slot)
            assert sampler.state_log_density(slot) == pytest.approx(-state @ state)
        assert len(target.evaluated) == old_calls


def test_bilinear_geometry_support():
    geometry = core.BilinearPolytope(
        np.vstack((np.eye(2), -np.eye(2))),
        np.ones(4),
        1,
        [np.ones((1, 1))],
        np.array([0.5]),
    )
    problem = core.Problem(geometry, core.UniformTarget())
    samples = bind(problem).draw(32, 1, 1)
    assert np.all(np.abs(samples) <= 1 + 1e-12)
    assert np.all(samples[0] * samples[1] <= 0.5 + 1e-12)
    with pytest.raises(TypeError, match="geometry"):
        bind(problem, "DikinWalkKernel")


def test_validation_leaves_existing_resources_usable():
    x = np.zeros(2)
    problem = core.Problem(box(), core.UniformTarget())
    ladder = core.TemperatureLadder.linear(2)
    kernels = [core.UniformHitAndRunKernel(x) for _ in range(2)]
    rngs = [core.RandomNumberGenerator(1, i) for i in range(2)]
    with pytest.raises(ValueError, match="one kernel"):
        core.bind_parallel_tempering(
            problem, ladder, kernels[:1], [x] * 2, rngs, rngs[0]
        )
    with pytest.raises(ValueError, match="distinct RNG streams"):
        core.bind_parallel_tempering(problem, ladder, kernels, [x] * 2, rngs, rngs[0])

    exchange_rng = core.RandomNumberGenerator(1, 2)
    a = core.bind_parallel_tempering(
        problem, ladder, kernels, [x] * 2, rngs, exchange_rng
    )
    b = core.bind_parallel_tempering(
        problem, ladder, kernels, [x] * 2, rngs, exchange_rng
    )
    chain = core.bind_chain(problem, kernels[0], x)
    chain.step(rngs[0])
    np.testing.assert_array_equal(a.draw(40, 1, 1), b.draw(40, 1, 1))
    with pytest.raises(RuntimeError, match="already in use"):
        core.bind_parallel_tempering(
            problem, ladder, kernels, [x] * 2, rngs, exchange_rng
        )


def test_gil_release_and_same_sampler_exclusion():
    native.release()  # Allow the initial target evaluations during binding.
    sampler = bind(core.Problem(box(), core.CapsuleTarget(native.make(0, 9))))
    native.reset()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(sampler.draw, 2, 1, 1)
        deadline = time.monotonic() + 3
        while not native.started() and time.monotonic() < deadline:
            time.sleep(0.001)
        try:
            assert native.started()
            with pytest.raises(RuntimeError, match="already in use"):
                sampler.sampling_state(0)
        finally:
            native.release()
        assert future.result(timeout=3).shape == (2, 2)


@pytest.mark.parametrize("capsule", [False, True])
def test_parallel_matches_serial_with_identical_streams(capsule):
    coordinates = core.AffineCoordinates(np.eye(2) * 1.5, np.array([0.1, -0.2]))

    def setup():
        if capsule:
            replicas = [core.CapsuleTarget(native.make(1)) for _ in range(4)]
            target = replicas[0]
        else:
            replicas = ()
            target = core.GaussianTarget(np.zeros(2), np.eye(2))
        return bind(
            core.Problem(box(), target, coordinates),
            "CSmMALAKernel",
            replica_targets=replicas,
        )

    a, b = setup(), setup()
    for count in (0, 25, 17):
        np.testing.assert_array_equal(a.draw(count, 2, 3), b.draw_parallel(count, 2, 3))
    assert a.replica_indices == b.replica_indices
    assert a.mean_exchange_rejections() == b.mean_exchange_rejections()
    assert a.round_trips == b.round_trips


def test_rounded_reference_matches_coordinate_walk_and_preserves_positive_beta():
    A = np.vstack((np.eye(2), -np.eye(2)))
    b = np.ones(4) * 3
    basis = np.array([[0.5, -0.3], [1.1, 1.7]])
    start = np.array([0.1, -0.2])
    target = core.GaussianTarget(np.zeros(2), np.eye(2))
    problem = core.Problem(core.LinearPolytope(A, b), target)

    sampler = bind(problem, "CSmMALAKernel", states=[start] * 4, reference_basis=basis)
    cold_control = bind(problem, "CSmMALAKernel", states=[start] * 4)
    rounded_start = np.zeros(2)
    rounded = core.Problem(
        core.LinearPolytope(A @ basis, b - A @ start),
        core.UniformTarget(),
        core.AffineCoordinates(basis, start),
    )
    walk = core.bind_chain(
        rounded,
        core.UniformRandomCoordinateHitAndRunKernel(rounded_start),
        rounded_start,
    )
    reference_samples = walk.draw(
        core.RandomNumberGenerator(123, stream=0), 40, thinning=3
    )
    for expected in reference_samples.T:
        assert sampler.explore(0, 3) == 3
        np.testing.assert_allclose(sampler.target_state(0), expected, atol=1e-12)
        assert sampler.state_log_density(0) == pytest.approx(
            target.log_density(expected)
        )
        assert sampler.proposal_log_density(0) is None
    assert sampler.explore(3, 11) == cold_control.explore(3, 11)
    np.testing.assert_array_equal(sampler.target_state(3), cold_control.target_state(3))


def test_rounded_reference_parallel_trace_matches_serial_through_exchanges():
    basis = np.array([[1.0, 0.2], [0.3, 0.5]])
    target = core.GaussianTarget(np.zeros(2), np.eye(2))
    problem = core.Problem(box(), target)
    serial = bind(
        problem, "CSmMALAKernel", reference_basis=basis, reference_max_beta=0.5
    )
    parallel = bind(
        problem, "CSmMALAKernel", reference_basis=basis, reference_max_beta=0.5
    )
    expected = serial.draw_traced(160, 2, 5)
    actual = parallel.draw_traced(160, 2, 5, parallel=True, workers=2)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    samples, ids, densities = actual
    destination = np.argmax(ids[:, 1:] == ids[-1, :-1], axis=0)
    np.testing.assert_allclose(
        densities[destination, np.arange(1, 161)],
        [target.log_density(x) for x in samples.T],
    )
    assert np.max(np.abs(samples)) < 3
    assert sum(serial.round_trips) > 0
    assert serial.round_trips == parallel.round_trips


def test_rounded_coordinate_kernel_targets_power_at_inclusive_threshold():
    A = np.vstack((np.eye(2), -np.eye(2)))
    b = np.ones(4) * 3
    basis = np.array([[1.2, 0.5], [0.3, 0.4]])
    start = np.array([0.1, -0.2])
    ladder = core.TemperatureLadder([0.0, 0.5, 0.75, 1.0])
    target = core.GaussianTarget(np.zeros(2), np.eye(2))
    problem = core.Problem(core.LinearPolytope(A, b), target)
    sampler = bind(
        problem,
        "CSmMALAKernel",
        states=[start] * 4,
        ladder=ladder,
        reference_basis=basis,
        reference_max_beta=0.5,
    )
    control = bind(problem, "CSmMALAKernel", states=[start] * 4, ladder=ladder)
    coords = core.AffineCoordinates(basis, start)
    powered = core.GaussianTarget(np.zeros(2), 2 * np.eye(2))
    rounded = core.Problem(
        core.LinearPolytope(A @ basis, b - A @ start), powered, coords
    )
    zero = np.zeros(2)
    walk = core.bind_chain(
        rounded, core.UniformRandomCoordinateHitAndRunKernel(zero), zero
    )
    rng = core.RandomNumberGenerator(123, stream=1)
    rejected = 0
    for _ in range(40):
        accepted = sampler.explore(1)
        assert accepted == walk.step(rng)
        rejected += 1 - accepted
        np.testing.assert_allclose(
            sampler.target_state(1), walk.target_state, atol=1e-12
        )
        assert sampler.state_log_density(1) == pytest.approx(
            target.log_density(walk.target_state)
        )
    assert rejected > 0
    assert sampler.explore(2, 9) == control.explore(2, 9)
    np.testing.assert_array_equal(sampler.target_state(2), control.target_state(2))


def test_per_slot_step_counts_match_manual_mixed_exploration():
    problem = core.Problem(box(), core.GaussianTarget(np.zeros(2), np.eye(2)))
    steps = [10, 5, 1, 1]

    def setup():
        return bind(
            problem, "CSmMALAKernel", reference_basis=np.eye(2), reference_max_beta=0.5
        )

    manual = setup()
    expected = []
    for _ in range(24):
        for slot, count in enumerate(steps):
            manual.explore(slot, count)
        expected.append(manual.target_state(3))
        manual.communicate()
    a = setup().draw_parallel(24, workers=2, steps_per_slot=steps)
    b = setup().draw_traced(24, steps_per_slot=steps)
    c = setup().draw_traced(24, parallel=True, workers=2, steps_per_slot=steps)
    for samples in (a, b[0], c[0]):
        np.testing.assert_array_equal(samples, np.column_stack(expected))
    for serial, parallel in zip(b[1:], c[1:]):
        np.testing.assert_array_equal(serial, parallel)


@pytest.mark.parametrize("steps", [[], [10, 1], [10, 10, 0, 1], [10, 10, -1, 1]])
def test_per_slot_step_counts_require_one_positive_count_for_each_slot(steps):
    sampler = bind(core.Problem(box(), core.UniformTarget()))
    for draw in (sampler.draw_parallel, sampler.draw_traced):
        with pytest.raises(ValueError, match="one positive step count per slot"):
            draw(0, steps_per_slot=steps)
    assert sampler.replica_indices == list(range(4))


def test_csmmala_rebuilds_state_cache_after_hot_reference_moves_replica():
    geometry = box()
    problem = core.Problem(geometry, core.UniformTarget())
    ladder = core.TemperatureLadder.linear(2)
    initial = np.zeros(2)
    sampler = bind(
        problem,
        "CSmMALAKernel",
        states=[initial] * 2,
        ladder=ladder,
        reference_basis=np.eye(2),
    )
    rng = core.RandomNumberGenerator(123, stream=1)
    warm = core.bind_chain(problem, core.CSmMALAKernel(initial), initial)
    sampler.explore(1, 3)
    first = warm.draw(rng, 3)[:, -1]
    np.testing.assert_array_equal(sampler.sampling_state(1), first)
    sampler.communicate()
    sampler.communicate()
    assert sampler.replica_indices == [1, 0]

    hot = core.bind_chain(
        problem, core.UniformRandomCoordinateHitAndRunKernel(first), first
    )
    after_hot = hot.draw(rng, 6)[:, -1]
    assert sampler.explore(0, 6) == 6
    np.testing.assert_allclose(sampler.sampling_state(0), after_hot, atol=1e-14)
    sampler.communicate()
    sampler.communicate()
    assert sampler.replica_indices == [0, 1]

    fresh = core.bind_chain(problem, core.CSmMALAKernel(after_hot), after_hot)
    sampler.explore(1, 3)
    np.testing.assert_allclose(
        sampler.sampling_state(1), fresh.draw(rng, 3)[:, -1], atol=1e-14
    )


@pytest.mark.parametrize("basis", [np.zeros((2, 2)), np.eye(3), np.ones((2, 1))])
def test_rounded_reference_validates_basis(basis):
    problem = core.Problem(box(), core.GaussianTarget(np.zeros(2), np.eye(2)))
    with pytest.raises(ValueError, match="Reference basis"):
        bind(problem, "CSmMALAKernel", reference_basis=basis)


@pytest.mark.parametrize("parallel", [False, True])
def test_replica_trace_tracks_exchanges_and_cached_log_density(parallel):
    target = core.GaussianTarget(np.zeros(2), np.eye(2))
    problem = core.Problem(box(), target)
    sampler = bind(problem)
    untouched = bind(problem)
    samples, ids, log_densities = sampler.draw_traced(0, parallel=parallel)
    assert samples.shape == (2, 0)
    assert ids[:, 0].tolist() == list(range(4))
    assert log_densities.shape == (4, 1)

    samples, ids, log_densities = sampler.draw_traced(40, 2, 3, parallel=parallel)
    draw = untouched.draw_parallel if parallel else untouched.draw
    np.testing.assert_array_equal(samples, draw(40, 2, 3))
    assert ids.shape == log_densities.shape == (4, 41)
    np.testing.assert_array_equal(
        np.sort(ids, axis=0), np.tile(np.arange(4)[:, None], (1, 41))
    )
    assert ids[:, -1].tolist() == sampler.replica_indices

    # Each cold draw is recorded before the exchange, so look up its replica
    # in the following post-exchange snapshot to find its cached log density.
    scans = np.arange(40)
    destination = np.argmax(ids[:, 1:] == ids[-1, :-1], axis=0)
    np.testing.assert_allclose(
        log_densities[destination, scans + 1],
        [target.log_density(x) for x in samples.T],
        atol=1e-14,
    )


def test_parallel_capsule_context_validation():
    target = core.CapsuleTarget(native.make(1))
    problem = core.Problem(box(), target)
    sampler = bind(problem)
    with pytest.raises(ValueError, match="one native capsule per replica"):
        sampler.draw_parallel(2)
    with pytest.raises(ValueError, match="one private capsule target"):
        bind(problem, replica_targets=[target])
    with pytest.raises(ValueError, match="distinct native contexts"):
        bind(problem, replica_targets=[target] * 4)


def test_parallel_native_callbacks_overlap():
    native.release()  # Binding evaluates each capsule once.
    targets = [core.CapsuleTarget(native.make(0, 9)) for _ in range(4)]
    sampler = bind(core.Problem(box(), targets[0]), replica_targets=targets)
    del targets
    native.reset()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(sampler.draw_parallel, 1, 1, 1)
        deadline = time.monotonic() + 3
        while native.calls_started() < 4 and time.monotonic() < deadline:
            time.sleep(0.001)
        try:
            assert native.calls_started() == 4
            with pytest.raises(RuntimeError, match="already in use"):
                sampler.communicate()
        finally:
            native.release()
        assert future.result(timeout=3).shape == (2, 1)


def test_parallel_worker_failure_is_propagated_before_exchange():
    targets = [core.CapsuleTarget(native.make(1)) for _ in range(3)]
    targets.append(core.CapsuleTarget(native.make(1, 10)))
    sampler = bind(
        core.Problem(box(), targets[0]), "CSmMALAKernel", replica_targets=targets
    )
    with pytest.raises(RuntimeError, match="19"):
        sampler.draw_parallel(3, 1, 1)
    assert sampler.replica_indices == list(range(4))
    assert [edge.attempts for edge in sampler.exchange_statistics] == [0, 0, 0]
