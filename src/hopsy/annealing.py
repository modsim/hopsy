"""

"""


class _core:
    from .core import (
        MarkovChain,
        Problem,
        Proposal,
        RandomNumberGenerator,
        Uniform,
        UniformCoordinateHitAndRunProposal,
        multiprocessing_context,
        round,
        transform,
    )
    from .definitions import multiprocessing_context


_c = _core


class _submodules:
    import os
    from dataclasses import dataclass
    from multiprocessing.managers import SharedMemoryManager
    from multiprocessing.shared_memory import SharedMemory
    from typing import Dict, List, Optional, Tuple

    import numpy
    from numpy.typing import ArrayLike
    from scipy.interpolate import PchipInterpolator

    if "JPY_PARENT_PID" in os.environ:
        import tqdm.notebook as tqdm
    else:
        import tqdm


_s = _submodules


@_s.dataclass
class SharedMemoryArray:
    name: str
    shared_memory: _s.SharedMemory
    shape: _s.Tuple[int, ...]
    dtype: _s.numpy.dtype

    def view(self) -> _s.ArrayLike:
        return _s.numpy.ndarray(
            self.shape, dtype=_s.numpy.dtype(self.dtype), buffer=self.shared_memory.buf
        )


@_s.dataclass
class SharedData:
    samples: SharedMemoryArray
    densities: SharedMemoryArray
    acceptance_rates: SharedMemoryArray
    rejection_rates: SharedMemoryArray
    machine_idx: SharedMemoryArray
    temperatures: SharedMemoryArray
    global_comm_barriers: SharedMemoryArray

    def to_numpy(self) -> _s.Dict[str, _s.ArrayLike]:
        """
        Return a dict mapping each shared-data attribute name
        to its NumPy array view.
        """
        return {
            attr: getattr(self, attr).view().copy()
            for attr in self.__dataclass_fields__
        }


def _initialize_shared_arrays(
    smm: _s.SharedMemoryManager,
    n_chains: int,
    n_rounds: int,
    state_dim: int,
    starting_point: _s.ArrayLike,
) -> SharedData:
    # total number of scan‐steps = sum_{r=0..n_rounds-1} 2**r
    n_scans = sum(2**r for r in range(n_rounds))

    def make_block(name: str, shape: tuple, dtype) -> SharedMemoryArray:
        n_bytes = int(_s.numpy.prod(shape) * _s.numpy.dtype(dtype).itemsize)
        shared_memory: _s.SharedMemory = smm.SharedMemory(size=n_bytes)
        return SharedMemoryArray(
            name=name, shared_memory=shared_memory, shape=shape, dtype=dtype
        )

    samples = make_block(
        "samples", (n_scans + 1, n_chains, state_dim), starting_point.dtype
    )
    densities = make_block("densities", (n_chains, n_chains), _s.numpy.float64)
    acc_rates = make_block(
        "acceptance_rates", (n_scans + 1, n_chains), _s.numpy.float64
    )
    rej_rates = make_block("rejection_rates", (n_rounds, n_chains), _s.numpy.float64)
    machine_idx = make_block("machine_idx", (n_scans + 1, n_chains), _s.numpy.int32)
    temps = make_block("temperatures", (n_rounds, n_chains), _s.numpy.float64)
    global_comm_barriers = make_block(
        "global_comm_barriers", (n_rounds, n_chains), _s.numpy.float64
    )

    # — initialize contents —
    samples_view = samples.view()
    samples_view.fill(0.0)
    samples_view[0] = starting_point.ravel()

    idx_array = machine_idx.view()
    idx_array[0, :] = _s.numpy.arange(n_chains, dtype=_s.numpy.int32)

    temps_view = temps.view()
    temps_view.fill(0.0)
    temps_view[0, :] = _s.numpy.linspace(0.0, 1.0, n_chains)

    return SharedData(
        samples=samples,
        densities=densities,
        acceptance_rates=acc_rates,
        rejection_rates=rej_rates,
        machine_idx=machine_idx,
        temperatures=temps,
        global_comm_barriers=global_comm_barriers,
    )


def _likelihood_idx(n_chains: int, even: bool):
    groups = []
    cur_group = []
    for i in range(n_chains):
        cur_group.append(i)
        if i % 2 != even:
            groups.append(cur_group)
            cur_group = []

        if i == n_chains - 1 and len(cur_group) > 0:
            groups.append(cur_group)

    cur_group_idx = 0
    cur_machine_idx = 0

    samples_idx = []

    while cur_machine_idx < n_chains:
        samples_to_consider_for_current_machine = list(groups[cur_group_idx])

        if cur_group_idx > 0:
            samples_to_consider_for_current_machine += groups[cur_group_idx - 1]
        if cur_group_idx < len(groups) - 1:
            samples_to_consider_for_current_machine += [groups[cur_group_idx + 1][0]]

        if cur_machine_idx == groups[cur_group_idx][-1]:
            cur_group_idx += 1

        cur_machine_idx += 1

        samples_idx.append(samples_to_consider_for_current_machine)

    pairs = []

    for i in range(n_chains):
        for j in samples_idx[i]:
            pairs.append((i, j))
            pairs.append((j, i))

    return list(set(pairs))


def _swap_probability(densities, idx_curr: int, idx_next: int):
    log_likelihood_ratio = (
        densities[idx_curr, idx_next]
        + densities[idx_next, idx_curr]
        - densities[idx_curr, idx_curr]
        - densities[idx_next, idx_next]
    )

    if _s.numpy.isnan(log_likelihood_ratio):
        raise ValueError("log_likelihood_ratio is nan")

    return 1 if log_likelihood_ratio > 0 else min(1, _s.numpy.exp(log_likelihood_ratio))


def _worker_density(
    target_id: int,
    sample_id: int,
    scan_idx: int,
    round_idx: int,
    chain: _c.MarkovChain,
    temperatures: _s.ArrayLike,
    samples: _s.ArrayLike,
    transition_probs: _s.ArrayLike,
):
    transition_probs[target_id, sample_id] = temperatures[
        round_idx, target_id
    ] * chain.model.log_density(samples[scan_idx, sample_id])


def _worker_sample(
    scan_idx: int,
    round_idx: int,
    process_id: int,
    reference_chain: _c.MarkovChain,
    chain: _c.MarkovChain,
    rng: _c.RandomNumberGenerator,
    thinning: int,
    machine_idx: _s.ArrayLike,
    temperatures: _s.ArrayLike,
    samples: _s.ArrayLike,
    acc_rates: _s.ArrayLike,
):
    chain_idx = _s.numpy.where(machine_idx[scan_idx] == process_id)[0][0]
    beta = temperatures[round_idx, chain_idx]

    if beta <= 1e-9:
        thinning_factor = max(len(chain.state), int(len(chain.state) ** 2 // 6))
        reference_chain.state = chain.state
        acceptance_rate, new_state = reference_chain.draw(rng, thinning_factor)
        chain.state = new_state
    else:
        chain.coldness = beta
        acceptance_rate, new_state = chain.draw(rng, thinning)

    # 5. Safe write to shared arrays
    samples[scan_idx, chain_idx] = new_state
    acc_rates[scan_idx, chain_idx] = acceptance_rate


def _launch_worker(
    target,
    reference,
    proposal_cls,
    starting_point,
    seed_base,
    task_queue: _c.multiprocessing_context.JoinableQueue,
    shared_data: SharedData,
    process_id: int,
    proposal_args: _s.Optional[_s.Dict] = None,
    thinning: int = 1,
):
    machine_idx = shared_data.machine_idx.view()
    temperatures = shared_data.temperatures.view()
    transition_probs = shared_data.densities.view()
    samples = shared_data.samples.view()
    acc_rates = shared_data.acceptance_rates.view()

    reference_chain = _c.MarkovChain(reference, _c.UniformCoordinateHitAndRunProposal)

    chain = _c.MarkovChain(target, proposal_cls, starting_point=starting_point)
    if proposal_args:
        for k in proposal_args:
            setattr(chain.proposal, k, proposal_args[k])
    rng = _c.RandomNumberGenerator(seed_base + (process_id + 1) * 100)

    while True:
        task = task_queue.get()

        # Check if you want the work to sample or to compute the transition probabilities
        if len(task) == 2:
            scan_idx, round_idx = task
            _worker_sample(
                scan_idx,
                round_idx,
                process_id,
                reference_chain,
                chain,
                rng,
                thinning,
                machine_idx,
                temperatures,
                samples,
                acc_rates,
            )
        else:
            target_id, sample_id, scan_idx, round_idx = task
            _worker_density(
                target_id,
                sample_id,
                scan_idx,
                round_idx,
                chain,
                temperatures,
                samples,
                transition_probs,
            )

        task_queue.task_done()


def scan(
    scan_idx: int,
    round_idx: int,
    executors: _s.List[_c.multiprocessing_context.JoinableQueue],
    sync_rng: _c.RandomNumberGenerator,
    shared_data: SharedData,
    even: bool = False,
):
    n_chains = len(executors)

    # local step
    for executor in executors:
        executor.put((scan_idx, round_idx))
    for executor in executors:
        executor.join()

    idx_list = _likelihood_idx(n_chains, even)
    for i, j in idx_list:
        executors[i].put((i, j, scan_idx, round_idx))
    for executor in executors:
        executor.join()

    machine_idx = shared_data.machine_idx.view()
    r = shared_data.rejection_rates.view()
    samples = shared_data.samples.view()
    densities = shared_data.densities.view()
    # swap
    for i in range(n_chains - 1):
        alpha = _swap_probability(densities, i, i + 1)
        # swap
        if even == i % 2:
            if _c.Uniform()(sync_rng) < alpha:
                samples_row = samples[scan_idx]
                samples_row[[i, i + 1]] = samples_row[
                    [i + 1, i]
                ]  # this creates an internal copy of the RHS

                idx_row = machine_idx[scan_idx]
                idx_row[[i, i + 1]] = idx_row[[i + 1, i]]
        # update rejection rates
        r_diff = max(1.0 - alpha, 1e-10)
        r[round_idx, i + 1] += r_diff


def _optimize_schedule(rejection_rates: _s.ArrayLike, temperatures: _s.ArrayLike):
    n_chains = len(temperatures)
    temperatures_next = _s.numpy.zeros(n_chains)
    temperatures_next[-1] = 1

    cum_rejection_rates = rejection_rates.cumsum()
    comm_barrier = cum_rejection_rates[-1]
    interpolator = _s.PchipInterpolator(
        cum_rejection_rates, temperatures, extrapolate=True
    )
    for k in range(1, n_chains - 1):
        temperatures_next[k] = interpolator(k / n_chains * comm_barrier)

    return comm_barrier, temperatures_next


def sample_pt(
    n_chains: int,
    n_rounds: int,
    target: _c.Problem,
    starting_point: _s.ArrayLike,
    proposal_cls: _c.Proposal,
    seed: int,
    proposal_args: _s.Dict = None,
    thinning: int = 1,
    deo: bool = True,
    progress_bar: bool = False,
):
    assert target.model is not None

    # Setup random number generator for synchronization
    sync_rng = _c.RandomNumberGenerator(seed)

    # Create special chain for efficiently sampling the uniform reference
    reference = _c.Problem(target.A, target.b)
    if target.shift is None and target.transformation is None:
        reference.starting_point = starting_point
        reference = _c.round(reference)
    else:
        reference.shift = target.shift
        reference.transformation = target.transformation
        reference.starting_point = _c.transform(reference, [starting_point])[0]

    with _s.SharedMemoryManager() as smm:
        # Allocate shared memory and initialize parallel workers
        shared_data = _initialize_shared_arrays(
            smm=smm,
            n_chains=n_chains,
            n_rounds=n_rounds,
            state_dim=target.A.shape[1],
            starting_point=starting_point,
        )
        executors = [
            _c.multiprocessing_context.JoinableQueue() for _ in range(n_chains)
        ]
        workers = [
            _c.multiprocessing_context.Process(
                target=_launch_worker,
                args=(
                    target,
                    reference,
                    proposal_cls,
                    starting_point,
                    seed,
                    executors[chain_idx],
                    shared_data,
                    chain_idx,
                    proposal_args,
                    thinning,
                ),
            )
            for chain_idx in range(n_chains)
        ]
        for worker in workers:
            worker.start()

        # Run parallel tempering for the given number of rounds
        temperatures = shared_data.temperatures.view()
        rejection_rates = shared_data.rejection_rates.view()
        global_comm_barriers = shared_data.global_comm_barriers.view()
        scan_idx = 1
        for round_idx in range(n_rounds):
            # Perform 2 ** round_idx scans
            if progress_bar:
                iterator = _s.tqdm.trange(
                    2**round_idx, desc=f"{round_idx + 1}/{n_rounds}"
                )
            else:
                iterator = range(2**round_idx)
            for t in iterator:
                if scan_idx <= 2**n_rounds - 1:
                    machine_idx = shared_data.machine_idx.view()
                    machine_idx[scan_idx] = machine_idx[scan_idx - 1].copy()

                shared_data.densities.view().fill(_s.numpy.nan)

                even = (t % 2) if deo else (_c.Uniform()(sync_rng) < 0.5)
                scan(scan_idx, round_idx, executors, sync_rng, shared_data, even=even)
                scan_idx += 1

            # Compute rejection rates (until now, the array contains the total number of rejections)
            rejection_rates[round_idx] = rejection_rates[round_idx] / (2**round_idx)

            # Adapt schedule
            if round_idx < n_rounds - 1:
                comm_barrier, temperatures_next = _optimize_schedule(
                    rejection_rates[round_idx], temperatures[round_idx]
                )
                global_comm_barriers[round_idx] = comm_barrier
                temperatures[round_idx + 1] = temperatures_next

        # Save the results
        result_stats = shared_data.to_numpy()

        for worker in workers:
            worker.kill()

    return result_stats
