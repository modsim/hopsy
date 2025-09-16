"""

"""


class _core:
    from .core import (
        Problem,
        Proposal,
        RandomNumberGenerator,
        Uniform,
        UniformCoordinateHitAndRunProposal,
    )
    from .definitions import multiprocessing_context
    from .misc import MarkovChain, round, transform


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
    r"""Storage for a numpy array in shared memory"""
    name: str
    shared_memory: _s.SharedMemory
    shape: _s.Tuple[int, ...]
    dtype: _s.numpy.dtype

    def view(self) -> _s.ArrayLike:
        r""""""
        return _s.numpy.ndarray(
            self.shape, dtype=_s.numpy.dtype(self.dtype), buffer=self.shared_memory.buf
        )


@_s.dataclass
class SharedData:
    r"""Shared data for parallel tempering"""

    samples: SharedMemoryArray
    densities: SharedMemoryArray
    acceptance_rates: SharedMemoryArray
    rejection_rates: SharedMemoryArray
    machine_idx: SharedMemoryArray
    temperatures: SharedMemoryArray
    global_comm_barriers: SharedMemoryArray

    def to_numpy(self) -> _s.Dict[str, _s.ArrayLike]:
        r"""
        Return a dict mapping each shared-data attribute name
        to its NumPy array view.
        """
        return {
            attr: getattr(self, attr).view().copy()
            for attr in self.__dataclass_fields__
        }


def _initialize_shared_arrays(
    smm: _s.SharedMemoryManager,
    chains: _s.List[_c.MarkovChain],
    n_rounds: int = 1,
    n_samples: int = 1,
) -> SharedData:

    if n_rounds == 1:
        n_scans = n_samples
    else:
        # total number of scan‐steps = sum_{r=0..n_rounds-1} 2**r
        n_scans = sum(2**r for r in range(n_rounds))

    def make_block(name: str, shape: tuple, dtype) -> SharedMemoryArray:
        n_bytes = int(_s.numpy.prod(shape) * _s.numpy.dtype(dtype).itemsize)
        shared_memory: _s.SharedMemory = smm.SharedMemory(size=n_bytes)
        return SharedMemoryArray(
            name=name, shared_memory=shared_memory, shape=shape, dtype=dtype
        )

    n_chains = len(chains)
    states = [chains[i].state for i in range(n_chains)]
    state_dim = states[0].shape[0]
    dtype = states[0].dtype
    assert all(s.shape == (state_dim,) for s in states), "All chains must have the same state dimension"
    assert all(s.dtype == dtype for s in states), "All chains must have the same state dtype"

    samples = make_block(
        "samples", (n_scans + 1, n_chains, state_dim), dtype
    )
    densities = make_block("densities", (1, n_chains), _s.numpy.float64)
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
    samples_view[0] = _s.numpy.array(states)

    idx_array = machine_idx.view()
    idx_array[0, :] = _s.numpy.arange(n_chains, dtype=_s.numpy.int32)

    temps_view = temps.view()
    temps_view.fill(0.0)
    temps_view[0, :] = _s.numpy.array([chains[i].coldness for i in range(n_chains)])
    print("Initial temperatures:", temps_view[0, :])
    return SharedData(
        samples=samples,
        densities=densities,
        acceptance_rates=acc_rates,
        rejection_rates=rej_rates,
        machine_idx=machine_idx,
        temperatures=temps,
        global_comm_barriers=global_comm_barriers,
    )


def _swap_probability(densities, temperatures, idx_curr: int, idx_next: int):
    log_likelihood_ratio = (
        temperatures[idx_curr] * densities[idx_next]
        + temperatures[idx_next] * densities[idx_curr]
        - temperatures[idx_curr] * densities[idx_curr]
        - temperatures[idx_next] * densities[idx_next]
    )

    return 1 if log_likelihood_ratio > 0 else min(1, _s.numpy.exp(log_likelihood_ratio))


def _launch_worker(
    chain : _c.MarkovChain,
    rng : _c.RandomNumberGenerator,
    reference: _c.Problem,
    task_queue: _c.multiprocessing_context.JoinableQueue,
    shared_data: SharedData,
    process_id: int,
    proposal_args: _s.Dict = None,
    thinning: int = 1,
):
    machine_idx = shared_data.machine_idx.view()
    temperatures = shared_data.temperatures.view()
    samples = shared_data.samples.view()
    acc_rates = shared_data.acceptance_rates.view()
    densities = shared_data.densities.view()

    reference_chain = _c.MarkovChain(reference, _c.UniformCoordinateHitAndRunProposal)

    if proposal_args:
        for k in proposal_args:
            setattr(chain.proposal, k, proposal_args[k])

    while True:
        task = task_queue.get()

        scan_idx, round_idx = task

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
        densities[0, process_id] = chain.model.log_density(new_state)

        task_queue.task_done()


def _scan(
    scan_idx: int,
    round_idx: int,
    executors: _s.List[_c.multiprocessing_context.JoinableQueue],
    sync_rng: _c.RandomNumberGenerator,
    shared_data: SharedData,
    even: bool = False,
):
    r"""

    :param scan_idx:
    :param round_idx:
    :param executors:
    :param sync_rng:
    :param shared_data:
    :param even:
    :return:
    """
    n_chains = len(executors)

    # local step
    for executor in executors:
        executor.put((scan_idx, round_idx))
    for executor in executors:
        executor.join()

    machine_idx = shared_data.machine_idx.view()
    r = shared_data.rejection_rates.view()
    samples = shared_data.samples.view()
    densities = shared_data.densities.view()
    temperatures = shared_data.temperatures.view()

    # swap
    for i in range(n_chains - 1):
        alpha = _swap_probability(densities[0], temperatures[round_idx], i, i + 1)
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


def _optimize_schedule(rejection_rates: _s.ArrayLike, temperatures: _s.ArrayLike, tune_schedule: bool = True):
    n_chains = len(temperatures)
    temperatures_next = _s.numpy.zeros(n_chains)
    temperatures_next[-1] = 1

    cum_rejection_rates = rejection_rates.cumsum()
    comm_barrier = cum_rejection_rates[-1]
    interpolator = _s.PchipInterpolator(
        cum_rejection_rates, temperatures, extrapolate=True
    )
    for k in range(1, n_chains - 1):
        temperatures_next[k] = interpolator(k / n_chains * comm_barrier) if tune_schedule else temperatures[k]

    return comm_barrier, temperatures_next


def nrpt(
    mcs: _s.List[_c.MarkovChain],
    rngs: _s.List[_c.RandomNumberGenerator],
    seed: int,
    n_rounds: int = 0,
    n_samples: int = 0,
    thinning : int = 1,
    deo: bool = True,
    tune_schedule: bool = True,
    progress_bar: bool = False,
    proposal_args: _s.Dict = None,
):
    """
    Non-Reversible Parallel Tempering

    Parameters
    ----------
    mcs : List[MarkovChain]
        List of Markov chains to use for sampling.
    rngs : List[_c.RandomNumberGenerator]
        List of random number generators to use for sampling.
    n_samples : int
        Number of samples to draw.
    n_rounds : int
        Number of rounds of parallel tempering to perform.
    deo : bool
        Whether to use deterministic even-odd (True) or random (False) exchanges.
    tune_schedule : bool
        Whether to adapt the temperature schedule or not.
    progress_bar : bool
        Whether to show a progress bar or not.
    proposal_args : Dict
        Additional arguments to pass to the proposal.
    """

    assert (n_samples > 0) ^ (n_rounds > 0), "Either n_samples or n_rounds must be specified, but not both."
    assert len(mcs) == len(rngs), "Number of Markov chains must match number of random number generators."

    if n_samples > 0:
        assert not tune_schedule, "Tuning the schedule is supported only for round-based tuning. For schedule tuning, specify n_rounds instead of n_samples."

    target = mcs[-1].problem
    starting_point = target.starting_point

    # Setup random number generator for synchronization
    sync_rng = _c.RandomNumberGenerator(seed)

    # Create special chain for efficiently sampling the uniform reference
    reference = _c.Problem(target.A, target.b)
    n_chains = len(mcs)

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
            chains=mcs,
            n_rounds=n_rounds,
        )

        executors = [
            _c.multiprocessing_context.JoinableQueue() for _ in range(n_chains)
        ]

        workers = [
            _c.multiprocessing_context.Process(
                target=_launch_worker,
                args=(
                    mcs[i],
                    rngs[i],
                    reference,
                    executors[i],
                    shared_data,
                    i,
                    proposal_args,
                    thinning,
                ),
            )
            for i in range(n_chains)
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
                _scan(scan_idx, round_idx, executors, sync_rng, shared_data, even=even)
                scan_idx += 1

            # Compute rejection rates (until now, the array contains the total number of rejections)
            rejection_rates[round_idx] = rejection_rates[round_idx] / (2**round_idx)

            # Adapt schedule
            if round_idx < n_rounds - 1:
                comm_barrier, temperatures_next = _optimize_schedule(
                    rejection_rates[round_idx], temperatures[round_idx], tune_schedule
                )
                global_comm_barriers[round_idx] = comm_barrier
                temperatures[round_idx + 1] = temperatures_next

        # Save the results
        result_stats = shared_data.to_numpy()
        final_samples = None

        if n_rounds > 0:
            final_samples = result_stats["samples"][-2**(n_rounds-1) :, :]
        else:
            final_samples = result_stats["samples"]

        for worker in workers:
            worker.kill()

    return result_stats, final_samples
