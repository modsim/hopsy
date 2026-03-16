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
    from .misc import MarkovChain, round, transform


_c = _core


class _submodules:
    import os
    from dataclasses import dataclass
    from threading import ThreadPoolExecutor
    from typing import Dict, List, Optional, Tuple

    import numpy
    from numpy.typing import ArrayLike
    from scipy.interpolate import PchipInterpolator

    if "JPY_PARENT_PID" in os.environ:
        import tqdm.notebook as tqdm
    else:
        import tqdm


_s = _submodules

class NRPTResult:

    def __init__(self,
        chains: _s.List[_c.MarkovChain],
        n_rounds: int = 1,
        n_samples: int = 1,
    ):

        if n_rounds == 1:
            n_scans = n_samples
        else:
            # total number of scan‐steps = sum_{r=0..n_rounds-1} 2**r
            n_scans = sum(2**r for r in range(n_rounds))

        self.n_chains = len(chains)
        states = [chains[i].state for i in range(self.n_chains)]
        state_dim = len(states[0])
        dtype = states[0].dtype
        assert all(s.shape == (state_dim,) for s in states), "All chains must have the same state dimension"
        assert all(s.dtype == dtype for s in states), "All chains must have the same state dtype"

        samples = _s.numpy.zeros((n_scans + 1, self.n_chains, state_dim))
        samples[0] = _s.numpy.array(states) # log the initial starting point of each chain
    
        densities = _s.numpy.zeros((n_scans + 1, self.n_chains))
        densities[0] = _s.numpy.array(chain.state_log_density for chain in chains) # initialize with log density of initial state 

        machine_idx = _s.numpy.zeros((n_scans + 1, self.n_chains), _s.numpy.int32)
        machine_idx[0] = _s.numpy.arange(self.n_chains, dtype=_s.numpy.int32) # initialize indices by list order

        acc_rates = _s.numpy.zeros((n_scans, self.n_chains))
    
        rej_rates = _s.numpy.zeros((n_rounds, self.n_chains))
        inv_temperatures = _s.numpy.zeros((n_rounds, self.n_chains))
        global_comm_barriers = _s.numpy.zeros((n_rounds, self.n_chains))


def _swap_probability(densities, inv_temperatures, idx_curr: int, idx_next: int):
    log_likelihood_ratio = (
        inv_temperatures[idx_curr] * densities[idx_next]
        + inv_temperatures[idx_next] * densities[idx_curr]
        - inv_temperatures[idx_curr] * densities[idx_curr]
        - inv_temperatures[idx_next] * densities[idx_next]
    )

    return 1 if log_likelihood_ratio > 0 else min(1, _s.numpy.exp(log_likelihood_ratio))


def _explore(
    chain : _c.MarkovChain,
    rng : _c.RandomNumberGenerator,
    reference: _c.Problem,
    data: NRPTResult,
    scan_idx: int, 
    round_idx: int,
    thread_id: int,
    thinning: int = 1,
):
    machine_idx = data.machine_idx
    temperatures = data.temperatures
    samples = data.samples
    acc_rates = data.acceptance_rates
    densities = data.densities

    chain_idx = _s.numpy.where(machine_idx[scan_idx] == thread_id)[0][0]
    beta = temperatures[round_idx, chain_idx]

    if beta <= 1e-9:
        reference_chain = _c.MarkovChain(reference, _c.UniformCoordinateHitAndRunProposal)
        thinning_factor = max(len(chain.state), int(len(chain.state) ** 2 // 6))
        reference_chain.state = chain.state
        acceptance_rate, new_state = reference_chain.draw(rng, thinning_factor)
        chain.state = new_state
        densities[scan_idx, thread_id] = chain.model.log_density(new_state)
    else:
        chain.coldness = beta
        acceptance_rate, new_state = chain.draw(rng, thinning)
        densities[scan_idx, thread_id] = chain.state_log_density

    # 5. Safe write to shared arrays
    samples[scan_idx, chain_idx] = new_state
    acc_rates[scan_idx, chain_idx] = acceptance_rate


def _scan(
    scan_idx: int,
    round_idx: int,
    executor: _s.ThreadPoolExecutor,
    sync_rng: _c.RandomNumberGenerator,
    data: NRPTResult,
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
    n_chains = executor.max_workers

    # Local exploration step and density computation (parallel)
    executor.map(_explore, ...)

    machine_idx = data.machine_idx
    r = data.rejection_rates
    samples = data.samples
    densities = data.densities
    temperatures = data.temperatures

    # Compute swap probability and swap
    for i in range(n_chains - 1):
        alpha = _swap_probability(densities[scan_idx], temperatures[round_idx], i, i + 1)
        if even == i % 2:
            if _c.Uniform()(sync_rng) < alpha:
                samples_row = samples[scan_idx]
                samples_row[[i, i + 1]] = samples_row[[i + 1, i]]  # this creates an internal copy of the RHS
                idx_row = machine_idx[scan_idx]
                idx_row[[i, i + 1]] = idx_row[[i + 1, i]]
        
        # Update rejection rates
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

# see https://pmc.ncbi.nlm.nih.gov/articles/PMC3038348/
def stepping_stone(densities: _s.numpy.array, betas: _s.numpy.array):
    tempered_densities = (densities * betas)
    max_densities = _s.numpy.max(tempered_densities, axis=0)
    betas = _s.numpy.array([betas[t] - betas[t-1] for t in range(1, len(betas))])
    n_samples= tempered_densities.shape[0]
    fac1 = (max_densities[1:] * betas).sum()
    fac2 = _s.numpy.log(_s.numpy.exp(tempered_densities[:,1:] * betas - max_densities[1:]).sum(axis=0) / n_samples).sum()

    return fac1 + fac2


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

    # Initialize regular Markov chains
    for mc in mcs:
        if proposal_args:
            for k in proposal_args:
                setattr(mc.proposal, k, proposal_args[k])

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

    data = NRPTResult(mcs, n_rounds, n_samples)

    with _s.ThreadPoolExecutor(n_chains) as executor:
        # Run parallel tempering for the given number of rounds
        temperatures = data.temperatures
        rejection_rates = data.rejection_rates
        global_comm_barriers = data.global_comm_barriers
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
                    machine_idx = data.machine_idx
                    machine_idx[scan_idx] = machine_idx[scan_idx - 1].copy()

                even = (t % 2) if deo else (_c.Uniform()(sync_rng) < 0.5)
                _scan(scan_idx, round_idx, executor, sync_rng, data, even=even)
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
        final_samples = None
        if n_rounds > 0:
            final_samples = data.samples[-2**(n_rounds-1) :, :]
        else:
            final_samples = data.samples

    return data, final_samples
