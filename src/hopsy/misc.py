"""

"""


class _core:
    from .callback import Callback
    from .core import (
        BilliardWalkProposal,
        Gaussian,
        GaussianHitAndRunProposal,
        MarkovChain,
        Problem,
        Proposal,
        RandomNumberGenerator,
        TruncatedGaussianProposal,
        Uniform,
        UniformCoordinateHitAndRunProposal,
        UniformInt,
    )
    from .lp import LP


_c = _core


class _submodules:
    import atexit
    import time
    import warnings

    import arviz
    import numpy
    import pandas

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        from PolyRound.api import PolyRoundApi
        from PolyRound.mutable_classes import polytope
        from PolyRound.static_classes.lp_utils import ChebyshevFinder
        from PolyRound.static_classes.lp_utils import OptlangInterfacer
        from PolyRound.static_classes.rounding.maximum_volume_ellipsoid import (
            MaximumVolumeEllipsoidFinder,
        )

    import multiprocessing
    import os
    from multiprocessing import shared_memory

    if "JPY_PARENT_PID" in os.environ:
        import tqdm.notebook as tqdm
    else:
        import tqdm
    import typing
    import warnings

    import numpy.typing


_s = _submodules


def create_shared_memory(state_shape, num_blocks: int, state_type=_s.numpy.float64):
    """
    It is the responsibility of the caller of create_shared_memory to call release_shared_memory.
    Multiprocessing will give a warning if leaked memory exists at the end of the program

    Parameters
    ----------
    state_shape
    state_type

    Returns names of shared memory buffers
    -------

    """
    if state_type != _s.numpy.float64:
        raise ValueError("state_types != np.float64 not yet supported.")
    # add two to first dimension for likelihood and coldness
    shape = (state_shape[0] + 2, *state_shape[1:])
    shared_memory_size = int(_s.numpy.dtype(state_type).itemsize * _s.numpy.prod(shape))
    shared_state_memory = [
        _s.shared_memory.SharedMemory(size=shared_memory_size, create=True)
        for i in range(num_blocks)
    ]
    _s.atexit.register(lambda: [release_shared_memory(s) for s in shared_state_memory])
    return shared_state_memory


def release_shared_memory(shm):
    """
    It is the responsibility of the caller of create_shared_memory to call release_shared_memory
    Multiprocessing will give a warning if leaked memory exists at the end of the program

    Parameters
    ----------
    name: str
        Name of the shared memory block to release

    Returns
    -------
    None

    """
    try:
        # If memory was already unlinked, it will throw. Because we register the cleanup with atexit,
        # we don't know for sure if the memory still exists or not. We just clean up any handle we ever had.
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass


class PyParallelTemperingChain:
    """
    Wrapper for MarkovChain for performing parallel tempering using Multiprocessing (instead of MPI).
    """

    def __init__(
        self,
        markov_chain: _c.MarkovChain,
        chain_index: int,
        num_chains_in_ensemble: int,
        sync_rng: _c.RandomNumberGenerator,
        barrier: _s.multiprocessing.Barrier,
        shared_memory: _s.typing.List[_s.shared_memory.SharedMemory] = None,
        draws_per_exchange_attempt: float = 10,
    ):
        self.markov_chain = markov_chain
        self.parallel_tempering_sync_rng = sync_rng
        self.shared_memory = shared_memory
        self.draws_per_exchange_attempt = draws_per_exchange_attempt
        self.chain_index = chain_index
        self.barrier = barrier
        self.num_chains_in_ensemble = num_chains_in_ensemble
        self.draw_offset = 0
        self.draw_tracker = 0
        self.exchange_tracker = 0

    def __del__(self):
        if self.chain_index == 0:
            for n in self.shared_memory:
                release_shared_memory(n)
        self.barrier.abort()

    @property
    def coldness(self):
        return self.markov_chain.coldness

    @property
    def state_log_density(self):
        return self.markov_chain.state_log_density

    @property
    def state(self):
        return self.markov_chain.state

    @state.setter
    def state(self, value):
        self.markov_chain.state = value

    @property
    def proposal(self):
        return self.markov_chain.proposal

    @proposal.setter
    def proposal(self, value):
        self.markov_chain.proposal = value

    @property
    def problem(self):
        return self.markov_chain.problem

    def draw(self, rng: _c.RandomNumberGenerator, thinning: int = 1):
        """
        :param rng:
        :param thinning:
        :return: acceptance rate, state
        """
        acceptance_rate = 0.0

        for i in range(thinning):
            if (i + self.draw_offset) % self.draws_per_exchange_attempt == 0:
                acceptance_rate += self._parallel_tempering_exchange()[0]
                self.exchange_tracker += 1
            acceptance_rate, state = self.markov_chain.draw(rng=rng, thinning=1)
            acceptance_rate += acceptance_rate
            self.draw_tracker += 1
        self.draw_offset = (
            self.draw_offset + thinning
        ) % self.draws_per_exchange_attempt
        return acceptance_rate / thinning, state

    def _parallel_tempering_exchange(self):
        accepted = False
        partner_index = self._find_partner_for_swap()
        expected_shape = (
            self.markov_chain.state.shape[0] + 2,
            *self.markov_chain.state.shape[1:],
        )
        if partner_index != -1:
            this_index = self.chain_index
            shm = self.shared_memory[this_index]

            write_destination = _s.numpy.ndarray(
                shape=expected_shape, dtype=_s.numpy.float64, buffer=shm.buf
            )
            # swaps out cold likelihoods, i.e., original likelihoods
            write_destination[0] = self.markov_chain.state_log_density
            write_destination[1] = self.coldness
            write_destination[2:] = self.markov_chain.state

        self.barrier.wait()
        if partner_index != -1:
            shm = self.shared_memory[partner_index]
            other_chain = _s.numpy.ndarray(
                shape=expected_shape, dtype=_s.numpy.float64, buffer=shm.buf
            )
            likelihood_difference = self.markov_chain.state_log_density - other_chain[0]
            coldness_difference = self.coldness - other_chain[1]
            log_acceptance_prob = likelihood_difference * coldness_difference
            if (
                _s.numpy.log(_c.Uniform(a=0, b=1)(self.parallel_tempering_sync_rng))
                < log_acceptance_prob
            ):
                self.state = other_chain[2:]
                accepted = True
        else:
            self.parallel_tempering_sync_rng()

        self.barrier.wait()
        return 1.0 if accepted else 0.0, self.markov_chain.state

    def _find_partner_for_swap(self):
        even_odd = _c.UniformInt(a=0, b=1)(self.parallel_tempering_sync_rng)
        if even_odd % 2 == 0:
            # even communication:
            if self.chain_index % 2 == 0:
                # this chain is even and communicates with chainIndex +1
                partner_index = self.chain_index + 1
            else:
                # this chain is odd and communicates with chainIndex -1
                partner_index = self.chain_index - 1
        else:
            # odd communication
            if self.chain_index % 2 == 0:
                # this chain is even and communicates with chainIndex +1
                partner_index = self.chain_index - 1
            else:
                # this chain is odd and communicates with chainIndex -1
                partner_index = self.chain_index + 1

        # check if swap partner is valid:
        if self.num_chains_in_ensemble % 2 == 0:
            # even number of chains: you can always find a partner
            if partner_index == -1:
                partner_index = self.num_chains_in_ensemble - 1
            partner_index = partner_index % self.num_chains_in_ensemble
        else:
            # odd number of chains: you can NOT always find a partner
            if partner_index == self.num_chains_in_ensemble:
                partner_index = -1

        return partner_index


def get_samples_with_temperature(
    temperature: float,
    temperature_ladder: _s.typing.List[float],
    samples: _s.numpy.ndarray,
) -> _s.numpy.ndarray:
    """
    Given a set of samples, the temperature ladder used for parallel tempering and a temperature value,
    this function extracts the subset of samples for the given temperature.

    Parameters
    ----------
    temperature: float
    temperature_ladder: _s.typing.List[float]
    samples: np.ndarray

    Returns
    -------
    np.ndarray
        subset of samples

    """
    sample_index = temperature_ladder.index(temperature)
    if samples.shape[0] % len(temperature_ladder) != 0:
        raise RuntimeError(
            "Number of markov chains does not fit to temperature ladder."
        )
    return samples[sample_index :: len(temperature_ladder), :, :]


def create_py_parallel_tempering_ensembles(
    markov_chains: _s.typing.Union[_c.MarkovChain, _s.typing.List[_c.MarkovChain]],
    temperature_ladder: _s.typing.List[float],
    sync_rngs: _s.typing.Union[
        _c.RandomNumberGenerator, _s.typing.List[_c.RandomNumberGenerator]
    ],
    draws_per_exchange_attempt: float = 100,
):
    """

    Parameters
    ----------
    markov_chains: hopsy.MarkovChain
        markov chain replicates to be used for parallel tempering
    temperature_ladder: List
        list of temperatures for parallel tempering. List must be sorted (ascending or descending)
    parallel_tempering_sync_rng: List[hopsy.RandomNumberGenerator]
        random number generator used for syncing parallel chains without explicit communication
    exchange_attempt_probability: float
        how often parallel chains should attempt to exchange states on average

    Returns
    -------
    List[PyParallelTemperingChain]
        For each markov chain given, this function creates an ensemble of len(temperature_ladder) chains.
        Each ensemble is an indepedent replicate and can be used for convergence check.
        All chains of all ensembles are returned in a single list for compatibility with hopsy.sample
        To parse our states given a temperature/coldness, use  hopsy.parse_states_from_ensembles
    """
    if isinstance(markov_chains, _c.MarkovChain) != isinstance(
        sync_rngs, _c.RandomNumberGenerator
    ):
        raise RuntimeError(
            "markov chains and sync rng need to both be lists or their respective types"
        )

    if isinstance(markov_chains, _c.MarkovChain):
        markov_chains = [markov_chains]
    if isinstance(markov_chains, _c.RandomNumberGenerator):
        sync_rngs = [sync_rngs]

    if len(markov_chains) != len(sync_rngs):
        raise RuntimeError(
            "Exactly one sync rng is required for every ensemble (markov chain replicate)."
        )

    _sorted_temps = sorted(temperature_ladder)
    if (
        _sorted_temps != temperature_ladder
        and list(reversed(_sorted_temps)) != temperature_ladder
    ):
        raise RuntimeError(
            "temperature_ladder needs to be sorted in either ascending or descending order."
        )
    if 1.0 not in temperature_ladder:
        raise RuntimeError(
            "temperature_ladder should contain 1.0, i.e., a chain that samples the original likelihood."
        )
    if any(t < 0 for t in temperature_ladder) or any(
        t > 1.0 for t in temperature_ladder
    ):
        raise RuntimeError(
            "temperature_ladder should only contain values in the interval [0, 1]"
        )

    pte = []
    m = _s.multiprocessing.Manager()
    for i, markov_chain in enumerate(markov_chains):
        shm = create_shared_memory(
            markov_chain.state.shape, num_blocks=len(temperature_ladder)
        )
        barrier = m.Barrier(len(temperature_ladder))

        for chain_index, coldness in enumerate(temperature_ladder):
            _pt_mc = MarkovChain(
                problem=_c.Problem(
                    A=markov_chain.problem.A,
                    b=markov_chain.problem.b,
                    shift=markov_chain.problem.shift,
                    transformation=markov_chain.problem.transformation,
                    starting_point=markov_chain.problem.starting_point,
                    model=markov_chain.problem.model,
                ),
                proposal=markov_chain.proposal,
            )
            _pt_mc.coldness = coldness
            pte.append(
                PyParallelTemperingChain(
                    markov_chain=_pt_mc,
                    chain_index=chain_index,
                    num_chains_in_ensemble=len(temperature_ladder),
                    sync_rng=sync_rngs[i],
                    draws_per_exchange_attempt=draws_per_exchange_attempt,
                    barrier=barrier,
                    shared_memory=shm,
                )
            )
    return pte


def MarkovChain(
    problem: _c.Problem,
    proposal: _s.typing.Union[_c.Proposal, _s.typing.Type[_c.Proposal]] = None,
    starting_point: _s.numpy.typing.ArrayLike = None,
    parallel_tempering_sync_rng: _c.RandomNumberGenerator = None,
    exchange_attempt_probability: float = 0.1,
    coldness: float = 1.0,
):
    _proposal = None

    if proposal is None and isinstance(problem.model, _c.Gaussian):
        proposal = _c.TruncatedGaussianProposal
    elif proposal is None and problem.model is None:
        proposal = _c.UniformCoordinateHitAndRunProposal
    elif proposal is None:
        proposal = _c.GaussianHitAndRunProposal

    if isinstance(proposal, type):
        if starting_point is not None:
            _proposal = proposal(problem, starting_point=starting_point)
        elif problem.starting_point is not None:
            _proposal = proposal(problem, starting_point=problem.starting_point)
        else:
            _starting_point = compute_chebyshev_center(problem)
            _proposal = proposal(problem, starting_point=_starting_point)
    else:
        _proposal = proposal

    return _c.MarkovChain(
        _proposal,
        problem,
        parallel_tempering_sync_rng=parallel_tempering_sync_rng,
        exchange_attempt_probability=exchange_attempt_probability,
        coldness=coldness,
    )


MarkovChain.__doc__ = _core.MarkovChain.__doc__  # propagate docstring


def add_box_constraints(
    problem: _c.Problem,
    lower_bound: _s.typing.Union[_s.numpy.typing.ArrayLike, float],
    upper_bound: _s.typing.Union[_s.numpy.typing.ArrayLike, float],
    simplify=True,
):
    r"""Adds box constraints to all dimensions. This will extend :attr:`hopsy.Problem.A` and :attr:`hopsy.Problem.b` of the returned :class:`hopsy.Problem` to have :math:`m+2n` rows.
    Box constraints are added naively, meaning that we do neither check whether the dimension may be already
    somehow bound nor check whether the very same constraint already exists. You can remove redundant constraints
    efficiently using the `PolyRound <https://pypi.org/project/PolyRound/>`_ toolbox or by using the :func:`hopsy.round` function, uses PolyRound to remove redundant constraints and also rounds the polytope.

    If ``lower_bound`` and ``upper_bound`` are both ``float``, then every dimension :math:`i` will be bound as
    :math:`lb \leq x_i \leq ub`. If `lower_bound`` and ``upper_bound`` are both ``numpy.ndarray`` with
    appropriate length, then every dimension :math:`i` will be bound as :math:`lb_i \leq x_i \leq ub_i`.

    :param hopsy.Problem problem: Problem which should be constrained and which contains the matrix :math:`A` and vector :math:`b` in :math:`Ax \leq b`.

    :param lower_bound: Specifies the lower bound(s).
    :type lower_bound: numpy.ndarray[float64[n,1]] or float

    :param upper_bound: Specifies the upper bound(s).
    :type upper_bound: numpy.ndarray[float64[n,1]] or float

    :return: A :class:`hopsy.Problem` bounded in all dimensions.
    :rtype: hopsy.Problem
    """
    if problem.A.shape[1] == 0:
        raise ValueError("Cannot determine dimension for empty inequality Ax <= b.")

    if hasattr(lower_bound, "__len__") and len(lower_bound) != problem.A.shape[1]:
        raise TypeError(
            "Length of array-like lower_bound has to match column space dimension of the problem."
        )
    if hasattr(upper_bound, "__len__") and len(upper_bound) != problem.A.shape[1]:
        raise TypeError(
            "Length of array-like upper_bound has to match column space dimension of the problem."
        )

    dim = problem.A.shape[1]

    _l = (
        _s.numpy.array(lower_bound)
        if hasattr(lower_bound, "__len__")
        else _s.numpy.array([lower_bound] * dim)
    )
    _l = -_l  # flip sign comes from flipping -x<-l to x>l
    _u = (
        _s.numpy.array(upper_bound)
        if hasattr(upper_bound, "__len__")
        else _s.numpy.array([upper_bound] * dim)
    )

    A = _s.numpy.vstack([problem.A, -_s.numpy.eye(dim), _s.numpy.eye(dim)])
    b = _s.numpy.hstack([problem.b.flatten(), _l.flatten(), _u.flatten()]).reshape(-1)

    _problem = _c.Problem(
        A,
        b,
        problem.model,
        problem.starting_point,
        problem.transformation,
        problem.shift,
    )

    if simplify:
        with _s.warnings.catch_warnings():
            _s.warnings.simplefilter("ignore")

            _problem = _simplify(_problem)

    return _problem


def add_equality_constraints(
    problem: _c.Problem, A_eq: _s.numpy.ndarray, b_eq: _s.numpy.typing.ArrayLike
):
    r"""Adds equality constraints as specified. This will change :attr:`hopsy.Problem.A` and :attr:`hopsy.Problem.b`.
    The equality constraints are incorporated into the transformation of the original problem.
    :param hopsy.Problem problem: Problem which should be constrained and which contains the matrix :math:`A` and vector :math:`b` in :math:`Ax \leq b`.
    In order to obtain useful results, the problem is automatically simplified.

    :param A_eq: equality constraint matrix (lhs)
    :type A_eq: numpy.ndarray[float64[n,m]]

    :param b_eq: equality constraint vector (rhs)
    :type b_eq: numpy.ndarray[float64[n,1]]

    :return: A :class:`hopsy.Problem` which has incorporated the equality constraints into the transformation and most likely reduced the dimensionality of the problem.
    :rtype: hopsy.Problem
    """
    if problem.A.shape[1] == 0:
        raise ValueError("Cannot determine dimension for empty inequality Ax <= b.")

    if hasattr(A_eq, "shape") and A_eq.shape[1] != problem.A.shape[1]:
        raise TypeError(
            "Dimensionality missmatch in equality and inequality constraints!"
        )
    if problem.transformation is not None or problem.shift is not None:
        raise RuntimeError(
            "Problem contains transformation and shift. "
            "This is not supported yet."
            "Set both to None to continue."
        )

    try:
        polytope = _s.polytope.Polytope(A=problem.A, b=problem.b, S=A_eq, h=b_eq)
        with _s.warnings.catch_warnings():
            _s.warnings.simplefilter("ignore")
            polytope = _s.PolyRoundApi.simplify_polytope(polytope, _c.LP().settings)

        # transform_polytope carries out dimension reduction due to equality constraints if possible
        polytope = _s.PolyRoundApi.transform_polytope(polytope, _c.LP().settings)

        _problem = _c.Problem(
            polytope.A,
            polytope.b,
            problem.model,
            problem.starting_point,
            polytope.transformation.values,
            polytope.shift.values,
        )

        if problem.starting_point is not None:
            _problem.starting_point = transform(_problem, [problem.starting_point])[0]

            if _s.numpy.any((_problem.b - _problem.A @ _problem.starting_point) < 0):
                _s.warnings.warn(
                    "Applying equality constraints to starting point in problem failed. "
                    "Please provide a new starting point."
                )
                _problem.starting_point = None

        return _problem
    except ValueError as e:
        raise ValueError(
            "Adding these equality constraints makes the problem infeasible! Check the problem and/or the LP().settings"
        )


def is_polytope_empty(
    A: _s.numpy.ndarray,
    b: _s.numpy.ndarray,
    S: _s.numpy.ndarray = None,
    h: _s.numpy.ndarray = None,
):
    """
    Checks whether the polytope given by Ax < b and optionally Sx=h has a solution x.

    The result is only as precise as the LP solver and settings used.
    We recommend using Gurobi and the NumericFocus option if problems are hard.
    See hopsy.LP to access LP solver settings.
    """
    if not (S is not None and h is not None) and not (S is None and h is None):
        raise RuntimeError("Either S and h must both be None OR both must have values")
    polytope = _s.polytope.Polytope(A=A, b=b, S=S, h=h)
    _s.warnings.filterwarnings("ignore", category=DeprecationWarning)
    optlang_model = _s.OptlangInterfacer.polytope_to_optlang(polytope, _c.LP().settings)
    status = optlang_model.optimize()
    _s.warnings.simplefilter("always")
    if status != "optimal":
        return True
    else:
        return False


def is_problem_polytope_empty(problem: _c.Problem):
    """
    Checks whether the problem with polytope given by Ax < b and optionally Sx=h has a solution x.

    The result is only as precise as the LP solver and settings used.
    We recommend using Gurobi and the NumericFocus option if problems are hard.
    See hopsy.LP to access LP solver settings.
    """
    return is_polytope_empty(problem.A, problem.b)


def compute_chebyshev_center(problem: _c.Problem, original_space: bool = False):
    """
    Computes the Chebyshev center, that is the midpoint of a (non-unique) largest inscribed ball in the polytope defined by :math:`Ax \leq b`.
    Note that if A and b are transformed (e.g. rounded), the chebyshev center is computed in the transformed space. To trigger a backtransform, use the parameter `original_space=True`.

    :param hopsy.Problem problem: Problem for which the Chebyshev center should be computed and which contains the matrix :math:`A` and vector :math:`b` in :math:`Ax \leq b`.
    :param bool original_space: If the problem has been transformed (e.g. rounded). the chebyshev center is computed in the rounded space by default. If the chebyshev center is required in the original space, use original_space=True. Only works if the transformation and shift are stored in the problem.

    :return: The Chebyshev center of the passed problem.
    :rtype: numpy.ndarray[float64[n,1]]
    """
    polytope = _s.polytope.Polytope(problem.A, problem.b)
    cheby_result = _s.ChebyshevFinder.chebyshev_center(polytope, _c.LP().settings)
    chebyshev_center = cheby_result[0]
    distance_to_border = cheby_result[1]
    if distance_to_border <= 0:
        raise ValueError(
            "Chebyshev center is outside of polytope. To solve check polytope feasibility or change LP settings"
        )
    if original_space:
        return back_transform(problem=problem, points=[chebyshev_center])[0]

    return chebyshev_center


def _compute_maximum_volume_ellipsoid(problem: _c.Problem):
    with _s.warnings.catch_warnings():
        _s.warnings.simplefilter("ignore")

        polytope = _s.polytope.Polytope(problem.A, problem.b)

        polytope = _s.PolyRoundApi.simplify_polytope(polytope, _c.LP().settings)

        if polytope.S is not None:
            polytope = _s.PolyRoundApi.transform_polytope(polytope, _c.LP().settings)
        else:
            number_of_reactions = polytope.A.shape[1]
            polytope.transformation = _s.pandas.DataFrame(
                _s.numpy.identity(number_of_reactions)
            )
            polytope.transformation.index = [
                str(i) for i in range(polytope.transformation.to_numpy().shape[0])
            ]
            polytope.transformation.columns = [
                str(i) for i in range(polytope.transformation.to_numpy().shape[1])
            ]
            polytope.shift = _s.pandas.Series(_s.numpy.zeros(number_of_reactions))

        _s.MaximumVolumeEllipsoidFinder.iterative_solve(polytope, _c.LP().settings)
        return polytope.transform.values


def simplify(problem: _c.Problem):
    r"""simplify(problem)

    Simplifies the polytope defined in the ``problem`` by removing redundant constraints and refunction inequality constraints to equality constraints in case of dimension width less than thresh.
    Thresh is defined in the LP settings singleton and refers to `PolyRound <https://pypi.org/project/PolyRound/>`_ settings.
    Simplification is typically the first step before sampling. It is called automatically when round is called, because it is required for efficient and effective rounding.

    Parameters
    ----------
    problem: hopsy.Problem for which the polytope should be simplified

    Returns
    -------
    hopsy.Problem
        Problem with simplified polytope.
    """

    with _s.warnings.catch_warnings():
        _s.warnings.simplefilter("ignore")

        polytope = _s.polytope.Polytope(problem.A, problem.b)

        polytope = _s.PolyRoundApi.simplify_polytope(
            polytope, settings=_c.LP().settings
        )
        if polytope.S is not None:
            polytope = _s.PolyRoundApi.transform_polytope(polytope, _c.LP().settings)
        else:
            number_of_reactions = polytope.A.shape[1]
            polytope.transformation = _s.pandas.DataFrame(
                _s.numpy.identity(number_of_reactions)
            )
            polytope.transformation.index = [
                str(i) for i in range(polytope.transformation.to_numpy().shape[0])
            ]
            polytope.transformation.columns = [
                str(i) for i in range(polytope.transformation.to_numpy().shape[1])
            ]
            polytope.shift = _s.pandas.Series(_s.numpy.zeros(number_of_reactions))

        problem.A = polytope.A.values
        problem.b = polytope.b.values

        return problem


_simplify = simplify


def round(problem: _c.Problem):
    """
    Rounds the polytope defined by the inequality :math:`Ax \leq b` using
    `PolyRound <https://pypi.org/project/PolyRound/>`_.
    This function also strips redundant constraints.
    The unrounding transformation :math:`T` and shift :math:`s` will be stored in :attr:`hopsy.UniformProblem.transformation`
    and :attr:`hopsy.UniformProblem.shift` of the returned problem. Its left-hand side operator :attr:`hopsy.UniformProblem.A` and
    the right-hand side :attr:`hopsy.UniformProblem.b` of the polytope will be transformed as :math:``
    inequality

    :param hopsy.Problem problem: Problem that should be rounded and which contains the matrix :math:`A` and vector :math:`b` in :math:`Ax \leq b`.

    :return: The rounded problem.
    :rtype: hopsy.Problem
    """
    with _s.warnings.catch_warnings():
        _s.warnings.simplefilter("ignore")

        polytope = _s.polytope.Polytope(problem.A, problem.b)

        polytope = _s.PolyRoundApi.simplify_polytope(polytope, _c.LP().settings)

        existing_transform = problem.transformation
        existing_shift = problem.shift

        if polytope.S is not None:
            polytope = _s.PolyRoundApi.transform_polytope(polytope, _c.LP().settings)
        else:
            number_of_reactions = polytope.A.shape[1]
            polytope.transformation = _s.pandas.DataFrame(
                _s.numpy.identity(number_of_reactions)
            )
            polytope.transformation.index = [
                str(i) for i in range(polytope.transformation.to_numpy().shape[0])
            ]
            polytope.transformation.columns = [
                str(i) for i in range(polytope.transformation.to_numpy().shape[1])
            ]
            polytope.shift = _s.pandas.Series(_s.numpy.zeros(number_of_reactions))

        polytope = _s.PolyRoundApi.round_polytope(polytope, _c.LP().settings)

        complete_transform = (
            polytope.transformation.values
            if existing_transform is None
            else existing_transform @ polytope.transformation.values
        )
        complete_shift = (
            polytope.shift.values
            if existing_shift is None
            else existing_shift + existing_transform @ polytope.shift.values
        )

        _problem = _c.Problem(
            polytope.A.values,
            polytope.b.values,
            problem.model,
            transformation=complete_transform,
            shift=complete_shift,
        )

        _problem.original_A = problem.A
        _problem.original_b = problem.b

        if problem.starting_point is not None:
            intermediate_problem = _c.Problem(
                polytope.A.values,
                polytope.b.values,
                transformation=polytope.transformation.values,
                shift=polytope.shift.values,
            )
            _problem.starting_point = transform(
                intermediate_problem, [problem.starting_point]
            )[0]

        return _problem


def back_transform(problem: _c.Problem, points: _s.numpy.typing.ArrayLike):
    """
    Transforms samples back from the sampling space (typically rounded) to the original parameter space.
    """
    transformed_points = []

    for point in points:
        _point = (
            problem.transformation @ point
            if problem.transformation is not None
            else point
        )
        _point = (
            _point + problem.shift.reshape(_point.shape)
            if problem.shift is not None
            else _point
        )

        transformed_points.append(_point)

    return transformed_points


def transform(problem: _c.Problem, points: _s.numpy.typing.ArrayLike):
    """
    Transforms samples from the parameter space to the sampling space (typically rounded).

    """
    transformed_points = []

    is_square = problem.transformation.shape[0] == problem.transformation.shape[1]
    solver = (
        _s.numpy.linalg.solve
        if is_square
        else lambda A, b: _s.numpy.linalg.lstsq(A, b, rcond=None)[0]
    )

    for point in points:
        _point = point - problem.shift if problem.shift is not None else point
        _point = (
            solver(problem.transformation, _point)
            if problem.transformation is not None
            else _point
        )

        transformed_points.append(_point)

    return transformed_points


def _sequential_sampling(
    markov_chain: _c.MarkovChain,
    rng: _c.RandomNumberGenerator,
    n_samples: int,
    thinning: int,
    chain_idx: int,
    in_memory: bool,
    record_meta=None,
    callback: _c.Callback = None,
    progress_bar: bool = False,
):
    states = [None] * n_samples

    meta = None
    if record_meta is None or record_meta is False:
        meta = []
    else:
        meta = {field: [] for field in record_meta}

    sample_range = (
        _s.tqdm.trange(n_samples, desc="chain {}".format(chain_idx))
        if progress_bar
        else range(n_samples)
    )
    for i in sample_range:
        accrate, state = markov_chain.draw(rng, thinning)

        curr_meta = None
        if record_meta is None or record_meta is False:
            curr_meta = accrate
        else:
            curr_meta = {}
            for field in record_meta:
                if field == "acceptance_rate":  # treat acceptance rate differently,
                    # as it is no attribute of the markov chain
                    curr_meta[field] = accrate
                else:
                    # recurse through the attribute name and record the final value
                    attrs = field.split(".")
                    base = markov_chain
                    for attr in attrs:
                        base = getattr(base, attr)

                    curr_meta[field] = base

        if in_memory:
            states[i] = state

            if record_meta is None or record_meta is False:
                meta.append(curr_meta)
            else:
                for field in record_meta:
                    meta[field].append(curr_meta[field])

        if callback is not None:
            callback.record(
                chain_idx,
                state,
                curr_meta
                if isinstance(curr_meta, dict)
                else {"acceptance_rate": curr_meta},
            )

    if (record_meta is None or record_meta is False) and in_memory:
        meta = _s.numpy.mean(meta)

    if callback is not None:
        callback.finish()

    return meta, _s.numpy.array(states)


def _sample_parallel_chain(
    markov_chain: _c.MarkovChain,
    rng: _c.RandomNumberGenerator,
    n_samples: int,
    thinning: int,
    chain_idx: int,
    in_memory: bool,
    record_meta=None,
    queue: _s.multiprocessing.Queue = None,
    # barrier: _s.multiprocessing.Barrier = None,
):
    states = [None] * n_samples

    meta = None
    if record_meta is None or record_meta is False:
        meta = []
    else:
        meta = {field: [] for field in record_meta}

    for i in range(n_samples):
        accrate, state = markov_chain.draw(rng, thinning)

        curr_meta = None
        if record_meta is None or record_meta is False:
            curr_meta = accrate
        else:
            curr_meta = {}
            for field in record_meta:
                if field == "acceptance_rate":  # treat acceptance rate differently,
                    # as it is no attribute of the markov chain
                    curr_meta[field] = accrate
                else:
                    # recurse through the attribute name and record the final value
                    attrs = field.split(".")
                    base = markov_chain
                    for attr in attrs:
                        base = getattr(base, attr)

                    curr_meta[field] = base

        if in_memory:
            states[i] = state

            if record_meta is None or record_meta is False:
                meta.append(curr_meta)
            else:
                for field in record_meta:
                    meta[field].append(curr_meta[field])

        if queue is not None:
            queue.put((chain_idx, state, curr_meta))

    if (record_meta is None or record_meta is False) and in_memory:
        meta = _s.numpy.mean(meta)

    if queue is not None:
        queue.put((chain_idx, None, None))

    return meta, _s.numpy.array(states), markov_chain.proposal.state, rng.state


def _process_record_meta(
    chain: MarkovChain, record_meta
) -> _s.typing.Tuple[
    _s.typing.List[str], _s.typing.List[_s.typing.List[int]], _s.typing.Dict[str, None]
]:
    shapes = []
    missing_fields = {}
    if isinstance(record_meta, list):
        record_meta = record_meta.copy()
        for field in record_meta:
            if field != "acceptance_rate":
                attrs = field.split(".")
                base = chain

                for attr in attrs:
                    if hasattr(base, attr):
                        base = getattr(base, attr)
                    else:
                        record_meta.remove(field)
                        missing_fields[field] = None

                if field not in missing_fields:
                    shape = []
                    if hasattr(base, "shape"):
                        shape = [i for i in base.shape]
                    shapes += [shape]
    else:
        shapes += [[]]
    return record_meta, shapes, missing_fields


def _parallel_sampling(
    args: _s.typing.List[_s.typing.Any],
    n_procs: int,
    callback: _c.Callback,
    progress_bar: bool,
):
    result_queue = (
        _s.multiprocessing.Manager().Queue()
        if callback is not None or progress_bar
        else None
    )
    for i in range(len(args)):
        args[i] += (result_queue,)

    if callback is not None or progress_bar:
        workers = _s.multiprocessing.Pool(n_procs)
        if "SLURM_JOB_ID" in _s.os.environ:
            _s.warnings.warn(
                "Warning: progress bars or callbacks within SLURM are not officially supported. Proceed with caution and make "
                "a feature request in our gitlab, if you require progress bars & SLURM"
            )
        result = workers.starmap_async(_sample_parallel_chain, args)
        pbars = (
            [
                _s.tqdm.trange(args[i][2], desc="chain {}".format(i))
                for i in range(len(args))
            ]
            if progress_bar
            else None
        )
        finished = [False for i in range(len(args))]
        while not _s.numpy.all(finished):
            chain_idx, state, meta = result_queue.get()
            if state is not None:
                if progress_bar:
                    pbars[chain_idx].update()
                if callback is not None:
                    callback.record(
                        chain_idx,
                        state,
                        meta if isinstance(meta, dict) else {"acceptance_rate": meta},
                    )
            else:
                finished[chain_idx] = True
                if progress_bar:
                    pbars[chain_idx].close()

        if callback is not None:
            callback.finish()
        workers.close()
        workers.join()
        return result.get()
    else:
        with _s.multiprocessing.Pool(n_procs) as workers:
            result = workers.starmap(_sample_parallel_chain, args)
        return result


def sample(
    markov_chains: _s.typing.Union[_c.MarkovChain, _s.typing.List[_c.MarkovChain]],
    rngs: _s.typing.Union[
        _c.RandomNumberGenerator, _s.typing.List[_c.RandomNumberGenerator]
    ],
    n_samples: int,
    thinning: int = 1,
    n_threads: int = 1,
    n_procs: int = 1,
    record_meta=None,
    in_memory: bool = True,
    callback: _c.Callback = None,
    progress_bar: bool = False,
):
    r"""sample(markov_chains, rngs, n_samples, thinning=1, n_procs=1)

    Draw ``n_samples`` from every passed chain in ``markov_chains``
    using the respective random number generator from ``rngs``.
    Thus, ``len(rngs)`` has to match ``len(markov_chains)``.

    Parameters
    ----------
    markov_chains : list[hopsy.MarkovChain] or hopsy.MarkovChain
        (List of) Markov chain(s) to simulate to generate samples.
    rngs : list[hopsy.RandomNumberGenerator] or hopsy.RandomNumberGenerator
        (List of) random number generator(s) to simulate the Markov chains.
        If a single :class:`hopsy.MarkovChain` was passed to ``sample``, then ``rng`` also must be a single
        :class:`hopsy.RandomNumberGenerator`.
    n_samples : int
        Number of samples to draw from every chain.
    thinning : int
        Number of samples to discard inbetween two saved states.
        This will increase the number of samples actually produced by the chain
        to ``thinning * n_samples``.
    n_threads : int
        (deprecated) Number of parallel processes to use.
        Parallelization is achieved using ``multiprocessing``.
        The worker pool size will be ``min(n_procs, len(markov_chains))``
    n_procs : int
        Number of parallel processes to use.
        Parallelization is achieved using ``multiprocessing``.
        The worker pool size will be ``min(n_procs, len(markov_chains))``
    record_meta : list[str] or bool
        Strings defining :class:`hopsy.MarkovChain` attributes or ``acceptance_rate``, which will
        then be recorded and returned. All attributes of :class:`hopsy.MarkovChain` can be used here,
        e.g. ``record_meta=['state_negative_log_likelihood', 'proposal.proposal']``.
    in_memory : bool
        Flag for enabling or disabling in-memory saving of states and metadata.
    callback : derived from hopsy.Callback
        Observer callback to which states and metadata are passed during the run. The callback is e.g. used
        to write the obtained information online to permanent storage. This enables online analysis of the
        MCMC run.

    Returns
    -------
    optional[tuple[list or dict, numpy.ndarray]]
        First value of the tuple holds meta information about the states. Without using ``record_meta``,
        this is a list containing the acceptance rates of each chain.
        If ``record_meta`` is used, then this is a dict containing the values of the :class:`hopsy.MarkovChain`
        attributes defined in ``record_meta``.
        If the attribute was not found (e.g. because of a typo), it will have value ``None``.

        Second value of the tuple holds produced states. Will have shape ``(n_chains, n_draws, dim)``. For single chains, it will
        thus be ``(1, n_draws, dim)``.

        If ``in_memory=False``, ``None`` will be returned.

    """

    # multiprocessing and mpi (parallel tempering) do not work together yet
    # because forked processes do not get a correct rank.
    if n_procs != 1 and any(
        [
            not isinstance(mc, PyParallelTemperingChain)
            and mc.parallel_tempering_sync_rng is not None
            for mc in markov_chains
        ]
    ):
        raise ValueError(
            "n_procs>1 does not work together with parallel tempering with is based on mpi."
        )

    # if both are lists, they have to match in size
    if (
        hasattr(markov_chains, "__len__")
        and hasattr(rngs, "__len__")
        and len(markov_chains) != len(rngs)
    ):
        raise ValueError(
            "Number of Markov chains has to match number of random number generators."
        )

    # if only one is a list, also fail
    elif not (hasattr(markov_chains, "__len__") and hasattr(rngs, "__len__")) and (
        hasattr(markov_chains, "__len__") or hasattr(rngs, "__len__")
    ):
        raise ValueError(
            "markov_chains and rngs have to be either both scalar or both lists with matching size."
        )

    if not hasattr(markov_chains, "__len__") and not hasattr(rngs, "__len__"):
        markov_chains = [markov_chains]
        rngs = [rngs]

    # remove invalid entries from record_meta
    record_meta, shapes, missing_fields = _process_record_meta(
        markov_chains[0], record_meta
    )

    # initialize backend
    if callback is not None:
        meta_names = (
            record_meta if isinstance(record_meta, list) else ["acceptance_rate"]
        )
        callback.setup(
            len(markov_chains),
            n_samples,
            len(markov_chains[0].state),
            meta_names,
            shapes,
        )

    result = []

    if (
        n_procs != 1
        or n_threads != 1
        or any([isinstance(mc, PyParallelTemperingChain) for mc in markov_chains])
    ):
        if n_threads != n_procs and n_threads != 1:
            n_procs = n_threads

        if n_procs < 0:
            n_procs = min(
                len(markov_chains), _s.multiprocessing.cpu_count()
            )  # do not use more procs than available cpus
        result_states = _parallel_sampling(
            [
                (
                    markov_chains[chain_idx],
                    rngs[chain_idx],
                    n_samples,
                    thinning,
                    chain_idx,
                    in_memory,
                    record_meta,
                )
                for chain_idx in range(len(markov_chains))
            ],
            n_procs,
            callback,
            progress_bar,
        )
        for i, chain_result in enumerate(result_states):
            result.append((chain_result[0], chain_result[1]))
            markov_chains[i].proposal.state = chain_result[2]
            rngs[i].state = chain_result[3]
    else:
        for chain_idx in range(len(markov_chains)):
            _accrates, _states = _sequential_sampling(
                markov_chains[chain_idx],
                rngs[chain_idx],
                n_samples,
                thinning,
                chain_idx,
                in_memory,
                record_meta,
                callback,
                progress_bar,
            )
            result.append((_accrates, _states))

    if in_memory:
        states = []
        meta = (
            []
            if record_meta is None or record_meta is False
            else {field: [] for field in record_meta}
        )

        for _meta, _states in result:
            states.append(_states)

            if record_meta is None or record_meta is False:
                meta.append(_meta)
            else:
                for field in meta:
                    meta[field].append(_meta[field])

        if record_meta is not None and record_meta is not False:
            for field in meta:
                meta[field] = _s.numpy.array(meta[field])
            meta.update(missing_fields)

        return meta, _s.numpy.array(states)


def _parallel_execution(
    func: _s.typing.Callable, args: _s.typing.List[_s.typing.Any], n_procs: int
):
    with _s.multiprocessing.Pool(n_procs) as workers:
        return workers.starmap(func, args)


def _is_constant_chains(data: _s.numpy.typing.ArrayLike):
    data = _s.numpy.array(data)
    assert len(data.shape) == 3
    return _s.numpy.sum(_s.numpy.abs(_s.numpy.diff(data, axis=1))) == 0


def _compute_statistic(
    i: int,
    n_chains: int,
    dim: int,
    f: _s.typing.Callable,
    data: _s.numpy.typing.ArrayLike,
    args,
    kwargs,
):
    # if chains are constant, ess = 1, no matter what
    if _is_constant_chains(data[:, :i]) and f == _s.arviz.ess:
        relative = (
            args[3]
            if len(args) > 4
            else kwargs["relative"]
            if "relative" in kwargs
            else False
        )
        return [1 / (n_chains * i) if relative else 1] * dim
    else:
        return f(_s.arviz.convert_to_inference_data(data[:, :i]), **kwargs).x.data


def _arviz(
    f: _s.typing.Callable,
    data: _s.numpy.typing.ArrayLike,
    series: int = 0,
    n_procs: int = 1,
    *args,
    **kwargs
):
    data = _s.numpy.array(data)
    assert len(data.shape) == 3
    n_chains, n_samples, dim = data.shape
    result = []

    if series:
        if n_procs != 1:
            indices = list(range(series, n_samples, series))
            if n_procs < 0:
                n_procs = min(
                    len(indices), _s.multiprocessing.cpu_count()
                )  # do not use more processes than available cpus
            result = _parallel_execution(
                _compute_statistic,
                [(i, n_chains, dim, f, data, args, kwargs) for i in indices],
                n_procs,
            )
        else:
            i = series
            while i <= n_samples:
                result.append(
                    _compute_statistic(i, n_chains, dim, f, data, args, kwargs)
                )
                i += series
    else:
        if _is_constant_chains(data) and f == _s.arviz.ess:
            relative = (
                args[3]
                if len(args) > 4
                else kwargs["relative"]
                if "relative" in kwargs
                else False
            )
            _result = [1 / (n_chains * n_samples) if relative else 1] * dim
        else:
            _result = f(
                _s.arviz.convert_to_inference_data(data), *args, **kwargs
            ).x.data
        result.append(_result)

    return _s.numpy.array(result)


def ess(*args, **kwargs):
    r"""ess(data, series=0, method="bulk", relative=False, prob=None, dask_kwargs=None)

    Calculate estimate of the effective sample size (ess).

    Parameters
    ----------
    data : numpy.ndarray
        MCMC samples with ``data.shape == (n_chains, n_draws, dim)``.
    series : int
        Compute a series of effective sample sizes every ``series`` samples,
        so ess will be computed for ``data[:,:n] for n in range(series, n_draws+1, series)``.
        For the default value ``series==0``, ess will be computed only once for the whole data.
    method : str
        Select ess method. Valid methods are:

        - "bulk"
        - "tail"     # prob, optional
        - "quantile" # prob
        - "mean" (old ess)
        - "sd"
        - "median"
        - "mad" (mean absolute deviance)
        - "z_scale"
        - "folded"
        - "identity"
        - "local"
    relative : bool
        Return relative ess
        `ress = ess / n`
    prob : float, or tuple of two floats, optional
        probability value for "tail", "quantile" or "local" ess functions.
    n_procs : int = 1
        In combination with "series": compute series of ess in parallel using ``n_procs``
        subprocesses.
    dask_kwargs : dict, optional
        Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.

    Returns
    -------
    numpy.ndarray
        Return the effective sample size, :math:`\hat{N}_{eff}`

    Notes
    -----
    The basic ess (:math:`N_{\mathit{eff}}`) diagnostic is computed by:

    .. math:: \hat{N}_{\mathit{eff}} = \frac{MN}{\hat{\tau}}

    .. math:: \hat{\tau} = -1 + 2 \sum_{t'=0}^K \hat{P}_{t'}

    where :math:`M` is the number of chains, :math:`N` the number of draws,
    :math:`\hat{\rho}_t` is the estimated _autocorrelation at lag :math:`t`, and
    :math:`K` is the last integer for which :math:`\hat{P}_{K} = \hat{\rho}_{2K} +
    \hat{\rho}_{2K+1}` is still positive.

    The current implementation is similar to Stan, which uses Geyer's initial monotone sequence
    criterion (Geyer, 1992; Geyer, 2011).

    References
    ----------
    * Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008
    * https://arviz-devs.github.io/arviz/api/generated/arviz.ess.html
    * https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html
      Section 15.4.2
    * Gelman et al. BDA (2014) Formula 11.8

    """
    return _arviz(_s.arviz.ess, *args, **kwargs)


def mcse(*args, **kwargs):
    r"""mcse(data, series=0, method="mean", prob=None, dask_kwargs=None)

    Calculate Markov Chain Standard Error statistic.

    Parameters
    ----------
    data : numpy.ndarray
        MCMC samples with ``data.shape == (n_chains, n_draws, dim)``.
    series : int
        Compute a series of statistics every ``series`` samples,
        so mcse will be computed for ``data[:,:n] for n in range(series, n_draws+1, series)``.
        For the default value ``series==0``, mcse will be computed only once for the whole data.
    method : str
        Select mcse method. Valid methods are:
        - "mean"
        - "sd"
        - "median"
        - "quantile"

    prob : float
        Quantile information.
    n_procs : int = 1
        In combination with "series": compute series of mcse in parallel using ``n_procs``
        subprocesses.
    dask_kwargs : dict, optional
        Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.

    Returns
    -------
    numpy.ndarray
        Return the msce dataset

    References
    ----------
    * https://arviz-devs.github.io/arviz/api/generated/arviz.mcse.html

    """
    return _arviz(_s.arviz.mcse, *args, **kwargs)


def rhat(data, *args, **kwargs):
    r"""rhat(data, series=0, method="rank", dask_kwargs=None)

    Compute estimate of rank normalized splitR-hat for a set of traces.

    The rank normalized R-hat diagnostic tests for lack of convergence by comparing the variance
    between multiple chains to the variance within each chain. If convergence has been achieved,
    the between-chain and within-chain variances should be identical. To be most effective in
    detecting evidence for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.

    Parameters
    ----------
    data : numpy.ndarray
        MCMC samples with ``data.shape == (n_chains, n_draws, dim)``.
    series : int
        Compute a series of R-hat statistics every ``series`` samples,
        so R-hat will be computed for ``data[:,:n] for n in range(series, n_draws+1, series)``.
        For the default value ``series==0``, R-hat will be computed only once for the whole data.
    method : str
        Select R-hat method. Valid methods are:
        - "rank"        # recommended by Vehtari et al. (2019)
        - "split"
        - "folded"
        - "z_scale"
        - "identity"
    n_procs : int = 1
        In combination with "series": compute series of R-hat in parallel using ``n_procs``
        subprocesses.
    dask_kwargs : dict
        Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.

    Returns
    -------
    numpy.ndarray
      Returns dataset of the potential scale reduction factors, :math:`\hat{R}`

    Notes
    -----
    The diagnostic is computed by:

      .. math:: \hat{R} = \frac{\hat{V}}{W}

    where :math:`W` is the within-chain variance and :math:`\hat{V}` is the posterior variance
    estimate for the pooled rank-traces. This is the potential scale reduction factor, which
    converges to unity when each of the traces is a sample from the target posterior. Values
    greater than one indicate that one or more chains have not yet converged.

    Rank values are calculated over all the chains with `scipy.stats.rankdata`.
    Each chain is split in two and normalized with the z-transform following Vehtari et al. (2019).

    Note that if all chains contain the same constant for some parameter (due to equality constraints),
    rhat will be 1 for this parameter.

    References
    ----------
    * Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008
    * Gelman et al. BDA (2014)
    * Brooks and Gelman (1998)
    * Gelman and Rubin (1992)
    * https://arviz-devs.github.io/arviz/api/generated/arviz.rhat.html

    """
    # next lines finds dimensions where samples for all chains are constant
    s = data.reshape(-1, data.shape[2])
    diff = _s.numpy.isclose(_s.numpy.diff(s, axis=0), 0)
    to_drop = _s.numpy.all(diff, axis=0)
    for i in reversed(range(len(to_drop))):
        if to_drop[i]:
            data = _s.numpy.delete(data, i, axis=2)
    rhat = _arviz(_s.arviz.rhat, data, *args, **kwargs)
    # reinclude rhat=1 for constant dimensions
    for i in range(len(to_drop)):
        if to_drop[i]:
            rhat = _s.numpy.insert(rhat, i, 1.0, axis=1)
    return rhat


def _svd_rounding(samples, polytope):
    """
    Polytope rounding based on samples, as suggested in https://drops.dagstuhl.de/opus/volltexte/2021/13820/pdf/LIPIcs-SoCG-2021-21.pdf
    This is used in the multipase_sampling function
    """
    # We concatenate them samples to [n_dim, n_iterations] for rounding
    stacked_samples = _s.numpy.vstack(
        [samples[i, :, :] for i in range(samples.shape[0])]
    )

    mean = _s.numpy.mean(stacked_samples, axis=0)
    stacked_samples = stacked_samples - mean
    U, S, Vh = _s.numpy.linalg.svd(stacked_samples)
    # Rescaling as mentioned in  https://drops.dagstuhl.de/opus/volltexte/2021/13820/pdf/LIPIcs-SoCG-2021-21.pdf
    s_ratio = _s.numpy.max(S) / _s.numpy.min(S)
    S = S / _s.numpy.min(S)
    if _s.numpy.max(S) >= 2.0:
        S[_s.numpy.where(S < 2.0)] = 1.0
    else:
        S = _s.numpy.ones(S.shape)
        Vh = _s.numpy.identity(Vh.shape[0])

    rounding_matrix = _s.numpy.transpose(Vh).dot(_s.numpy.diag(S))

    # Transforms current last samples into new polytope
    sub_problem = _c.Problem(
        polytope.A, polytope.b, transformation=rounding_matrix, shift=mean
    )
    starting_points = transform(
        sub_problem, [samples[i, -1, :] for i in range(samples.shape[0])]
    )
    polytope.apply_shift(mean)
    polytope.apply_transformation(rounding_matrix)

    return s_ratio, starting_points, polytope, sub_problem


def run_multiphase_sampling(
    problem: _c.Problem,
    seeds: _s.typing.List,
    steps_per_phase: int,
    starting_points: _s.typing.List,
    target_ess=1000,
    proposal=_c.BilliardWalkProposal,
    n_procs=1,
    limit_singular_value_ratio=2.3,
):
    """
    runs multiphase sampling as suggested in https://drops.dagstuhl.de/opus/volltexte/2021/13820/pdf/LIPIcs-SoCG-2021-21.pdf
    limit_singular_value_ratio=2.3

    Beware this algorithm works well for uniform sampling, BUT it is not officially supported for non-uniform targets!
    """
    limit_singular_value_ratio = 2.3
    assert len(starting_points) == len(seeds)
    rngs = [_c.RandomNumberGenerator(s) for s in seeds]

    polytope = _s.polytope.Polytope(A=problem.A, b=problem.b)
    polytope.normalize()
    current_ess = 0
    iterations = 0
    s_ratio = limit_singular_value_ratio + 1
    sampling_time = 0
    samples = None
    last_iteration_did_rounding = True
    while current_ess < target_ess:
        iterations += 1
        internal_polytope = polytope
        p = _c.Problem(internal_polytope.A, internal_polytope.b)
        markov_chains = [
            MarkovChain(proposal=proposal, problem=p, starting_point=s)
            for s in starting_points
        ]

        start = _s.time.perf_counter()
        acceptance_rate, _samples = sample(
            markov_chains, rngs, n_samples=steps_per_phase, thinning=1, n_procs=4
        )
        end = _s.time.perf_counter()
        sampling_time += end - start

        if s_ratio > limit_singular_value_ratio:
            samples = _samples
            # also measures the rounding time
            start = _s.time.perf_counter()
            s_ratio, starting_points, internal_polytope, sub_problem = _svd_rounding(
                samples, internal_polytope
            )
            end = _s.time.perf_counter()
            sampling_time += end - start
            last_iteration_did_rounding = True
        else:
            if last_iteration_did_rounding:
                # next operation transforms last samples to current space before concatenating
                for j in range(samples.shape[0]):
                    samples[j] = transform(
                        sub_problem, [samples[j, i, :] for i in range(samples.shape[1])]
                    )
                last_iteration_did_rounding = False

            samples = _s.numpy.concatenate((samples, _samples), axis=1)
            starting_points = [samples[i, -1, :] for i in range(samples.shape[0])]
        current_ess = ess(samples)
        current_ess = _s.numpy.min(current_ess)
        steps_per_phase += 100

    # transforms back to full space
    _samples = _s.numpy.zeros(
        (samples.shape[0], samples.shape[1], polytope.transformation.shape[0])
    )
    for j in range(samples.shape[0]):
        _samples[j] = back_transform(
            _c.Problem(
                A=internal_polytope.A,
                b=internal_polytope.b,
                transformation=internal_polytope.transformation,
                shift=internal_polytope.shift,
            ),
            [samples[j, i, :] for i in range(samples.shape[1])],
        )

    return _samples, iterations, current_ess, sampling_time
