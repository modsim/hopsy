"""

"""

class _core:
    from .core import GaussianHitAndRunProposal
    from .core import MarkovChain
    from .core import Problem
    from .core import Proposal
    from .core import RandomNumberGenerator

_c = _core


class _submodules:
    import numpy
    import arviz
    import pandas

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        from PolyRound.api import PolyRoundApi
        from PolyRound.mutable_classes import polytope
        from PolyRound.settings import PolyRoundSettings

    #from tqdm.auto import tqdm

    import multiprocessing

    import numpy.typing
    import typing

_s = _submodules


def MarkovChain(problem: _c.Problem, 
                proposal: _s.typing.Union[_c.Proposal, _s.typing.Type[_c.Proposal]] = _c.GaussianHitAndRunProposal, 
                starting_point: _s.numpy.typing.ArrayLike = None):
    _proposal = None
    if isinstance(proposal, type):
        if starting_point is not None:
            _proposal = proposal(problem, starting_point=starting_point)
        elif problem.starting_point is not None:
            _proposal = proposal(problem, starting_point=problem.starting_point)
        else:
            _proposal = proposal(problem)
    else:
        _proposal = proposal

    return _c.MarkovChain(_proposal, problem)
    

MarkovChain.__doc__ = _core.MarkovChain.__doc__ # propagate docstring


def add_box_constraints(problem: _c.Problem, 
                        lower_bound: _s.typing.Union[_s.numpy.typing.ArrayLike, float], 
                        upper_bound: _s.typing.Union[_s.numpy.typing.ArrayLike, float], 
                        simplify = True):
    """

    """
    if problem.A.shape[1] == 0:
        raise ValueError("Cannot determine dimension for empty inequality Ax <= b.")

    if hasattr(lower_bound, "__len__") and len(lower_bound) != problem.A.shape[1]:
        raise TypeError("Length of array-like lower_bound has to match column space dimension of the problem.")
    if hasattr(upper_bound, "__len__") and len(upper_bound) != problem.A.shape[1]:
        raise TypeError("Length of array-like upper_bound has to match column space dimension of the problem.")

    dim = problem.A.shape[1]

    _l = _s.numpy.array(lower_bound) if hasattr(lower_bound, "__len__") else _s.numpy.array([lower_bound] * dim)
    _l = -_l # flip sign comes from flipping -x<-l to x>l
    _u = _s.numpy.array(upper_bound) if hasattr(upper_bound, "__len__") else _s.numpy.array([upper_bound] * dim)

    A = _s.numpy.vstack([problem.A, -_s.numpy.eye(dim), _s.numpy.eye(dim)])
    b = _s.numpy.hstack([problem.b.flatten(), _l.flatten(), _u.flatten()]).reshape(-1)
   
    _problem = _c.Problem(A, b, problem.model, problem.starting_point, problem.transformation, problem.shift)

    if simplify:
        with _s.warnings.catch_warnings():
            _s.warnings.simplefilter("ignore")

            _problem = _simplify(_problem)

    return _problem


def compute_chebyshev_center(problem: _c.Problem):
    """

    """
    pass

def _compute_maximum_volume_ellipsoid(problem: _c.Problem):
    with _s.warnings.catch_warnings():
        _s.warnings.simplefilter("ignore")

        polytope = _s.polytope.Polytope(problem.A, problem.b)

        polytope = _s.PolyRoundApi.simplify_polytope(polytope)

        number_of_reactions = polytope.A.shape[1]
        polytope.transformation = _s.pandas.DataFrame(_s.numpy.identity(number_of_reactions))
        polytope.transformation.index = [str(i) for i in range(polytope.transformation.to_numpy().shape[0])]
        polytope.transformation.columns = [str(i) for i in range(polytope.transformation.to_numpy().shape[1])]
        polytope.shift = _s.pandas.Series(_s.numpy.zeros(number_of_reactions))

        MaximumVolumeEllipsoidFinder.iterative_solve(polytope, _s.PolyRoundSettings())
        return polytope.transform.values

def simplify(problem: _c.Problem):
    """

    """
    with _s.warnings.catch_warnings():
        _s.warnings.simplefilter("ignore")

        polytope = _s.polytope.Polytope(problem.A, problem.b)

        polytope = _s.PolyRoundApi.simplify_polytope(polytope)

        problem.A = polytope.A.values
        problem.b = polytope.b.values

        return problem


_simplify = simplify


def round(problem: _c.Problem):
    """

    """
    with _s.warnings.catch_warnings():
        _s.warnings.simplefilter("ignore")

        polytope = _s.polytope.Polytope(problem.A, problem.b)

        polytope = _s.PolyRoundApi.simplify_polytope(polytope)

        number_of_reactions = polytope.A.shape[1]
        polytope.transformation = _s.pandas.DataFrame(_s.numpy.identity(number_of_reactions))
        polytope.transformation.index = [str(i) for i in range(polytope.transformation.to_numpy().shape[0])]
        polytope.transformation.columns = [str(i) for i in range(polytope.transformation.to_numpy().shape[1])]
        polytope.shift = _s.pandas.Series(_s.numpy.zeros(number_of_reactions))

        polytope = _s.PolyRoundApi.round_polytope(polytope)

        _problem = _c.Problem(polytope.A.values, polytope.b.values, problem.model, transformation=polytope.transformation.values, shift=polytope.shift.values)

        if problem.starting_point is not None:
            _problem.starting_point = transform(_problem, [problem.starting_point])[0]

        return _problem


def transform(problem: _c.Problem, points: _s.numpy.typing.ArrayLike):
    """

    """
    transformed_points = []

    for point in points:
        _point = problem.transformation @ point if problem.transformation is not None else point
        _point = _point + problem.shift if problem.shift is not None else _point

        transformed_points.append(_point)

    return transformed_points


def back_transform(problem: _c.Problem, points: _s.numpy.typing.ArrayLike):
    """

    """
    transformed_points = []

    for point in points:
        _point = point - problem.shift if problem.shift is not None else point
        _point = _s.numpy.linalg.solve(problem.transformation, _point) if problem.transformation is not None else _point

        transformed_points.append(_point)

    return transformed_points


def _sample(markov_chain: _c.MarkovChain, 
            rng: _c.RandomNumberGenerator, 
            n_samples: int, 
            thinning: int):
    accrates, states = [], []
    for i in range(n_samples):
        accrate, state = markov_chain.draw(rng, thinning)
        accrates.append(accrate)
        states.append(state)

    return _s.numpy.mean(accrates), _s.numpy.array(states)


def sample(markov_chains: _s.typing.Union[_c.MarkovChain, _s.typing.List[_c.MarkovChain]], 
           rngs: _s.typing.Union[_c.RandomNumberGenerator, _s.typing.List[_c.RandomNumberGenerator]], 
           n_samples: int, 
           thinning: int = 1, 
           n_threads: int = 1):
    r"""sample(markov_chains, rngs, n_samples, thinning=1, n_threads=1)

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
        Number of parallel threads to use.
        Parallelization is achieved using ``multiprocessing``.
        The worker pool size will be ``min(n_threads, len(markov_chains))``

    Returns
    -------
    numpy.ndarray
        The produced states. Will have shape ``(n_chains, n_draws, dim)``. For single chains, it will
        thus be ``(1, n_draws, dim)``.

    """

    # if both are lists, they have to match in size
    if hasattr(markov_chains, "__len__") and hasattr(rngs, "__len__") and len(markov_chains) != len(rngs):
        raise ValueError("Number of Markov chains has to match number of random number generators.")

    # if only one is a list, also fail
    elif not (hasattr(markov_chains, "__len__") and hasattr(rngs, "__len__")) and (hasattr(markov_chains, "__len__") or hasattr(rngs, "__len__")):
        raise ValueError("markov_chains and rngs have to be either both scalar or both lists with matching size.")

    if not hasattr(markov_chains, "__len__") and not hasattr(rngs, "__len__"):
        markov_chains = [markov_chains]
        rngs = [rngs]

    if n_threads != 1:
        if n_threads < 0: 
            n_threads = min(len(markov_chains), _s.multiprocessing.cpu_count())

        with _s.multiprocessing.Pool(n_threads) as workers:
            result = workers.starmap(_sample, [(markov_chains[i], rngs[i], n_samples, thinning) for i in range(len(markov_chains))])

            accrates, states = [], []
            for accrate, state in result:
                accrates.append(accrate)
                states.append(state)

        return accrates, _s.numpy.array(states)
    else:
        accrates, states = [], []
        for i in range(len(markov_chains)):
            _accrates, _states = _sample(markov_chains[i], rngs[i], n_samples, thinning)
            accrates.append(_accrates)
            states.append(_states)

        return accrates, _s.numpy.array(states)


def _is_constant_chains(data: _s.numpy.typing.ArrayLike):
    data = _s.numpy.array(data)
    assert len(data.shape) == 3
    return _s.numpy.sum(_s.numpy.abs(_s.numpy.diff(data, axis=1))) == 0


def _arviz(f: _s.typing.Callable, 
           data: _s.numpy.typing.ArrayLike, 
           series: int = 0, 
           *args, **kwargs):
    data = _s.numpy.array(data)
    assert len(data.shape) == 3
    n_chains, n_samples, dim = data.shape
    result = []
    if series:
        i = series
        while i <= n_samples:
            # if chains are constant, ess = 1, no matter what
            if _is_constant_chains(data[:,:i]) and f == _s.arviz.ess:
                relative = args[3] if len(args) > 4 else kwargs['relative'] if 'relative' in kwargs else False
                _result = [1 / (n_chains*i) if relative else 1] * dim
            else:
                _result = f(_s.arviz.convert_to_inference_data(data[:,:i]), *args, **kwargs).x.data
            result.append(_result)
            i += series
    else:
        if _is_constant_chains(data) and f == _s.arviz.ess:
            relative = args[3] if len(args) > 4 else kwargs['relative'] if 'relative' in kwargs else False
            _result = [1 / (n_chains*n_samples) if relative else 1] * dim
        else:
            _result = f(_s.arviz.convert_to_inference_data(data), *args, **kwargs).x.data
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


def rhat(*args, **kwargs):
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

        References
        ----------
        * Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008
        * Gelman et al. BDA (2014)
        * Brooks and Gelman (1998)
        * Gelman and Rubin (1992)
        * https://arviz-devs.github.io/arviz/api/generated/arviz.rhat.html

    """
    return _arviz(_s.arviz.rhat, *args, **kwargs)


