"""

"""

class _core:
    from .core import MarkovChain
    from .core import Problem
    from .core import Proposal
    from .core import RandomNumberGenerator

_c = _core


class _submodules:
    import numpy
    import arviz
    import pandas

    from PolyRound.api import PolyRoundApi
    from PolyRound.mutable_classes import polytope

    from tqdm.auto import tqdm
    from PolyRound.settings import PolyRoundSettings

    import multiprocessing

    import numpy.typing
    import typing

_s = _submodules


def MarkovChain(proposal: _s.typing.Union[_c.Proposal, _s.typing.Type[_c.Proposal]], 
                problem: _c.Problem, 
                starting_point: _s.numpy.typing.ArrayLike = None):
    _proposal = None
    if isinstance(proposal, type):
        if starting_point is not None:
            _proposal = proposal(problem, starting_point=starting_point)
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
    _u = _s.numpy.array(upper_bound) if hasattr(upper_bound, "__len__") else _s.numpy.array([upper_bound] * dim)

    A = _s.numpy.vstack([problem.A, -_s.numpy.eye(dim), _s.numpy.eye(dim)])
    b = _s.numpy.hstack([problem.b.flatten(), _l.flatten(), _u.flatten()]).reshape(-1)
   
    _problem = _c.Problem(A, b, problem.model, problem.starting_point, problem.transformation, problem.shift)

    if simplify:
        _problem = _simplify(_problem)

    return _problem


def compute_chebyshev_center(problem: _c.Problem):
    """

    """
    pass

def _compute_maximum_volume_ellipsoid(problem: _c.Problem):
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
    polytope = _s.polytope.Polytope(problem.A, problem.b)

    polytope = _s.PolyRoundApi.simplify_polytope(polytope)

    problem.A = polytope.A.values
    problem.b = polytope.b.values

    return problem


_simplify = simplify


def round(problem: _c.Problem):
    """

    """
    polytope = _s.polytope.Polytope(problem.A, problem.b)

    polytope = _s.PolyRoundApi.simplify_polytope(polytope)

    number_of_reactions = polytope.A.shape[1]
    polytope.transformation = _s.pandas.DataFrame(_s.numpy.identity(number_of_reactions))
    polytope.transformation.index = [str(i) for i in range(polytope.transformation.to_numpy().shape[0])]
    polytope.transformation.columns = [str(i) for i in range(polytope.transformation.to_numpy().shape[1])]
    polytope.shift = _s.pandas.Series(_s.numpy.zeros(number_of_reactions))

    polytope = _s.PolyRoundApi.round_polytope(polytope)

    _problem = Problem(polytope.A.values, polytope.b.values, problem.model, transformation=polytope.transformation.values, shift=polytope.shift.values)

    if problem.starting_point is not None:
        _problem.starting_point = transform(_problem, [problem.starting_point])

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
            n_thinning: int):
    accrates, states = [], []
    for i in range(n_samples):
        accrate, state = markov_chain.draw(rng, n_thinning)
        accrates.append(accrate)
        states.append(state)

    return accrates, _s.numpy.array(states)


def sample(markov_chains: _s.typing.Union[_c.MarkovChain, _s.typing.List[_c.MarkovChain]], 
           rngs: _s.typing.Union[_c.RandomNumberGenerator, _s.typing.List[_c.RandomNumberGenerator]], 
           n_samples: int, 
           n_thinning: int = 1, 
           n_threads: int = 1):
    """

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
            result = workers.starmap(_sample, [(markov_chains[i], rngs[i], n_samples, n_thinning) for i in range(len(markov_chains))])

            accrates, states = [], []
            for accrate, state in result:
                accrates.append(accrate)
                states.append(state)

        return accrates, _s.numpy.array(states)
    else:
        accrates, states = [], []
        for i in range(len(markov_chains)):
            _accrates, _states = _sample(markov_chains[i], rngs[i], n_samples, n_thinning)
            accrates.append(_accrates)
            states.append(_states)

        return accrates, _s.numpy.array(states)


def _arviz(f: _s.typing.Callable, data: _s.numpy.typing.ArrayLike, series: int = 0, *args, **kwargs):
    n_chains, n_samples, dim = data.shape
    result = []
    if series:
        i = series
        while i < n_samples:
            result.append(f(_s.arviz.convert_to_inference_data(data[:,:i]), *args, **kwargs).x.data)
            i += series
    else:
        result.append(f(_s.arviz.convert_to_inference_data(data), *args, **kwargs).x.data)

    return _s.numpy.array(result)


def ess(*args, **kwargs):
    """

    """
    return _arviz(_s.arviz.ess, *args, **kwargs)


def mcse(*args, **kwargs):
    """

    """
    return _arviz(_s.arviz.mcse, *args, **kwargs)


def rhat(*args, **kwargs):
    """

    """
    return _arviz(_s.arviz.rhat, *args, **kwargs)


