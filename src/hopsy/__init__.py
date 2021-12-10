from . core import *

import numpy as np
import pandas as pd

from PolyRound.api import PolyRoundApi as prapi
from PolyRound.mutable_classes import polytope as prp

from tqdm.auto import tqdm

#class Problem:
#    def __init__(self, A, b, model = None, starting_point = None, transformation = None, shift = None):
#        self.A = np.array(A)
#        self.b = np.array(b)
#        self.model = model
#        self.starting_point = np.array(starting_point) if starting_point is not None else None
#        self.transformation = np.array(transformation) if transformation is not None else None
#        self.shift = np.array(shift) if shift is not None else None
#
#
#    def __repr__(self):
#        return "Problem(A=" + self.A.__repr__() + ", " + \
#               "b=" + self.b.__repr__() + ", " + \
#               "model=" + self.model.__repr__() + ", " + \
#               "starting_point=" + self.starting_point.__repr__() + ", " + \
#               "transformation=" + self.transformation.__repr__() + ", " + \
#               "shift=" + self.shift.__repr__() + \
#               ")" 


def add_box_constraints(problem, lower_bound, upper_bound, simplify = True):
    if problem.A.shape[1] == 0:
        raise ValueError("Cannot determine dimension for empty inequality Ax <= b.")

    if hasattr(lower_bound, "__len__") and len(lower_bound) != problem.A.shape[1]:
        raise TypeError("Length of array-like lower_bound has to match column space dimension of the problem.")
    if hasattr(upper_bound, "__len__") and len(upper_bound) != problem.A.shape[1]:
        raise TypeError("Length of array-like upper_bound has to match column space dimension of the problem.")

    dim = problem.A.shape[1]

    _l = np.array(lower_bound) if hasattr(lower_bound, "__len__") else np.array([lower_bound] * dim)
    _u = np.array(upper_bound) if hasattr(upper_bound, "__len__") else np.array([upper_bound] * dim)

    A = np.vstack([problem.A, -np.eye(dim), np.eye(dim)])
    b = np.hstack([problem.b.flatten(), _l.flatten(), _u.flatten()]).reshape(-1)
   
    _problem = Problem(A, b, problem.model, problem.starting_point, problem.transformation, problem.shift)

    if simplify:
        _problem = _simplify(_problem)

    return _problem


def compute_chebyshev_center(problem):
    pass


def simplify(problem):
    polytope = prp.Polytope(problem.A, problem.b)

    polytope = prapi.simplify_polytope(polytope)

    problem.A = polytope.A.values
    problem.b = polytope.b.values

    return problem


_simplify = simplify


def round(problem):
    polytope = prp.Polytope(problem.A, problem.b)

    polytope = prapi.simplify_polytope(polytope)

    number_of_reactions = polytope.A.shape[1]
    polytope.transformation = pd.DataFrame(np.identity(number_of_reactions))
    polytope.transformation.index = [str(i) for i in range(polytope.transformation.to_numpy().shape[0])]
    polytope.transformation.columns = [str(i) for i in range(polytope.transformation.to_numpy().shape[1])]
    polytope.shift = pd.Series(np.zeros(number_of_reactions))

    polytope = prapi.round_polytope(polytope)

    _problem = Problem(polytope.A.values, polytope.b.values, problem.model, transformation=polytope.transformation.values, shift=polytope.shift.values)

    if problem.starting_point is not None:
        _problem.starting_point = transform(_problem, [problem.starting_point])

    return _problem


def transform(problem, points):
    transformed_points = []

    for point in points:
        _point = problem.transformation @ point if problem.transformation is not None else point
        _point = _point + problem.shift if problem.shift is not None else _point

        transformed_points.append(_point)

    return transformed_points


def back_transform(problem, points):
    transformed_points = []

    for point in points:
        _point = point - problem.shift if problem.shift is not None else point
        _point = np.linalg.solve(problem.transformation, _point) if problem.transformation is not None else _point

        transformed_points.append(_point)

    return transformed_points
    

class Sampler:
    def __init__(self, proposal, problem, starting_points = None, n_chains = 1, n_threads = 1, seed = 0):
        #self.proposal_dist = proposal  # here we should call create_markov_chain(proposal)
                                        # which would decide based on the type of proposal
        self.proposal = proposal
        self.problem = problem

        self.starting_points = starting_points if problem.starting_point is None else problem.starting_point
        assert self.starting_points is not None

        self.n_chains = n_chains
        self.n_threads = n_threads

        self.states = None
        self.markov_chains = []
        self.rngs = []

        k = 0
        for i in range(n_chains):
            self.markov_chains.append(MarkovChain(self.proposal, self.problem.model, starting_points[k]))
            self.rngs.append(RandomNumberGenerator(seed, i))
            k += 1 if k-1 < len(starting_points) else 0


    def sample(self, n_samples, thinning = 1):
        states = [[] for i in range(self.n_chains)]

        for i in tqdm(range(n_samples)):
            for j in range(self.n_chains):
                states[j].append(self.markov_chains[j].draw(self.rngs[j], thinning))

        self.states = np.array(states)
        return self.states


