import sys
from typing import List

import numpy as np

import hopsy


class PyProposalWithFullBoilerplate:
    def __init__(
        self, A: np.ndarray, b: np.ndarray, state: np.ndarray, cov: np.ndarray
    ):
        self.A = A
        self.b = b
        self.state = state
        self.cov = cov
        self.__stepsize = 1
        self.proposal = state

    def propose(self, rng):
        mean = np.zeros((len(cov),))
        y = np.random.multivariate_normal(mean, cov).reshape(-1, 1)
        self.proposal = self.state + self.stepsize * y
        return self.proposal

    def accept_proposal(self):
        self.state = self.proposal
        return self.state

    def log_acceptance_probability(self) -> float:
        if ((self.A @ self.proposal - self.b) >= 0).any():
            return -np.inf
        return 0

    @property
    def stepsize(self):
        return self.__stepsize

    def has_stepsize(self) -> bool:
        return True

    def name(self) -> str:
        return "PyGaussianProposal"


class PyProposalMinimalBoilerplate:
    def __init__(
        self, A: np.ndarray, b: np.ndarray, state: np.ndarray, cov: np.ndarray
    ):
        self.proposal = None
        self.A = A
        self.b = b
        self.state = state
        self.cov = cov

    def propose(self, rng):
        mean = np.zeros((len(cov),))
        y = np.random.multivariate_normal(mean, cov).reshape(-1, 1)
        self.proposal = self.state + y
        return self.proposal

    def log_acceptance_probability(self) -> float:
        if ((self.A @ self.proposal - self.b) >= 0).any():
            return -np.inf
        return 0

    def accept_proposal(self):
        self.state = self.proposal
        return self.state


if __name__ == "__main__":
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array([[1], [1], [0], [0]])

    x0 = np.array([[0.1], [0.1]])

    mu = np.zeros((2, 1))
    cov = 0.1 * np.identity(2)

    model = hopsy.Gaussian(mu, cov)
    problem = hopsy.Problem(A, b, model)

    mc = hopsy.MarkovChain(
        problem=problem,
        proposal=hopsy.UniformCoordinateHitAndRunProposal,
        starting_point=mu,
    )

    assert mc.proposal.name == "CoordinateHitAndRun"

    mc.proposal = hopsy.GaussianHitAndRunProposal(problem, x0)
    assert mc.proposal.name == "HitAndRun"

    mc.proposal = hopsy.UniformHitAndRunProposal(problem, x0)
    assert mc.proposal.name == "HitAndRun"

    mc.proposal = PyProposalWithFullBoilerplate(A, b, x0, cov)
    assert mc.proposal.name == "PyGaussianProposal"

    # tests sampling with custom proposal
    hopsy.sample(mc, hopsy.RandomNumberGenerator(42), n_samples=10)

    # Wrapping proposal in python in order to fill in boilerplate code
    proposal = hopsy.PyProposal(PyProposalMinimalBoilerplate(A, b, x0, cov))
    assert proposal.name == "PyProposalMinimalBoilerplate"
    mc.proposal = proposal
    assert mc.proposal.name == "PyProposalMinimalBoilerplate"
    _, samples = hopsy.sample(mc, hopsy.RandomNumberGenerator(42), n_samples=10)

    print(mc.proposal.name)
    print(_, samples)

    # Letting C++ wrap the proposal for you
    proposal = PyProposalMinimalBoilerplate(A, b, x0, cov)
    # proposal.name does not exist in python now!
    mc.proposal = proposal
    # mc.proposal.name is default generated in C++ backend and is just the class name
    assert mc.proposal.name == "PyProposalMinimalBoilerplate"

    _, samples = hopsy.sample(mc, hopsy.RandomNumberGenerator(42), n_samples=10)
    print(mc.proposal.name)
    print(_, samples)
    print(mc.proposal.proposal)
