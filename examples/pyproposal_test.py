import pickle
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
        self.state = state.flatten()
        self.cov = cov
        self.__stepsize = 1
        self.proposal = state.flatten()

    def propose(self, rng):
        mean = np.zeros((len(cov),))
        y = np.random.multivariate_normal(mean, cov).reshape(-1, 1).flatten()
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
        self.proposal = state.flatten()
        self.A = A
        self.b = b
        self.state = state.flatten()
        self.cov = cov

    def propose(self, rng):
        mean = np.zeros((len(cov),))
        y = np.random.multivariate_normal(mean, cov).reshape(-1, 1).flatten()
        print("y", y)
        self.proposal = self.state + y
        print("py proposing ", self.proposal)
        return self.proposal

    def log_acceptance_probability(self) -> float:
        print("py acceptance prob for", self.proposal)
        if ((self.A @ self.proposal - self.b) >= 0).any():
            return -np.inf
        return 0


if __name__ == "__main__":
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array([[1], [1], [0], [0]])

    x0 = np.array([[0.1], [0.1]])

    mu = np.zeros((2, 1))
    cov = 0.1 * np.identity(2)

    model = hopsy.Gaussian(mu, cov)
    problem = hopsy.Problem(A, b, model)

    mcs = [
        hopsy.MarkovChain(
            problem=problem,
            proposal=hopsy.UniformCoordinateHitAndRunProposal,
            starting_point=mu,
        )
        for i in range(4)
    ]
    mc = mcs[0]
    assert mc.proposal.name == "CoordinateHitAndRun"
    # _, samples = hopsy.sample(mcs, [hopsy.RandomNumberGenerator(42 + i) for i in range(4)], n_samples=10, n_procs=4)

    mc.proposal = hopsy.GaussianHitAndRunProposal(problem, x0)
    for i in range(4):
        mcs[i].proposal = mc.proposal
    assert mc.proposal.name == "HitAndRun"

    mc.proposal = hopsy.UniformHitAndRunProposal(problem, x0)
    for i in range(4):
        mcs[i].proposal = mc.proposal
    assert mc.proposal.name == "HitAndRun"
    # _, samples = hopsy.sample(mcs, [hopsy.RandomNumberGenerator(42 + i) for i in range(4)], n_samples=10, n_procs=4)

    mc.proposal = PyProposalWithFullBoilerplate(A, b, x0, cov)
    for i in range(4):
        mcs[i].proposal = mc.proposal
    assert mc.proposal.name == "PyGaussianProposal"
    # _, samples = hopsy.sample(mcs, [hopsy.RandomNumberGenerator(42 + i) for i in range(4)], n_samples=10, n_procs=4)

    # tests sampling with custom proposal
    # hopsy.sample(mc, hopsy.RandomNumberGenerator(42), n_samples=10)

    # Wrapping proposal in python in order to fill in boilerplate code
    proposal = hopsy.PyProposal(PyProposalMinimalBoilerplate(A, b, x0, cov))
    assert proposal.name == "PyProposalMinimalBoilerplate"
    mc.proposal = proposal
    for i in range(4):
        mcs[i].proposal = mc.proposal
    assert mc.proposal.name == "PyProposalMinimalBoilerplate"
    dump = pickle.dumps(mc)
    new_mc = pickle.loads(dump)
    print("pickled", mc)
    print("loaded", new_mc)
    # _, samples = hopsy.sample(mc, hopsy.RandomNumberGenerator(42), n_samples=10)
    _, samples = hopsy.sample(
        mcs,
        [hopsy.RandomNumberGenerator(42 + i) for i in range(4)],
        n_samples=10,
        n_procs=4,
    )

    # Letting C++ wrap the proposal for you
    proposal = PyProposalMinimalBoilerplate(A, b, x0, cov)
    # proposal.name does not exist in python now!
    mc.proposal = proposal
    for i in range(4):
        mcs[i].proposal = mc.proposal
    # mc.proposal.name is default generated in C++ backend and is just the class name
    assert mc.proposal.name == "PyProposalMinimalBoilerplate"

    # # tests sequential sampling
    # _, samples = hopsy.sample( mc, hopsy.RandomNumberGenerator(42), n_samples=10, n_procs=1)
    # tests parallel sampling
    _, samples = hopsy.sample(
        mcs,
        [hopsy.RandomNumberGenerator(42 + i) for i in range(4)],
        n_samples=10,
        n_procs=4,
    )
    print(mc.proposal.name)
    # print(mc.proposal.proposal)
