import sys

import numpy as np
from typing import List

import hopsy


class PyProposal:
    def __init__(
        self, A: np.ndarray, b: np.ndarray, state: np.ndarray, cov: np.ndarray
    ):
        self.A = A
        self.b = b
        self.__state = state
        self.cov = cov
        self.__stepsize = 1
        self.__proposal = state

    def propose(self):
        mean = np.zeros((len(cov),))
        y = np.random.multivariate_normal(mean, cov).reshape(-1, 1)
        self.__proposal = self.__state + self.stepsize * y
        return self.__proposal

    def accept_proposal(self):
        self.__state = self.__proposal
        return self.__state

    def compute_log_acceptance_probability(self) -> float:
        if ((self.A @ self.__proposal - self.b) >= 0).any():
            return -np.inf
        return 0

    @property
    def state(self):
        return self.__state

    @property
    def proposal(self):
        return self.__proposal

    @property
    def stepsize(self):
        return self.__stepsize


    def has_stepsize(self) -> bool:
        return True

    def get_name(self) -> str:
        return "PyGaussianProposal"



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

print("mc.proposal")
print(mc.proposal)

mc.proposal = hopsy.GaussianHitAndRunProposal(problem, x0)
print("mc.proposal")
print(mc.proposal)
print(mc.proposal.__dir__())
mc.proposal = hopsy.UniformHitAndRunProposal(problem, x0)
print("mc.proposal")
print(mc.proposal)
print(PyProposal(A, b, x0, cov).__dir__())
mc.proposal = PyProposal(A, b, x0, cov)
print("mc.proposal")
print(mc.proposal)



