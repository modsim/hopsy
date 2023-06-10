import sys
import time

import numpy as np

import hopsy


class PyGaussianProposal:
    def __init__(
        self, A: np.ndarray, b: np.ndarray, starting_point: np.ndarray, cov: np.ndarray
    ):
        self.A = A
        self.b = b
        self.__state = starting_point.reshape(-1, 1)
        self.cov = cov
        self.__stepsize = 1
        self.__proposal = self.state
        self.__dimension_names = ['x_' + str(i) for i in range(self.state.shape[0])]

    def propose(self, rng: hopsy.RandomNumberGenerator=None):
        mean = np.zeros((len(cov),))
        y = np.random.multivariate_normal(mean, cov).reshape(-1, 1)
        self.__proposal = self.__state + self.stepsize * y
        return self.__proposal

    def accept_proposal(self):
        self.__state = self.__proposal
        return self.__state

    def log_acceptance_probability(self) -> float:
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

    @property
    def dimension_names(self):
        # can be empty, e.g., when model has dimension_names.
        return self.__dimension_names

    @dimension_names.setter
    def dimension_names(self, value): 
        self.__dimension_names = value 

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

sampler = hopsy.MarkovChain(problem, proposal=hopsy.GaussianProposal, starting_point=x0)


times = []

rng = hopsy.RandomNumberGenerator(42)

for i in range(10):
    start = time.time()
    hopsy.sample(sampler, rng, n_samples=10000)
    end = time.time()
    times.append(end - start)

cpp_time = (times[-1], np.mean(times))
times = []

sampler.proposal = PyGaussianProposal(A, b, x0, cov)

for i in range(10):
    start = time.time()
    hopsy.sample(sampler, rng, n_samples=10000)
    end = time.time()
    times.append(end - start)

py_time = (times[-1], np.mean(times))

print('cpp_time', cpp_time, 'py_time', py_time)

