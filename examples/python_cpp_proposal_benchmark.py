import hopsy
import numpy as np
import sys
import matplotlib.pyplot as plt

import time

class GaussianProposal:
    def __init__(self, A: np.ndarray, b: np.ndarray, x: np.ndarray, cov: np.ndarray):
        self.A = A
        self.b = b
        self.x = x
        self.cov = cov
        self.r = 1
        self.proposal = x

    def propose(self):
        mean = np.zeros((len(cov),))
        y = np.random.multivariate_normal(mean, cov).reshape(-1, 1)
        self.proposal = self.x + self.r * y 

    def accept_proposal(self):
        self.x = self.proposal

    def get_log_acceptance_probability(self) -> float:
        if ((self.A @ self.proposal - self.b) >= 0).any():
            return -np.inf
        return 0

    def get_state(self) -> np.ndarray:
        return self.x

    def set_state(self, new_state: np.ndarray):
        self.x = new_state.reshape(-1,1)

    def get_proposal(self) -> np.ndarray:
        return self.proposal

    def get_stepsize(self) -> float:
        return self.r

    def set_stepsize(self, new_stepsize: float):
        self.r = new_stepsize

    def get_name(self) -> str:
        return "PyGaussianProposal"


A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
b = np.array([[1], [1], [0], [0]]);

x0 = np.array([[0.1], [0.1]])

mu = np.zeros((2,1))
cov = 0.1*np.identity(2)

proposal = GaussianProposal(A, b, x0, 0.5*np.identity(2))

model = hopsy.MultivariateGaussianModel(mu, cov)
problem = hopsy.Problem(A, b, model)

run = hopsy.Run(problem, proposal)
run2 = hopsy.Run(problem, "Gaussian")

## alternatively use (which internally happens anyways)
# run = hopsy.Run(problem, hopsy.PyProposal(proposal))

run.set_starting_points([x0])
run2.set_starting_points([x0])

times = []

for i in range(10):
    start = time.time()
    run.sample(10000)
    end = time.time()
    times.append(end - start)

print(times[-1], np.mean(times))
times = []

for i in range(10):
    start = time.time()
    run2.sample(10000)
    end = time.time()
    times.append(end - start)

print(times[-1], np.mean(times))

if len(sys.argv) == 1 or sys.argv[1] != "test":
    states = np.array(run.get_data().get_states()[0])

    fig = plt.figure(figsize=(35,35))
    fig.patch.set_alpha(1)
    ax = fig.gca()
    ax.scatter(states[:,0], states[:,1])
    plt.show()
