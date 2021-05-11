import hopsy
import numpy as np

import sys

 
class GaussianProposal:
    def __init__(self, A: np.ndarray, b: np.ndarray, x: np.ndarray, cov: np.ndarray):
        self.A = A
        self.b = b
        self.x = x
        self.cov = cov
        self.r = 1
        self.proposal = x

    def propose(self):
        mean = np.zeros((len(self.cov),))
        y = np.random.multivariate_normal(mean, self.cov).reshape(-1, 1)
        self.proposal = self.x + self.r * y 

    def accept_proposal(self):
        self.x = self.proposal

    def calculate_log_acceptance_probability(self) -> float:
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


class AdaptiveGaussianProposal:
    def __init__(self, A: np.ndarray, b: np.ndarray, x: np.ndarray, eps = 0.001):
        self.A = A
        self.b = b
        self.x = x
        self.eps = eps
        self.r = 1
        self.proposal = x
        self.t = 0
        self.cov = np.identity(len(x))
        self.mean = np.zeros(x.shape)


    def propose(self):
        new_mean = (self.t * self.mean + self.x) / (self.t + 1)
        self.cov = ((self.t - 1) * self.cov + self.t * np.outer(self.mean, self.mean) - (self.t + 1) * np.outer(new_mean, new_mean) + np.outer(self.x, self.x) + self.eps * np.identity(len(self.x))) / self.t if self.t > 0 else self.cov
        self.t += 1

        proposal_mean = np.zeros((len(self.cov),))
        y = np.random.multivariate_normal(proposal_mean, self.cov).reshape(-1, 1)
        self.proposal = self.x + self.r * y 

    def accept_proposal(self):
        self.x = self.proposal

    def calculate_log_acceptance_probability(self) -> float:
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
        return "AdaptiveGaussianPyProposal"

A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
b = np.array([[1], [1], [0], [0]]);

x0 = np.array([[0.1], [0.1]])

mu = np.zeros((2,1))
cov = 0.1*np.identity(2)

gaussian_proposal = GaussianProposal(A, b, x0, np.identity(2))
adaptive_proposal = AdaptiveGaussianProposal(A, b, x0)

model = hopsy.MultivariateGaussianModel(mu, cov)
problem = hopsy.Problem(A, b, model)

gaussian_run = hopsy.Run(problem, gaussian_proposal)
adaptive_run = hopsy.Run(problem, adaptive_proposal)

gaussian_run.set_starting_points([x0])
adaptive_run.set_starting_points([x0])

gaussian_run.sample(10000)
adaptive_run.sample(10000)

gaussian_stepsize = gaussian_run.get_stepsize()
adaptive_stepsize = adaptive_run.get_stepsize()

gaussian_acc_rate = hopsy.compute_acceptance_rate(gaussian_run.data)[0]
adaptive_acc_rate = hopsy.compute_acceptance_rate(adaptive_run.data)[0]

gaussian_esjd = hopsy.compute_expected_squared_jump_distance(gaussian_run.data)[0]
adaptive_esjd = hopsy.compute_expected_squared_jump_distance(adaptive_run.data)[0]

if len(sys.argv) == 1 or sys.argv[1] != "test":
    print("         | Gaussian proposal"                         + " | Adaptive proposal")
    print("---------+------------------"                         + "-+------------------")
    print("Stepsize |               " + str(gaussian_stepsize)   + " |               " + str(adaptive_stepsize))
    print("Acc Rate |             " + str(gaussian_acc_rate)[:5] + " |             " + str(adaptive_acc_rate)[:5]) 
    print("ESJD     |             " + str(gaussian_esjd)[:5]     + " |             " + str(adaptive_esjd)[:5])

    import matplotlib.pyplot as plt
    gaussian_states = np.array(gaussian_run.data.states[0])
    adaptive_states = np.array(adaptive_run.data.states[0])

    fig = plt.figure(figsize=(35,35))
    fig.patch.set_alpha(1)
    ax = fig.gca()
    ax.scatter(adaptive_states[:,0], adaptive_states[:,1])
    ax.scatter(gaussian_states[:,0], gaussian_states[:,1])
    plt.show()
