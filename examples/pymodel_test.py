import hopsy
import numpy as np
import sys

class GaussianModel:
    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov

    def compute_negative_log_likelihood(self, x):
        return (0.5 * (x.reshape(-1, 1) - self.mu).T @ np.linalg.inv(self.cov) @ (x.reshape(-1, 1) - self.mu))[0,0]

    def compute_expected_fisher_information(self, x):
        return np.linalg.inv(self.cov)

    def compute_log_likelihood_gradient(self, x):
        return -np.linalg.inv(self.cov) @ (x - self.mu)

A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
b = np.array([[1], [1], [0], [0]]);

mu = np.zeros((2,1))
cov = 0.1*np.identity(2)

model = GaussianModel(mu, cov)
problem = hopsy.Problem(A, b, model)
## alternatively use (which internally happens anyways)
# problem = hopsy.Problem(A, b, hopsy.PyModel(model))

run = hopsy.Run(problem, starting_points = [np.array([[0.1], [0.1]])])

run.sample(10000)

if len(sys.argv) == 1 or sys.argv[1] != "test":
    import matplotlib.pyplot as plt
    states = np.array(run.data.states[0])

    fig = plt.figure(figsize=(35,35))
    fig.patch.set_alpha(1)
    ax = fig.gca()
    ax.scatter(states[:,0], states[:,1])
    plt.show()
