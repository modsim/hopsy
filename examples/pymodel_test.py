import hopsy
import numpy as np
import sys

class GaussianModel2:
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

model1 = hopsy.Gaussian(mu, cov)
model2 = GaussianModel2(2*mu, cov)

problem = hopsy.Problem(A, b, model2)

print('problem.model')
print(problem.model)
print('problem.model.mu')
print(problem.model.mu)

problem.model = model1
print('problem.model')
print(problem.model)
print('problem.model.mean')
print(problem.model.mean)


problem.model = model2
print('problem.model')
print(problem.model)
print('problem.model.mean')
print(problem.model.mu)



mc = hopsy.MarkovChain(problem=problem, proposal=hopsy.UniformCoordinateHitAndRunProposal, starting_point=mu)

print('markov chain model switches')
print(mc.model)

print('switch to model 2')
mc.model = model2
print('switch to model 1')
mc.model = model1
print('switch to model 2')
mc.model = model2
print('switch to model 1')
mc.model = model1

print(mc.model)

## alternatively use (which internally happens anyways)
# problem = hopsy.Problem(A, b, hopsy.PyModel(model))

