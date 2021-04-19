import hopsy
import numpy as np

A = np.array([[1, 0], [0, 1], [-1, 0], [-1, 0]])
b = np.array([[1], [1], [0], [0]]);

mu = np.zeros((2,1))
cov = np.identity(2)

model = hopsy.MultivariateGaussianModel(mu, cov)
problem = hopsy.Problem(A, b, model)
run = hopsy.Run(problem, "HitAndRun", 100, 1)

run.sample()

