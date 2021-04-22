import hopsy
import numpy as np

#A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
#b = np.array([[1], [1], [0], [0]]);

A = np.array([[1, 1], [-1, 0], [0, -1]])
b = np.array([[1], [0], [0]]);

mu = np.zeros((2,1))
cov = np.identity(2)

model = hopsy.MultivariateGaussianModel(mu, cov)
problem = hopsy.Problem(A, b, model)
run = hopsy.Run(problem)

run.init()
print(run.get_starting_points())

run.sample()

