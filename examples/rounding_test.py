import hopsy
import numpy as np

A = np.array([[-1, 0, 0, 0], 
              [0, -1, 0, 0], 
              [0, 0, -1, 0],
              [0, 0, 0, -1],
              [1, 1, 1, 1]])

b = np.array([[0], 
              [0],
              [0],
              [0],
              [1]]);

mu = np.zeros((2,1))
cov = np.identity(2)

model = hopsy.MultivariateGaussianModel(mu, cov)
problem = hopsy.Problem(A, b, model)

rounded_problem = hopsy.round(problem)
expected_transformation = np.array([[0.199999997299948, 0, 0, 0,],
								    [-0.049999999324932, 0.193649164696071, 0, 0,],
								    [-0.049999999324932, -0.064549721565262, 0.182574183370304, 0,],
								    [-0.049999999324932, -0.064549721565262, -0.091287091684950, 0.158113880873998]])

expected_center = np.array([0.200000000299977, 0.200000000299977, 0.200000000299977, 0.200000000299977])

assert (np.abs(rounded_problem.unrounding_transformation - expected_transformation) < 0.00001).all()

