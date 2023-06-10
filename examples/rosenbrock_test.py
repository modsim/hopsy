import numpy as np

import hopsy


A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
b = np.array([10, 10, 10, 10]).reshape(-1, 1)

model = hopsy.Rosenbrock(0.1, np.array([[1]]))
problem = hopsy.Problem(A, b, model)

mc = hopsy.MarkovChain(problem, starting_point=np.zeros((2,1)))
mc.proposal.stepsize = 0.1

hopsy.sample(mc, rngs=hopsy.RandomNumberGenerator(1), n_samples=10000, thinning=100)

