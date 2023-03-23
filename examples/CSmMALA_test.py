import numpy as np

import hopsy

if __name__ == "__main__":
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array([[1], [1], [0], [0]])

    mu = np.zeros((2, 1))
    cov = np.identity(2)

    model = hopsy.MultivariateGaussianModel(mu, cov)
    problem = hopsy.Problem(A, b, model)
    run = hopsy.Run(problem)

    run.starting_points = [np.array([[0.5], [0.5]])]
    # run.enable_rounding()

    run.sample()
