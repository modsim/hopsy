import numpy as np

import hopsy

if __name__ == "__main__":
    A = np.array([[1, 1], [-1, 0], [0, -1]])
    b = np.array([[1], [0], [0]])

    mu = np.zeros((2, 1))
    cov = np.identity(2)

    model = hopsy.MultivariateGaussianModel(mu, cov)
    problem = hopsy.Problem(A, b, model)
    run = hopsy.Run(problem)

    x0 = hopsy.compute_chebyshev_center(problem)

    run.starting_points = [x0]

    # this computes the chebyshev center as no other starting point has been
    # previously set
    run.init()

    run.sample()
