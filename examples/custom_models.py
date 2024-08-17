import numpy as np

import hopsy

# Important note:
# Functionality within multiprocessing requires that the main module be importable by the children. See https://docs.python.org/3/library/multiprocessing.html#using-a-pool-of-workers. This means, that the gaussian mixture needs to be imported from a file and not defined here


class GaussianMixture:
    def __init__(self, mu1, mu2):
        epsilon = 0.05
        cov = epsilon * np.eye(2, 2)
        self.model1 = hopsy.Gaussian(mean=mu1, covariance=cov)
        self.model2 = hopsy.Gaussian(mean=mu2, covariance=cov)

    def log_density(self, x):
        return np.log(
            np.exp(-self.model1.compute_negative_log_likelihood(x))
            + np.exp(-self.model2.compute_negative_log_likelihood(x))
        )


class CustomModel:
    """The custom model is created by you and its implementation is determined by your domain.
    If you have a simulator for your problem domain, you can interface with it in this class, which should adapt your simulator to the interface that hopsy expects.

    Note, that this model does NOT have to inheret from anything, as long as it has the correct methods.
    """

    def compute_negative_log_likelihood(self, parameter):
        """
        computes the negative log likelihood of a parameter vector according to your custom model
        """

        # dummy implementation, in this case a gaussian
        mu = np.array([[-10], [0]])
        cov = np.array([[10, 2.5], [2.5, 10]])
        return (
            (parameter.reshape(-1, 1) - mu).T
            @ np.linalg.inv(cov)
            @ (parameter.reshape(-1, 1) - mu)
        )[0, 0]

    # The properties A and b and not mandatory, but for convenience we put them here into the model. They define the polytope as Ax < b. In this example, A and b will shape a 2d simplex.
    @property
    def A(self):
        return np.array([[1, 1]])

    @property
    def b(self):
        return np.array([1])


if __name__ == "__main__":
    pass
