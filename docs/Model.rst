Models
============================

hopsy ships with a number of simple models, also known as target functions. 
Together with the polytope 

.. math::
   \mathcal{P} := \{ x : Ax \leq b \}

the model forms a ``hopsy.Problem``.

The simple models provided are 

* ``hopsy.DegenerateMultivariateGaussianModel``: A multivariate Gaussian with invariant dimensions, meaning it ignores some dimensions of the input vector. This is interesting for simulating non-identifiabilities.

* ``hopsy.MultimodalMultivariateGaussianModel``: A mixture Gaussian model, which is a weighted linear combination of multivariate Gaussians.

* ``hopsy.MultivariateGaussianModel``: A multivariate Gaussian, consisting of mean vector and covariance matrix.

* ``hopsy.RosenbrockModel``: Rosenbrock's banana function, a popular test function for computational optimization.

* ``hopsy.UniformModel``: The uniform distribution on the Polytope :math:`\mathcal{P}`


Custom Python models
--------------------

Often, e.g. in physics or parameter inference, one wants to sample more complex target functions. 
hopsy allows to do so, by providing ``hopsy.PyModel`` which acts as an interface between an arbitrary Python model and HOPS.
This works by simply delegating any function calls to ``hopsy.PyModel`` (from within HOPS) to the respective Python method. 
Obviously, this can only work, as long as the Python model implements the  ``calculate_negative_log_likelihood`` method.
Depending on the algorithm you aim to use, it might also be necessary to further implement

* ``calculate_expected_fisher_information``

* ``calculate_log_likelihood_gradient``

For more details on signature and return type, please refer to ``hopsy.PyModel``. 
We present a quick example code for a Python-implemented Gaussian target function:

::

   import hopsy
   import numpy as np
   
   # define the Gaussian model
   class GaussianModel:
       def __init__(self, mu, cov):
           self.mu = mu
           self.cov = cov
   
       def calculate_negative_log_likelihood(self, x):
           return (0.5 * (x.reshape(-1, 1) - self.mu).T @ np.linalg.inv(self.cov) @ (x.reshape(-1, 1) - self.mu))[0,0]
   
       def calculate_expected_fisher_information(self, x):
           return np.linalg.inv(self.cov)
   
       def calculate_log_likelihood_gradient(self, x):
           return -np.linalg.inv(self.cov) @ (x - self.mu)
   
   # the polytope is defined as 
   #          P := {x : Ax <= b}
   # thus we need to define A and b. these constraints form the simple box [0,1]^2.
   A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
   b = np.array([[1], [1], [0], [0]]);
   
   # next we define our target distribution as an isotropic Gaussian with mean 0 and 
   # scaled identity covariance.
   mu = np.zeros((2,1))
   cov = 0.1*np.identity(2)

   # unlike the example on the title page, we now use our own Python implementation GaussianModel
   model = GaussianModel(mu, cov)

   # the hopsy.Problem() factory can take any Python object and will wrap it in hopsy.PyModel
   problem = hopsy.Problem(A, b, model)

   # alternatively we could wrap it ourselves, though it makes no difference
   # problem = hopsy.Problem(A, b, hopsy.PyModel(model))
  
   # instantiate the run using the problem with the custom Python model...
   run = hopsy.Run(problem)

   # ... and sample
   run.sample()


Available hopsy models in detail
--------------------------------

.. currentmodule:: hopsy

.. autosummary::
   :toctree: _generate

   DegenerateMultivariateGaussianModel
   MultimodalMultivariateGaussianModel
   MultivariateGaussianModel
   PyModel
   RosenbrockModel
   UniformModel


