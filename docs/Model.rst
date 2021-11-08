Models
============================

hopsy ships with a number of simple models, also known as target functions. 
Together with the polytope 

.. math::
   \mathcal{P} := \{ x : Ax \leq b \}

the model forms a ``hopsy.Problem``.

The simple models provided are 

* :class:`hopsy.GaussianMixture``: A mixture Gaussian model, which is a weighted linear combination of multivariate Gaussians.

* :class:`hopsy.Mixture``: A mixture model, which is a weighted linear combination of arbitrary model components.

* :class:`hopsy.MultivariateGaussian`: A multivariate Gaussian which can also have invariant dimensions, meaning it ignores some dimensions of the input vector. This is interesting for simulating non-identifiabilities.

* :class:`hopsy.Rosenbrock``: Rosenbrock's banana function, a popular test function for computational optimization.

* :class:`hopsy.Uniform``: The uniform distribution on the Polytope :math:`\mathcal{P}`


Custom Python models
--------------------

Often, e.g. in physics or parameter inference, one wants to sample more complex target functions. 
hopsy allows to do so, by providing ``hopsy.PyModel`` which acts as an interface between an arbitrary Python model and HOPS.
This works by simply delegating any function calls to ``hopsy.PyModel`` (from within HOPS) to the respective Python method. 
Obviously, this can only work, as long as the Python model implements the  required methods. 
In the simplest case, this is only the ``compute_negative_log_likelihood`` method, which should return the negative
log-likelihood of the target distribution.
Depending on the algorithm you aim to use, it might also be necessary to further implement

* ``compute_log_likelihood_gradient``

* ``compute_expected_fisher_information``

For more details on signature and return type, please refer to ``hopsy.PyModel`` and/or consider the code below


Example code
^^^^^^^^^^^^

We present a quick example code for a Python-implemented Gaussian target function:

::

   import hopsy
   import numpy as np
   
   # define the Gaussian model
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


Reference
---------

.. currentmodule:: hopsy

.. autoclass:: GaussianMixture

   .. automethod:: __init__
   .. automethod:: compute_negative_log_likelihood
   .. automethod:: compute_log_likelihood_gradient
   .. automethod:: compute_expected_fisher_information

.. autoclass:: Mixture

   .. automethod:: __init__
   .. automethod:: compute_negative_log_likelihood
   .. automethod:: compute_log_likelihood_gradient
   .. automethod:: compute_expected_fisher_information

.. autoclass:: MultivariateGaussian

   .. automethod:: __init__
   .. automethod:: compute_negative_log_likelihood
   .. automethod:: compute_log_likelihood_gradient
   .. automethod:: compute_expected_fisher_information

.. autoclass:: PyModel

   .. automethod:: __init__
   .. automethod:: compute_negative_log_likelihood
   .. automethod:: compute_log_likelihood_gradient
   .. automethod:: compute_expected_fisher_information

.. autoclass:: Rosenbrock

   .. automethod:: __init__
   .. automethod:: compute_negative_log_likelihood
   .. automethod:: compute_log_likelihood_gradient
   .. automethod:: compute_expected_fisher_information

.. autoclass:: Uniform

   .. automethod:: __init__
   .. automethod:: compute_negative_log_likelihood
   .. automethod:: compute_log_likelihood_gradient
   .. automethod:: compute_expected_fisher_information

