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
Obviously, this can only work, as long as the Python model implements the  ``compute_negative_log_likelihood`` method.
Depending on the algorithm you aim to use, it might also be necessary to further implement

* ``compute_log_likelihood_gradient``

* ``compute_expected_fisher_information``

For more details on signature and return type, please refer to ``hopsy.PyModel``. 


Example code
------------

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


.. class:: DegenerateMultivariateGaussianModel

   A degenerate multivariate Gaussian model is a Gaussian model which is invariant in some dimensions of the
   input vector. As an example, consider the one-dimensional squared exponential as a function of two input 
   variables

   .. math::
      f(x_1, x_2) = \exp\big\{ -x_1^2 \big\}

   then this function is invariant under the second dimension. We also say, that the second component of the 
   input vector :math:`(x_1, x_2)` is inactive. The degenerate multivaraite Gaussian is defined
   as a regular Gaussian in :math:`n-k` dimensions, where the input vector has :math:`n` dimensions but :math:`k`
   of its components are inactive.

   Technically, this is realized by removing the rows and columns from the mean vector and covariance matrix, that 
   correspond to the inactive dimensions. This then basically constructs a Gaussian in :math:`n-k` dimensions. However,
   unlike a standard multivariate Gaussian model, this model will still (and only) accept input vectors of dimension 
   :math:`n-k`.

   **Attributes:**

   .. attribute:: mean: numpy.ndarray[float64[n,1]]

      The full mean vector over all - active and inactive - input components.

   .. attribute:: covariance: numpy.ndarray[float64[n,n]]

      The full covariance matrix over all - active and inactive - input components.

   .. attribute:: inactives: List[int]

      A list of indices of the inactive dimensions. If we are in :math:`n`-dimensional space, then its elements
      should be integers between 0 and :math:`n-1`.

   **Methods:**

   .. method:: DegenerateMultivariateGaussianModel(mean, covariance, inactives)

      :param mean: Gaussian mean vector 
      :type mean: numpy.ndarray[float64[n,1]]

      :param covariance: Gaussian covariance matrix 
      :type covariance: numpy.ndarray[float64[n,n]]

      :param inactives: List of inactive dimensions 
      :type inactives: List[int]

      :return: 
      :rtype: hopsy.DegenerateMultivariateGaussianModel

      Constructs a ``DegenerateMultivariateGaussianModel`` with given mean and covariance and `deactivates` 
      all dimensions specified in ``inactives``. This works by removing the corresponding rows and columns
      from the mean and covariance. 
      
      Passing an empty list as ``inactives`` will actually define a standard multivariate Gaussian.

   .. method:: compute_negative_log_likelihood(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :return: 
      :rtype: float

      Computes the negative logarithm of the probability density function of a multivariate Gaussian model in 
      :math:`m-k` dimensions at ``x``. Note that `x` still has to have dimension :math:`n`.

   .. method:: compute_log_likelihood_gradient(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :return:
      :rtype: numpy.ndarray[float64[n,1]] 

      Computes the gradient of the logarithm of the probability density function of a multivariate Gaussian 
      model in :math:`n-k` dimensions at ``x``. Note that `x` still has to have dimension :math:`n`.

   .. method:: compute_expected_fisher_information(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :return:
      :rtype:numpy.ndarray[float64[n,n]] 

      Computes the expected fisher information of a multivariate Gaussian model 
      in :math:`n-k` dimensions at ``x``. This turns out to be just the reduced covariance matrix. 
      Note that `x` still has to have dimension :math:`n`.


.. class:: MixtureModel

   The ``MixtureModel`` is a weighted sum of 
   ``PyModel`` components and thus arbitrary components can be used.

   **Attributes:**

   .. attribute:: components: List[hopsy.PyModel]

   .. attribute:: weights: List[float]

   **Methods:**

   .. method:: MixtureModel(components)

      :param components: Model components
      :type components: List[hopsy.PyModel] 

      Construct a ``MixtureModel`` as (equally weighted) sum over the elements from ``components``.

   .. method:: MixtureModel(components, weights)

      :param components: Model components
      :type components: List[hopsy.PyModel] 

      :param weights: Model weights
      :type weights: List[float]

      Construct a ``MixtureModel`` as weighted sum over the elements from ``components``.

   .. method:: compute_negative_log_likelihood(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :return: 
      :rtype: float

   .. method:: compute_log_likelihood_gradient(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :return:
      :rtype: numpy.ndarray[float64[n,1]] 

   .. method:: compute_expected_fisher_information(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :return:
      :rtype: numpy.ndarray[float64[n,n]] 


.. class:: MultivariateGaussianModel

   **Attributes:**

   .. attribute:: mean: numpy.ndarray[float64[n,1]]

   .. attribute:: covariance: numpy.ndarray[float64[n,n]]

   **Methods:**

   .. method:: MultivariateGaussianModel(mean, covariance)

      :param mean:
      :type mean:

      :param covariance:
      :type covariance:

      :return:
      :rtype: hopsy.MultivariateGaussianModel

   .. method:: compute_negative_log_likelihood(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :return: 
      :rtype: float

   .. method:: compute_log_likelihood_gradient(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :return:
      :rtype: numpy.ndarray[float64[n,1]] 

   .. method:: compute_expected_fisher_information(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :return:
      :rtype: numpy.ndarray[float64[n,n]] 


.. class:: PyModel

   The ``PyModel`` class allows you to interface arbitrary models implemented in Python to work correctly with
   hops, given that the ``model`` object you pass implements the required functions:
   
   * ``compute_negative_log_likelihood``

   * ``compute_log_likelihood_gradient``

   * ``compute_expected_fisher_information``

   **Attributes:**

   .. attribute:: model: object

   **Methods:**

   .. method:: PyModel(model)

      :param model:
      :type model: object

      :return:
      :rtype: hopsy.PyModel

      Wraps the passed object such that it can be correctly called from hops. 

   .. method:: compute_negative_log_likelihood(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :return: 
      :rtype: float

      Returns the value of ``model.compute_negative_log_likelihod(x)``.

   .. method:: compute_log_likelihood_gradient(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :return:
      :rtype: numpy.ndarray[float64[n,1]] 

      Returns the value of ``model.compute_log_likelihood_gradient(x)``.

   .. method:: compute_expected_fisher_information(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :return:
      :rtype:numpy.ndarray[float64[n,n]] 

      Returns the value of ``model.compute_expected_fisher_information(x)``.


.. class:: RosenbrockModel

   **Attributes:**

   .. attribute:: scale: float

   .. attribute:: shift: numpy.ndarray[float64[k,1]]

   **Methods:**

   .. method:: RosenbrockModel(scale, shift)

      :param scale: 
      :type scale: float

      :param shift: 
      :type shift: numpy.ndarray[float64[k,1]]

      :return: 
      :rtype: float

   .. method:: compute_negative_log_likelihood(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :return: 
      :rtype: float

   .. method:: compute_log_likelihood_gradient(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :return:
      :rtype: numpy.ndarray[float64[n,1]] 

   .. method:: compute_expected_fisher_information(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :return:
      :rtype: numpy.ndarray[float64[n,n]] 


.. class:: UniformModel

    The ``hopsy.UniformModel`` defines the uniform target distribution on the polytope

    .. math::
        \pi(x) := \frac{1}{Z} \mathbf{1}_{\mathcal{P}}(x)

    where

    .. math::
        Z = \int_{\mathcal{P}} \mathbf{1}_{\mathcal{P}}(x) \mathrm{d}x

    Note that the uniform distribution is only well-defined, if the volume of the polytope is finite and
    thus if the polytope is bounded in all dimensions. So for example, :math:`A = 1` and :math:`b = 0` 
    define the the inequality :math:`Ax = x \leq 0 = b` which defines the negative real line. This can be
    seen as an unbounded one-dimensional polytope and the uniform distribution is thus not well defined on it.

    To prevent your polytope from being unbounded, you can use ``hopsy.add_box_constraints`` to add box constraints,
    that guarantee your polytope to be bounded. For more details, please refer :doc:`here<Problem>`.

   **Methods:**

   .. method:: UniformModel(x)

      :return: 
      :rtype: hopsy.UniformModel

   .. method:: compute_negative_log_likelihood(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :raises RuntimeError: always

      :return: 
      :rtype: float

      The negative log-likelihood for the uniform model is the unknown constant :math:`\frac{1}{Z}`, 
      which depends on the volume of the support of the density. The volume is further only well-defined in 
      dependence of the polytope, which is not known to the ``hopsy.UniformModel``.
      In the Metropolis-Hastings algorithm, this constant cancels out and is not needed for computing the 
      acceptance rate. Thus, this function is only available for technical reasons and **will always throw 
      an exception, when being called.**

   .. method:: compute_log_likelihood_gradient(x)

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :raises RuntimeError: always

      :return:
      :rtype: numpy.ndarray[float64[n,1]] 

      For similar reasons as with the ``compute_negative_log_likelihod``, this function **will always throw an 
      exception when being called.**

   .. method:: compute_expected_fisher_information(x) 

      :param x: Input vector
      :type x: numpy.ndarray[float64[n,1]]

      :raises RuntimeError: always

      :return:
      :rtype: numpy.ndarray[float64[n,n]] 

      For similar reasons as with the ``compute_negative_log_likelihod``, this function **will always throw an 
      exception when being called.**


