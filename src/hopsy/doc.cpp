#include "doc.hpp"

const char* hopsy::doc::base = R"pbdoc(
)pbdoc";



const char* hopsy::doc::RandomNumberGenerator::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::RandomNumberGenerator::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::RandomNumberGenerator::__call__ = R"pbdoc(
)pbdoc";



const char* hopsy::doc::Uniform::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Uniform::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Uniform::__call__ = R"pbdoc(
)pbdoc";



const char* hopsy::doc::Normal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Normal::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Normal::__call__ = R"pbdoc(
)pbdoc";



/*
 *  Model
 */

const char* hopsy::doc::Model::base = R"pbdoc(
Base model class. While custom models are not required to inherit from this base class, they should still
implement these functions so that all proposal mechanisms work.
See the methods for more details on how to implement the  methods.
)pbdoc";


const char* hopsy::doc::Model::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Model::computeNegativeLogLikelihood = R"pbdoc(compute_negative_log_likelihood(self, x)
deprecated:: 1.4
   Use :func:`log_density` instead.

This method is required for  any custom model and should be implemented for any custom models.

Parameters
----------
:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

Returns
-------
:return: The value of ``model.compute_negative_log_likelihod(x)``
:rtype: float
)pbdoc";


const char* hopsy::doc::Model::computeLogLikelihoodGradient = R"pbdoc(compute_log_likelihood_gradient(self, x)
deprecated:: 1.4
   Use :func:`log_gradient` instead.
For some proposals, the gradient will help converging faster as long as the gradient computation is not too slow.
If you can not compute a useful or fast enough gradient for your custom model, you can just return a zero vector with the correct dimensionality (number of rows equal to number of parameters).

Parameters
----------
x : numpy.ndarray[n, 1]
    Input vector

Returns
-------
numpy.ndarray[n, 1]
    The gradient of the (unnormalized) log-likelihood
)pbdoc";


const char* hopsy::doc::Model::computeExpectedFisherInformation = R"pbdoc(
deprecated:: 1.4
   Use :func:`log_curvature` instead.

For some proposals, the expected fisher information will help converging faster as long as the gradient computation is not too slow.
If you can not compute a useful or fast enough expected fisher information for your custom model, you can just return a zero matrix with the correct dimensionality (number of rows and cols each equal to number of parameters).

Parameters
----------
:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

Returns
-------
:return: The value of ``model.compute_expected_fisher_information(x)``
:rtype: numpy.ndarray[float64[n,n]]
)pbdoc";


const char* hopsy::doc::Model::logDensity = R"pbdoc(log_density(self, x)

This method is required for  any custom model and should be implemented for any custom models.

Parameters
----------
:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

Returns
-------
:return: The value of ``model.log_density(x)``
:rtype: float
)pbdoc";


const char* hopsy::doc::Model::logGradient = R"pbdoc(log_gradient(self, x)

For some proposals, the gradient will help converging faster as long as the gradient computation is not too slow.
If you can not compute a useful or fast enough gradient for your custom model, you can just return a zero vector with the correct dimensionality (number of rows equal to number of parameters).

Parameters
----------
x : numpy.ndarray[n, 1]
    Input vector

Returns
-------
numpy.ndarray[n, 1]
    The gradient of the (unnormalized) log_density
)pbdoc";


const char* hopsy::doc::Model::logCurvature = R"pbdoc(log_curvature(self x)

For some proposals, the curvature will help converging faster as long as the gradient computation is not too slow.
The curvature is a square matrix which is (semi-)positive definit. For example one can use the fisher information, the hessian, linear approximations to the hessian and so on.
If you can not compute a useful or fast enough curvature for your custom model, you can just return a zero matrix with the correct dimensionality (number of rows and cols each equal to number of parameters).
Alternatively do not implement it and it will not be used.

Parameters
----------
:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

Returns
-------
:return: The value of ``model.log_curvature(x)``
:rtype: numpy.ndarray[float64[n,n]]
)pbdoc";


/*
 *  Gaussian
 */

const char* hopsy::doc::Gaussian::base = R"pbdoc(
hopsy.Gaussian(self, mean=[0, 0], covariance=[[1, 0], [0, 1]], inactives=[])

Gaussian model which can be invariant in some dimensions of the
input vector. As an example, consider the one-dimensional squared exponential as a function of two input
variables

.. math::
   f(x_1, x_2) = \exp\big\{ -x_1^2 \big\}

then this function is invariant under the second dimension. We also say, that the second component of the
input vector :math:`(x_1, x_2)` is inactive. The degenerate multivariate Gaussian is defined
as a regular Gaussian in :math:`n-k` dimensions, where the input vector has :math:`n` dimensions but :math:`k`
of its components are inactive.

Technically, this is realized by removing the rows and columns from the mean vector and covariance matrix, that
correspond to the inactive dimensions. This then basically constructs a Gaussian in :math:`n-k` dimensions.
However, unlike a standard multivariate Gaussian model, this model will still (and only) accept input vectors
of dimension :math:`n`.

Passing an empty list as ``inactives`` will define a common multivariate Gaussian.

Parameters
----------
mean : numpy.ndarray[n, 1]
    Gaussian mean vector
covariance : numpy.ndarray[n, n]
    Gaussian covariance matrix
inactives : list[int]
    List of inactive dimensions, so entries should be between 0 and :math:`n-1`.

)pbdoc";


const char* hopsy::doc::Gaussian::__init__ = R"pbdoc(
Constructs a ``hopsy.Gaussian`` with given mean and covariance and `deactivates`
all dimensions specified in ``inactives``. This works by removing the corresponding rows and columns
from the mean and covariance.

Passing an empty list as ``inactives`` will define a common multivariate Gaussian.

Parameters
----------
mean : numpy.ndarray[n, 1]
    Gaussian mean vector
covariance : numpy.ndarray[n, n]
    Gaussian covariance matrix
inactives : list[int]
    List of inactive dimensions, so entries should be between 0 and :math:`n-1`.

)pbdoc";


const char* hopsy::doc::Gaussian::mean = R"pbdoc(
The Gaussian's mean vector in full space, having :math:`n` entries.
)pbdoc";


const char* hopsy::doc::Gaussian::covariance = R"pbdoc(
The Gaussian's covariance matrix in full space, thus having :math:`n^2` entries.
)pbdoc";


const char* hopsy::doc::Gaussian::inactives = R"pbdoc(
List of indices of the inactive dimensions. E.g. ``inactives = [0, 1]`` will render dimension 0 and 1 inactive.
)pbdoc";


const char* hopsy::doc::Gaussian::computeNegativeLogLikelihood = R"pbdoc(compute_negative_log_likelihood(self, x)
deprecated:: 1.4
   Use :func:`log_density` instead.

Computes the negative logarithm of the probability density function of a multivariate Gaussian model in
:math:`m-k` dimensions at ``x``. Note that `x` still has to have dimension :math:`n`.

Parameters
----------
x : numpy.ndarray[n, 1]
    Input vector

Returns
-------
float
    The (unnormalized) negative log-likelihood

)pbdoc";


const char* hopsy::doc::Gaussian::computeLogLikelihoodGradient = R"pbdoc(compute_log_likelihood_gradient(self, x)
deprecated:: 1.4
   Use :func:`log_gradient` instead.

Computes the gradient of the logarithm of the probability density function of a multivariate Gaussian
model in :math:`n-k` dimensions at ``x``. Note that `x` still has to have dimension :math:`n`.

Parameters
----------
x : numpy.ndarray[n, 1]
    Input vector

Returns
-------
numpy.ndarray[n, 1]
    The gradient of the (unnormalized) log-likelihood

)pbdoc";


const char* hopsy::doc::Gaussian::computeExpectedFisherInformation = R"pbdoc(compute_expected_fisher_information(self, x)
deprecated:: 1.4
   Use :func:`log_curvature` instead.

Computes the expected fisher information of a multivariate Gaussian model
in :math:`n-k` dimensions at ``x``. This turns out to be just the reduced covariance matrix.
Note that `x` still has to have dimension :math:`n`.

Parameters
----------
x : numpy.ndarray[n, 1]
    Input vector

Returns
-------
numpy.narray[n, n]
    The expected Fisher information matrix

)pbdoc";

const char* hopsy::doc::Gaussian::logDensity = R"pbdoc(log_density(self, x)

Computes the probability density function of a multivariate Gaussian model in
:math:`m-k` dimensions at ``x``. Note that `x` still has to have dimension :math:`n`.

Parameters
----------
x : numpy.ndarray[n, 1]
    Input vector

Returns
-------
float
    The (unnormalized) density

)pbdoc";



const char* hopsy::doc::Gaussian::logGradient = R"pbdoc(log_gradient(self, x)

Computes the gradient of the logarithm of the probability density function of a multivariate Gaussian
model in :math:`n-k` dimensions at ``x``. Note that `x` still has to have dimension :math:`n`.

Parameters
----------
x : numpy.ndarray[n, 1]
    Input vector

Returns
-------
numpy.ndarray[n, 1]
    The gradient of the (unnormalized) log_density

)pbdoc";


const char* hopsy::doc::Gaussian::logCurvature = R"pbdoc(log_curvature(self x)

Computes the expected fisher information of a multivariate Gaussian model
in :math:`n-k` dimensions at ``x``. This turns out to be just the reduced covariance matrix.
Note that `x` still has to have dimension :math:`n`.

Parameters
----------
x : numpy.ndarray[n, 1]
    Input vector

Returns
-------
numpy.narray[n, n]
    The expected Fisher information matrix, which we call log_curvature in this context.
)pbdoc";


/*
 *  MixtureModel
 */

const char* hopsy::doc::Mixture::base = R"pbdoc(hopsy.Mixture(self, components, weights = [1, ..., 1])

The ``Mixture`` is a weighted sum of :math:`n` components, so its unnormalized density is
given as

.. math::
  f(x) = \sum_{i=1}^n w_i f_i(x)

The components may be arbitrary python objects implementing the methods as required in a :class:`hopsy.PyModel`

Parameters
----------
components : list[object]
    The Mixture's model components.
weights : list[float]
    Component weights. If none are given, they will be assumed to be all 1.

)pbdoc";


const char* hopsy::doc::Mixture::components = R"pbdoc(
A list of model components, where every components is supposed to be a Python object implementing `hopsy.Model`
or being wrapped inside `hopsy.PyModel`.
)pbdoc";


const char* hopsy::doc::Mixture::weights = R"pbdoc(
A list of component weights, which has to match the number of components.
)pbdoc";


const char* hopsy::doc::Mixture::__init__ = R"pbdoc(__init__(self, components, weights = [1, ..., 1])

Construct a ``Mixture`` as weighted sum over the elements from ``components``.

Parameters
----------
components : list[object]
    The Mixture's model components.
weights : list[float]
    Component weights. If none are given, they will be assumed to be all 1.

)pbdoc";


const char* hopsy::doc::Mixture::computeNegativeLogLikelihood = R"pbdoc(compute_negative_log_likelihood(self, x)
deprecated:: 1.4
   Use :func:`log_density` instead.

Computes the negative logarithm of the weighted sum of the probability density functions of the model
components

.. math::
  \log f(x) = \log \sum_{i=1}^n w_i f_i(x).

Parameters
----------
x : numpy.ndarray[n, 1]
    Input vector

Returns
-------
float
    The (unnormalized) negative log-likelihood

)pbdoc";


const char* hopsy::doc::Mixture::computeLogLikelihoodGradient = R"pbdoc(compute_log_likelihood_gradient(self, x)
deprecated:: 1.4
   Use :func:`log_gradient` instead.

Computes the gradient of the logarithm of the weighted sum of the probability density functions of the model
components

.. math::
  \nabla \log f(x) = \nabla \log \sum_{i=1}^n w_i f_i(x).

Parameters
----------
x : numpy.ndarray[n, 1]
    Input vector

Returns
-------
numpy.ndarray[n, 1]
    The gradient of the (unnormalized) log-likelihood
)pbdoc";


const char* hopsy::doc::Mixture::computeExpectedFisherInformation = R"pbdoc(compute_expected_fisher_information(self, x)
deprecated:: 1.4
   Use :func:`log_curvature` instead.

This method is not implemented, as there exists no closed-form solution to
computing the expected Fisher information of a general mixture model.

Parameters
----------
x : numpy.ndarray[n, 1]
    Input vector

Returns
-------
None

)pbdoc";


const char* hopsy::doc::Mixture::logDensity = R"pbdoc(log_density(self, x)

Computes the log probability density of the model
components

.. math::
  \log f(x) = \log \sum_{i=1}^n w_i f_i(x).

Parameters
----------
x : numpy.ndarray[n, 1]
    Input vector

Returns
-------
float
    The (unnormalized) log_density

)pbdoc";


const char* hopsy::doc::Mixture::logGradient = R"pbdoc(log_gradient(self, x)

Computes the gradient of the logarithm of the weighted sum of the probability density functions of the model
components

.. math::
  \nabla \log f(x) = \nabla \log \sum_{i=1}^n w_i f_i(x).

Parameters
----------
x : numpy.ndarray[n, 1]
    Input vector

Returns
-------
numpy.ndarray[n, 1]
    The gradient of the (unnormalized) log_density
)pbdoc";


const char* hopsy::doc::Mixture::logCurvature = R"pbdoc(log_curvature(self x)

This method is not implemented, as there exists no closed-form solution to
computing the log curvature (typically defined as the expected Fisher information) of a general mixture model.

Parameters
----------
x : numpy.ndarray[n, 1]
    Input vector

Returns
-------
None

)pbdoc";



/*
 *  PyModel
 */

const char* hopsy::doc::PyModel::base = R"pbdoc(
The ``PyModel`` class allows you to interface arbitrary models implemented in Python to work correctly with
hops, given that the ``model`` object you pass implements the required functions:

* ``compute_negative_log_likelihood``

* ``compute_log_likelihood_gradient``

* ``compute_expected_fisher_information``

**Methods:**
)pbdoc";


const char* hopsy::doc::PyModel::__init__ = R"pbdoc(__init__(self, model)

:param model:
:type model: object

Wraps the passed object such that it can be correctly called from hops.
)pbdoc";


const char* hopsy::doc::PyModel::model = R"pbdoc(
The wrapped user-defined model.
)pbdoc";


const char* hopsy::doc::PyModel::computeNegativeLogLikelihood = R"pbdoc(compute_negative_log_likelihood(self, x)
deprecated:: 1.4
   Use :func:`log_density` instead.

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The value of ``model.compute_negative_log_likelihod(x)``
:rtype: float
)pbdoc";


const char* hopsy::doc::PyModel::computeLogLikelihoodGradient = R"pbdoc(compute_log_likelihood_gradient(self, x)
deprecated:: 1.4
   Use :func:`log_gradient` instead.

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The value of ``model.compute_log_likelihood_gradient(x)``
:rtype: numpy.ndarray[float64[n,1]]
)pbdoc";


const char* hopsy::doc::PyModel::computeExpectedFisherInformation = R"pbdoc(compute_expected_fisher_information(self, x)
deprecated:: 1.4
   Use :func:`log_curvature` instead.

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The value of ``model.compute_expected_fisher_information(x)``
:rtype: numpy.ndarray[float64[n,n]]
)pbdoc";


const char* hopsy::doc::PyModel::logDensity = R"pbdoc(log_density(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The value of ``model.log_density(x)``
:rtype: float
)pbdoc";


const char* hopsy::doc::PyModel::logGradient = R"pbdoc(log_gradient(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The value of ``model.log_gradient(x)``
:rtype: numpy.ndarray[float64[n,1]]
)pbdoc";


const char* hopsy::doc::PyModel::logCurvature = R"pbdoc(log_curvature(self x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The value of ``model.log_curvature(x)``
:rtype: numpy.ndarray[float64[n,n]]
)pbdoc";



/*
 *  Rosenbrock
 */

const char* hopsy::doc::Rosenbrock::base = R"pbdoc(

A multi-dimensional Rosenbrock function in :math:`2n` dimensions.

Reference: https://doi.org/10.1162/evco.2009.17.3.437

**Methods:**
)pbdoc";


const char* hopsy::doc::Rosenbrock::__init__ = R"pbdoc(__init__(self, scale = 1, shift = [0])

:param scale:
:type scale: float

:param shift:
:type shift: numpy.ndarray[float64[n,1]]
)pbdoc";


const char* hopsy::doc::Rosenbrock::scale = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Rosenbrock::shift = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Rosenbrock::computeNegativeLogLikelihood = R"pbdoc(compute_negative_log_likelihood(self, x)
deprecated:: 1.4
   Use :func:`log_density` instead.

:param x: Input vector
:type x: numpy.ndarray[float64[2n,1]]

:return: The (unnormalized) negative log-likelihood
:rtype: float
)pbdoc";


const char* hopsy::doc::Rosenbrock::computeLogLikelihoodGradient = R"pbdoc(compute_log_likelihood_gradient(self, x)
deprecated:: 1.4
   Use :func:`log_gradient` instead.

:param x: Input vector
:type x: numpy.ndarray[float64[2n,1]]

:return: The gradient of the (unnormalized) log-likelihood
:rtype: numpy.ndarray[float64[2n,1]]
)pbdoc";


const char* hopsy::doc::Rosenbrock::computeExpectedFisherInformation = R"pbdoc(compute_expected_fisher_information(self, x)
deprecated:: 1.4
   Use :func:`log_curvature` instead.

:param x: Input vector
:type x: numpy.ndarray[float64[2n,1]]

:return: The expected Fisher information matrix
:rtype: numpy.ndarray[float64[2n,2n]]
)pbdoc";


const char* hopsy::doc::Rosenbrock::logDensity = R"pbdoc(log_density(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[2n,1]]

:return: The (unnormalized) log_density
:rtype: float
)pbdoc";


const char* hopsy::doc::Rosenbrock::logGradient = R"pbdoc(log_gradient(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[2n,1]]

:return: The gradient of the (unnormalized) log_density
:rtype: numpy.ndarray[float64[2n,1]]
)pbdoc";


const char* hopsy::doc::Rosenbrock::logCurvature = R"pbdoc(log_curvature(self x)

:param x: Input vector
:type x: numpy.ndarray[float64[2n,1]]

:return: The log curvature (in this case the expected Fisher information matrix)
:rtype: numpy.ndarray[float64[2n,2n]]
)pbdoc";



/*
 *  UniformModel
 */

const char* hopsy::doc::UniformModel::base = R"pbdoc(
The :class:`hopsy.UniformModel` defines the uniform target distribution on the polytope

.. math::
   \pi(x) := \frac{1}{Z} \mathbf{1}_{\mathcal{P}}(x)

where

.. math::
   Z = \int_{\mathcal{P}} \mathbf{1}_{\mathcal{P}}(x) \mathrm{d}x

Note that the uniform distribution is only well-defined, if the volume of the polytope is finite and
thus if the polytope is bounded in all dimensions. So for example, :math:`A = 1` and :math:`b = 0`
define the the inequality :math:`Ax = x \leq 0 = b` which defines the negative real line. This can be
seen as an unbounded one-dimensional polytope and the uniform distribution is thus not well defined on it.

To prevent your polytope from being unbounded, you can use :class:`hopsy.add_box_constraints` to add box constraints,
that guarantee your polytope to be bounded. For more details, please refer :doc:`here<Problem>`.

**Methods:**
)pbdoc";


const char* hopsy::doc::UniformModel::__init__ = R"pbdoc(__init__(self)
)pbdoc";


const char* hopsy::doc::UniformModel::computeNegativeLogLikelihood = R"pbdoc(compute_negative_log_likelihood(self, x)
deprecated:: 1.4
   Use :func:`log_density` instead.

The negative log-likelihood for the uniform model is the unknown constant :math:`\frac{1}{Z}`,
which depends on the volume of the support of the density. The volume is further only well-defined in
dependence of the polytope, which is not known to the ``hopsy.UniformModel``.
In the Metropolis-Hastings algorithm, this constant cancels out and is not needed for computing the
acceptance rate. Thus, this function is only available for technical reasons and **will always throw
an exception, when being called.**

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The (unnormalized) negative log-likelihood
:rtype: float

:raises RuntimeError: always
)pbdoc";


const char* hopsy::doc::UniformModel::computeLogLikelihoodGradient = R"pbdoc(compute_log_likelihood_gradient(self, x)
deprecated:: 1.4
   Use :func:`log_gradient` instead.

For similar reasons as with the ``compute_negative_log_likelihod``, this function **will always throw
an exception, when being called.**

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The gradient of the (unnormalized) log-likelihood
:rtype: numpy.ndarray[float64[n,1]]

:raises RuntimeError: always
)pbdoc";


const char* hopsy::doc::UniformModel::computeExpectedFisherInformation = R"pbdoc(compute_expected_fisher_information(self, x)
deprecated:: 1.4
   Use :func:`log_curvature` instead.

For similar reasons as with the ``compute_negative_log_likelihod``, this function **will always throw
an exception, when being called.**

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The expected Fisher information matrix
:rtype: numpy.ndarray[float64[n,n]]

:raises RuntimeError: always
)pbdoc";


const char* hopsy::doc::UniformModel::logDensity = R"pbdoc(log_density(self, x)

The log density for the uniform model is the unknown constant :math:`\frac{1}{Z}`,
which depends on the volume of the support of the density. The volume is further only well-defined in
dependence of the polytope, which is not known to the ``hopsy.UniformModel``.
In the Metropolis-Hastings algorithm, this constant cancels out and is not needed for computing the
acceptance rate. Thus, this function is only available for technical reasons and **will always throw
an exception, when being called.**

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The (unnormalized) log density
:rtype: float

:raises RuntimeError: always
)pbdoc";


const char* hopsy::doc::UniformModel::logGradient = R"pbdoc(log_gradient(self, x)

For similar reasons as with the ``log_density``, this function **will always throw
an exception, when being called.**

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The gradient of the (unnormalized) log_density
:rtype: numpy.ndarray[float64[n,1]]

:raises RuntimeError: always
)pbdoc";


const char* hopsy::doc::UniformModel::logCurvature = R"pbdoc(log_curvature(self x)

For similar reasons as with the ``log_density``, this function **will always throw
an exception, when being called.**

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The log curvature
:rtype: numpy.ndarray[float64[n,n]]

:raises RuntimeError: always
)pbdoc";




/*
 *  Problem
 */

const char* hopsy::doc::Problem::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Problem::__init__ = R"pbdoc(Problem(A, b, model = hopsy.Uniform(), starting_point = numpy.ndarray[shape[0]], transformation = numpy.ndarray[shape[0,0]], shift = numpy.ndarray[shape[0]])

Creates a Problem object of appropriate type using the passed ``model``. The actual return type depends on the type of ``model``.
If ``model`` is an arbitrary python object, it will be wrapped inside :class:`hopsy.PyModel`,
which works as an interface between HOPS and any Python-implemented model class.
Any calls from within HOPS to the model will be delegated by calling the appropriate Python function.

:param numpy.ndarray[shape[m,n]] A: the left-hand side of the polytope inequality :math:`Ax \leq b`.

:param numpy.ndarray[shape[m,1]] b: the right-hand side of the polytope inequality :math:`Ax \leq b`

:param object model: defines the target distribution and may be any Python object.
:param numpy.ndarray[shape[n,1]] starting_point: the starting point.

:param numpy.ndarray[shape[n,n]] transformation: the matrix :math:`T` in a linear transformation :math:`Tx + h`.

:param numpy.ndarray[shape[n,1]] shift: the vector :math:`h` in a linear transformation :math:`Tx + h`.

:return:
:rtype:
)pbdoc";


const char* hopsy::doc::Problem::A = R"pbdoc(numpy.ndarray[shape[m,n]]: Left-hand side operator :math:`A` of the polytope-defining inequality :math:`Ax \leq b`. ``A`` has ``m`` rows and ``n`` columns, defining ``m`` constraints in a ``n``-dimensional space.
)pbdoc";


const char* hopsy::doc::Problem::b = R"pbdoc(numpy.ndarray[shape[m,1]]: Right-hand side of the polytope-defining inequality :math:`Ax \leq b`. Note that ``b`` has to match the matrix ``A`` in the number of rows (=constraints).
)pbdoc";


const char* hopsy::doc::Problem::model = R"pbdoc(object: The ``model`` object defines the target distribution inside the polytope via its :meth:`compute_negative_log_likelihood`. If no model is passed, then a uniform target on the polytope will be assumed. **Note that a uniform distribution on a polytope is only defined for bounded polytopes.**
)pbdoc";


const char* hopsy::doc::Problem::startingPoint = R"pbdoc(numpy.ndarray[float64[n,1]]: A single starting point as part of the problem. Since the asymptotic behaviour of a (well-defined) Markov chain is independent of the starting distribution, the starting points are usually rather considered part of the chain than part of the problem. However, for particular problems, where one wishes to fix the starting point, this field can be used. If no starting point is passed, it will be initialized as an empty vector.
)pbdoc";


const char* hopsy::doc::Problem::transformation = R"pbdoc(numpy.ndarray[float64[n,n]]: For preconditioning (e.g. polytope rounding) one may wish to transform the polytope defined by :math:`Ax \leq b` as :math:`ATy \leq b - As` where `x = Ty + s`. If a non-zero sized ``transformation`` is passed, then it will be used as the matrix :math:`T` to transform all recorded states :math:`x` as `x = Ty`. The matrix ``A`` will be assumed to be the left-hand side operator of the already transformed problem.
)pbdoc";


const char* hopsy::doc::Problem::shift = R"pbdoc(numpy.ndarray[float64[n,1]]: For preconditioning (e.g. polytope rounding) one may wish to transform the polytope defined by :math:`Ax \leq b` as :math:`ATy \leq b - As` where `x = Ty + s`. If a non-zero sized ``shift`` is passed, then it will be used as the vector :math:`s` to transform all recorded states :math:`x` as `x = y + s`. The vector ``b`` will be assumed to be the right-hand side of the already transformed problem.
)pbdoc";

const char* hopsy::doc::Problem::slacks = R"pbdoc(problem.slacks(point)

Given a `point`` computes polytope slacks, i.e., `b - A * x`. Note, that if A and b have been transformed, then the slacks are computed in the transformed space, e.g. after rounding.

Parameters
----------
:param point: point at which to compute slacks
:type point: numpy.ndarray[float64[n,1]]

Returns
-------
:return: The value of the slacks at ``point``
:rtype: numpy.ndarray[float64[n,1]]
)pbdoc";


const char* hopsy::doc::Proposal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Proposal::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Proposal::propose = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Proposal::acceptProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Proposal::logAcceptanceProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Proposal::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Proposal::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Proposal::getParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Proposal::setParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Proposal::getStepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Proposal::name = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Proposal::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";


const char* hopsy::doc::Proposal::proposalNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`proposal_log_density` instead.
)pbdoc";


const char* hopsy::doc::Proposal::hasNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use :attr:`has_log_density` instead.
)pbdoc";

const char* hopsy::doc::Proposal::stateLogDensity = R"pbdoc(state_log_density()

Returns the log density of the current state of the proposal.

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::Proposal::proposalLogDensity = R"pbdoc(proposal_log_density()
Returns the log density of the proposal. If no proposal has been porposed, it returns 0

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::Proposal::hasLogDensity = R"pbdoc(has_log_density()

Returns whether the proposal knows about the log density or not.
Most proposals dont know about the log density. Exceptions are usually proposals which use log_gradient().

Returns
-------
bool
    whether proposal knows log density
)pbdoc";


const char* hopsy::doc::Proposal::copyProposal = R"pbdoc(
)pbdoc";



const char* hopsy::doc::AdaptiveMetropolisProposal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::propose = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::acceptProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::logAcceptanceProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::getParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::setParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::getStepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::name = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::proposalNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`proposal_log_density` instead.
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::hasNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use :attr:`has_log_density` instead.
)pbdoc";

const char* hopsy::doc::AdaptiveMetropolisProposal::stateLogDensity = R"pbdoc(state_log_density()

Returns the log density of the current state of the proposal.

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::proposalLogDensity = R"pbdoc(proposal_log_density()
Returns the log density of the proposal. If no proposal has been porposed, it returns 0

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::hasLogDensity = R"pbdoc(has_log_density()

Returns whether the proposal knows about the log density or not.
Most proposals dont know about the log density. Exceptions are usually proposals which use log_gradient().

Returns
-------
bool
    whether proposal knows log density
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::copyProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::boundaryCushion = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::epsilon = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AdaptiveMetropolisProposal::warmUp = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::propose = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::acceptProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::logAcceptanceProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::getParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::setParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::getStepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::name = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::proposalNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`proposal_log_density` instead.
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::hasNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use :attr:`has_log_density` instead.
)pbdoc";

const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::stateLogDensity = R"pbdoc(state_log_density()

Returns the log density of the current state of the proposal.

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::proposalLogDensity = R"pbdoc(proposal_log_density()
Returns the log density of the proposal. If no proposal has been porposed, it returns 0

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::hasLogDensity = R"pbdoc(has_log_density()

Returns whether the proposal knows about the log density or not.
Most proposals dont know about the log density. Exceptions are usually proposals which use log_gradient().

Returns
-------
bool
    whether proposal knows log density
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::copyProposal = R"pbdoc(
)pbdoc";

const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::maxReflections = R"pbdoc(
)pbdoc";

const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::boundaryCushion = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::epsilon = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::stepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardAdaptiveMetropolisProposal::warmUp = R"pbdoc(
)pbdoc";

const char* hopsy::doc::BallWalkProposal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BallWalkProposal::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BallWalkProposal::propose = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BallWalkProposal::acceptProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BallWalkProposal::logAcceptanceProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BallWalkProposal::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BallWalkProposal::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BallWalkProposal::getParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BallWalkProposal::setParameter = R"pbdoc(
)pbdoc";

const char* hopsy::doc::BallWalkProposal::getStepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BallWalkProposal::name = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BallWalkProposal::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";


const char* hopsy::doc::BallWalkProposal::proposalNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`proposal_log_density` instead.
)pbdoc";


const char* hopsy::doc::BallWalkProposal::hasNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use :attr:`has_log_density` instead.
)pbdoc";

const char* hopsy::doc::BallWalkProposal::stateLogDensity = R"pbdoc(state_log_density()

Returns the log density of the current state of the proposal.

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::BallWalkProposal::proposalLogDensity = R"pbdoc(proposal_log_density()
Returns the log density of the proposal. If no proposal has been porposed, it returns 0

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::BallWalkProposal::hasLogDensity = R"pbdoc(has_log_density()

Returns whether the proposal knows about the log density or not.
Most proposals dont know about the log density. Exceptions are usually proposals which use log_gradient().

Returns
-------
bool
    whether proposal knows log density
)pbdoc";

const char* hopsy::doc::BallWalkProposal::copyProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BallWalkProposal::stepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::propose = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::acceptProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::logAcceptanceProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::getParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::setParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::getStepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::name = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::proposalNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`proposal_log_density` instead.
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::hasNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use :attr:`has_log_density` instead.
)pbdoc";

const char* hopsy::doc::BilliardMALAProposal::stateLogDensity = R"pbdoc(state_log_density()

Returns the log density of the current state of the proposal.

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::proposalLogDensity = R"pbdoc(proposal_log_density()
Returns the log density of the proposal. If no proposal has been porposed, it returns 0

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::hasLogDensity = R"pbdoc(has_log_density()

Returns whether the proposal knows about the log density or not.
Most proposals dont know about the log density. Exceptions are usually proposals which use log_gradient().

Returns
-------
bool
    whether proposal knows log density
)pbdoc";

const char* hopsy::doc::BilliardMALAProposal::copyProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::maxReflections = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardMALAProposal::stepSize = R"pbdoc(
)pbdoc";

const char* hopsy::doc::BilliardWalkProposal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::propose = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::acceptProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::logAcceptanceProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::getParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::setParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::getStepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::name = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::proposalNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`proposal_log_density` instead.
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::hasNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use :attr:`has_log_density` instead.
)pbdoc";

const char* hopsy::doc::BilliardWalkProposal::stateLogDensity = R"pbdoc(state_log_density()

Returns the log density of the current state of the proposal.

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::proposalLogDensity = R"pbdoc(proposal_log_density()
Returns the log density of the proposal. If no proposal has been porposed, it returns 0

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::hasLogDensity = R"pbdoc(has_log_density()

Returns whether the proposal knows about the log density or not.
Most proposals dont know about the log density. Exceptions are usually proposals which use log_gradient().

Returns
-------
bool
    whether proposal knows log density
)pbdoc";

const char* hopsy::doc::BilliardWalkProposal::copyProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::maxReflections = R"pbdoc(
)pbdoc";


const char* hopsy::doc::BilliardWalkProposal::stepSize = R"pbdoc(
)pbdoc";

const char* hopsy::doc::CSmMALAProposal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::propose = R"pbdoc(
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::acceptProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::logAcceptanceProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::getParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::setParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::getStepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::name = R"pbdoc(
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::proposalNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`proposal_log_density` instead.
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::hasNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use :attr:`has_log_density` instead.
)pbdoc";

const char* hopsy::doc::CSmMALAProposal::stateLogDensity = R"pbdoc(state_log_density()

Returns the log density of the current state of the proposal.

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::proposalLogDensity = R"pbdoc(proposal_log_density()
Returns the log density of the proposal. If no proposal has been porposed, it returns 0

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::hasLogDensity = R"pbdoc(has_log_density()

Returns whether the proposal knows about the log density or not.
Most proposals dont know about the log density. Exceptions are usually proposals which use log_gradient().

Returns
-------
bool
    whether proposal knows log density
)pbdoc";

const char* hopsy::doc::CSmMALAProposal::copyProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::fisherWeight = R"pbdoc(
)pbdoc";


const char* hopsy::doc::CSmMALAProposal::stepSize = R"pbdoc(
)pbdoc";



const char* hopsy::doc::DikinWalkProposal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::propose = R"pbdoc(
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::acceptProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::logAcceptanceProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::getParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::setParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::getStepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::name = R"pbdoc(
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::proposalNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`proposal_log_density` instead.
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::hasNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use :attr:`has_log_density` instead.
)pbdoc";

const char* hopsy::doc::DikinWalkProposal::stateLogDensity = R"pbdoc(state_log_density()

Returns the log density of the current state of the proposal.

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::proposalLogDensity = R"pbdoc(proposal_log_density()
Returns the log density of the proposal. If no proposal has been porposed, it returns 0

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::hasLogDensity = R"pbdoc(has_log_density()

Returns whether the proposal knows about the log density or not.
Most proposals dont know about the log density. Exceptions are usually proposals which use log_gradient().

Returns
-------
bool
    whether proposal knows log density
)pbdoc";

const char* hopsy::doc::DikinWalkProposal::copyProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::boundaryCushion = R"pbdoc(
)pbdoc";


const char* hopsy::doc::DikinWalkProposal::stepSize = R"pbdoc(
)pbdoc";



const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::__init__ = R"pbdoc(
)pbdoc";



const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::propose = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::acceptProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::logAcceptanceProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::getParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::setParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::getStepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::name = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";


const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::proposalNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`proposal_log_density` instead.
)pbdoc";


const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::hasNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use :attr:`has_log_density` instead.
)pbdoc";

const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::stateLogDensity = R"pbdoc(state_log_density()

Returns the log density of the current state of the proposal.

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::proposalLogDensity = R"pbdoc(proposal_log_density()
Returns the log density of the proposal. If no proposal has been porposed, it returns 0

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::hasLogDensity = R"pbdoc(has_log_density()

Returns whether the proposal knows about the log density or not.
Most proposals dont know about the log density. Exceptions are usually proposals which use log_gradient().

Returns
-------
bool
    whether proposal knows log density
)pbdoc";

const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::copyProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianCoordinateHitAndRunProposal::stepSize = R"pbdoc(
)pbdoc";



const char* hopsy::doc::GaussianHitAndRunProposal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianHitAndRunProposal::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianHitAndRunProposal::propose = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianHitAndRunProposal::acceptProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianHitAndRunProposal::logAcceptanceProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianHitAndRunProposal::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianHitAndRunProposal::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianHitAndRunProposal::getParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianHitAndRunProposal::setParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianHitAndRunProposal::getStepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianHitAndRunProposal::name = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianHitAndRunProposal::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";


const char* hopsy::doc::GaussianHitAndRunProposal::proposalNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`proposal_log_density` instead.
)pbdoc";


const char* hopsy::doc::GaussianHitAndRunProposal::hasNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use :attr:`has_log_density` instead.
)pbdoc";

const char* hopsy::doc::GaussianHitAndRunProposal::stateLogDensity = R"pbdoc(state_log_density()

Returns the log density of the current state of the proposal.

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::GaussianHitAndRunProposal::proposalLogDensity = R"pbdoc(proposal_log_density()
Returns the log density of the proposal. If no proposal has been porposed, it returns 0

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::GaussianHitAndRunProposal::hasLogDensity = R"pbdoc(has_log_density()

Returns whether the proposal knows about the log density or not.
Most proposals dont know about the log density. Exceptions are usually proposals which use log_gradient().

Returns
-------
bool
    whether proposal knows log density
)pbdoc";

const char* hopsy::doc::GaussianHitAndRunProposal::copyProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianHitAndRunProposal::stepSize = R"pbdoc(
)pbdoc";



const char* hopsy::doc::GaussianProposal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianProposal::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianProposal::propose = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianProposal::acceptProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianProposal::logAcceptanceProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianProposal::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianProposal::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianProposal::getParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianProposal::setParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianProposal::getStepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianProposal::name = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianProposal::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";


const char* hopsy::doc::GaussianProposal::proposalNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`proposal_log_density` instead.
)pbdoc";


const char* hopsy::doc::GaussianProposal::hasNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use :attr:`has_log_density` instead.
)pbdoc";

const char* hopsy::doc::GaussianProposal::stateLogDensity = R"pbdoc(state_log_density()

Returns the log density of the current state of the proposal.

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::GaussianProposal::proposalLogDensity = R"pbdoc(proposal_log_density()
Returns the log density of the proposal. If no proposal has been porposed, it returns 0

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::GaussianProposal::hasLogDensity = R"pbdoc(has_log_density()

Returns whether the proposal knows about the log density or not.
Most proposals dont know about the log density. Exceptions are usually proposals which use log_gradient().

Returns
-------
bool
    whether proposal knows log density
)pbdoc";

const char* hopsy::doc::GaussianProposal::copyProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::GaussianProposal::stepSize = R"pbdoc(
)pbdoc";

const char* hopsy::doc::ReversibleJumpProposal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ReversibleJumpProposal::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ReversibleJumpProposal::propose = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ReversibleJumpProposal::acceptProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ReversibleJumpProposal::logAcceptanceProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ReversibleJumpProposal::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ReversibleJumpProposal::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ReversibleJumpProposal::getParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ReversibleJumpProposal::setParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ReversibleJumpProposal::getStepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ReversibleJumpProposal::name = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ReversibleJumpProposal::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";


const char* hopsy::doc::ReversibleJumpProposal::proposalNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`proposal_log_density` instead.
)pbdoc";


const char* hopsy::doc::ReversibleJumpProposal::hasNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use :attr:`has_log_density` instead.
)pbdoc";

const char* hopsy::doc::ReversibleJumpProposal::stateLogDensity = R"pbdoc(state_log_density()

Returns the log density of the current state of the proposal.

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::ReversibleJumpProposal::proposalLogDensity = R"pbdoc(proposal_log_density()
Returns the log density of the proposal. If no proposal has been porposed, it returns 0

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::ReversibleJumpProposal::hasLogDensity = R"pbdoc(has_log_density()

Returns whether the proposal knows about the log density or not.
Most proposals dont know about the log density. Exceptions are usually proposals which use log_gradient().

Returns
-------
bool
    whether proposal knows log density
)pbdoc";

const char* hopsy::doc::ReversibleJumpProposal::copyProposal = R"pbdoc(
)pbdoc";

const char* hopsy::doc::ReversibleJumpProposal::modelJumpProbability = R"pbdoc(
)pbdoc";

const char* hopsy::doc::ReversibleJumpProposal::activationProbability = R"pbdoc(
)pbdoc";

const char* hopsy::doc::ReversibleJumpProposal::deactivationProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::PyProposal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::PyProposal::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::PyProposal::propose = R"pbdoc(
)pbdoc";


const char* hopsy::doc::PyProposal::acceptProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::PyProposal::logAcceptanceProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::PyProposal::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::PyProposal::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::PyProposal::getParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::PyProposal::setParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::PyProposal::getStepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::PyProposal::name = R"pbdoc(
)pbdoc";


const char* hopsy::doc::PyProposal::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";


const char* hopsy::doc::PyProposal::proposalNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`proposal_log_density` instead.
)pbdoc";


const char* hopsy::doc::PyProposal::hasNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use :attr:`has_log_density` instead.
)pbdoc";

const char* hopsy::doc::PyProposal::stateLogDensity = R"pbdoc(state_log_density()

Returns the log density of the current state of the proposal.

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::PyProposal::proposalLogDensity = R"pbdoc(proposal_log_density()
Returns the log density of the proposal. If no proposal has been porposed, it returns 0

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::PyProposal::hasLogDensity = R"pbdoc(has_log_density()

Returns whether the proposal knows about the log density or not.
Most proposals dont know about the log density. Exceptions are usually proposals which use log_gradient().

Returns
-------
bool
    whether proposal knows log density
)pbdoc";

const char* hopsy::doc::PyProposal::copyProposal = R"pbdoc(
)pbdoc";

const char* hopsy::doc::TruncatedGaussianProposal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::TruncatedGaussianProposal::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::TruncatedGaussianProposal::propose = R"pbdoc(
)pbdoc";


const char* hopsy::doc::TruncatedGaussianProposal::acceptProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::TruncatedGaussianProposal::logAcceptanceProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::TruncatedGaussianProposal::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::TruncatedGaussianProposal::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::TruncatedGaussianProposal::getParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::TruncatedGaussianProposal::setParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::TruncatedGaussianProposal::getStepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::TruncatedGaussianProposal::name = R"pbdoc(
)pbdoc";


const char* hopsy::doc::TruncatedGaussianProposal::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";


const char* hopsy::doc::TruncatedGaussianProposal::proposalNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`proposal_log_density` instead.
)pbdoc";


const char* hopsy::doc::TruncatedGaussianProposal::hasNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use :attr:`has_log_density` instead.
)pbdoc";

const char* hopsy::doc::TruncatedGaussianProposal::stateLogDensity = R"pbdoc(state_log_density()

Returns the log density of the current state of the proposal.

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::TruncatedGaussianProposal::proposalLogDensity = R"pbdoc(proposal_log_density()
Returns the log density of the proposal. If no proposal has been porposed, it returns 0

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::TruncatedGaussianProposal::hasLogDensity = R"pbdoc(has_log_density()

Returns whether the proposal knows about the log density or not.
Most proposals dont know about the log density. Exceptions are usually proposals which use log_gradient().

Returns
-------
bool
    whether proposal knows log density
)pbdoc";

const char* hopsy::doc::TruncatedGaussianProposal::copyProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformCoordinateHitAndRunProposal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformCoordinateHitAndRunProposal::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformCoordinateHitAndRunProposal::propose = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformCoordinateHitAndRunProposal::acceptProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformCoordinateHitAndRunProposal::logAcceptanceProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformCoordinateHitAndRunProposal::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformCoordinateHitAndRunProposal::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformCoordinateHitAndRunProposal::getParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformCoordinateHitAndRunProposal::setParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformCoordinateHitAndRunProposal::getStepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformCoordinateHitAndRunProposal::name = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformCoordinateHitAndRunProposal::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";


const char* hopsy::doc::UniformCoordinateHitAndRunProposal::proposalNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`proposal_log_density` instead.
)pbdoc";


const char* hopsy::doc::UniformCoordinateHitAndRunProposal::hasNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use :attr:`has_log_density` instead.
)pbdoc";

const char* hopsy::doc::UniformCoordinateHitAndRunProposal::stateLogDensity = R"pbdoc(state_log_density()

Returns the log density of the current state of the proposal.

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::UniformCoordinateHitAndRunProposal::proposalLogDensity = R"pbdoc(proposal_log_density()
Returns the log density of the proposal. If no proposal has been porposed, it returns 0

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::UniformCoordinateHitAndRunProposal::hasLogDensity = R"pbdoc(has_log_density()

Returns whether the proposal knows about the log density or not.
Most proposals dont know about the log density. Exceptions are usually proposals which use log_gradient().

Returns
-------
bool
    whether proposal knows log density
)pbdoc";

const char* hopsy::doc::UniformCoordinateHitAndRunProposal::copyProposal = R"pbdoc(
)pbdoc";



const char* hopsy::doc::UniformHitAndRunProposal::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformHitAndRunProposal::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformHitAndRunProposal::propose = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformHitAndRunProposal::acceptProposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformHitAndRunProposal::logAcceptanceProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformHitAndRunProposal::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformHitAndRunProposal::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformHitAndRunProposal::getParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformHitAndRunProposal::setParameter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformHitAndRunProposal::getStepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformHitAndRunProposal::name = R"pbdoc(
)pbdoc";


const char* hopsy::doc::UniformHitAndRunProposal::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";


const char* hopsy::doc::UniformHitAndRunProposal::proposalNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`proposal_log_density` instead.
)pbdoc";


const char* hopsy::doc::UniformHitAndRunProposal::hasNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use :attr:`has_log_density` instead.
)pbdoc";

const char* hopsy::doc::UniformHitAndRunProposal::stateLogDensity = R"pbdoc(state_log_density()

Returns the log density of the current state of the proposal.

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::UniformHitAndRunProposal::proposalLogDensity = R"pbdoc(proposal_log_density()
Returns the log density of the proposal. If no proposal has been porposed, it returns 0

Returns
-------
float
    the log density
)pbdoc";


const char* hopsy::doc::UniformHitAndRunProposal::hasLogDensity = R"pbdoc(has_log_density()

Returns whether the proposal knows about the log density or not.
Most proposals dont know about the log density. Exceptions are usually proposals which use log_gradient().

Returns
-------
bool
    whether proposal knows log density
)pbdoc";

const char* hopsy::doc::UniformHitAndRunProposal::copyProposal = R"pbdoc(
)pbdoc";



/*
 *  MarkovChain
 */

const char* hopsy::doc::MarkovChain::base = R"pbdoc(MarkovChain(problem, proposal=hopsy.GaussianHitAndRun, starting_point=None)

Given a hopsy.Problem a MarkovChain object can be constructed. The Markov chain keeps track of the internal state and the selected proposal mechanism, see proposal argument.
Several Markov chain objects can be sampled in parallel by passing them as a list to hopsy.sample.
)pbdoc";


const char* hopsy::doc::MarkovChain::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::MarkovChain::draw = R"pbdoc(draw(rng, thinning=1)

Draws a new state from the chain.

Parameters
----------
rng : hopsy.RandomNumberGenerator
    Random number generator to produce the random sample and evaluate the Metropolis criterion.
thinning : int
    Performs a total of ``thinning`` steps, discarding the intermediate steps.

Returns
-------
(float, numpy.ndarray)
    A tuple consisting of the scalar acceptance rate and the vector-valued state.
)pbdoc";


const char* hopsy::doc::MarkovChain::state = R"pbdoc(
)pbdoc";


const char* hopsy::doc::MarkovChain::model = R"pbdoc(
)pbdoc";


const char* hopsy::doc::MarkovChain::exchangeAttemptProbability = R"pbdoc(
)pbdoc";


const char* hopsy::doc::MarkovChain::proposal = R"pbdoc(
)pbdoc";


const char* hopsy::doc::MarkovChain::stateNegativeLogLikelihood = R"pbdoc(
deprecated:: 1.4
   Use -:attr:`state_log_density` instead.
)pbdoc";

const char* hopsy::doc::MarkovChain::stateLogDensity = R"pbdoc(state_log_density()
Returns the log density of the current state of the markov chain.

Returns
-------
float
    the log density
)pbdoc";



/*
 *  AcceptanceRateTarget
 */

const char* hopsy::doc::TuningTarget::base = R"pbdoc(
)pbdoc";



/*
 *  AcceptanceRateTarget
 */

const char* hopsy::doc::AcceptanceRateTarget::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AcceptanceRateTarget::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AcceptanceRateTarget::markovChains = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AcceptanceRateTarget::numberOfTestSamples = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AcceptanceRateTarget::acceptanceRate = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AcceptanceRateTarget::__call__= R"pbdoc(
)pbdoc";


/*
 *  ExpectedSquaredJumpDistanceTarget
 */

const char* hopsy::doc::ExpectedSquaredJumpDistanceTarget::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ExpectedSquaredJumpDistanceTarget::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ExpectedSquaredJumpDistanceTarget::markovChains = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ExpectedSquaredJumpDistanceTarget::numberOfTestSamples = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ExpectedSquaredJumpDistanceTarget::lags = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ExpectedSquaredJumpDistanceTarget::considerTimeCost = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ExpectedSquaredJumpDistanceTarget::__call__= R"pbdoc(
)pbdoc";



/*
 *  PyTuningTarget
 */

const char* hopsy::doc::PyTuningTarget::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::PyTuningTarget::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::PyTuningTarget::__call__= R"pbdoc(
)pbdoc";



/*
 *  ThompsonSampling
 */

const char* hopsy::doc::ThompsonSampling::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ThompsonSampling::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ThompsonSampling::posteriorUpdateIterations = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ThompsonSampling::pureSamplingIterations = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ThompsonSampling::iterationsForConvergence = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ThompsonSampling::posteriorUpdateIterationsNeeded = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ThompsonSampling::stepSizeGridSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ThompsonSampling::stepSizeLowerBound = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ThompsonSampling::stepSizeUpperBound = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ThompsonSampling::smoothingLength = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ThompsonSampling::randomSeed = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ThompsonSampling::recordData = R"pbdoc(
)pbdoc";




/*
 *  tune
 */

const char* hopsy::doc::tune = R"pbdoc(
)pbdoc";



/*
 *  Statistics
 */

const char* hopsy::doc::computeAcceptanceRate = R"pbdoc(compute_acceptance_rate(data, sqrt_covariance = numpy.ndarray[float[0, 0]])

Compute the average acceptance rate of the chains in ``data``.
Acceptance rates are returned in an ``m`` x ``1`` column vector,
where ``m`` is the number of chains stored in ``data``.

The acceptance rate is
actually also logged after every chain iteration and stored in the ChainData,
but this initializes the acceptance_rate field inside the Data object and thus
allows to discard the samples.

)pbdoc";


const char* hopsy::doc::computeEffectiveSampleSize = R"pbdoc(compute_effective_sample_size(data)

Compute the effective sample size of the chains in ``data``.
The effective sample size is computed for every dimension individually and is then
returned in an ``m`` x ``1`` column vector,
where ``m`` is the dimension of the states.

)pbdoc";


const char* hopsy::doc::computeExpectedSquaredJumpDistance = R"pbdoc(compute_expected_squared_jump_distance(data)

Compute the expected squared jump distance of the chains in ``data``.
The expected squared jump distance is computed for every chain individually and is then
returned in an ``m`` x ``1`` column vector,
where ``m`` is the number of chains stored in ``data``.
)pbdoc";


const char* hopsy::doc::computePotentialScaleReductionFactor = R"pbdoc(compute_potential_scale_reduction_factor(data)

Compute the potential scale reduction factor (also known as R-hat)
of the chains in ``data``.
The potential scale reduction factor is computed for every dimension individually and is then
returned in an ``m`` x ``1`` column vector,
where ``m`` is the dimension of the states.
)pbdoc";


const char* hopsy::doc::computeTotalTimeTaken = R"pbdoc(compute_total_time_taken(data)

Compute the total time taken of the chains in ``data``.
Times are returned in an ``m`` x ``1`` column vector,
where ``m`` is the number of chains stored in ``data``.

Timestamps are actually also logged after every chain iteration and stored in the ChainData,
so this function just takes the difference of the last and first timestamp.
It also initializes the total_time_taken field inside the Data object and thus
allows to discard the samples.
)pbdoc";
