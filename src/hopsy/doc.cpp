#include "doc.hpp"

/*
 *  DegenerateMultivariateGaussianModel
 */

const char* hopsy::doc::DegenerateMultivariateGaussianModel::base = R"pbdoc(
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
:math:`n`.

**Methods:**
)pbdoc";


const char* hopsy::doc::DegenerateMultivariateGaussianModel::__init__ = R"pbdoc(__init__(self, mean, covariance, inactives)

Constructs a ``DegenerateMultivariateGaussianModel`` with given mean and covariance and `deactivates` 
all dimensions specified in ``inactives``. This works by removing the corresponding rows and columns
from the mean and covariance. 

Passing an empty list as ``inactives`` will actually define a standard multivariate Gaussian.

:param mean: Gaussian mean vector 
:type mean: numpy.ndarray[float64[n,1]]

:param covariance: Gaussian covariance matrix 
:type covariance: numpy.ndarray[float64[n,n]]

:param inactives: List of inactive dimensions, so entries should be between 0 and :math:`n-1`.
:type inactives: list[int]
)pbdoc";


const char* hopsy::doc::DegenerateMultivariateGaussianModel::computeNegativeLogLikelihood = R"pbdoc(compute_negative_log_likelihood(self, x)

Computes the negative logarithm of the probability density function of a multivariate Gaussian model in 
:math:`m-k` dimensions at ``x``. Note that `x` still has to have dimension :math:`n`.

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The (unnormalized) negative log-likelihood
:rtype: float
)pbdoc";


const char* hopsy::doc::DegenerateMultivariateGaussianModel::computeLogLikelihoodGradient = R"pbdoc(compute_log_likelihood_gradient(self, x)

Computes the gradient of the logarithm of the probability density function of a multivariate Gaussian 
model in :math:`n-k` dimensions at ``x``. Note that `x` still has to have dimension :math:`n`.

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The gradient of the (unnormalized) log-likelihood
:rtype: numpy.ndarray[float64[n,1]] 
)pbdoc";


const char* hopsy::doc::DegenerateMultivariateGaussianModel::computeExpectedFisherInformation = R"pbdoc(compute_expected_fisher_information(self, x)

Computes the expected fisher information of a multivariate Gaussian model 
in :math:`n-k` dimensions at ``x``. This turns out to be just the reduced covariance matrix. 
Note that `x` still has to have dimension :math:`n`.

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The expected Fisher information matrix
:rtype: numpy.ndarray[float64[n,n]] 
)pbdoc";
        

/*
 *  GaussianMixtureModel
 */

const char* hopsy::doc::GaussianMixtureModel::base = R"pbdoc(
The ``GaussianMixtureModel`` is a weighted sum of :math:`n` components, so its unnormalized density is
given as

.. math::
  f(x) = \sum_{i=1}^n w_i f_i(x)

The components have to be of type ``hopsy.MultivariateGaussian``. If you want to use arbitrary component types, consider using 
``hopsy.MixtureModel``.

**Methods:**

)pbdoc";
        

const char* hopsy::doc::GaussianMixtureModel::__init__ = R"pbdoc(__init__(self, components, weights = [1, ..., 1])

Construct a ``GaussianMixtureModel`` as weighted sum over the elements from ``components``.

:param components: Model components
:type components: list[object] 

:param weights: Component weights. If none are given, they will be assumed to be all 1.
:type weights: list[float] 
)pbdoc";
        

const char* hopsy::doc::GaussianMixtureModel::computeNegativeLogLikelihood = R"pbdoc(compute_negative_log_likelihood(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The (unnormalized) negative log-likelihood
:rtype: float
)pbdoc";
        

const char* hopsy::doc::GaussianMixtureModel::computeLogLikelihoodGradient = R"pbdoc(compute_log_likelihood_gradient(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The gradient of the (unnormalized) log-likelihood
:rtype: numpy.ndarray[float64[n,1]] 
)pbdoc";
        

const char* hopsy::doc::GaussianMixtureModel::computeExpectedFisherInformation = R"pbdoc(compute_expected_fisher_information(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The expected Fisher information matrix
:rtype: numpy.ndarray[float64[n,n]] 
)pbdoc";
        

/*
 *  MixtureModel
 */

const char* hopsy::doc::MixtureModel::base = R"pbdoc(
The ``MixtureModel`` is a weighted sum of :math:`n` components, so its unnormalized density is
given as

.. math::
  f(x) = \sum_{i=1}^n w_i f_i(x)

The components may be arbitrary python objects implementing the methods as required in a ``hopsy.PyModel``
If you plan to use ``hopsy.MultivariateGaussian`` as component type, consider using 
``hopsy.GaussianMixtureModel`` for performance reasons.

**Methods:**

)pbdoc";
        

const char* hopsy::doc::MixtureModel::__init__ = R"pbdoc(__init__(self, components, weights = [1, ..., 1])

Construct a ``MixtureModel`` as weighted sum over the elements from ``components``.

:param components: Model components
:type components: list[object] 

:param weights: Component weights. If none are given, they will be assumed to be all 1.
:type weights: list[float] 
)pbdoc";
        

const char* hopsy::doc::MixtureModel::computeNegativeLogLikelihood = R"pbdoc(compute_negative_log_likelihood(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The (unnormalized) negative log-likelihood
:rtype: float
)pbdoc";
        

const char* hopsy::doc::MixtureModel::computeLogLikelihoodGradient = R"pbdoc(compute_log_likelihood_gradient(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The gradient of the (unnormalized) log-likelihood
:rtype: numpy.ndarray[float64[n,1]] 
)pbdoc";
        

const char* hopsy::doc::MixtureModel::computeExpectedFisherInformation = R"pbdoc(compute_expected_fisher_information(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The expected Fisher information matrix
:rtype: numpy.ndarray[float64[n,n]] 
)pbdoc";
        

/*
 *  MultivariateGaussianModel
 */

const char* hopsy::doc::MultivariateGaussianModel::base = R"pbdoc(
**Methods:**
)pbdoc";
        

const char* hopsy::doc::MultivariateGaussianModel::__init__ = R"pbdoc(__init__(self, mean = [0, 0], covariance = [[1, 0], [0, 1]])

:param mean:
:type mean:

:param covariance:
:type covariance:
)pbdoc";
        

const char* hopsy::doc::MultivariateGaussianModel::computeNegativeLogLikelihood = R"pbdoc(compute_negative_log_likelihood(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The (unnormalized) negative log-likelihood
:rtype: float
)pbdoc";
        

const char* hopsy::doc::MultivariateGaussianModel::computeLogLikelihoodGradient = R"pbdoc(compute_log_likelihood_gradient(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The gradient of the (unnormalized) log-likelihood
:rtype: numpy.ndarray[float64[n,1]] 
)pbdoc";
        

const char* hopsy::doc::MultivariateGaussianModel::computeExpectedFisherInformation = R"pbdoc(compute_expected_fisher_information(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The expected Fisher information matrix
:rtype: numpy.ndarray[float64[n,n]] 
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
        

const char* hopsy::doc::PyModel::computeNegativeLogLikelihood = R"pbdoc(compute_negative_log_likelihood(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The value of ``model.compute_negative_log_likelihod(x)``
:rtype: float
)pbdoc";
        

const char* hopsy::doc::PyModel::computeLogLikelihoodGradient = R"pbdoc(compute_log_likelihood_gradient(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The value of ``model.compute_log_likelihood_gradient(x)``
:rtype: numpy.ndarray[float64[n,1]] 
)pbdoc";
        

const char* hopsy::doc::PyModel::computeExpectedFisherInformation = R"pbdoc(compute_expected_fisher_information(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The value of ``model.compute_expected_fisher_information(x)``
:rtype: numpy.ndarray[float64[n,n]] 
)pbdoc";

        

/*
 *  RosenbrockModel
 */

const char* hopsy::doc::RosenbrockModel::base = R"pbdoc(
**Methods:**
)pbdoc";
        

const char* hopsy::doc::RosenbrockModel::__init__ = R"pbdoc(__init__(self, scale = 1, shift = [0])

:param scale: 
:type scale: float

:param shift: 
:type shift: numpy.ndarray[float64[k,1]]
)pbdoc";
        

const char* hopsy::doc::RosenbrockModel::computeNegativeLogLikelihood = R"pbdoc(compute_negative_log_likelihood(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The (unnormalized) negative log-likelihood
:rtype: float
)pbdoc";
        

const char* hopsy::doc::RosenbrockModel::computeLogLikelihoodGradient = R"pbdoc(compute_log_likelihood_gradient(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The gradient of the (unnormalized) log-likelihood
:rtype: numpy.ndarray[float64[n,1]] 
)pbdoc";
        

const char* hopsy::doc::RosenbrockModel::computeExpectedFisherInformation = R"pbdoc(compute_expected_fisher_information(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The expected Fisher information matrix
:rtype: numpy.ndarray[float64[n,n]] 
)pbdoc";

        

/*
 *  UniformModel
 */

const char* hopsy::doc::UniformModel::base = R"pbdoc(
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
)pbdoc";
        

const char* hopsy::doc::UniformModel::__init__ = R"pbdoc(__init__(self)
)pbdoc";
        

const char* hopsy::doc::UniformModel::computeNegativeLogLikelihood = R"pbdoc(compute_negative_log_likelihood(self, x)

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

For similar reasons as with the ``compute_negative_log_likelihod``, this function **will always throw 
an exception, when being called.**

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The gradient of the (unnormalized) log-likelihood
:rtype: numpy.ndarray[float64[n,1]] 

:raises RuntimeError: always
)pbdoc";
        

const char* hopsy::doc::UniformModel::computeExpectedFisherInformation = R"pbdoc(compute_expected_fisher_information(self, x)

For similar reasons as with the ``compute_negative_log_likelihod``, this function **will always throw 
an exception, when being called.**

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The expected Fisher information matrix
:rtype: numpy.ndarray[float64[n,n]] 

:raises RuntimeError: always
)pbdoc";

        

/*
 *  Problem
 */

const char* hopsy::doc::Problem::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Problem::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Problem::A = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Problem::b = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Problem::model = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Problem::startingPoint = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Problem::unroundingTransformation = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Problem::unroundingShift = R"pbdoc(
)pbdoc";


const char* hopsy::doc::addBoxConstraints = R"pbdoc(
)pbdoc";


const char* hopsy::doc::computeChebyshevCenter = R"pbdoc(
)pbdoc";


const char* hopsy::doc::round = R"pbdoc(
)pbdoc";



/*
 *  Run
 */

const char* hopsy::doc::Run::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Run::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Run::init = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Run::sample = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Run::data = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Run::problem = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Run::startingPoints = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Run::numberOfChains = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Run::numberOfSamples = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Run::thinning = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Run::stepSize = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Run::fisherWeight = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Run::randomSeed = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Run::samplingUntilConvergence = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Run::convergenceThreshold = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Run::maxRepetitions = R"pbdoc(
)pbdoc";


/*
 *  Tuning Targets
 */

const char* hopsy::doc::AcceptanceRateTarget::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AcceptanceRateTarget::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::AcceptanceRateTarget::acceptanceRate = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ExpectedSquaredJumpDistanceTarget::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ExpectedSquaredJumpDistanceTarget::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ExpectedSquaredJumpDistanceTarget::lags = R"pbdoc(
)pbdoc";


const char* hopsy::doc::ExpectedSquaredJumpDistanceTarget::considerTimeCost = R"pbdoc(
)pbdoc";


/*
 *  tune
 */

const char* hopsy::doc::tune = R"pbdoc(
)pbdoc";


/*
 *  Data
 */

const char* hopsy::doc::Data::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Data::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Data::flatten = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Data::reset = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Data::subsample = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Data::thin = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Data::write = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Data::acceptanceRates = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Data::negativeLogLikelihood = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Data::parameterNames = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Data::states = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Data::timestamps = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Data::numberOfChains = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Data::numberOfSamples = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Data::dims = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Data::shape = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Data::__getitem__ = R"pbdoc(
)pbdoc";


/*
 *  Statistics
 */

const char* computeAcceptanceRate = R"pbdoc(
)pbdoc";


const char* computeEffectiveSampleSize = R"pbdoc(
)pbdoc";


const char* computeExpectedSquaredJumpDistance = R"pbdoc(
)pbdoc";


const char* computePotentialScaleReductionFactor = R"pbdoc(
)pbdoc";


const char* computeTotalTimeTaken = R"pbdoc(
)pbdoc";



