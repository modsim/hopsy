#include "doc.hpp"

const char* hopsy::doc::numberOfThreads = R"pbdoc(
)pbdoc";

/*
 *  Model
 */

const char* hopsy::doc::Model::base = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Model::__init__ = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Model::computeNegativeLogLikelihood = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Model::computeLogLikelihoodGradient = R"pbdoc(
)pbdoc";


const char* hopsy::doc::Model::computeExpectedFisherInformation = R"pbdoc(
)pbdoc";
        

/*
 *  DegenerateMultivariateGaussianModel
 */

const char* hopsy::doc::DegenerateGaussian::base = R"pbdoc(
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


const char* hopsy::doc::DegenerateGaussian::__init__ = R"pbdoc(__init__(self, mean, covariance, inactives)

Constructs a ``DegenerateGaussian`` with given mean and covariance and `deactivates` 
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


const char* hopsy::doc::DegenerateGaussian::mean = R"pbdoc(
)pbdoc";


const char* hopsy::doc::DegenerateGaussian::covariance = R"pbdoc(
)pbdoc";


const char* hopsy::doc::DegenerateGaussian::inactives = R"pbdoc(
)pbdoc";


const char* hopsy::doc::DegenerateGaussian::computeNegativeLogLikelihood = R"pbdoc(compute_negative_log_likelihood(self, x)

Computes the negative logarithm of the probability density function of a multivariate Gaussian model in 
:math:`m-k` dimensions at ``x``. Note that `x` still has to have dimension :math:`n`.

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The (unnormalized) negative log-likelihood
:rtype: float
)pbdoc";


const char* hopsy::doc::DegenerateGaussian::computeLogLikelihoodGradient = R"pbdoc(compute_log_likelihood_gradient(self, x)

Computes the gradient of the logarithm of the probability density function of a multivariate Gaussian 
model in :math:`n-k` dimensions at ``x``. Note that `x` still has to have dimension :math:`n`.

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The gradient of the (unnormalized) log-likelihood
:rtype: numpy.ndarray[float64[n,1]] 
)pbdoc";


const char* hopsy::doc::DegenerateGaussian::computeExpectedFisherInformation = R"pbdoc(compute_expected_fisher_information(self, x)

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

The components have to be of type :class:`hopsy.MultivariateGaussian`. If you want to use arbitrary component types, consider using 
:class:`hopsy.MixtureModel`.

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

const char* hopsy::doc::Mixture::base = R"pbdoc(
The ``Mixture`` is a weighted sum of :math:`n` components, so its unnormalized density is
given as

.. math::
  f(x) = \sum_{i=1}^n w_i f_i(x)

The components may be arbitrary python objects implementing the methods as required in a :class:`hopsy.PyModel`
If you plan to use :class:`hopsy.MultivariateGaussian` as component type, consider using 
:class:`hopsy.GaussianMixture` for performance reasons.

**Methods:**

)pbdoc";
        

const char* hopsy::doc::Mixture::components = R"pbdoc(
)pbdoc";
        

const char* hopsy::doc::Mixture::weights = R"pbdoc(
)pbdoc";
        

const char* hopsy::doc::Mixture::__init__ = R"pbdoc(__init__(self, components, weights = [1, ..., 1])

Construct a ``Mixture`` as weighted sum over the elements from ``components``.

:param components: Model components
:type components: list[object] 

:param weights: Component weights. If none are given, they will be assumed to be all 1.
:type weights: list[float] 
)pbdoc";
        

const char* hopsy::doc::Mixture::computeNegativeLogLikelihood = R"pbdoc(compute_negative_log_likelihood(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The (unnormalized) negative log-likelihood
:rtype: float
)pbdoc";
        

const char* hopsy::doc::Mixture::computeLogLikelihoodGradient = R"pbdoc(compute_log_likelihood_gradient(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The gradient of the (unnormalized) log-likelihood
:rtype: numpy.ndarray[float64[n,1]] 
)pbdoc";
        

const char* hopsy::doc::Mixture::computeExpectedFisherInformation = R"pbdoc(compute_expected_fisher_information(self, x)

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
        

const char* hopsy::doc::PyModel::model = R"pbdoc(
The wrapped user-defined model.
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
 *  Rosenbrock
 */

const char* hopsy::doc::Rosenbrock::base = R"pbdoc(
**Methods:**
)pbdoc";
        

const char* hopsy::doc::Rosenbrock::__init__ = R"pbdoc(__init__(self, scale = 1, shift = [0])

:param scale: 
:type scale: float

:param shift: 
:type shift: numpy.ndarray[float64[k,1]]
)pbdoc";
        

const char* hopsy::doc::Rosenbrock::scale = R"pbdoc(
)pbdoc";
        

const char* hopsy::doc::Rosenbrock::shift = R"pbdoc(
)pbdoc";
        

const char* hopsy::doc::Rosenbrock::computeNegativeLogLikelihood = R"pbdoc(compute_negative_log_likelihood(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The (unnormalized) negative log-likelihood
:rtype: float
)pbdoc";
        

const char* hopsy::doc::Rosenbrock::computeLogLikelihoodGradient = R"pbdoc(compute_log_likelihood_gradient(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The gradient of the (unnormalized) log-likelihood
:rtype: numpy.ndarray[float64[n,1]] 
)pbdoc";
        

const char* hopsy::doc::Rosenbrock::computeExpectedFisherInformation = R"pbdoc(compute_expected_fisher_information(self, x)

:param x: Input vector
:type x: numpy.ndarray[float64[n,1]]

:return: The expected Fisher information matrix
:rtype: numpy.ndarray[float64[n,n]] 
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


const char* hopsy::doc::Problem::unroundingTransformation = R"pbdoc(numpy.ndarray[float64[n,n]]: For preconditioning (e.g. polytope rounding) one may wish to transform the polytope defined by :math:`Ax \leq b` as :math:`ATy \leq b - As` where `x = Ty + s`. If a non-zero sized ``transformation`` is passed, then it will be used as the matrix :math:`T` to transform all recorded states :math:`x` as `x = Ty`. The matrix ``A`` will be assumed to be the left-hand side operator of the already transformed problem.
)pbdoc";


const char* hopsy::doc::Problem::unroundingShift = R"pbdoc(numpy.ndarray[float64[n,1]]: For preconditioning (e.g. polytope rounding) one may wish to transform the polytope defined by :math:`Ax \leq b` as :math:`ATy \leq b - As` where `x = Ty + s`. If a non-zero sized ``shift`` is passed, then it will be used as the vector :math:`s` to transform all recorded states :math:`x` as `x = y + s`. The vector ``b`` will be assumed to be the right-hand side of the already transformed problem.
)pbdoc";


/*
 *  Problem helper
 */

const char* hopsy::doc::addBoxConstraintsToProblem = R"pbdoc(add_box_constraints(problem, lower_bound, upper_bound)

Adds box constraints to all dimensions. This will extend :attr:`hopsy.UniformProblem.A` and :attr:`hopsy.UniformProblem.A` of the returned :class:`hopsy.Problem` to have :math:`m+2n` rows.
Box constraints are added naively, meaning that we do neither check whether the dimension may be already 
somehow bound nor check whether the very same constraint already exists. You can remove unnecessary constraints
efficiently using the PolyRound\ [#f1]_ toolbox or by using the :func:`hopsy.round` function, which however will also round
the polytope.

If ``lower_bound`` and ``upper_bound`` are both ``float``, then every dimension :math:`i` will be bound as 
:math:`lb \leq x_i \leq ub`. If `lower_bound`` and ``upper_bound`` are both ``numpy.ndarray`` with 
appropriate length, then every dimension :math:`i` will be bound as :math:`lb_i \leq x_i \leq ub_i`.

:param hopsy.Problem problem: Problem which should be constrained and which contains the matrix :math:`A` and vector :math:`b` in :math:`Ax \leq b`.

:param lower_bound: Specifies the lower bound(s). 
:type lower_bound: numpy.ndarray[float64[n,1]] or float

:param upper_bound: Specifies the upper bound(s). 
:type upper_bound: numpy.ndarray[float64[n,1]] or float

:return: A :class:`hopsy.Problem` bounded in all dimensions.
:rtype: hopsy.Problem
)pbdoc";


const char* hopsy::doc::computeChebyshevCenter = R"pbdoc(compute_chebyshev_center(problem)

Computes the Chebyshev center, that is the midpoint of a (non-unique) largest inscribed ball in the polytope defined by :math:`Ax \leq b`. 

:param hopsy.Problem problem: Problem for which the Chebyshev center should be computed and which contains the matrix :math:`A` and vector :math:`b` in :math:`Ax \leq b`.

:return: The Chebyshev center of the passed problem.
:rtype: numpy.ndarray[float64[n,1]]
)pbdoc";


const char* hopsy::doc::round = R"pbdoc(round(problem)

Rounds the polytope defined by the inequality :math:`Ax \leq b` using PolyRound\ [#f1]_. 
This also strips unnecessary constraints, that is constraints, which can never be active.
The unrounding transformation :math:`T` and shift :math:`s` will be stored in :attr:`hopsy.UniformProblem.transformation`
and :attr:`hopsy.UniformProblem.shift` of the returned problem. Its left-hand side operator :attr:`hopsy.UniformProblem.A` and 
the right-hand side :attr:`hopsy.UniformProblem.b` of the polytope will be transformed as :math:``
inequality

:param hopsy.Problem problem: Problem that should be rounded and which contains the matrix :math:`A` and vector :math:`b` in :math:`Ax \leq b`.

:return: The rounded problem.
:rtype: hopsy.Problem
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



