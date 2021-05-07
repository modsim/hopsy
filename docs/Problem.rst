Problem
============================

The ``hopsy.Problem`` classes contain the full description of the problem, we aim to sample.
A problem consists mainly of the domain, that is the inequality :math:`Ax \leq b`, and the target distribution, the ``hopsy.Model`` object.


.. currentmodule:: hopsy

.. class:: Problem

.. function:: Problem(A: numpy.ndarray[numpy.float64[m, n]], b: numpy.ndarray[numpy.float64[m, 1]], model: hopsy.Model)

    Creates a Problem object of appropriate type using the passed ``model``. The actual return type depends on the type of ``model``.

   * ``A`` the left-hand side of the polytope inequality :math:`Ax \leq b`.

   * ``b`` the right-hand side of the polytope inequality :math:`Ax \leq b`

   * ``model`` defines the target distribution and may be any ``hopsy.Model``.


.. function:: Problem(A: numpy.ndarray[numpy.float64[m, n]], b: numpy.ndarray[numpy.float64[m, 1]], model: object)

    Creates a Problem object of appropriate type using the passed ``model``. The actual return type depends on the type of ``model``.

   * ``A`` the left-hand side of the polytope inequality :math:`Ax \leq b`.

   * ``b`` the right-hand side of the polytope inequality :math:`Ax \leq b`

   * ``model`` defines the target distribution and may be any Python object. Internally, ``model``
     will be wrapped inside ``hopsy.PyModel``, which works as an interface between HOPS and any Python-implemented model class. Any calls
     from within HOPS to the model will be delegated by calling the appropriate Python function. 


.. autosummary::
   :toctree: _generate

   DegenerateMultivariateGaussianProblem
   MultimodalMultivariateGaussianProblem
   MultivariateGaussianProblem
   PyProblem
   RosenbrockProblem
   UniformProblem
