Problem
============================

The ``hopsy.Problem`` classes contain the full description of the problem, we aim to sample.
A problem consists mainly of the domain, that is the inequality :math:`Ax \leq b`, and the target distribution, the ``hopsy.Model`` object.


Linear Transformations
----------------------

Often, it is desirable to apply linear transformations on the polytope to precondition the problem.
However, the states and the space of interest, may be the untransformed space.
For polytopes, the most common transformation is a "rounding" transformation.
Thus, it is possible to set unrounding transformations and shifts for the problem,
such the produced states lie in the correct space.
This assumes, that the transformation was already applied to the :math:`A` and :math:`b` matrix,
which are passed to the model.
In order to compute rounding transformations, we recommend to use the PolyRound\ [#f1]_ toolbox.
For more details about rounding, please refer to xyz.



Reference
---------

.. currentmodule:: hopsy

.. function:: Problem(A: numpy.ndarray[numpy.float64[m, n]], b: numpy.ndarray[numpy.float64[m, 1]], model: hopsy.Model = hopsy.UniformModel())

    Creates a Problem object of appropriate type using the passed ``model``. The actual return type depends on the type of ``model``.

   :param numpy.ndarray[shape[m,n]] A: the left-hand side of the polytope inequality :math:`Ax \leq b`.

   :param numpy.ndarray[shape[m,1]] b: the right-hand side of the polytope inequality :math:`Ax \leq b`

   :param hopsy.Model model: defines the target distribution and may be any ``hopsy.Model``. If none is passed, then it is assumed to be a uniform model.


.. function:: Problem(A: numpy.ndarray[numpy.float64[m, n]], b: numpy.ndarray[numpy.float64[m, 1]], model: object)


    Creates a Problem object of appropriate type using the passed ``model``. The actual return type depends on the type of ``model``.

   :param numpy.ndarray[shape[m,n]] A: the left-hand side of the polytope inequality :math:`Ax \leq b`.

   :param numpy.ndarray[shape[m,1]] b: the right-hand side of the polytope inequality :math:`Ax \leq b`

   :param hopsy.Model model: defines the target distribution and may be any Python object. Internally, ``model``
     will be wrapped inside ``hopsy.PyModel``, which works as an interface between HOPS and any Python-implemented model class. Any calls
     from within HOPS to the model will be delegated by calling the appropriate Python function. 

.. autoclass:: DegenerateMultivariateGaussianProblem
.. autoclass:: MultimodalMultivariateGaussianProblem
.. autoclass:: MultivariateGaussianProblem
.. autoclass:: PyProblem
.. autoclass:: RosenbrockProblem
.. autoclass:: UniformProblem

.. [#f1] https://pypi.org/project/PolyRound/
