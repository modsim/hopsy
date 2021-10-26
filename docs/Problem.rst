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

.. function:: Problem(A, b, model = hopsy.UniformModel())

    Creates a Problem object of appropriate type using the passed ``model``. The actual return type depends on the type of ``model``.

   :param numpy.ndarray[shape[m,n]] A: the left-hand side of the polytope inequality :math:`Ax \leq b`.

   :param numpy.ndarray[shape[m,1]] b: the right-hand side of the polytope inequality :math:`Ax \leq b`

   :param hopsy.Model model: defines the target distribution and may be any ``hopsy.Model``. If none is passed, then it is assumed to be a uniform model.

   :return:
   :rtype: 


.. function:: Problem(A, b, model)

    Creates a Problem object of appropriate type using the passed ``model``. The actual return type depends on the type of ``model``.

   :param numpy.ndarray[shape[m,n]] A: the left-hand side of the polytope inequality :math:`Ax \leq b`.

   :param numpy.ndarray[shape[m,1]] b: the right-hand side of the polytope inequality :math:`Ax \leq b`

   :param object model: defines the target distribution and may be any Python object. Internally, ``model``
     will be wrapped inside ``hopsy.PyModel``, which works as an interface between HOPS and any Python-implemented model class. Any calls
     from within HOPS to the model will be delegated by calling the appropriate Python function. 

   :return:
   :rtype:


.. note:: The following class reference refers to the non-existent class ``hopsy.Problem``.
          It applies to any of the available problem classes, which are listed at the end of this page.


.. class:: Problem

   **Attributes:**

   .. attribute:: A: numpy.ndarray[float64[m,n]]

   .. attribute:: b: numpy.ndarray[float64[m,1]]

   .. attribute:: starting_point: numpy.ndarray[float64[n,1]]

   .. attribute:: unrounding_transformation: numpy.ndarray[float64[m,n]]

   .. attribute:: unrounding_shift: numpy.ndarray[float64[m,1]]


Miscellaneous functions
~~~~~~~~~~~~~~~~~~~~~~~

.. function:: add_box_constraints(A, b, lower_bound, upper_bound)

   :param numpy.ndarray[float64[m,n]] A: Left hand-side of the inequality :math:`Ax \leq b`.

   :param numpy.ndarray[float64[m,1]] b: Right hand-side of the inequality :math:`Ax \leq b`.

   :param lower_bound: Specifies the lower bound(s). 
   :type lower_bound: numpy.ndarray[float64[n,1]] or float
   
   :param upper_bound: Specifies the upper bound(s). 
   :type upper_bound: numpy.ndarray[float64[n,1]] or float

   :return:
   :rtype: (numpy.ndarray[float64[m,n]], numpy.ndarray[float64[m,1]])

   Adds box constraints to all dimensions. This will extend ``A`` and ``b`` to have :math:`m+2n` rows.
   Box constraints are added naively, meaning that we do neither check whether the dimension may be already 
   somehow bound nor check whether the very same constraint already exists. You can remove unnecessary constraints
   efficiently using the PolyRound\ [#f1]_ toolbox or by using the ``hopsy.round`` function, which however will also round
   the polytope.
   
   If ``lower_bound`` and ``upper_bound`` are both ``float``, then every dimension :math:`i` will be bound as 
   :math:`lb \leq x_i \leq ub`. If `lower_bound`` and ``upper_bound`` are both ``numpy.ndarray`` with 
   appropriate length, then every dimension :math:`i` will be bound as :math:`lb_i \leq x_i \leq ub_i`.


.. function:: add_box_constraints(problem, lower_bound, upper_bound)

   :param hopsy.Problem problem: Problem which contains the matrix :math:`A` and vector :math:`b` in :math:`Ax \leq b`.

   :param lower_bound: Specifies the lower bound(s). 
   :type lower_bound: numpy.ndarray[float64[n,1]] or float
   
   :param upper_bound: Specifies the upper bound(s). 
   :type upper_bound: numpy.ndarray[float64[n,1]] or float

   :return:
   :rtype: hopsy.Problem
   
   Adds box constraints to all dimensions. This will extend ``A`` and ``b`` of the returned ``hopsy.Problem`` to have :math:`m+2n` rows.
   Box constraints are added naively, meaning that we do neither check whether the dimension may be already 
   somehow bound nor check whether the very same constraint already exists. You can remove unnecessary constraints
   efficiently using the PolyRound\ [#f1]_ toolbox or by using the ``hopsy.round`` function, which however will also round
   the polytope.
   
   If ``lower_bound`` and ``upper_bound`` are both ``float``, then every dimension :math:`i` will be bound as 
   :math:`lb \leq x_i \leq ub`. If `lower_bound`` and ``upper_bound`` are both ``numpy.ndarray`` with 
   appropriate length, then every dimension :math:`i` will be bound as :math:`lb_i \leq x_i \leq ub_i`.


.. function:: compute_chebyshev_center(problem)

   :param hopsy.Problem problem:

   :return:
   :rtype: numpy.ndarray[float64[n,1]]


.. function:: round(problem)

   :param hopsy.Problem problem: The problem, which should be rounded.

   :return:
   :rtype: hopsy.Problem

   Rounds the polytope defined by the inequality :math:`Ax \leq b` using PolyRound\ [#f1]_. 
   This also strips unnecessary constraints, that is constraints, which can never be active.
   The unrounding transformation and shift will be stored in ``problem.unrounding_transformation``
   and ``problem.unrounding_shift``.


Possible Problem-types
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MixtureProblem
.. autoclass:: MultivariateGaussianProblem
.. autoclass:: PyProblem
.. autoclass:: RosenbrockProblem
.. autoclass:: UniformProblem

.. [#f1] https://pypi.org/project/PolyRound/
