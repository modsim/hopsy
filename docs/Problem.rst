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

.. autofunction:: Problem

.. autoclass:: UniformProblem

   .. autoproperty:: A
   .. autoproperty:: b
   .. autoproperty:: model
   .. autoproperty:: starting_point
   .. autoproperty:: transformation
   .. autoproperty:: shift


Miscellaneous functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: add_box_constraints
.. autofunction:: compute_chebyshev_center
.. autofunction:: round

Possible Problem-types
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GaussianMixtureProblem
.. autoclass:: MixtureProblem
.. autoclass:: MultivariateGaussianProblem
.. autoclass:: PyModelProblem
.. autoclass:: RosenbrockProblem
.. autoclass:: UniformProblem

.. [#f1] https://pypi.org/project/PolyRound/
