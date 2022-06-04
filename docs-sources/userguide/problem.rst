Problem Description
===================

``hopsy`` ships with a number of simple models, also known as target functions. 
Together with the polytope 

.. math::
   \mathcal{P} := \{ x : Ax \leq b \}

the model forms a :class:`hopsy.Problem`.

The simple models provided are 

* :class:`hopsy.Gaussian`: A multivariate Gaussian which can also have invariant dimensions, 
  meaning it ignores some dimensions of the input vector. 
  This is interesting for simulating non-identifiabilities.

* :class:`hopsy.Mixture`: A mixture model, which is a weighted linear combination of arbitrary model components.

* :class:`hopsy.Rosenbrock`: Rosenbrock's banana function, 
  a popular test function for computational optimization.

In order to implement custom models, the :class:`hopsy.PyModel` class is provided, which interfaces
Python-implemented models with the ``hops``-C++-API.
The custom model has to at least implement the :meth:`hopsy.PyModel.compute_negative_log_likelihood` method, 
which should return the negative log-likelihood of the target distribution.
Depending on the algorithm you aim to use, it might also be necessary to further implement

* :meth:`hopsy.PyModel.compute_log_likelihood_gradient`

* :meth:`hopsy.PyModel.compute_expected_fisher_information`

For more details on signature and return type, please refer to :class:`hopsy.PyModel` or 

Problem
^^^^^^^

The ``hopsy.Problem`` classes contain the full description of the problem, we aim to sample.
A problem consists mainly of the domain, that is the inequality :math:`Ax \leq b`, and the target distribution, the ``hopsy.Model`` object.


Linear Transformations
^^^^^^^^^^^^^^^^^^^^^^

Often, it is desirable to apply linear transformations on the polytope to precondition the problem.
However, the states and the space of interest, may be the untransformed space.
For polytopes, the most common transformation is a "rounding" transformation.
Thus, it is possible to set unrounding transformations and shifts for the problem,
such the produced states lie in the correct space.
This assumes, that the transformation was already applied to the :math:`A` and :math:`b` matrix,
which are passed to the model.
In order to compute rounding transformations, we recommend to use the PolyRound\ [#f1]_ toolbox.
For more details about rounding, please refer to xyz.

.. [#f1] https://pypi.org/project/PolyRound/

Reference
^^^^^^^^^

.. currentmodule:: hopsy

Models
""""""

.. autosummary::
   :toctree: ../generated/
   :template: ../_templates/autosummary/class.rst

   hopsy.Model
   hopsy.Gaussian
   hopsy.Mixture
   hopsy.PyModel
   hopsy.Rosenbrock

Problem
"""""""

.. autosummary::
   :toctree: ../generated/
   :template: ../_templates/autosummary/class.rst

   hopsy.Problem
   hopsy.add_box_constraints
   hopsy.compute_chebyshev_center
   hopsy.round


