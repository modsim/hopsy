Markov Chain Monte Carlo Sampling
=================================

Markov Chains
^^^^^^^^^^^^^


Proposal Distributions
^^^^^^^^^^^^^^^^^^^^^^

``hopsy`` is not particulary designed for development of polytope samplers, 
but instead aims at making the proposals available in hops
easily usable to solve the practitioner's problems at hand.
As the proposal - together with the Metropolis filter and the models likelihood function - 
forms the core of any Markov chain Monte Carlo algorithm, we believe that for the sake of performance 
proposals should be optimized and implemented in C++ and ``hops`` itself.

Nevertheless, it might be desirable to do some rapid prototyping to test promising ideas.
For this case, 
it is possible to implement Python proposals and to instruct ``hopsy`` (and ultimately ``hops``) to use those.

Similarily to the custom Python models, 
this works by wrapping the Python proposal in the :class:`hopsy.PyProposal` class.
:class:`hopsy.PyProposal` implements all functions necessary for the proposal class to be usable in 
``hops`` by delegating the function calls to the stored Python proposal.
Thus, it is obviously needed, that the Python proposal implements the corresponding functions.

The required functions are

.. note:: THIS SECTION IS DEPRECATED AND NEEDS TO BE REWRITTEN.

* ``propose() -> None`` generates a new proposal and store it internally

* ``accept_proposal() -> None`` sets the current state, which also has to be stored internally, to the current proposal

* ``calculate_log_acceptance_probability() -> float`` compute the logarithm of :math:`\frac{P(x|y)}{P(y|x)}`, 
  if :math:`x` is the current state and :math:`y` the proposal. 
  For symmetric proposals this is e.g. 0. For infeasible proposals, which e.g. lie outside the polytope, it is :math:`-\infty`.

* ``get_state() -> numpy.ndarray[shape[d,1]]`` returns the current state.

* ``set_state(new_state: numpy.ndarray[shape[d,1]]) -> None`` sets the current state.

* ``get_proposal() -> numpy.ndarray[shape[d,1]]`` returns the current proposal.

* (optional) ``get_stepsize() -> float`` sets the stepsize, if it is available. 

* (optional) ``set_stepsize(new_stepsize: float)`` gets the stepsize, if it is available.

* (optional) ``get_name() -> str`` gets the algorithms name, if it is available.

Example code
^^^^^^^^^^^^

Reference
^^^^^^^^^

Proposals
"""""""""

.. autosummary::
   :toctree: ../generated/
   :template: ../_templates/autosummary/class.rst

   hopsy.AdaptiveMetropolisProposal
   hopsy.BallWalkProposal
   hopsy.CSmMALAProposal
   hopsy.DikinWalkProposal
   hopsy.GaussianCoordinateHitAndRunProposal
   hopsy.GaussianHitAndRunProposal
   hopsy.GaussianProposal
   hopsy.UniformCoordinateHitAndRunProposal
   hopsy.UniformHitAndRunProposal

Markov Chain
""""""""""""

.. autosummary::
   :toctree: ../generated/
   :template: ../_templates/autosummary/class.rst

   hopsy.MarkovChain
   hopsy.sample

Random
""""""

.. autosummary::
   :toctree: ../generated/
   :template: ../_templates/autosummary/random_class.rst

   hopsy.RandomNumberGenerator
   hopsy.Normal
   hopsy.Uniform


