Run
============================

   The ``Run`` classes encapsulate the actual Markov chains and allow to sample a ``Problem``. 
   Further, they provide methods to control the sampling process. 
   Note, that there does not actually exist a class ``Run``. 
   Instead there exists a number of ``Run``-classes, which all share the same methods, but are in fact different instantiations of
   the HOPS template class ``Run``.

   The ``Run()`` method, which aims to simulate the look and feel of a constructor, which automatically deduces the correct template
   parameters, acts as a factory for the different ``Run`` classes. 
   At the end of this page, the existing ``Run`` classes are listed for completeness. 
   However, since they all share the same methods and in fact only differ in the type of the ``hopsy.Problem`` instance, they store,
   their methods will be listed and described here.

   A noteworthy feature of hopsy is the possibility to implement proposal algorithms in Python and then let the Markov chains run with them.
   This is realized by wrapping the Python proposal into a ``hopsy.PyProposal`` class which acts as an interface for HOPS by simply delegating
   the calls to its function to the respective Python methods. For more details on how to use this, refer to ``hopsy.PyProposal``.


Reference
---------

.. currentmodule:: hopsy

.. function:: Run(problem: hopsy.Problem, proposal_string: str = "HitAndRun", number_of_samples: int = 1000, number_of_chains: int = 1)

   Creates a run object of appropriate type using the passed ``problem``. The actual return type depends on the type of ``problem``.
   
   :param hopsy.Problem problem: the problem which we aim to sample.
   :param str proposal_string: may be any string identifier of propsals included in hops. Valid identifiers are:

      * ``"BallWalk"``
      * ``"CoordinateHitAndRun"``
      * ``"CSmMALA"``
      * ``"CSmMALANoGradient"``
      * ``"DikinWalk"``
      * ``"Gaussian"``
      * ``"HitAndRun"``

   :param int number_of_samples: sets the number of samples which will be drawn when calling ``run.sample()``. 
   :param int number_of_chains: sets the number of chains which will be run. 
   
   Note that the total number of samples generated will be ``number_of_samples * number_of_chains``.


.. function:: Run(problem: hopsy.Problem, proposal: object, number_of_samples: int = 1000, number_of_chains: int = 1)

   Creates a run object of appropriate type using the passed ``problem``. 
   
   :param hopsy.Problem problem: the problem which we aim to sample.
   :param object proposal: may be any Python object. Internally, ``proposal``
      will be wrapped inside ``hopsy.PyProposal``, which works as an interface between HOPS and any Python-implemented proposal class. Any calls
      from within HOPS to the proposal will be delegated by calling the appropriate Python function. 

   :param int number_of_samples: sets the number of samples which will be drawn when calling ``run.sample()``. 
   :param int number_of_chains: sets the number of chains which will be run. 
   
   Note that the total number of samples generated will be ``number_of_samples * number_of_chains``.


.. function:: Run(problem: hopsy.Problem, proposal: hopsy.PyProposal, number_of_samples: int = 1000, number_of_chains: int = 1)

   Creates a run object of appropriate type using the passed ``problem``. 
   
   ``proposal`` has to be a PyProposal. This method is just a more
   explicit version of the above.. 
   
   ``number_of_samples`` sets the number of samples which will be drawn when calling ``run.sample()``. 
   
   ``number_of_chains`` sets the number
   of chains which will be run. 
   
   Note that the total number of samples generated will be ``number_of_samples * number_of_chains``.

The following class reference refers to a non-existent class ``hopsy.AnyRun``. 
It applies to any of the available run classes, which are listed at the end of this page.

.. class:: AnyRun

   .. method:: Run()
   .. method:: get_data() -> hopsy.Data
   .. method:: init() -> None
   .. method:: sample() -> None
   .. method:: sample(number_of_samples: int, thinning: int = 1) -> None
   .. method:: get_problem
   .. method:: set_starting_points(starting_points: list[numpy.ndarray[numpy.float64[m, 1]]]) -> None
   .. method:: get_starting_points() -> list[numpy.ndarray[numpy.float64[m, 1]]] 
   .. method:: set_markov_chain_type
   .. method:: get_markovv_chain_type
   .. method:: set_number_of_chains(number_of_chains: int) -> None
   .. method:: get_number_of_chains() -> int
   .. method:: set_number_of_samples(number_of_samples: int) -> None
   .. method:: get_number_of_samples() -> int
   .. method:: set_thinning(thinning: int) -> None
   .. method:: get_thinning() -> int
   .. method:: enable_rounding() -> None
   .. method:: disable_rounding() -> None
   .. method:: set_stepsize(stepsize: float) -> None
   .. method:: get_stepsize() -> float
   .. method:: set_fisher_weight(fisher_weight: float) -> None
   .. method:: get_fisher_weight() -> float
   .. method:: set_random_seed(seed: float) -> None
   .. method:: get_random_seed() -> float
   .. method:: set_sampling_until_convergence(sample_until_convergence: bool, diagnostics_threshold: float, max_repetitions: int) -> None
   .. method:: unset_sampling_until_convergence() -> None
   .. method:: get_diagnostics_threshold() -> float
   .. method:: get_max_repetitions() -> int

.. autosummary::
   :toctree: _generate

   DegenerateMultivariateGaussianRun
   MultimodalMultivariateGaussianRun
   MultivariateGaussianRun
   PyRun
   RosenbrockRun
   UniformRun


