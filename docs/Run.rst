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


.. currentmodule:: hopsy

.. function:: Run(problem: hopsy.Problem, proposal_string: str = "HitAndRun", number_of_samples: int = 1000, number_of_chains: int = 1)

    Creates a run object of appropriate type using the passed ``problem``. The actual return type depends on the type of ``problem``.
   
    * ``problem`` which we aim to sample.

    * ``proposal_string`` may be any string identifier of propsals included in HOPS. Valid identifiers are:

      * ``"BallWalk"``
      * ``"CoordinateHitAndRun"``
      * ``"CSmMALA"``
      * ``"CSmMALANoGradient"``
      * ``"DikinWalk"``
      * ``"Gaussian"``
      * ``"HitAndRun"``

   * ``number_of_samples`` sets the number of samples which will be drawn when calling ``run.sample()``. 
   
   * ``number_of_chains`` sets the number of chains which will be run. 
   
   Note that the total number of samples generated will be ``number_of_samples * number_of_chains``.


.. function:: Run(problem: hopsy.Problem, proposal: object, number_of_samples: int = 1000, number_of_chains: int = 1)

   Creates a run object of appropriate type using the passed ``problem``. 
   
   ``proposal`` may be any Python object. Internally, ``proposal``
   will be wrapped inside ``hopsy.PyProposal``, which works as an interface between HOPS and any Python-implemented proposal class. Any calls
   from within HOPS to the proposal will be delegated by calling the appropriate Python function. 
   
   ``number_of_samples`` sets the number of samples which will be drawn when calling ``run.sample()``. 
   
   ``number_of_chains`` sets the number
   of chains which will be run. 
   
   Note that the total number of samples generated will be ``number_of_samples * number_of_chains``.


.. function:: Run(problem: hopsy.Problem, proposal: hopsy.PyProposal, number_of_samples: int = 1000, number_of_chains: int = 1)

   Creates a run object of appropriate type using the passed ``problem``. 
   
   ``proposal`` has to be a PyProposal. This method is just a more
   explicit version of the above.. 
   
   ``number_of_samples`` sets the number of samples which will be drawn when calling ``run.sample()``. 
   
   ``number_of_chains`` sets the number
   of chains which will be run. 
   
   Note that the total number of samples generated will be ``number_of_samples * number_of_chains``.


.. autosummary::
   :toctree: _generate

   DegenerateMultivariateGaussianRun
   MultimodalMultivariateGaussianRun
   MultivariateGaussianRun
   PyRun
   RosenbrockRun
   UniformRun


