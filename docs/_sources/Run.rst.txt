Run
============================

   The ``hopsy.Run`` classes encapsulate the actual Markov chains and allow to sample a ``hopsy.Problem``. 
   Further, they provide methods to control the sampling process. 
   Note, that there does not actually exist a class ``hopsy.Run``. 
   Instead there exists a number of ``hopsy.Run``-classes, which all share the same methods, but are in fact different instantiations of
   the hops template class ``hopsy.Run``.

   The ``hopsy.Run()`` method, which aims to simulate the look and feel of a constructor, which automatically deduces the correct template
   parameters, acts as a factory for the different ``hopsy.Run`` classes. 
   At the end of this page, the existing ``hopsy.Run`` classes are listed for completeness. 
   However, since they all share the same methods and in fact only differ in the type of the ``hopsy.Problem`` instance, they store,
   their methods will be listed and described here.

   A noteworthy feature of hopsy is the possibility to implement proposal algorithms in Python and then let the Markov chains run with them.
   This is realized by wrapping the Python proposal into a ``hopsy.PyProposal`` class which acts as an interface for hops by simply delegating
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
   
      For more information on the proposal algorithms available, please refer to the Proposals section.

   :param int number_of_samples: sets the number of samples which will be drawn when calling ``run.sample()``. 
   :param int number_of_chains: sets the number of chains which will be run. 
   
   Note that the total number of samples generated will be ``number_of_samples * number_of_chains``.


.. function:: Run(problem: hopsy.Problem, proposal: object, number_of_samples: int = 1000, number_of_chains: int = 1)

   Creates a run object of appropriate type using the passed ``problem``. 
   
   :param hopsy.Problem problem: the problem which we aim to sample.
   :param object proposal: may be any Python object. Internally, ``proposal``
      will be wrapped inside ``hopsy.PyProposal``, which works as an interface between hops and any Python-implemented proposal class. Any calls
      from within hops to the proposal will be delegated by calling the appropriate Python function. 

   :param int number_of_samples: sets the number of samples which will be drawn when calling ``run.sample()``. 
   :param int number_of_chains: sets the number of chains which will be run. 
   
   Note that the total number of samples generated will be ``number_of_samples * number_of_chains``.


.. function:: Run(problem: hopsy.Problem, proposal: hopsy.PyProposal, number_of_samples: int = 1000, number_of_chains: int = 1)

   Creates a run object of appropriate type using the passed ``problem``. 
   
   :param hopsy.Problem problem: the problem which we aim to sample.
   :param hopsy.PyProposal proposal: the proposal class wrapping the actual proposal object. This method is just a more explicit version of the above. 

   :param int number_of_samples: sets the number of samples which will be drawn when calling ``run.sample()``. 
   :param int number_of_chains: sets the number of chains which will be run. 
   
   Note that the total number of samples generated will be ``number_of_samples * number_of_chains``.

.. note:: The following class reference refers to the non-existent class ``hopsy.Run``.
          It applies to any of the available run classes, which are listed at the end of this page.

.. class:: Run

   **Attributes:**

   .. attribute:: data: hopsy.Data

        A ``hopsy.Data`` object containing most prominently the states, as well as other data, that the chain produced.
        Read-only access.

   .. attribute:: problem: hopsy.Problem

        A``hopsy.Problem`` object that contains the problem formulation. This consists most prominently of the polytope inequality :math:`Ax \leq b`
        and the target function, a ``hopsy.Model``.
        Read-only access.

   .. attribute:: starting_points: List[numpy.ndarray[float64[m,1]]]

        The chains' starting points.
        Changing the chains starting points will lead to a reinitialization of the run object. 
        Note that ``len(starting_points)`` should match ``number_of_chains``. 
        If not enough starting points are supplied and no default starting point was set in ``problem``, then you will receive an exception.

   .. attribute:: proposal_name: str

        The name of the used proposal. May not be defined, if a ``hopsy.PyProposal`` is used, which does not implement the ``get_name`` method.
        Changing the ``proposal_name`` to one of the supplied ones above, will change the chains type, if it was not using a self-implemented proposal
        algorithm before.
        In the latter case, changing the name will have no effect in changing the chains proposal, but will lead to a reinitialization of the run object.

   .. attribute:: number_of_chains: int

        The number of chains.
        Changing ``number_of_chains`` will lead to a reinitialization of the run object.

   .. attribute:: number_of_samples: int

        The number of samples, each call to ``sample()`` will produce. 
        If ``sample_until_convergence == True``, then ``number_of_samples`` samples will be produced each round before computing the 
        convergence diagnostics. 

   .. attribute:: thinning: int

        Only every ``thinning``-th sample will be kept.
        If ``problem`` has a unrounding transformation, this transformation will be applied only on the samples being kept.
        This may speed up some chains in comparison to applying thinning only afterwards.

   .. attribute:: stepsize: float

        The stepsize parameter of the proposal algorithms. Note that not all proposal algorithms, especially self-implemented ones, may actually have
        a stepsize.
        Changing ``stepsize`` will lead to a reinitialization of the run object.

   .. attribute:: fisher_weight: float 

        The Fisher weight of the ``CSmMALA`` algorithm.
        Changing ``fisher_weight`` will lead to a reinitialization of the run object.

   .. attribute:: random_seed: float

        Random seed of the hops random number generators. Note that for self-implemented proposal algorithms, this seed has no effect.
        Changing ``random_seed`` will lead to a reinitialization of the run object.

   .. attribute:: sample_until_convergence: bool
    
        Flag, whether ``sample()`` should try to sample until convergence is diagnosed. 
        Convergence is diagnosed, if the potential scale reduction factor is below ``diagnostics_threshold``, which by default is set to 1.05.

   .. attribute:: diagnostics_threshold: float

        The threshold, below which convergence is diagnosed using the potential scale reduction factor.

   .. attribute:: max_repetitions: int

        The maximal number of repetitions when sampling until convergence. This prevents endless loops when convergence is not reached.

   **Methods:**

   .. method:: init() -> None

       Initializes the run object. This initializes the ``data`` object, if it was not initialized before and **erases all of its content.**
       It initializes the random number generators and initializes missing starting points, if a default starting point was set in problem.
       Further, it sets up the chains and preallocates memory for the samples.

   .. method:: sample() -> None

      Draws samples from the given problem using the passed settings. If the run object is uninitialized, then it will be initialized first.
      If ``sample_until_convergence == True``, then the run will repeatedly draw samples and compute the potential scale reduction factor, 
      to determine whether convergence has been reached or not. 
      This may produce up to ``max_repetitions * number_of_samples`` samples.

   .. method:: sample(number_of_samples: int, thinning: int = 1) -> None

      Draws samples from the given problem using the passed settings, except using the passed ``number_of_samples`` and ``thinning``, instead
      of the settings stored in the run object.
      If the run object is uninitialized, then it will be initialized first.
      If ``sample_until_convergence == True``, then the run will repeatedly draw samples and compute the potential scale reduction factor, 
      to determine whether convergence has been reached or not. 
      This may produce up to ``max_repetitions * number_of_samples`` samples.

Possible Run-types
~~~~~~~~~~~~~~~~~~

.. autoclass:: DegenerateMultivariateGaussianRun
.. autoclass:: MixtureRun
.. autoclass:: MultivariateGaussianRun
.. autoclass:: PyRun
.. autoclass:: RosenbrockRun
.. autoclass:: UniformRun


