.. image:: hopsy.png
   :align: center
   :width: 400
   :alt: hopsy logo

hopsy - Python bindings for hops
================================

A Python interface for hops - the **H**\ ighly **O**\ ptimized toolbox for **P**\ olytope **S**\ ampling.
Built using pybind11.

   
Installation
------------

hopsy can be easily installed from the Python Package Index using ``pip install hopsy``.
Alternatively, you can download the source code from our GitHub repository with

::

 git clone https://github.com/modsim/hopsy --recursive
 
and compile either a binary wheel using pip

::

 pip wheel --no-deps hopsy/

or use the standard CMake routine

::

 mkdir hopsy/cmake-build-release && cd hopsy/cmake-build-release
 cmake ..
 make 

Note however that the binary wheel produced from ``pip`` can be actually installed using ``pip``, using

::

 pip install hopsy-x.y.z-tag.whl

where the version ``x.y.z`` and tag ``tag`` will depend on the verison you downloaded and your build environment.


Introduction
------------

**hopsy** is the Python interface for **hops**, a highly template-based C++-library for Markov chain Monte Carlo sampling on convex polytopes

.. math:: \mathcal{P} = \{ x : Ax \leq b \}.

The key functionality of hops is provided by encapsulating the key components of the most common use-cases within four to five "classes":

* The ``hopsy.Model``, which defines the target function to be sampled. hopsy ships with a number of simple target functions
  which can be used out of the box, but also allows to use Python-implemented target functions by wrapping them internally in an appropriate C++-class.

* The ``hopsy.Problem`` collects the key components of a convex polytope sampling problem: the polytope-defining inequality constraints and the target 
  function or, as we call it, the model.

* The ``hopsy.Run`` takes a ``hopsy.Problem`` and then constructs the Markov chains to actually sample the problem. 
  Invoking the ``hopsy.Run.sample()`` method then does the actual work of drawing samples.

* Finally, the ``hopsy.Data`` class collects all data produced when sampling. 
  This consists most prominently of the generated states. 
  Further data stored, are e.g. timestamps, negative log-likelihood values and a few statistics.

These are the "classes", which we deemed most useful and important for any practitioner, who wants to sample a linearily constrained problem at hand. 

.. note:: Note, that we write classes in quotes, as the mentioned ``hopsy.Model``, ``hopsy.Problem`` and ``hopsy.Run`` "classes" are not actually 
   classes.
   In fact, ``hopsy.Model`` rather refers to a non-existent base class of the models implemented in hops and ``hopsy.Problem`` and ``hopsy.Run`` refer to
   the hops template classes ``hops::Problem`` and ``hops::Run``. 
   Since Python does not provide template functionality, we instantiated the ``hops::Problem`` and ``hops::Run`` template classes 
   with the models we provide. 
   This leads to a number of different problem and run classes, which however all behave the same except that they act on different models.
   In order to make this trick more transparent to the user, ``hopsy.Problem`` and ``hopsy.Run`` are two factory methods, 
   which aim at simulating the look and feel of a templated constructor with automatic template deduction.
   Throughout this documentation, we will however use ``hopsy.Model``, ``hopsy.Problem`` and ``hopsy.Run`` also as a placeholder for any of the actually
   available models, problems and runs.


Example code
------------

A short example on how to sample a Gaussian target distribution restricted to :math:`\mathcal{P} = [0,1]^2`.

::

   import hopsy
   import numpy as np
   
   # the polytope is defined as 
   #          P := {x : Ax <= b}
   # thus we need to define A and b. these constraints form the simple box [0,1]^2.
   A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
   b = np.array([[1], [1], [0], [0]]);
   
   # next we define our target distribution as an isotropic Gaussian with mean 0 and 
   # identity covariance.
   mu = np.zeros((2,1))
   cov = np.identity(2)
   
   model = hopsy.MultivariateGaussianModel(mu, cov)
   
   # the complete problem is defined by the target distribution and the constrained domain, 
   # defined by the above mentioned inequality
   problem = hopsy.Problem(A, b, model)
   
   # the run object contains and constructs the markov chains. in the default case, the
   # Run object will have a single chain using the Hit-and-Run proposal algorithm and is
   # set to produce 10,000 samples.
   run = hopsy.Run(problem)
   
   # we finally sample
   run.sample()
   
   # from the run, we can now extract the produced data
   data = run.data
   
   # the states is a list of lists of numpy.ndarrays, which can be casted to a numpy.ndarray
   # which then has the shape (m,n,d), where m is the number of chains, n the number of samples
   # and d the dimenion
   states  = data.states


Python-implemented proposals & models
-------------------------------------

For high flexibility when using hops, we made the proposal algorithm as well as the model defining the likelihood implementable in Python.
Consider the Metropolis criterion

.. math:: \alpha(\theta, \theta^*) = \min \Big\{ 1, \frac{\pi(\theta^*)q(\theta^*, \theta)}{\pi(\theta)q(\theta, \theta^*)} \Big\},

which computes the acceptance probability of a move :math:`\theta^*` generated with probability :math:`q(\theta, \theta^*)`,
where :math:`q` is the proposal distribution and :math:`\pi` the target distribution, from which we wish to draw samples.

In hopsy, we assume that :math:`\pi(x) = \exp\big\{-f(x)\big\}`, where :math:`f(x)` is the negative log likelihood, which you have to provide,
if you want to sample a self-implemented model. To read more about the details on how to sample custom models, please refer to :doc:`this page<Model>`.

Although hopsy ships with numerous proposal algorithms optimized to work well in linearily-constrained spaces and written in C++,
it is also possible to implement the proposal algorithm in Python. 
To read more about the details on how to implement custom proposal algorithms with hopsy, please refer to :doc:`this page<Proposal>`.


Contents
--------

.. toctree::
   :maxdepth: 2

   Model <Model.rst>
   Problem <Problem.rst>
   Run <Run.rst>
   Proposal <Proposal.rst>
   Data <Data.rst>


