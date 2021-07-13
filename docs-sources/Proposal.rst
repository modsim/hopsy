Proposals
=========

hopsy is not particulary designed for development of polytope samplers, but instead aims at making the proposals available in hops
easily usable to solve the practitioner's problems at hand.
Also, the proposal - together with the Metropolis filter and the models likelihood function - form the core of any Markov chain Monte Carlo algorithm.
Thus, we believe that for sake of performance, proposals should be optimized and implemented in C++ and hops itself.
For more informations on the proposal distributions implemented in hops, please refer to xyz.

Custom Python proposals
-----------------------

Nevertheless, sometimes it might be desirable to do some rapid prototyping to test promising ideas.
For this case, it is possible to implement Python proposals and to instruct hopsy (and ultimately hops) to use those.

Similarily to the custom Python models, this works by wrapping the Python proposal in the ``hopsy.PyProposal`` class.
``hopsy.PyProposal`` implements all functions necessary for the proposal class to be usable in hops by delegating the function
calls to the store Python proposal.
Thus, it is obviously needed, that the Python proposal implements the corresponding functions.

The functions are

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
------------

The following example implements the constraint Gaussian random walk and a constraint adaptive Metropolis 
algorithm, following the original idea of Haario et al., 2001.

::

 import hopsy
 import numpy as np
 
 class GaussianProposal:
     def __init__(self, A: np.ndarray, b: np.ndarray, x: np.ndarray, cov: np.ndarray):
         self.A = A
         self.b = b
         self.x = x
         self.cov = cov
         self.r = 1
         self.proposal = x
 
     def propose(self):
         mean = np.zeros((len(self.cov),))
         y = np.random.multivariate_normal(mean, self.cov).reshape(-1, 1)
         self.proposal = self.x + self.r * y 
 
     def accept_proposal(self):
         self.x = self.proposal
 
     def calculate_log_acceptance_probability(self) -> float:
         if ((self.A @ self.proposal - self.b) >= 0).any():
             return -np.inf
         return 0
 
     def get_state(self) -> np.ndarray:
         return self.x
 
     def set_state(self, new_state: np.ndarray):
         self.x = new_state.reshape(-1,1)
 
     def get_proposal(self) -> np.ndarray:
         return self.proposal
 
     def get_stepsize(self) -> float:
         return self.r
 
     def set_stepsize(self, new_stepsize: float):
         self.r = new_stepsize
 
     def get_name(self) -> str:
         return "PyGaussianProposal"
 
 
 class AdaptiveGaussianProposal:
     def __init__(self, A: np.ndarray, b: np.ndarray, x: np.ndarray, eps = 0.001):
         self.A = A
         self.b = b
         self.x = x
         self.eps = eps
         self.r = 1
         self.proposal = x
         self.t = 0
         self.cov = np.identity(len(x))
         self.mean = np.zeros(x.shape)
 
 
     def propose(self):
         new_mean = (self.t * self.mean + self.x) / (self.t + 1)
         self.cov = ((self.t - 1) * self.cov + self.t * np.outer(self.mean, self.mean) - (self.t + 1) * np.outer(new_mean, new_mean) + np.outer(self.x, self.x) + self.eps * np.identity(len(self.x))) / self.t if self.t > 0 else self.cov
         self.t += 1
 
         proposal_mean = np.zeros((len(self.cov),))
         y = np.random.multivariate_normal(proposal_mean, self.cov).reshape(-1, 1)
         self.proposal = self.x + self.r * y 
 
     def accept_proposal(self):
         self.x = self.proposal
 
     def calculate_log_acceptance_probability(self) -> float:
         if ((self.A @ self.proposal - self.b) >= 0).any():
             return -np.inf
         return 0
 
     def get_state(self) -> np.ndarray:
         return self.x
 
     def set_state(self, new_state: np.ndarray):
         self.x = new_state.reshape(-1,1)
 
     def get_proposal(self) -> np.ndarray:
         return self.proposal
 
     def get_stepsize(self) -> float:
         return self.r
 
     def set_stepsize(self, new_stepsize: float):
         self.r = new_stepsize
 
     def get_name(self) -> str:
         return "AdaptiveGaussianPyProposal"
 
 A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
 b = np.array([[1], [1], [0], [0]]);
 
 x0 = np.array([[0.1], [0.1]])
 
 mu = np.zeros((2,1))
 cov = 0.1*np.identity(2)
 
 gaussian_proposal = GaussianProposal(A, b, x0, np.identity(2))
 adaptive_proposal = AdaptiveGaussianProposal(A, b, x0)
 
 model = hopsy.MultivariateGaussianModel(mu, cov)
 problem = hopsy.Problem(A, b, model)
 
 gaussian_run = hopsy.Run(problem, gaussian_proposal)
 adaptive_run = hopsy.Run(problem, adaptive_proposal)
 
 gaussian_run.starting_points = [x0]
 adaptive_run.starting_points = [x0]
 
 gaussian_run.sample(10000)
 adaptive_run.sample(10000)
 
 gaussian_stepsize = gaussian_run.stepsize
 adaptive_stepsize = adaptive_run.stepsize
 
 gaussian_acc_rate = hopsy.compute_acceptance_rate(gaussian_run.data)[0]
 adaptive_acc_rate = hopsy.compute_acceptance_rate(adaptive_run.data)[0]
 
 gaussian_esjd = hopsy.compute_expected_squared_jump_distance(gaussian_run.data)[0]
 adaptive_esjd = hopsy.compute_expected_squared_jump_distance(adaptive_run.data)[0]
 
 print("         | Gaussian proposal"                         + " | Adaptive proposal")
 print("---------+------------------"                         + "-+------------------")
 print("Stepsize |               " + str(gaussian_stepsize)   + " |               " + str(adaptive_stepsize))
 print("Acc Rate |             " + str(gaussian_acc_rate)[:5] + " |             " + str(adaptive_acc_rate)[:5]) 
 print("ESJD     |             " + str(gaussian_esjd)[:5]     + " |             " + str(adaptive_esjd)[:5])

(Approximate) output:

.. code-block:: console

          | Gaussian proposal | Adaptive proposal
 ---------+-------------------+------------------
 Stepsize |               1.0 |               1.0
 Acc Rate |             0.046 |             0.293
 ESJD     |             0.006 |             0.018


Reference
---------

.. currentmodule:: hopsy

.. autoclass:: PyProposal

   Wraps any Python object as a valid C++-proposal, making it usable in hops.
   This works by simply delegating calls to its methods to the wrapped Python object.

   .. method:: PyProposal(proposal: object)
   
      Constructs the ``hopsy.PyProposal`` by wrapping the passed Python object.
      For the proposal to work correctly together with hops, it should provide the methods

      * ``propose() -> None``

      * ``accept_proposal() -> None``

      * ``calculate_log_acceptance_probability() -> float``

      * ``get_state() -> numpy.ndarray[shape[d,1]]``

      * ``set_state(new_state: numpy.ndarray[shape[d,1]]) -> None``

      * ``get_proposal() -> numpy.ndarray[shape[d,1]]``

      * ``get_stepsize() -> float``

      * ``set_stepsize(new_stepsize: float)``

      * ``get_name() -> str``

      :param object proposal: The Python-implemented proposal distribution, which should implement the above mentioned methods.
      :return: the hopsy.PyProposal object wrapping ``object``
      :rtype: hopsy.PyProposal

