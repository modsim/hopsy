Introduction
============



Python-implemented proposals & models
-------------------------------------

For high flexibility when using hops, we made the proposal algorithm 
as well as the model defining the likelihood implementable in Python.
Consider the Metropolis criterion

.. math:: \alpha(\theta, \theta^*) = \min \Big\{ 1, \frac{\pi(\theta^*)q(\theta^*, \theta)}{\pi(\theta)q(\theta, \theta^*)} \Big\},

which computes the acceptance probability of a move :math:`\theta^*` 
generated with probability :math:`q(\theta, \theta^*)`,
where :math:`q` is the proposal distribution and :math:`\pi` the target distribution, 
from which we wish to draw samples.

In hopsy, we assume that :math:`\pi(x) = \exp\big\{-f(x)\big\}`, 
where :math:`f(x)` is the negative log likelihood, which you have to provide,
if you want to sample a self-implemented model. 
To read more about the details on how to sample custom models, please refer to :doc:`this page<../userguide/problem>`.

Although hopsy ships with numerous proposal algorithms optimized 
to work well in linearily-constrained spaces and written in C++,
it is also possible to implement the proposal algorithm in Python. 
To read more about the details on how to implement custom proposal algorithms with hopsy, 
please refer to :doc:`this page<../userguide/sampling>`.

