hopsy - Highly Optimized Polytope Sampling in Python
====================================================


.. raw:: html

   <div class="container-xl">
      <div class="row">
         <div class="col-md-6">

         <br>


.. image:: _static/hopsy.png
   :align: center
   :width: 400
   :alt: hopsy logo


.. raw:: html

   <br>


:doc:`hopsy <index>` is a scientific Python tool for Markov chain Monte Carlo sampling on convex polytopes of the form

.. math:: \mathcal{P} := \{ x : Ax \leq b \},

a problem that often arises in 
`Bayesian metabolic flux analysis <https://onlinelibrary.wiley.com/doi/10.1002/bit.26379>`_.
It is built using `pybind11 <https://pybind11.readthedocs.io/en/stable/>`_ 
and works on the powerful C++ polytope sampling tool 
`HOPS <https://modsim.github.io/hops/doxygen/index.html>`_.
Its goal is to combine the efficiency of the C++-backend with the flexibility of the Python language whilst
maintaining a comprehensible API and interoperability with other Python libraries.

:doc:`hopsy <index>` offers not only a magnitude of efficiently implemented algorithms specialized for sampling 
linearily constrained spaces, but also allows for implementing custom samplers in C++.
Similarily, besides providing a few test :doc:`models<userguide/problem>`, :doc:`hopsy <index>` allows sampling
user-implemented models.
A few examples including :math:`^{13}\mathrm{C}`-metabolic flux analysis, bioreactor kinetics, 
statistical physics simulations and a SIR-model can be found in the examples section.

An important task, especially when sampling expensive to evaluate target functions, is tuning
of the proposal's hyperparameters.
:doc:`hopsy <index>` offers a Thompson Sampling-based approach which can perform tuning based on different objective
functions.
Please refer to :doc:`this chapter<userguide/tuning>` for more information.


Further resources
"""""""""""""""""

If you encounter bugs, please feel free to open an issue `on GitHub <https://github.com/modsim/hopsy/issues>`_,
where you can also find the `source code <https://github.com/modsim/hopsy>`_.
For questions, help or feedback regarding :doc:`hopsy <index>`, 
visit us `on our Gitter <https://gitter.im/modsim/hopsy>`_!


.. raw:: html

         </div>
         <div class="col-md-6">

    <form class="bd-search align-items-center" action="search.html" method="get">
      <input type="search" class="form-control search-front-page" name="q" id="search-input" placeholder="&#128269; Search the docs ..." aria-label="Search the docs ..." autocomplete="off">
    </form>


.. raw:: html

    <h2>Quick Start</h2>


**Installation** using pip

.. code::

   pip3 install hopsy

**Code example**

.. code::

   import hopsy

   problem = hopsy.Problem([[1, 1], [-1, 0], [0, -1]], [1, 0, 0])

   chain = hopsy.MarkovChain(problem, starting_point=[.5, .5])
   rng = hopsy.RandomNumberGenerator(seed=42)

   accrate, samples = hopsy.sample(chain, rng, n_samples=1000, thinning=10)


.. raw:: html

   <h2>Contents</h2>


.. toctree::
   :maxdepth: 1

   FirstSteps.ipynb

.. toctree::
   :maxdepth: 2

   userguide.rst
   examples.rst

.. toctree::
   :maxdepth: 1

   zreference.rst


.. raw:: html

         </div>
      </div>
   </div>




