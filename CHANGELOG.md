# Changelog

## v1.6.1
- [#171](https://jugit.fz-juelich.de/IBG0/ModSim/hopsy/-/issues/171): Make mcbackend optional dependency

## v1.6.0
- [#132](https://jugit.fz-juelich.de/IBG0/ModSim/hopsy/-/issues/132): Add support for Python 3.13
- [#154](https://jugit.fz-juelich.de/IBG0/ModSim/hopsy/-/issues/154): Add support for numpy 2.0 and newer

## v1.5.3
- [#169](https://jugit.fz-juelich.de/IBG0/ModSim/hopsy/-/issues/169): Move RandomNumberGenerator to hops (backend)
- [#168](https://jugit.fz-juelich.de/IBG0/ModSim/hopsy/-/issues/168): Add warning to help users avoid unintended constraint violations
- [#167](https://jugit.fz-juelich.de/IBG0/ModSim/hopsy/-/issues/167): Fix example code for GEM sampling
- [#166](https://jugit.fz-juelich.de/IBG0/ModSim/hopsy/-/issues/166): Add `simplify` kwarg to `hopsy.round`
- [#165](https://jugit.fz-juelich.de/IBG0/ModSim/hopsy/-/issues/165): Treat thinning=0 as no thinning instead of producing NaNs
- [#162](https://jugit.fz-juelich.de/IBG0/ModSim/hopsy/-/issues/162): Fix sign error in python implementation of parallel tempering
- [#161](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/161): Persist markov chain proposal after parallel sampling

## v1.5.2
- [#115](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/115): Improve initialization of custom PyProposals in MarkovChain
- [#114](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/114): Disable inheriting from PyProposal and PyModel

## v1.5.1
- [#138](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/138): Ignore fixed dimensions when computing ESS & rhat
- [#157](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/157): Pin numpy<2.0
- [#141](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/141): Add minimal example for genome-scale sampling
- [#152](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/152): Add test-case for selecting start point with equality constraints
- [#69](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/69): Add CLA
- [#155](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/155): Increase numerical accuracy for parallel tempering when using python multiprocessing
- [#153](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/153): Add tuning target selection to setup function

## v1.5.0
- [#145](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/145): Improve docs
- [#127](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/127): Add method to heuristically setup good sampling configurations depending on the problem
- [#139](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/139): Add transformation of start points during add\_equality\_constraints
- [#88](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/88): Fix parallelization for sampling with mcbackend as sample storage
- [#123](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/123): Add parallel tempering support for non-MPI users
- [#137](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/137): Fix rare bug for users implementing their own PyProposals for non-uniform sampling
- [#136](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/136): Improve numerical stability of mixture models
- [#133](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/133): Add default starting point to proposals and markov chains when none is given
- [#128](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/128): Add stepsize tuning with Python TuningTargets
- [#122](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/122): Fix rounding for Reversible Jump MCMC
- [#125](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/125): Update Readme.md & Fix automatic docs generation
- [#106](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/106): Add official support for python 3.12
- [#130](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/130): Update requirements-dev.txt
- [#108](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/108): For development: Fix setup.py warning on missing commit ID
- [#119](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/119): Fix numpy deprecation warnings in unit tests
- [#121](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/121): Update docs for problem

## v.1.4.1
- [#120](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/120): Fix parallel sampling with additional equality constraints
- [#117](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/117): Update Optlang, sympy and numpy dependencies
- [#118](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/118): Fix back\_transform when manually adding equality constraints and rounding

## v1.4.0
- [#113](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/113): Fix small bug in back\_transform
- [#111](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/111): Update documentation
- [#92](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/92): Add convenience functions for generating hybercubes, simplices and Birkhoff polytopes
- [#110](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/110): Add convenience method to problem for computing slacks
- [#82](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/82): Add option to compute\_chebyshev to return point in transformed or original space
- [#102](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/102): Add RJMCMC
- [#104](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/104): Add multiphase sampling
- [#83](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/83): Readd vectorization for hopsy
- [#103](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/103): Fix callback example
- [#105](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/105): Drop support for unsupported python 3.7
- [#101](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/101): Add helper functions for testing if polytope is empty
- [#99](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/99): Fix benchmark example typo
- [#98](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/98): Consistent names for model functions
- [#97](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/97): Disable jekyll in automatic doc building
- [#96](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/96): Fix pickling for problem
- [#95](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/95): Fix parallel sampling for PyProposals
- [#91](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/91): Add benchmark example
- [#90](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/90): Reduce ci build times
- [#87](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/87): Reduce boilerplate for custom pyproposals
- [#71](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/71): Rename functions and deprecate old variants

## v1.3.1
- [#85](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/85): Fix release for Python 3.10 and Python 3.11 windows

## v1.3.0
- [#78](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/78): Make (mac+linux+win compatible) sdist
- [#79](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/79): Add installation example for arch linux
- [#33](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/33): Add direct support for equality constraints
- [#77](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/77): Add feasibility check to compute\_chebyshev\_center

## v1.2.0
- [#70](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/70): Add correct license
- [#64](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/64): Optional in memory saving of states
- [#58](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/58): Add mcbackend support
- [#50](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/50): Bugfix: Fix state copying for parallel chains
- [#49](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/49): Parallelize computation of statistics on samples (arviz)
- [#48](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/48): Replace keyword n\_threads by n\_procs in sample function
- [#47](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/47): Add mcbackend compatibility
- [#11](https://jugit.fz-juelich.de/IBG-1/ModSim/hopsy/-/issues/11): Add parallel tempering with MPI
