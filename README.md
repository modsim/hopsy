# HOPSY - Python bindings for HOPS

 [![pipeline status](https://jugit.fz-juelich.de/fluxomics/hopsy/badges/develop/pipeline.svg)](https://jugit.fz-juelich.de/fluxomics/hopsy/-/commits/develop)
 [![docstring coverage](https://jugit.fz-juelich.de/fluxomics/hopsy/-/jobs/218829/artifacts/raw/docs/docstrcov.svg?job=test_release)](https://jugit.fz-juelich.de/fluxomics/hopsy/-/jobs/218829/artifacts/raw/docs/.docstrcovreport?job=test_release)
 
A python interface for HOPS - the **H**ighly **O**ptimized toolbox for **P**olytope **S**ampling.
Built using `pybind11`

<img src="docs/hopsy.png" alt="HOPSY Logo" width="300"/>

**hopsy** is the attempt to offer some of the key functionatlity of **hops** through a Python interface.
hops is a highly template-based C++-library for Markov chain Monte Carlo sampling on convex polytopes
 
P = {x : Ax &#8804; b},

which often arises in metabolic flux analysis.


## Installation

hopsy can be easily installed from the Python Package Index using ``pip install hopsy``.
Alternatively, you can download the source code from our GitHub repository with

```bash
git clone https://github.com/modsim/hopsy --recursive
```

Note the `--recursive` option which is
needed for hops, eigen and pybind11 submodules.

Next, compile either a binary wheel using pip

```bash
pip wheel --no-deps hopsy/
```

or use the standard CMake routine

```bash
mkdir hopsy/cmake-build-release && cd hopsy/cmake-build-release
cmake ..
make 
```

Note however that the binary wheel produced from ``pip`` can be actually installed using ``pip``, using

```bash
pip install hopsy-x.y.z-tag.whl
```

where the version `x.y.z` and tag `tag` will depend on the verison you downloaded and your build environment.
If you use the CMake routine, the compiled shared library will be located in `build/` and can 
be used within the directory. 

To compile wheels for distribution via the Python Package Index (pypi.org), use the `makewheels.sh` script.


### Prerequisites for compiling from source

**On Unix (Linux, OS X)**

* A compiler with C++11 support
* CMake >= 3.4 or Pip 10+
* Ninja or Pip 10+
* Docker (optional, for building wheels)


## License


## Examples

A basic usage example is presented below. More examples can be found in `tests/` directory.

```python
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
```

[`cibuildwheel`]:          https://cibuildwheel.readthedocs.io
[FAQ]: http://pybind11.rtfd.io/en/latest/faq.html#working-with-ancient-visual-studio-2009-builds-on-windows
[vs2015_runtime]: https://www.microsoft.com/en-us/download/details.aspx?id=48145
