Installation
============

``hopsy`` can be easily installed from the Python Package Index using 

::

  pip install hopsy


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

 pip install dist/hopsy-x.y.z-tag.whl

where the version ``x.y.z`` and tag ``tag`` will depend on the verison you downloaded and your build environment.


