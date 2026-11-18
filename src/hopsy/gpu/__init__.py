"""CUDA backend of hopsy (experimental).

Only usable with the GPU build of hopsy. In the CPU build the compiled
extension ``hopsy.gpu._core`` is missing and importing this package fails.
"""

from ._core import *
