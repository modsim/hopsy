"""CUDA backend of hopsy (experimental).

Only usable with the GPU build of hopsy. In the CPU build the compiled
extension ``hopsy.gpu._core`` is missing and importing this package raises an
ImportError that explains how to build hopsy with GPU support.
"""

# No installable GPU wheel exists yet, so the hint points to a source build.
_GPU_BUILD_COMMAND = "pip install . -Ccmake.define.HOPSY_BUILD_GPU=ON"

try:
    from ._core import *
except ModuleNotFoundError as err:
    # Only a missing _core means "this is the CPU build". Any other load error
    # (e.g. a missing CUDA library) is a real problem and is passed on unchanged.
    if err.name != f"{__name__}._core":
        raise
    raise ImportError(
        "hopsy.gpu is not available: this is the CPU-only build of hopsy. "
        "To use the GPU backend, build hopsy from source with GPU support "
        "(Linux x86_64, CUDA toolkit >= 12.8 with cuBLAS, cuSPARSE and cuRAND). "
        f"From a hopsy source checkout:\n    {_GPU_BUILD_COMMAND}"
    ) from err
