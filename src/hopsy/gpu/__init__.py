"""CUDA backend of hopsy (experimental).

Only usable with the GPU build of hopsy. In the CPU build the compiled
extension ``hopsy.gpu._core`` is missing and importing this package raises an
ImportError that explains how to install the GPU build.
"""

#TODO: Fill in the registry once decided
_GPU_INSTALL_COMMAND = (
    "pip install hopsy --extra-index-url "
    "<foo>"
)

try:
    from ._core import *
except ModuleNotFoundError as err:
    # Only a missing _core means "this is the CPU build". Any other load error
    # (e.g. a missing CUDA library) is a real problem and is passed on unchanged.
    if err.name != f"{__name__}._core":
        raise
    raise ImportError(
        "hopsy.gpu is not available: this is the CPU-only build of hopsy. "
        f"To use the GPU backend, install the GPU build:\n    {_GPU_INSTALL_COMMAND}"
    ) from err
