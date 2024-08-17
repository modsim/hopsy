"""
This file contains useful definitions. E.g., it sets the multiprocessing context for all of hopsy
"""


class _submodules:
    import multiprocessing


_s = _submodules

multiprocessing_context = _s.multiprocessing.get_context("spawn")
