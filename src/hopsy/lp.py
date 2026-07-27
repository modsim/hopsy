""" """

import threading as _threading

from ._rounding.settings import RoundingSettings


class LP:
    """Singleton for controlling rounding parameters for linear programming.

    Pattern from https://medium.com/analytics-vidhya/how-to-create-a-thread-safe-singleton-class-in-python-822e1170a7f6
    """

    __instance = None
    settings = None
    _lock = _threading.Lock()

    def __new__(self, *args, **kwargs):
        """thread safe instance instantiation"""
        if not self.__instance:
            with self._lock:
                # another thread could have created the instance
                # before we acquired the lock. So check that the
                # instance is still nonexistent.
                if not self.__instance:
                    self.__instance = super(LP, self).__new__(self)
                    self.settings = RoundingSettings()
        return self.__instance

    def reset(self):
        """Reset all rounding settings to their default values."""
        self.settings = RoundingSettings()
