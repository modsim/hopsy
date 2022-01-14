import unittest

from .. import *

class ModelTests(unittest.TestCase):
    def test_gaussian_pickling(self):
        problem = Gaussian()
        print(problem)
        print(problem.__getstate__())

