import pickle
import unittest

from hopsy import *


class RandomTests(unittest.TestCase):
    def test_randomnumbergenerator_pickling(self):
        rng = RandomNumberGenerator()
        # calls rng to advance it
        for i in range(10):
            rng()
        dump = pickle.dumps(rng)
        new_rng = pickle.loads(dump)

        self.assertEqual(rng(), new_rng())
