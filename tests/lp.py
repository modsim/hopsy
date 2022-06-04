import unittest

from hopsy.lp import LP

class LPTests(unittest.TestCase):
    def test_singleton_stores_changes_correctly(self):
        lp = LP()
        lp.settings.backend = 'glpk'
        self.assertEqual(lp.settings.backend, LP().settings.backend)
