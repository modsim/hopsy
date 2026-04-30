import unittest

from hopsy.lp import LP


class LPTests(unittest.TestCase):
    def test_singleton_stores_changes_correctly(self):
        lp = LP()
        lp.settings.thresh = 1e-8
        self.assertEqual(lp.settings.thresh, LP().settings.thresh)
