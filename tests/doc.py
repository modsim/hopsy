import unittest

import docov

import hopsy


class ExistingDocstring:
    def __call__(self, item):
        return item.__doc__ is not None


class DocTests(unittest.TestCase):
    def test_docstrings_exist(self):
        good, bad, cond = docov.analyze(
            hopsy,
            condition=ExistingDocstring(),
            ignore=["ProposalParameter", "MCBBackend", "multiprocessing_context"],
        )

        if len(bad) > 0:
            print()
            print("ATTENTION! No docstrings were found for:")
            for name in sorted([name for name, _ in bad]):
                print("  -- " + name)

        self.assertEqual(  # All symbols should have at least an empty docstring!
            len(bad), 0
        )
