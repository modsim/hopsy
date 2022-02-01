import unittest

_round = round

from .. import *

items = [
    AcceptanceRateTarget,
    AdaptiveMetropolisProposal,
    BallWalkProposal,
    CSmMALAProposal,
    DikinWalkProposal,
    ExpectedSquaredJumpDistanceTarget,
    Gaussian,
    GaussianCoordinateHitAndRunProposal,
    GaussianHitAndRunProposal,
    GaussianProposal,
    MarkovChain,
    Mixture,
    Model,
    Normal,
    Problem,
    Proposal,
    ProposalParameter,
    PyModel,
    PyProposal,
    PyTuningTarget,
    RandomNumberGenerator,
    Rosenbrock,
    ThompsonSamplingTuning,
    TuningTarget,
    Uniform,
    UniformCoordinateHitAndRunProposal,
    UniformHitAndRunProposal,
    add_box_constraints,
    back_transform,
    compute_chebyshev_center,
    round,
    simplify,
    transform,
    tune,
    sample
]

class DocTests(unittest.TestCase):

    @unittest.expectedFailure
    def test_has_docstring(self):
        sufficient_docstrings = True
        report = "Checking docstrings...\n"
        n_all = len(items)
        n_sufficient = 0

        for item in items:
            docstring = item.__doc__

            if docstring is None or len(docstring) < 20:
                report += "  -- Found insufficient docstring for hopsy." + item.__name__ + "\n"
                sufficient_docstrings = False
            else:
                n_sufficient += 1

            attributes = [item for item in dir(item) if not item.startswith('_')]
            n_all += len(attributes)

            for attribute in attributes:
                docstring = eval(item.__name__ + "." + attribute).__doc__
                if docstring is None or len(docstring) < 20:
                    report += "  -- Found insufficient docstring for hopsy." + item.__name__ + "." + attribute + "\n"
                    sufficient_docstrings = False
                else:
                    n_sufficient += 1

        docstr_coverage = _round(1. * n_sufficient / n_all, 2) * 100
        report += "Found" + str(n_all) + "items in hopsy of which" + str(n_sufficient) + "have sufficient docstrings." + "\n"
        report += "Docstring coverage: " + str(docstr_coverage) + "%\n"

        try:
            with open("docs/.docstrcovreport.txt", "w") as f:
                f.write(report)
        except Exception as e:
            print("Could not write docstring coverage report!\n", e)

        self.assertTrue(sufficient_docstrings)
