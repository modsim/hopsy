import pickle
import unittest

from hopsy import *

ProposalTypes = [
    AdaptiveMetropolisProposal,
    BallWalkProposal,
    BilliardMALAProposal,
    BilliardWalkProposal,
    CSmMALAProposal,
    DikinWalkProposal,
    GaussianCoordinateHitAndRunProposal,
    GaussianHitAndRunProposal,
    GaussianProposal,
    TruncatedGaussianProposal,
    UniformCoordinateHitAndRunProposal,
    UniformHitAndRunProposal,
]


class MarkovChainTests(unittest.TestCase):
    def test_automatic_downcasting(self):
        problem = Problem(
            [[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, 0]
        )

        for ProposalType in ProposalTypes:
            proposal = ProposalType(problem)
            mc = MarkovChain(problem, proposal)
            self.assertIsInstance(mc.proposal, ProposalType)

    def test_state_log_density(self):
        model = Gaussian()
        start = [0, 0]
        problem = Problem(
            [[1, 1], [-1, 0], [0, -1]], [1, 0, 0], model, starting_point=start
        )

        for ProposalType in ProposalTypes:
            proposal = ProposalType(problem)
            mc = MarkovChain(problem, proposal)
            self.assertAlmostEqual(
                mc.state_log_density, -mc.state_negative_log_likelihood
            )
            self.assertAlmostEqual(mc.state_log_density, model.log_density(start))
            self.assertAlmostEqual(
                mc.state_negative_log_likelihood, -model.log_density(start)
            )

    def test_empty_proposal_initialization(self):
        problem = Problem(
            [[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, 0]
        )

        for ProposalType in ProposalTypes:
            mc = MarkovChain(problem, ProposalType)
            self.assertIsInstance(mc.proposal, ProposalType)

    def test_markovchain_pickling(self):
        problem = Problem(
            [[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, 0]
        )

        for ProposalType in ProposalTypes:
            proposal = ProposalType(problem)
            mc = MarkovChain(problem, proposal)
            dump = pickle.dumps(mc)
            new_mc = pickle.loads(dump)
