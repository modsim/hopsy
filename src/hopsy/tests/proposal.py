from .. import *

import unittest

ProposalTypes = [
            AdaptiveMetropolisProposal,
            BallWalkProposal,
            CSmMALAProposal,
            DikinWalkProposal,
            GaussianCoordinateHitAndRunProposal,
            GaussianHitAndRunProposal,
            GaussianProposal,
            UniformCoordinateHitAndRunProposal,
            UniformHitAndRunProposal,
        ]

class ProposalTests(unittest.TestCase):
    def test_dimension_mismatch(self):
        pass

    def test_unbounded_uniform_problem(self):
        #problem = Problem([[1, 1], [0, -1]], [1, 0], starting_point=[0, 0]) ## DikinWalk can handle this problem?
        problem = Problem([[1, 1]], [1], starting_point=[0, 0])

        for ProposalType in ProposalTypes:

            if ProposalType == CSmMALAProposal or ProposalType == DikinWalkProposal:
                with self.assertRaises(RuntimeError):
                    proposal = ProposalType(problem)
                    self.assertTrue((proposal.state == problem.starting_point).all())

            else:
                proposal = ProposalType(problem)
                self.assertTrue((proposal.state == problem.starting_point).all())


    def test_bounded_uniform_problem(self):
        problem = Problem([[1, 1], [-1, 0], [0, -1]], [1, 0, 0], starting_point=[0, 0])

        for ProposalType in ProposalTypes:

            if ProposalType == CSmMALAProposal:
                with self.assertRaises(RuntimeError):
                    proposal = ProposalType(problem)
                    self.assertTrue((proposal.state == problem.starting_point).all())
            
            else:
                proposal = ProposalType(problem)
                self.assertTrue((proposal.state == problem.starting_point).all())


    def test_unbounded_nonuniform_problem(self):
        problem = Problem([[1, 1]], [1], Gaussian(), starting_point=[0, 0])

        for ProposalType in ProposalTypes:

            if ProposalType == DikinWalkProposal:
                with self.assertRaises(RuntimeError):
                    proposal = ProposalType(problem)
                    self.assertTrue((proposal.state == problem.starting_point).all())
            
            else:
                proposal = ProposalType(problem)
                self.assertTrue((proposal.state == problem.starting_point).all())


    def test_bounded_nonuniform_problem(self):
        problem = Problem([[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, 0])

        for ProposalType in ProposalTypes:
            proposal = ProposalType(problem)
            self.assertTrue((proposal.state == problem.starting_point).all())

            if hasattr(proposal, 'boundary_cushion'):
                proposal = ProposalType(problem, boundary_cushion=1.e3)
                self.assertEqual(proposal.boundary_cushion, 1.e3)

            if hasattr(proposal, 'epsilon'):
                proposal = ProposalType(problem, epsilon=1.e3)
                self.assertEqual(proposal.epsilon, 1.e3)

            if hasattr(proposal, 'fisher_weight'):
                proposal = ProposalType(problem, fisher_weight=1.e-2)
                self.assertEqual(proposal.fisher_weight, 1.e-2)

            if hasattr(proposal, 'stepsize'):
                proposal = ProposalType(problem, stepsize=1.e3)
                self.assertEqual(proposal.stepsize, 1.e3)

            if hasattr(proposal, 'warm_up'):
                proposal = ProposalType(problem, warm_up=int(1.e3))
                self.assertEqual(proposal.warm_up, 1.e3)


    def test_starting_points(self):
        problem = Problem([[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, 0])
        problem_no_start = Problem([[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian())
        x = [0.1, 0.2]

        for ProposalType in ProposalTypes:
            proposal = ProposalType(problem, starting_point=x)
            self.assertTrue((proposal.state == x).all())

            with self.assertRaises(RuntimeError):
                proposal = ProposalType(problem_no_start)


    @unittest.expectedFailure
    def test_automatic_downcasting(self):
        uniform_problem = Problem([[1, 1]], [1], starting_point=[0, 0])

        for ProposalType in ProposalTypes:
            proposal = ProposalType(Problem(A, b, starting_point=x))
            mc = MarkovChain(proposal, uniform_problem)
            self.assertIsInstance(mc.proposal, GaussianProposal)

