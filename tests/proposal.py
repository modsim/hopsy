import pickle
import unittest

import numpy as np

from hopsy import *

ProposalTypes = [
    AdaptiveMetropolisProposal,
    BilliardAdaptiveMetropolisProposal,
    BallWalkProposal,
    BilliardMALAProposal,
    CSmMALAProposal,
    DikinWalkProposal,
    GaussianCoordinateHitAndRunProposal,
    GaussianHitAndRunProposal,
    GaussianProposal,
    TruncatedGaussianProposal,
    UniformCoordinateHitAndRunProposal,
    UniformHitAndRunProposal,
]


class ProposalTests(unittest.TestCase):
    def test_dimension_mismatch(self):
        pass

    def test_unbounded_uniform_problem(self):
        problem = Problem([[1, 1]], [1], starting_point=[0, 0])

        for ProposalType in ProposalTypes:

            if ProposalType in [
                BilliardMALAProposal,
                CSmMALAProposal,
                DikinWalkProposal,
                TruncatedGaussianProposal,
            ]:
                with self.assertRaises(RuntimeError):
                    proposal = ProposalType(problem)
                    self.assertTrue((proposal.state == problem.starting_point).all())

            else:
                proposal = ProposalType(problem)
                self.assertTrue((proposal.state == problem.starting_point).all())

    def test_bounded_uniform_problem(self):
        problem = Problem([[1, 1], [-1, 0], [0, -1]], [1, 0, 0], starting_point=[0, 0])

        for ProposalType in ProposalTypes:

            if ProposalType in [
                CSmMALAProposal,
                BilliardMALAProposal,
                TruncatedGaussianProposal,
            ]:
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
        problem = Problem(
            [[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, 0]
        )

        for ProposalType in ProposalTypes:
            proposal = ProposalType(problem)
            self.assertTrue((proposal.state == problem.starting_point).all())

            if hasattr(proposal, "boundary_cushion"):
                proposal = ProposalType(problem, boundary_cushion=1.0e3)
                self.assertEqual(proposal.boundary_cushion, 1.0e3)

            if hasattr(proposal, "epsilon"):
                proposal = ProposalType(problem, epsilon=1.0e3)
                self.assertEqual(proposal.epsilon, 1.0e3)

            if hasattr(proposal, "fisher_weight"):
                proposal = ProposalType(problem, fisher_weight=1.0e-2)
                self.assertEqual(proposal.fisher_weight, 1.0e-2)

            if hasattr(proposal, "stepsize"):
                proposal = ProposalType(problem, stepsize=1.0e3)
                self.assertEqual(proposal.stepsize, 1.0e3)

            if hasattr(proposal, "warm_up"):
                proposal = ProposalType(problem, warm_up=int(1.0e3))
                self.assertEqual(proposal.warm_up, 1.0e3)

    def test_starting_points(self):
        problem = Problem(
            [[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, 0]
        )
        problem_no_start = Problem([[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian())
        x = [0.1, 0.2]

        for ProposalType in ProposalTypes:
            proposal = ProposalType(problem, starting_point=x)
            self.assertTrue((proposal.state == x).all())

            with self.assertRaises(RuntimeError):
                proposal = ProposalType(problem_no_start)

    def test_automatic_downcasting(self):
        problem = Problem(
            [[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, 0]
        )

        for ProposalType in ProposalTypes:
            proposal = ProposalType(problem)
            new_proposal = proposal.deepcopy()
            self.assertIsInstance(new_proposal, ProposalType)

    def test_proposal_pickling(self):
        problem = Problem(
            [[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, 0]
        )

        for ProposalType in ProposalTypes:
            proposal = ProposalType(problem)
            dump = pickle.dumps(proposal)
            new_proposal = pickle.loads(dump)
            self.assertIsInstance(new_proposal, ProposalType)

    def test_starting_point_validation(self):
        problem = Problem(
            [[1, 1], [-1, 0], [0, -1]], [1, 0, 0], Gaussian(), starting_point=[0, -0.1]
        )

        for ProposalType in ProposalTypes:
            with self.assertRaises(ValueError):
                proposal = ProposalType(problem)

            with self.assertRaises(ValueError):
                proposal = ProposalType(problem, starting_point=[0.1, 0.1])
                proposal.state = [0, -0.1]

    def test_gaussian_sampling(self):
        model = Gaussian(1)
        problem = Problem(np.array([1, -1]), 5 * np.ones(2), model)

        for ProposalType in ProposalTypes:
            if ProposalType in [
                UniformCoordinateHitAndRunProposal,
                UniformHitAndRunProposal,
            ]:
                with self.assertRaises(RuntimeError):
                    proposal = ProposalType(problem)
                    self.assertTrue((proposal.state == problem.starting_point).all())

            else:
                mc = MarkovChain(problem, ProposalType, [0.0])
                rng = RandomNumberGenerator(seed=42)

                num_samples = 25000
                _, samples = sample(mc, rng, num_samples)

                # checks sample mean is close to real mean of 0
                true_std = 1.0
                standard_error_of_mean = true_std / np.sqrt(ess(samples))
                #  checks that mean is within 2 standard errors
                self.assertTrue(
                    np.abs(np.mean(samples) - model.mean) < 2 * standard_error_of_mean
                )
                # checks sample std is close to real std of 1
                self.assertTrue(np.abs(np.std(samples, ddof=1) - true_std) < 1e-1)

    def test_gaussian_sampling_shifted(self):
        model = Gaussian(np.ones(1) * 5, 0.25 * np.identity(1))
        problem = Problem(np.array([1, -1]), 100 * np.ones(2), model)

        for ProposalType in ProposalTypes:
            if ProposalType in [
                UniformCoordinateHitAndRunProposal,
                UniformHitAndRunProposal,
            ]:
                with self.assertRaises(RuntimeError):
                    proposal = ProposalType(problem)
                    self.assertTrue((proposal.state == problem.starting_point).all())

            else:
                mc = MarkovChain(problem, ProposalType, [0.0])
                rng = RandomNumberGenerator(seed=42)

                num_samples = (
                    150_000 if ProposalType == AdaptiveMetropolisProposal else 25_000
                )

                _, samples = sample(mc, rng, num_samples)

                # checks sample mean is close to real mean of 0
                true_std = 0.5
                standard_error_of_mean = true_std / np.sqrt(ess(samples))
                #  checks that mean is within 2 standard errors
                self.assertTrue(
                    np.abs(np.mean(samples) - model.mean) < 2 * standard_error_of_mean
                )
                # checks sample std is close to real std of 1

    def test_truncated_proposals_simplex_sharp_gauss(self):
        # works for epsilon>=1e-2 but not for 1e-3
        epsilon = 1e-3
        A = np.array(
            [
                [-1.0, 0.0],
                [0.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, -1.0],
                [1.0, -1.0],
                [1.0, -1.0],
                [1.0, -1.0],
            ]
        )
        b = np.array([[0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, -0.0]]).T
        mean = np.array([[0.6, 1.0]]).T
        cov = epsilon * np.array([[1, 0.0], [0.0, 1]])

        model = Gaussian(mean, cov)
        problem = Problem(A, b, model)
        start = np.array([[1.46446609], [3.53553391]])
        mc = MarkovChain(problem, TruncatedGaussianProposal, start)
        rng = RandomNumberGenerator(seed=42)

        _, samples = sample(mc, rng, 1000)

        self.assertFalse(np.isnan(samples).any())
