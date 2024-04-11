import pickle
import unittest

import numpy as np
from scipy.special import gamma

from hopsy import *

ProposalTypes = [
    AdaptiveMetropolisProposal,
    BilliardAdaptiveMetropolisProposal,
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


class GammaPDF:
    """Test class for Reversible Jump proposal"""

    def __init__(self, data):
        self.data = data
        self.A = np.array(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ]
        )
        self.b = np.array([0.9, 0, 10, -0.1, 10, -0.1])

    def log_density(self, x):
        location = x[3]
        scale = x[4]
        shape = x[5]
        if scale <= 0 or shape <= 0:
            raise ValueError(f"invalid parameters! {scale}, {shape}")
        log_density = 0
        for datum in self.data:
            if datum - location < 0:
                continue
            density = (
                              (datum - location) ** (shape - 1) * np.exp(-(datum - location) / scale)
                      ) / (gamma(shape) * scale ** shape)
            log_density += np.log(density)

        return log_density


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
                pass

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

    def test_log_density_of_proposals(self):
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
        cov = np.array([[1, 0.0], [0.0, 1]])

        model = Gaussian(mean, cov)
        problem = Problem(A, b, model)
        start = np.array([[1.46446609], [3.53553391]])

        proposal_without_log_density = DikinWalkProposal(problem, starting_point=start)
        self.assertFalse(proposal_without_log_density.has_log_density)
        self.assertEqual(proposal_without_log_density.state_log_density, 0)
        self.assertEqual(proposal_without_log_density.proposal_log_density, 0)
        self.assertFalse(proposal_without_log_density.has_negative_log_likelihood)
        self.assertEqual(proposal_without_log_density.state_negative_log_likelihood, 0)
        self.assertEqual(
            proposal_without_log_density.proposal_negative_log_likelihood, 0
        )

        proposal_with_log_density = CSmMALAProposal(problem, starting_point=start)
        self.assertTrue(proposal_with_log_density.has_log_density)
        self.assertEqual(
            proposal_with_log_density.state_log_density, model.log_density(start)
        )
        self.assertEqual(proposal_with_log_density.proposal_log_density, 0)
        self.assertTrue(proposal_with_log_density.has_negative_log_likelihood)
        self.assertEqual(
            proposal_with_log_density.state_negative_log_likelihood,
            -model.log_density(start),
        )
        # for fresh proposals the log likelihood is still unitialized at 0
        self.assertEqual(proposal_with_log_density.proposal_negative_log_likelihood, 0)

    def test_rjmcmc(self):
        measurements = [1.0]
        gammaPDF = GammaPDF(measurements)
        problem = Problem(
            gammaPDF.A, gammaPDF.b, gammaPDF, starting_point=[0.5, 0.5, 0.5]
        )
        proposal = UniformHitAndRunProposal(problem)
        jumpIndices = np.array([0, 1])
        defaultValues = np.array([0, 1])
        rjmcmc_proposal = ReversibleJumpProposal(proposal, jumpIndices, defaultValues)

        mc = MarkovChain(problem=problem, proposal=rjmcmc_proposal)
        rng = RandomNumberGenerator(5)

        acc, samples = sample(mc, rng, n_samples=500_000, thinning=1)

        location_samples = samples[0, :, 3]
        scale_samples = samples[0, :, 4]
        shape_samples = samples[0, :, 5]

        actual_location_mean = np.mean(location_samples)
        actual_scale_mean = np.mean(scale_samples)
        actual_shape_mean = np.mean(shape_samples)

        actual_model_visits = [0, 0, 0, 0]
        for s in samples[0, :, :]:
            model_index = 0
            for j in range(jumpIndices.shape[0]):
                model_index += 2 ** (jumpIndices.shape[0] - 1 - j) * s[jumpIndices[j]]
            actual_model_visits[int(model_index)] += 1
        actual_model_probabilities = [
            float(v) / samples.shape[1] for v in actual_model_visits
        ]

        expected_location_mean = 0.25381601398469622
        expected_scale_mean = 1.5811482758198503
        expected_shape_mean = 1.7883893206783834

        expected_model_probabilities = [
            0.3147978599901968,
            0.15325738646688555,
            0.3485946848672095,
            0.18335006867570805,
        ]

        # In order to save time testing, we sample less and receive less accurate results.
        # The RJMCMC proposal is more rigorously tested in HOPS. Here we test that the python bindings work.
        self.assertAlmostEqual(expected_location_mean, actual_location_mean, places=1)
        self.assertAlmostEqual(expected_scale_mean, actual_scale_mean, places=1)
        self.assertAlmostEqual(expected_shape_mean, actual_shape_mean, places=1)
        for i in range(
                max(len(expected_model_probabilities), len(actual_model_probabilities))
        ):
            self.assertAlmostEqual(
                expected_model_probabilities[i], actual_model_probabilities[i], places=1
            )

    def test_rjmcmc_rounded(self):
        measurements = [1.0]
        gammaPDF = GammaPDF(measurements)
        problem = Problem(gammaPDF.A, gammaPDF.b, gammaPDF, starting_point=[0.5, 0.5, 0.5])

        rounded_problem = round(problem)

        rng = RandomNumberGenerator(42)
        proposal = UniformHitAndRunProposal(rounded_problem)

        jumpIndices = np.array([0, 1])
        # tip: perturb default values with epsilon to ensure they are not on the polytope borders
        defaultValues = np.array([1e-14, 1])
        # RJMCMC requires original polytope A and B to insert correct default values
        rjmcmc_proposal = ReversibleJumpProposal(proposal,
                                                 jumpIndices,
                                                 defaultValues,
                                                 A=rounded_problem.original_A,
                                                 b=rounded_problem.original_b)

        mc = MarkovChain(problem=rounded_problem, proposal=rjmcmc_proposal)

        acc, samples = sample(mc, rng, n_samples=500_000, thinning=1)

        location_samples = samples[0, :, 3]
        scale_samples = samples[0, :, 4]
        shape_samples = samples[0, :, 5]

        actual_location_mean = np.mean(location_samples)
        actual_scale_mean = np.mean(scale_samples)
        actual_shape_mean = np.mean(shape_samples)

        actual_model_visits = [0, 0, 0, 0]
        for s in samples[0, :, :]:
            model_index = 0
            for j in range(jumpIndices.shape[0]):
                model_index += 2 ** (jumpIndices.shape[0] - 1 - j) * s[jumpIndices[j]]
            actual_model_visits[int(model_index)] += 1
        actual_model_probabilities = [
            float(v) / samples.shape[1] for v in actual_model_visits
        ]

        expected_location_mean = 0.25381601398469622
        expected_scale_mean = 1.5811482758198503
        expected_shape_mean = 1.7883893206783834

        expected_model_probabilities = [
            0.3147978599901968,
            0.15325738646688555,
            0.3485946848672095,
            0.18335006867570805,
        ]

        # In order to save time testing, we sample less and receive less accurate results.
        # The RJMCMC proposal is more rigorously tested in HOPS. Here we test that the python bindings work.
        self.assertAlmostEqual(expected_location_mean, actual_location_mean, places=1)
        self.assertAlmostEqual(expected_scale_mean, actual_scale_mean, places=1)
        self.assertAlmostEqual(expected_shape_mean, actual_shape_mean, places=1)
        for i in range(
                max(len(expected_model_probabilities), len(actual_model_probabilities))
        ):
            self.assertAlmostEqual(
                expected_model_probabilities[i], actual_model_probabilities[i], places=1
            )

    def test_pickle_rjmcmc(self):
        measurements = [1.0]
        gammaPDF = GammaPDF(measurements)
        problem = Problem(
            gammaPDF.A, gammaPDF.b, gammaPDF, starting_point=[0.5, 0.5, 0.5]
        )
        internal_proposals = [
            UniformCoordinateHitAndRunProposal(problem),
            GaussianCoordinateHitAndRunProposal(problem),
            UniformHitAndRunProposal(problem),
            GaussianHitAndRunProposal(problem),
        ]

        jumpIndices = np.array([0, 1])
        defaultValues = np.array([0, 1])
        for proposal in internal_proposals:
            rjmcmc_proposal = ReversibleJumpProposal(
                proposal, jumpIndices, defaultValues
            )
            dump = pickle.dumps(rjmcmc_proposal)
            new_proposal = pickle.loads(dump)
            self.assertIsInstance(new_proposal, ReversibleJumpProposal)

    def test_pickle_rjmcmc_when_in_markov_chain(self):
        measurements = [1.0]
        gammaPDF = GammaPDF(measurements)
        problem = Problem(
            gammaPDF.A, gammaPDF.b, gammaPDF, starting_point=[0.5, 0.5, 0.5]
        )
        internal_proposals = [
            UniformCoordinateHitAndRunProposal(problem),
            GaussianCoordinateHitAndRunProposal(problem),
            UniformHitAndRunProposal(problem),
            GaussianHitAndRunProposal(problem),
        ]

        jumpIndices = np.array([0, 1])
        defaultValues = np.array([0, 1])
        for proposal in internal_proposals:
            rjmcmc = MarkovChain(
                proposal=ReversibleJumpProposal(proposal, jumpIndices, defaultValues),
                problem=problem,
            )
            dump = pickle.dumps(rjmcmc)
            new_rjmcmc = pickle.loads(dump)

    def test_rjmcmc_parallel(self):
        measurements = [1.0]
        gammaPDF = GammaPDF(measurements)
        problem = Problem(
            gammaPDF.A, gammaPDF.b, gammaPDF, starting_point=[0.5, 0.5, 0.5]
        )
        proposals = [
            UniformHitAndRunProposal(problem),
            GaussianHitAndRunProposal(problem),
            UniformCoordinateHitAndRunProposal(problem),
            GaussianCoordinateHitAndRunProposal(problem),
        ]
        jumpIndices = np.array([0, 1])
        defaultValues = np.array([0, 1])
        rjmcmc_proposals = [
            ReversibleJumpProposal(proposal, jumpIndices, defaultValues)
            for proposal in proposals
        ]

        mc = [
            MarkovChain(problem=problem, proposal=rjmcmc_proposal)
            for rjmcmc_proposal in rjmcmc_proposals
        ]
        rng = [RandomNumberGenerator(5 * i) for i in range(len(rjmcmc_proposals))]

        num_procs = 4
        acc, samples = sample(mc, rng, n_samples=500_000, thinning=1, n_procs=num_procs)

        samples = np.concatenate(
            tuple([samples[i, :, :] for i in range(len(rjmcmc_proposals))]), axis=0
        )

        location_samples = samples[:, 3]
        scale_samples = samples[:, 4]
        shape_samples = samples[:, 5]

        actual_location_mean = np.mean(location_samples)
        actual_scale_mean = np.mean(scale_samples)
        actual_shape_mean = np.mean(shape_samples)

        actual_model_visits = [0, 0, 0, 0]
        for s in samples[:, :]:
            model_index = 0
            for j in range(jumpIndices.shape[0]):
                model_index += 2 ** (jumpIndices.shape[0] - 1 - j) * s[jumpIndices[j]]
            actual_model_visits[int(model_index)] += 1
        actual_model_probabilities = [
            float(v) / samples.shape[0] for v in actual_model_visits
        ]

        expected_location_mean = 0.25381601398469622
        expected_scale_mean = 1.5811482758198503
        expected_shape_mean = 1.7883893206783834

        expected_model_probabilities = [
            0.3147978599901968,
            0.15325738646688555,
            0.3485946848672095,
            0.18335006867570805,
        ]

        # In order to save time testing, we sample less and receive less accurate results.
        # The RJMCMC proposal is more rigorously tested in HOPS. Here we test that the python bindings work.
        self.assertAlmostEqual(expected_location_mean, actual_location_mean, places=1)
        self.assertAlmostEqual(expected_scale_mean, actual_scale_mean, places=1)
        self.assertAlmostEqual(expected_shape_mean, actual_shape_mean, places=1)
        for i in range(
                max(len(expected_model_probabilities), len(actual_model_probabilities))
        ):
            self.assertAlmostEqual(
                expected_model_probabilities[i], actual_model_probabilities[i], places=2
            )
