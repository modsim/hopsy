import typing
import unittest

import numpy
import numpy as np

from hopsy import *

n_chains = 4
n_samples = 100
thinning = 10

seed = 0

uniform = Problem([[1, 1]], [1], starting_point=[0, 0])
problem = Problem([[1, 1]], [1], Gaussian(), starting_point=[0, 0])

ProposalTypes = [
    AdaptiveMetropolisProposal,
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


class MiscTests(unittest.TestCase):
    def test_is_polytope_empty_returns_false_on_non_empty(self):
        # test problem is x < 10, -x < 10
        A = np.array([[1], [-1]])
        b = np.array([10, 10])
        is_empty = is_polytope_empty(A, b, S=None, h=None)
        self.assertEqual(is_empty, False)

    def test_is_polytope_empty_throws_exceptions_on_bad_arguments(self):
        A = np.array([[1], [-1]])
        b = np.array([10, 10])
        with self.assertRaises(RuntimeError):
            is_polytope_empty(A, b, None, 1)
        with self.assertRaises(RuntimeError):
            is_polytope_empty(A, b, 1, None)

    def test_is_polytope_empty_returns_true_on_empty(self):
        # test problem is x < 1, x > 2
        A = np.array([[1], [-1]])
        b = np.array([1, -2])
        is_empty = is_polytope_empty(A, b)
        self.assertEqual(is_empty, True)

    def test_is_polytope_empty_returns_false_on_non_empty_with_equality_constraints(
        self,
    ):
        # test problem is x < 10, -x < 10, x=0
        A = np.array([[1], [-1]])
        b = np.array([10, 10])
        S = np.array([[1]])
        h = np.array([5])
        is_empty = is_polytope_empty(A, b, S=S, h=h)
        self.assertEqual(is_empty, False)

    def test_is_problem_empty_returns_false_on_non_empty(self):
        # test problem is x < 10, -x < 10
        A = np.array([[1], [-1]])
        b = np.array([10, 10])
        problem = Problem(A, b)
        is_empty = is_problem_polytope_empty(problem)
        self.assertEqual(is_empty, False)

    def test_is_problem_empty_returns_true_on_empty(self):
        # test problem is x < 1, x > 2
        A = np.array([[1], [-1]])
        b = np.array([1, -2])
        problem = Problem(A, b)
        is_empty = is_problem_polytope_empty(problem)
        self.assertEqual(is_empty, True)

    def test_default_starting_point_for_markov_chain(self):
        A = np.array([[1], [-1]])
        b = np.array([2, 1])
        p = Problem(A, b)
        mc = MarkovChain(p)
        assert np.all(b - A @ mc.state > 0)

    def test_sequential_sampling(self):
        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]

        accrates, states = sample(chains, rngs, n_samples, thinning)

        self.assertListEqual(list(states.shape), [n_chains, n_samples, 2])

        chains = [MarkovChain(uniform, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]

        accrates, states = sample(chains, rngs, n_samples, thinning)

        self.assertListEqual(list(states.shape), [n_chains, n_samples, 2])

    def test_parallel_sampling(self):
        n_procs = 10
        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]

        accrates, states = sample(chains, rngs, n_samples, thinning, n_procs)

        self.assertListEqual(list(states.shape), [n_chains, n_samples, 2])

        chains = [MarkovChain(uniform, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]

        accrates, states = sample(chains, rngs, n_samples, thinning, n_procs)

        self.assertListEqual(list(states.shape), [n_chains, n_samples, 2])

    def test_sampling_semigroup_property(self):
        # Test property for sequential chains
        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]
        _, states0 = sample(chains, rngs, int(n_samples // 2), thinning)
        _, states1 = sample(chains, rngs, int(n_samples // 2), thinning)

        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]
        _, states2 = sample(chains, rngs, n_samples, thinning)

        self.assertTrue(numpy.linalg.norm(states2[:, -1] - states1[:, -1]) < 1e-7)

        # Test property for parallel chains
        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]
        _, states0 = sample(chains, rngs, int(n_samples // 2), thinning, -1)
        _, states1 = sample(chains, rngs, int(n_samples // 2), thinning, -1)

        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]
        _, states2 = sample(chains, rngs, n_samples, thinning, -1)

        self.assertTrue(numpy.linalg.norm(states2[:, -1] - states1[:, -1]) < 1e-7)

    def test_add_box_constraints(self):
        uniform_problem = Problem(
            [
                [
                    1,
                    1,
                ]
            ],
            [1],
        )
        uniform_problem = add_box_constraints(uniform_problem, -2, 1)

        for ProposalType in ProposalTypes:
            if ProposalType in [
                BilliardMALAProposal,
                CSmMALAProposal,
                TruncatedGaussianProposal,
            ]:
                continue
            chain = MarkovChain(
                uniform_problem, ProposalType, starting_point=[0.1, 0.1]
            )

        gaussian_problem = Problem(
            [
                [
                    1,
                    1,
                ]
            ],
            [1],
            Gaussian(),
        )
        gaussian_problem = add_box_constraints(gaussian_problem, -1, 1)

        for ProposalType in ProposalTypes:
            chain = MarkovChain(
                gaussian_problem, ProposalType, starting_point=[0.1, 0.1]
            )

    def test_rounding(self):
        problem = Problem(
            [
                [
                    1,
                    1,
                ]
            ],
            [1],
        )
        problem = add_box_constraints(problem, -2, 1)
        problem = round(problem)

        problem = Problem(
            [
                [
                    1,
                    1,
                ]
            ],
            [1],
            starting_point=[0, 0],
        )
        problem = add_box_constraints(problem, -2, 1)
        problem = round(problem)

        self.assertTrue((problem.b - problem.A @ problem.starting_point >= 0).all())

    def test_ess(self):
        states = [[[0, 1, 2, 3, 4]] * 100] * 4
        states = [[[0, 1, 2, 3, 4]] * 100] * 4
        neff = ess(states)
        self.assertListEqual([1, 1, 1, 1, 1], neff[0].tolist())

        states = numpy.concatenate(
            [[[[0, 1, 2, 3, 4]] * 100] * 4, numpy.random.rand(4, 100, 5)], axis=1
        )
        neff = ess(states, series=100)
        self.assertListEqual([1, 1, 1, 1, 1], neff[0].tolist())

        states = numpy.concatenate(
            [[[[0, 1, 2, 3, 4]] * 100] * 4, numpy.random.rand(4, 100, 5)], axis=1
        )
        neff = ess(states, series=100, n_procs=-1)
        self.assertListEqual([1, 1, 1, 1, 1], neff[0].tolist())

        rel_ess = 1 / 400

        states = [[[0, 1, 2, 3, 4]] * 100] * 4
        states = [[[0, 1, 2, 3, 4]] * 100] * 4
        neff = ess(states, relative=True)
        self.assertListEqual([rel_ess] * 5, neff[0].tolist())
        self.assertListEqual([rel_ess] * 5, neff[0].tolist())

        states = numpy.concatenate(
            [[[[0, 1, 2, 3, 4]] * 100] * 4, numpy.random.rand(4, 100, 5)], axis=1
        )
        neff = ess(states, series=100, relative=True)
        self.assertListEqual([rel_ess] * 5, neff[0].tolist())
        self.assertListEqual([rel_ess] * 5, neff[0].tolist())

        states = numpy.concatenate(
            [[[[0, 1, 2, 3, 4]] * 100] * 4, numpy.random.rand(4, 100, 5)], axis=1
        )
        neff = ess(states, series=100, relative=True, n_procs=-1)
        self.assertListEqual([rel_ess] * 5, neff[0].tolist())
        self.assertListEqual([rel_ess] * 5, neff[0].tolist())

    def test_recording_meta_data(self):
        problem = Problem(
            [[1, 0], [0, 1], [-1, 0], [0, -1]], [5, 5, 0, 0], Gaussian(dim=2)
        )
        mcs = [
            MarkovChain(
                problem, proposal=GaussianHitAndRunProposal, starting_point=[0.5, 0.5]
            )
            for i in range(2)
        ]
        rngs = [RandomNumberGenerator(42, i) for i in range(2)]

        record_meta = ["state_negative_log_likelihood", "proposal.proposal"]
        meta, states = sample(mcs, rngs, n_samples=100, record_meta=record_meta)
        self.assertTrue(len(meta) == len(record_meta))
        self.assertTrue(meta["proposal.proposal"].shape == (2, 100, 2))

        record_meta = ["state_negative_log_likelihood", "proposal.proposal"]
        meta, states = sample(
            mcs, rngs, n_samples=100, n_procs=2, record_meta=record_meta
        )
        self.assertTrue(len(meta) == len(record_meta))
        self.assertTrue(meta["proposal.proposal"].shape == (2, 100, 2))

        record_meta = ["acceptance_rate"]
        meta, states = sample(mcs, rngs, n_samples=100, record_meta=record_meta)
        self.assertTrue(len(meta) == len(record_meta))

        record_meta = ["foo"]  # obviously not an attribute
        meta, states = sample(mcs, rngs, n_samples=100, record_meta=record_meta)
        self.assertTrue(len(meta) == len(record_meta))
        self.assertTrue(meta["foo"] is None)

        record_meta = False
        meta, states = sample(mcs, rngs, n_samples=100, record_meta=record_meta)
        self.assertTrue(len(meta) == 2)  # just usual acceptance rates
        self.assertTrue(type(meta) is list)  # just usual acceptance rates

    def test_callback(self):
        class TestCallback(Callback):
            def __init__(self, name: str = None):
                super(TestCallback, self).__init__()
                self.states = None
                self.state_idx = []
                self.meta = None

            def setup(
                self,
                n_chains: int,
                n_samples: int,
                n_dims: int,
                meta_names: typing.List[str],
                meta_shapes: typing.List[typing.List[int]],
            ) -> None:
                super(TestCallback, self).setup(
                    n_chains, n_samples, n_dims, meta_names, meta_shapes
                )
                self.states = np.zeros((self.n_chains, self.n_samples, self.n_dims))
                self.state_idx = [0 for i in range(n_chains)]
                self.meta = [None for i in range(n_chains)]

            def record(
                self,
                chain_idx: int,
                state: numpy.ndarray,
                meta: typing.Dict[str, typing.Union[float, numpy.ndarray]],
            ) -> None:
                self.states[chain_idx, self.state_idx[chain_idx]] = state
                self.state_idx[chain_idx] += 1
                if self.meta[chain_idx] is None:
                    self.meta[chain_idx] = {name: [meta[name]] for name in meta.keys()}
                else:
                    for name in meta.keys():
                        self.meta[chain_idx][name] += [meta[name]]

            def finish(self) -> None:
                for i in range(self.n_chain):
                    for field in self.meta:
                        self.meta[i][field] = numpy.array(self.meta[i][field])

        problem = Problem(
            [[1, 0], [0, 1], [-1, 0], [0, -1]], [5, 5, 0, 0], Gaussian(dim=2)
        )
        mcs = [
            MarkovChain(
                problem, proposal=GaussianHitAndRunProposal, starting_point=[0.5, 0.5]
            )
            for i in range(2)
        ]
        rngs = [RandomNumberGenerator(42, i) for i in range(2)]
        callback = TestCallback()
        record_meta = ["state_negative_log_likelihood", "proposal.proposal"]
        meta, states = sample(
            mcs, rngs, n_samples=100, record_meta=record_meta, callback=callback
        )
        self.assertTrue(np.all(states == callback.states))
        self.assertTrue(meta.keys() == callback.meta[0].keys())
        self.assertTrue(callback.meta_shapes == [[], [2]])
        self.assertTrue(
            np.all(
                [
                    np.all([meta[name][i] == callback.meta[i][name] for i in range(2)])
                    for name in meta.keys()
                ]
            )
        )

        callback = TestCallback()
        record_meta = False
        meta, states = sample(
            mcs,
            rngs,
            n_samples=100,
            n_procs=3,
            record_meta=record_meta,
            callback=callback,
        )
        self.assertTrue(np.all(states == callback.states))
        self.assertTrue(callback.meta_shapes == [[]])
        self.assertTrue(
            np.all(
                meta
                == np.mean(
                    [callback.meta[i]["acceptance_rate"] for i in range(2)], axis=1
                )
            )
        )

    def test_progress_bar(self):
        n_procs = 4

        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]

        accrates, states = sample(
            chains, rngs, n_samples, thinning, n_procs, progress_bar=True
        )

        self.assertListEqual(list(states.shape), [n_chains, n_samples, 2])

        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]

        accrates, states = sample(chains, rngs, n_samples, thinning, progress_bar=True)

        self.assertListEqual(list(states.shape), [n_chains, n_samples, 2])

    def test_not_in_memory(self):
        n_procs = 4

        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]

        ret = sample(chains, rngs, n_samples, thinning, n_procs, in_memory=False)

        self.assertEqual(ret, None)

        chains = [MarkovChain(problem, GaussianProposal) for i in range(n_chains)]
        rngs = [RandomNumberGenerator(seed, i) for i in range(n_chains)]

        ret = sample(chains, rngs, n_samples, thinning, in_memory=False)

        self.assertEqual(ret, None)

    def test_multiphase_sampling(self):
        polytope = BirkhoffPolytope(3)
        problem = Problem(polytope.A, polytope.b)
        cheby = compute_chebyshev_center(polytope)
        seeds = [42, 43, 44, 45]
        samples, iterations, ess, runtime = run_multiphase_sampling(
            problem=problem,
            seeds=seeds,
            steps_per_phase=100,
            starting_points=[cheby for s in seeds],
            n_procs=len(seeds),
        )

        self.assertGreater(ess, 1000)

    def test_multiphase_sampling_starting_in_rounded_space(self):
        polytope = BirkhoffPolytope(3)
        problem = round(Problem(polytope.A, polytope.b))
        cheby = compute_chebyshev_center(polytope)
        seeds = [42, 43, 44, 45]
        samples, iterations, ess, runtime = run_multiphase_sampling(
            problem=problem,
            seeds=seeds,
            steps_per_phase=100,
            starting_points=[cheby for s in seeds],
            n_procs=len(seeds),
        )

        self.assertGreater(ess, 1000)

    def test_compute_chebyshev_center_in_original_space(self):
        A = np.array([]).reshape(0, 3)
        b = np.array([])

        lb = [0.01, 1e-10, 1e-10]
        ub = [3, 0.1, 1]
        problem = Problem(A, b)
        problem = add_box_constraints(
            problem, lower_bound=lb, upper_bound=ub, simplify=True
        )
        chebyshev = compute_chebyshev_center(problem)
        self.assertGreater(np.min(problem.slacks(chebyshev)), 0)
        problem_rounded = round(problem)
        chebyshev_rounded = compute_chebyshev_center(
            problem_rounded, original_space=True
        )
        self.assertGreater(np.min(problem.slacks(chebyshev_rounded)), 0)

    def test_compute_chebyshev_center_in_original_space_with_equality_constraints_when_rounded(
        self,
    ):
        A = np.array([]).reshape((0, 3))
        b = np.array([])

        problem = Problem(A, b)

        lb = [-5, -5, -5]
        ub = [5, 5, 5]

        problem = add_box_constraints(problem, lb, ub, simplify=False)

        A_eq = np.array([[2.0, 1.0, 0]])
        b_eq = np.array([8])

        problem = add_equality_constraints(problem, A_eq, b_eq)

        chebychev = compute_chebyshev_center(problem, original_space=True)[:, 0]
        problem_rounded = round(problem)

        chebychev_rounded = compute_chebyshev_center(
            problem_rounded, original_space=True
        )[:, 0]

        self.assertEqual(chebychev.shape, chebychev_rounded.shape)

    def test_get_samples_with_temperature(self):
        temperature_ladder = [1.0, 0.3, 0.0]

        samples = np.array(
            [
                [
                    [1.0, 1.0, 1.0],
                    [1.1, 1.1, 1.1],
                ],
                [
                    [0.3, 0.3, 0.3],
                    [1.3, 1.3, 1.3],
                ],
                [
                    [0, 0, 0],
                    [0.1, 0.1, 0.1],
                ],
                [
                    [1.0, 1.0, 1.0],
                    [1.1, 1.1, 1.1],
                ],
                [
                    [0.3, 0.3, 0.3],
                    [1.3, 1.3, 1.3],
                ],
                [
                    [0, 0, 0],
                    [0.1, 0.1, 0.1],
                ],
                [
                    [1.0, 1.0, 1.0],
                    [1.1, 1.1, 1.1],
                ],
                [
                    [0.3, 0.3, 0.3],
                    [1.3, 1.3, 1.3],
                ],
                [
                    [0, 0, 0],
                    [0.1, 0.1, 0.1],
                ],
                [
                    [1.0, 1.0, 1.0],
                    [1.1, 1.1, 1.1],
                ],
                [
                    [0.3, 0.3, 0.3],
                    [1.3, 1.3, 1.3],
                ],
                [
                    [0, 0, 0],
                    [0.1, 0.1, 0.1],
                ],
            ]
        )

        expected_samples = {}
        expected_samples[0] = np.array(
            [[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]] for i in range(4)]
        )
        expected_samples[0.3] = np.array(
            [[[0.3, 0.3, 0.3], [1.3, 1.3, 1.3]] for i in range(4)]
        )
        expected_samples[1] = np.array(
            [[[1.0, 1.0, 1.0], [1.1, 1.1, 1.1]] for i in range(4)]
        )

        for t in temperature_ladder:
            samples_t = get_samples_with_temperature(t, temperature_ladder, samples)
            assert samples_t.shape == (4, 2, 3)
            assert np.all(samples_t == expected_samples[t])

    def test_starting_point_with_equality_constraints(self):
        A = np.array([]).reshape((0, 3))
        b = np.array([])

        starting_point = np.array([3, 2, 1])

        problem = Problem(A, b, starting_point=starting_point)

        lb = [-5, -5, -5]
        ub = [5, 5, 5]

        problem = add_box_constraints(problem, lb, ub, simplify=False)

        A_eq = np.array([[2.0, 1.0, 0]])
        b_eq = np.array([8])

        problem = add_equality_constraints(problem, A_eq, b_eq)

        self.assertIsNotNone(problem.starting_point)
        self.assertTrue(np.all((problem.b - problem.A @ problem.starting_point) > 0))

        original_starting_point = back_transform(problem, [problem.starting_point])[0]
        self.assertTrue(np.all(original_starting_point == starting_point))

    def test_starting_point_with_equality_constraints_when_rounded(self):
        A = np.array([]).reshape((0, 3))
        b = np.array([])

        starting_point = np.array([3, 2, 1])

        problem = Problem(A, b, starting_point=starting_point)

        lb = [-5, -5, -5]
        ub = [5, 5, 5]

        problem = add_box_constraints(problem, lb, ub, simplify=False)

        A_eq = np.array([[2.0, 1.0, 0]])
        b_eq = np.array([8])

        problem = add_equality_constraints(problem, A_eq, b_eq)
        problem = round(problem)

        self.assertIsNotNone(problem.starting_point)
        self.assertTrue(np.all((problem.b - problem.A @ problem.starting_point) > 0))

        original_starting_point = back_transform(problem, [problem.starting_point])[0]
        self.assertTrue(np.all(original_starting_point == starting_point))

    def test_starting_point_with_equality_constraints_simplify(self):
        A = np.array([]).reshape((0, 3))
        b = np.array([])

        starting_point = np.array([3, 2, 1])

        problem = Problem(A, b, starting_point=starting_point)

        lb = [-5, -5, -5]
        ub = [5, 5, 5]

        problem = add_box_constraints(problem, lb, ub, simplify=True)

        A_eq = np.array([[2.0, 1.0, 0]])
        b_eq = np.array([8])

        problem = add_equality_constraints(problem, A_eq, b_eq)

        self.assertIsNotNone(problem.starting_point)
        self.assertTrue(np.all((problem.b - problem.A @ problem.starting_point) > 0))

        original_starting_point = back_transform(problem, [problem.starting_point])[0]
        self.assertTrue(np.all(original_starting_point == starting_point))

    def test_starting_point_with_equality_constraints_when_rounded_simplify(self):
        A = np.array([]).reshape((0, 3))
        b = np.array([])

        starting_point = np.array([3, 2, 1])

        problem = Problem(A, b, starting_point=starting_point)

        lb = [-5, -5, -5]
        ub = [5, 5, 5]

        problem = add_box_constraints(problem, lb, ub, simplify=True)

        A_eq = np.array([[2.0, 1.0, 0]])
        b_eq = np.array([8])

        problem = add_equality_constraints(problem, A_eq, b_eq)
        problem = round(problem)

        self.assertIsNotNone(problem.starting_point)
        self.assertTrue(np.all((problem.b - problem.A @ problem.starting_point) > 0))

        original_starting_point = back_transform(problem, [problem.starting_point])[0]
        self.assertTrue(np.all(original_starting_point == starting_point))

        def test_equality_constraints_throw_on_existing_transformation(self):
            A = np.array([]).reshape((0, 3))
            b = np.array([])

            starting_point = np.array([3, 2, 1])

            problem = Problem(A, b, starting_point=starting_point, transform=A, shift=b)

            lb = [-5, -5, -5]
            ub = [5, 5, 5]

            with self.assertRaises(RuntimeError):
                add_box_constraints(problem, lb, ub, simplify=False)
