import pickle
import unittest

import numpy as np

from hopsy import *


class ProblemTests(unittest.TestCase):
    def test_repr(self):
        problem = Problem([[1, 1]], [1], starting_point=[0, 0])
        expected = "hopsy.Problem(A=array([[1., 1.]]), b=array([1.]), starting_point=array([0., 0.]))"
        self.assertEqual(str(problem), expected)

        problem = Problem([[1, 1]], [1], Gaussian(), starting_point=[0, 0])
        expected = """hopsy.Problem(A=array([[1., 1.]]), b=array([1.]), model=hopsy.Gaussian(mean=array([0., 0.]), covariance=array([[1., 0.],
       [0., 1.]])), starting_point=array([0., 0.]))"""
        self.assertEqual(str(problem), expected)

        problem = Problem(
            [[1, 1]], [1], Gaussian(), starting_point=[0, 0], shift=[1, 2]
        )
        expected = """hopsy.Problem(A=array([[1., 1.]]), b=array([1.]), model=hopsy.Gaussian(mean=array([0., 0.]), covariance=array([[1., 0.],
       [0., 1.]])), starting_point=array([0., 0.]), shift=array([1., 2.]))"""
        self.assertEqual(str(problem), expected)

    def test_model_id(self):
        gaussian = Gaussian()
        id_gaussian = id(gaussian)
        problem = Problem([[1, 1]], [1], gaussian, starting_point=[0, 0])
        self.assertNotEqual(id(gaussian), id(problem.model))

        new_gaussian = problem.model
        self.assertNotEqual(id(gaussian), id(new_gaussian))

        del problem
        self.assertEqual(id_gaussian, id(gaussian))
        self.assertNotEqual(id_gaussian, id(new_gaussian))

    def test_create_box_and_round_glpk(self):
        try:
            lp = LP()
            lp.settings.backend = "glpk"
            problem = add_box_constraints(
                Problem(np.zeros((0, 2)), np.ones((0))),
                [0.5, 1.0e-14],
                [0.95, 1.0e-7],
                simplify=True,
            )
            expected_A = np.array([[-1.0], [1.0]])
            expected_b = np.array([0.225, 0.225])
            self.assertTrue(np.isclose(problem.A, expected_A).all())
            self.assertTrue(np.isclose(problem.b, expected_b).all())
            problem = round(problem)
        except:
            self.fail("Rounding box created from np.zeros unexpectedly raised error.")

    def test_add_box_constraints(self):
        A = np.array([[1, 1, 1]])
        b = np.array([[1000]])

        old_A = np.array([[1, 1, 1]])
        old_b = np.array([[1000.0]])

        expected_A = np.array(
            [
                [1.0, 1.0, 1.0],
                [-1.0, -0.0, -0.0],
                [-0.0, -1.0, -0.0],
                [-0.0, -0.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        expected_b = np.array([1000.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        expected_b2 = np.array([1000.0, 1, 2, 3, 4, 5, 6])
        problem = Problem(A, b)

        new_problem = add_box_constraints(problem, -5, 5, simplify=False)

        assert (new_problem.A == expected_A).all()
        assert (new_problem.A == expected_A).all()
        assert (new_problem.b == expected_b).all()
        assert (problem.A == old_A).all()
        assert (problem.b == old_b).all()

        new_problem = add_box_constraints(
            problem,
            [
                -1,
                -2,
                -3,
            ],
            [4, 5, 6],
            simplify=False,
        )

        assert (new_problem.A == expected_A).all()
        assert (new_problem.b == expected_b2).all()
        assert (problem.A == old_A).all()
        assert (problem.b == old_b).all()

    def test_create_box_and_round_glpk_thresh_adjusted(self):
        try:
            lp = LP()
            lp.settings.backend = "glpk"
            lp.settings.thresh = 1e-8
            problem = add_box_constraints(
                Problem(np.zeros((0, 2)), np.ones((0))),
                [0.5, 1.0e-14],
                [0.95, 1.0e-7],
                simplify=True,
            )
            expected_A = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
            expected_b = np.array([-5.0e-01, -1.0e-14, 9.5e-01, 1.0e-07])
            self.assertTrue(np.isclose(problem.A, expected_A).all())
            self.assertTrue(np.isclose(problem.b, expected_b).all())
            problem = round(problem)
        except:
            self.fail("Rounding box created from np.zeros unexpectedly raised error.")

    def test_problem_pickling(self):
        problem = Problem([[1, 1]], [1], starting_point=[0, 0])
        data = pickle.dumps(problem)
        new = pickle.loads(data)
        self.assertListEqual(problem.A.tolist(), new.A.tolist())
        self.assertListEqual(problem.b.tolist(), new.b.tolist())
        self.assertEqual(problem.model, new.model)
        self.assertListEqual(
            problem.starting_point.tolist(), new.starting_point.tolist()
        )
        self.assertEqual(problem.transformation, new.transformation)
        self.assertEqual(problem.shift, new.shift)

        problem = Problem([[1, 1]], [1], Gaussian(), starting_point=[0, 0])
        data = pickle.dumps(problem)
        new = pickle.loads(data)
        self.assertListEqual(problem.A.tolist(), new.A.tolist())
        self.assertListEqual(problem.b.tolist(), new.b.tolist())
        self.assertEqual(problem.model.mean.tolist(), new.model.mean.tolist())
        self.assertEqual(
            problem.model.covariance.tolist(), new.model.covariance.tolist()
        )
        self.assertEqual(problem.model.inactives, new.model.inactives)
        self.assertListEqual(
            problem.starting_point.tolist(), new.starting_point.tolist()
        )
        self.assertEqual(problem.transformation, new.transformation)
        self.assertEqual(problem.shift, new.shift)

    def test_chebyshev_center(self):
        problem = Problem([[1, 1], [-1, 0], [0, -1]], [1, 0, 0])
        chebyshev = compute_chebyshev_center(problem).reshape(-1)

        for i in range(len(chebyshev)):
            self.assertAlmostEqual(chebyshev[i], 0.29289322)

    def test_chebyshev_center_when_problem_is_infeasible(self):
        A = [[-0.5]]
        b = [-2]
        lb = [-2]
        ub = [2]

        infeasible_problem = Problem(A, b)

        with self.assertRaises(ValueError):
            problem = add_box_constraints(
                infeasible_problem,
                lb,
                ub,
                simplify=False,
            )
            compute_chebyshev_center(problem)

        with self.assertRaises(ValueError):
            simplified_problem = add_box_constraints(
                infeasible_problem,
                lb,
                ub,
                simplify=True,
            )
            compute_chebyshev_center(simplified_problem)
