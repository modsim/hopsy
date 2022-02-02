import unittest
import pickle

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

        problem = Problem([[1, 1]], [1], Gaussian(), starting_point=[0, 0], shift=[1, 2])
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


    def test_problem_pickling(self):
        problem = Problem([[1, 1]], [1], starting_point=[0, 0])
        data = pickle.dumps(problem)
        new = pickle.loads(data)
        self.assertListEqual(problem.A.tolist(), new.A.tolist())
        self.assertListEqual(problem.b.tolist(), new.b.tolist())
        self.assertEqual(problem.model, new.model)
        self.assertListEqual(problem.starting_point.tolist(), new.starting_point.tolist())
        self.assertEqual(problem.transformation, new.transformation)
        self.assertEqual(problem.shift, new.shift)

        problem = Problem([[1, 1]], [1], Gaussian(), starting_point=[0, 0])
        data = pickle.dumps(problem)
        new = pickle.loads(data)
        self.assertListEqual(problem.A.tolist(), new.A.tolist())
        self.assertListEqual(problem.b.tolist(), new.b.tolist())
        self.assertEqual(problem.model.mean.tolist(), new.model.mean.tolist())
        self.assertEqual(problem.model.covariance.tolist(), new.model.covariance.tolist())
        self.assertEqual(problem.model.inactives, new.model.inactives)
        self.assertListEqual(problem.starting_point.tolist(), new.starting_point.tolist())
        self.assertEqual(problem.transformation, new.transformation)
        self.assertEqual(problem.shift, new.shift)
