import unittest

from .. import *

class ProblemTests(unittest.TestCase):

    def test_repr(self):
        problem = Problem([[1, 1]], [1], starting_point=[0, 0])
        expected = "hopsy.Problem(A=array([[1., 1.]]), b=array([1.]), starting_point=array([0., 0.]))"
        self.assertEqual(str(problem), expected)

        problem = Problem([[1, 1]], [1], Gaussian(), starting_point=[0, 0])
        expected = "hopsy.Problem(A=array([[1., 1.]]), b=array([1.]), model=hopsy.Gaussian(), starting_point=array([0., 0.]))"
        self.assertEqual(str(problem), expected)

        problem = Problem([[1, 1]], [1], Gaussian(), starting_point=[0, 0], shift=[1, 2])
        expected = "hopsy.Problem(A=array([[1., 1.]]), b=array([1.]), model=hopsy.Gaussian(), starting_point=array([0., 0.]), shift=array([1., 2.]))"
        self.assertEqual(str(problem), expected)

    def test_model_id(self):
        gaussian = Gaussian()
        id_gaussian = id(gaussian)
        problem = Problem([[1, 1]], [1], gaussian, starting_point=[0, 0])
        self.assertEqual(id(gaussian), id(problem.model))

        new_gaussian = problem.model
        self.assertEqual(id(gaussian), id(new_gaussian))

        del problem
        self.assertEqual(id_gaussian, id(gaussian))
        self.assertEqual(id_gaussian, id(new_gaussian))

