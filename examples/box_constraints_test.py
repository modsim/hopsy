import numpy as np
import hopsy

A = np.array([[1, 1, 1]])
b = np.array([[1]])

old_A = np.array([[1, 1, 1]])
old_b = np.array([[1]])

expected_A = np.array(
    [[ 1., 1., 1.],
     [ 1., 0., 0.],
     [ 0., 1., 0.],
     [ 0., 0., 1.],
     [-1.,-0.,-0.],
     [-0.,-1.,-0.],
     [-0.,-0.,-1.]])
expected_b = np.array([1., 5., 5., 5., 5., 5., 5.])
expected_b2 = np.array([1., 4., 5., 6., -1., -2., -3.])

new_A, new_b = hopsy.add_box_constraints(A, b, -5, 5)

assert (new_A == expected_A).all()
assert (new_b == expected_b).all()
assert (A == old_A).all()
assert (b == old_b).all()

new_A, new_b = hopsy.add_box_constraints(A, b, [1, 2, 3,], [4, 5, 6])

assert (new_A == expected_A).all()
assert (new_b == expected_b2).all()
assert (A == old_A).all()
assert (b == old_b).all()

problem = hopsy.Problem(A, b)

new_problem = hopsy.add_box_constraints(problem, -5, 5)

assert (new_problem.A == expected_A).all()
assert (new_problem.b == expected_b).all()
assert (problem.A == old_A).all()
assert (problem.b == old_b).all()

new_problem = hopsy.add_box_constraints(problem, [1, 2, 3,], [4, 5, 6])

assert (new_problem.A == expected_A).all()
assert (new_problem.b == expected_b2).all()
assert (problem.A == old_A).all()
assert (problem.b == old_b).all()


