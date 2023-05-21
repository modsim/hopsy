import pickle

import numpy as np

from hopsy import *


def create_box_and_round_glpk():
    lp = LP()
    # lp.settings.backend = "glpk"
    lp.settings.backend = "gurobi"
    p = Problem(np.zeros((0, 2)), np.ones((0)))
    problem = add_box_constraints(
        Problem(np.zeros((0, 2)), np.ones((0))),
        [0.5, 1.0e-14],
        [0.95, 1.0e-7],
        simplify=True,
    )
    expected_A = np.array([[-1.0], [1.0]])
    expected_b = np.array([0.225, 0.225])
    print('true',problem.A)
    print('expected', expected_A)
    print('true', problem.b)
    print('expected', expected_b)
    problem = round(problem)

if __name__=="__main__":
    create_box_and_round_glpk()

