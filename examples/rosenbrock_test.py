import sys

import hopsy
import numpy as np

#A = np.array([[1, 1], [-1, 0], [0, -1]])
#b = np.array([1, 0, 0]).reshape(-1, 1)

A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
b = np.array([10, 10, 10, 10]).reshape(-1, 1)

model = hopsy.RosenbrockModel(0.1, np.array([[1]])) 
problem = hopsy.Problem(A, b, model)
#run = hopsy.Run(problem, "HitAndRun")

run = hopsy.Run(problem, starting_points=[np.zeros((2,1))])

#run.stepsize = 0.1
#run.starting_points = [hopsy.compute_chebyshev_center(hopsy.Problem(A, b))]

run.sample(10000, 100)
#print(run.data.negative_log_likelihood)


if len(sys.argv) == 1 or sys.argv[1] != "test":
    import matplotlib.pyplot as plt
    states = np.array(run.data.states[0])
    plt.scatter(states[:,0], states[:,1])
    plt.show()
