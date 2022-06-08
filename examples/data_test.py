import hopsy as hp
import numpy as np
import hopsy.util as util
import matplotlib.pyplot as plt

from copy import deepcopy

import os
import sys
import shutil

data = hp.Data()
data.states = [[np.array([0, 0])], [np.array([0, 0])]]

model = hp.MultivariateGaussianModel()
problem = hp.Problem([[1, 0]], [5], model)
problem = hp.add_box_constraints(problem, -5, 5)

n_chains = 2

x0 = [0.2, 0.6]
starting_points = [deepcopy(x0) for x in range(n_chains)]

run = hp.Run(problem, number_of_chains = n_chains, starting_points=starting_points)

run.sample(1000, 10)
print(np.array(run.data.states).shape)

run.data.write("test_output", False)

if len(sys.argv) == 1 or sys.argv[1] != "test":
    fig, axs = util.jointplot(run.data, density_estimator=util.gaussian_mixture)
    plt.show()

data = util.load("test_output")

if len(sys.argv) == 1 or sys.argv[1] != "test":
    fig, axs = util.jointplot(data, density_estimator=util.gaussian_mixture)
    plt.show()

shutil.rmtree("test_output")
