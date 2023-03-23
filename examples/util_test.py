from copy import deepcopy
from importlib import reload

import matplotlib.pyplot as plt
import numpy as np
import x3cfluxpy as x3c

import hopsy as hp
import hopsy.util as util

# m = x3c.X3CModel("Spiralus.fml", "default")
m = hp.MultivariateGaussianModel()
# prob = hp.Problem(m.get_A(), m.get_b(), m)
prob = hp.Problem([[1, 0]], [5], m)
prob = hp.add_box_constraints(prob, -5, 5)
n_chains = 2
x0 = [0.2, 0.6]
starting_points = [deepcopy(x0) for x in range(n_chains)]
run = hp.Run(prob, number_of_chains=n_chains, starting_points=starting_points)

print("starting sampling")
run.sample(1000, 10)
print(np.array(run.data.states).shape)

run.data.write("test_output", False)

fig, axs = util.jointplot(run.data, density_estimator=util.gaussian_mixture)
plt.show()
