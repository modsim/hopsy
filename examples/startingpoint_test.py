import sys

import hopsy
import numpy as np

if __name__ == "__main__":
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array([[1], [1], [0], [0]])

    problem = hopsy.Problem(A, b)
    run = hopsy.Run(problem)

    run.starting_points = [np.array([[1.5], [1.5]])]

    try:
        run.init()
    except:
        pass

    run.starting_points = [np.array([[0.5], [0.5]])]
    run.sample()

    if len(sys.argv) == 1 or sys.argv[1] != "test":
        import matplotlib.pyplot as plt
        states = np.array(run.data.states[0])

        fig = plt.figure(figsize=(35,35))
        fig.patch.set_alpha(1)
        ax = fig.gca()
        ax.scatter(states[:,0], states[:,1])
        plt.show()

