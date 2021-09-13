try:
    from . _hopsy import *
except:
    from hopsy import *

import os 

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture


def load(path = "."):
    data = Data()

    abs_path = os.path.abspath(path)

    subdirs = os.listdir(path)
    subdirs = [os.path.join(abs_path, item) for item in subdirs]
    subdirs = [item for item in subdirs if os.path.isdir(item)]

    acceptance_rates = []
    negative_log_likelihood = []
    states = []
    timestamps = []

    for subdir in subdirs:
        items = os.listdir(subdir)
        items = [os.path.join(subdir, item) for item in items]
        items = [item for item in items if os.path.isfile(item)]

        for item in items:
            print(item)
            if "csv" not in item: 
                continue

            content = item.split("_")[-1]
            content = content.split(".csv")[0]

            if content == "acceptanceRates":
                acceptance_rates.append(np.loadtxt(item, delimiter=",").tolist())
            if content == "negativeLogLikelihood":
                negative_log_likelihood.append(np.loadtxt(item, delimiter=",").tolist())
            if content == "states":
                states.append(np.loadtxt(item, delimiter=",").tolist())
            if content == "timestamps":
                timestamps.append(np.loadtxt(item, delimiter=",").astype(int).tolist())

    data.acceptance_rates = acceptance_rates
    data.negative_log_likelihood = negative_log_likelihood
    data.states = states
    data.timestamps = timestamps

    return data


def kde(x,
        y,
        xbins = 100j,
        ybins = 100j,
        **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(**kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


def gaussian_mixture(x,
                     y,
                     xbins = 100j,
                     ybins = 100j,
                     **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = GaussianMixture(**kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


def scatterplot(data,
                fig = None,
                axs = None,
                distinguish_chains = False,
                dims = None,
                **kwargs):
    states = np.array(data.states)
    dim = len(states[0,0]) if dims is None else len(dims)
    number_of_chains = len(states)

    if fig is None and axs is None:
        fig = plt.figure(**fig_kwargs)

    if axs is None:
        rows, cols = dim-1, dim-1
        print(rows, cols)
        # only populate upper triangular part with subplots
        axs = [[fig.add_subplot(rows, cols, (i*rows)+j+1) for j in range(i, cols)] for i in range(rows)]
    else:
        rows, cols = len(axs), len(axs[0])

    # hack to save an if statement since int(True) == 1 and int(False) == 0
    colors = [cm.tab10(int((distinguish_chains) * i) % 10) for i in range(number_of_chains)]

    for i in range(rows):
        for c in range(number_of_chains):
            for j in range(1, cols-i):
                print(i, j, c)
                axs[i][j].scatter(states[c,:,i], states[c,:,j+i], color=colors[c], alpha = 1./number_of_chains, **kwargs)

    return fig, axs


def densityplot(data, 
                fig = None, 
                axs = None, 
                distinguish_chains = False, 
                dims = None, 
                density_estimator = kde,
                **kwargs):
    states = np.array(data.states)
    dim = len(states[0,0]) if dims is None else len(dims)
    number_of_chains = len(states)

    if fig is None and axs is None:
        fig = plt.figure()

    if axs is None:
        rows, cols = dim-1, dim-1
        print(rows, cols)
        # only populate upper triangular part with subplots
        axs = [[fig.add_subplot(rows, cols, (i*rows)+j+1) for j in range(i, cols)] for i in range(rows)]
    else:
        rows, cols = len(axs), len(axs[0])

    # hack to save an if statement since int(True) == 1 and int(False) == 0
    colors = [cm.tab10(int((distinguish_chains) * i) % 10) for i in range(number_of_chains)]

    for i in range(rows):
        for c in range(number_of_chains):
            for j in range(1, cols-i):
                xx, yy, zz = density_estimator(states[c,:,i], states[c,:,j+i], **kwargs)
                axs[i][j].contour(xx, yy, zz, alpha = 1./number_of_chains, **kwargs)

    return fig, axs


def scatterdensityplot(data, 
                       fig = None, 
                       axs = None, 
                       distinguish_chains = False, 
                       dims = None, 
                       density_estimator = kde,
                       **kwargs):
        fig, axs = scatterplot(data, fig, axs, distinguish_chains, dims, **kwargs)
        fig, axs = densityplot(data, fig, axs, distinguish_chains, dims, density_estimator, **kwargs)
        return fig, axs


def jointplot(data,
              fig = None,
              axs = None,
              distinguish_chains = True,
              dims = None,
              off_diag_plot=scatterdensityplot,
              **kwargs):
    states = np.array(data.states)
    dim = len(states[0,0]) if dims is None else len(dims)
    number_of_chains = len(states)

    if fig is None and axs is None:
        fig = plt.figure()

    if axs is None:
        rows, cols = dim, dim
        print(rows, cols)
        # only populate upper triangular part with subplots
        axs = [[fig.add_subplot(rows, cols, (i*rows)+j+1) for j in range(i, cols)] for i in range(rows)]

    # hack to save an if statement since int(True) == 1 and int(False) == 0
    colors = [cm.tab10(int((distinguish_chains) * i) % 10) for i in range(number_of_chains)]

    off_diag_axs = []
    for i in range(rows):
        off_diag_axs.append([])
        for c in range(number_of_chains):
            axs[i][0].hist(states[c,:,i], color=colors[c], alpha = 1./number_of_chains)
            for j in range(1, cols-i):
                off_diag_axs[i].append(axs[i][j])

    off_diag_plot(data, fig, off_diag_axs, distinguish_chains, dims, **kwargs)

    return fig, axs

