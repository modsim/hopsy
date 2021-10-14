import os 
import sys

import hopsy._hopsy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm 

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture


def load(path = "."):
    data = hopsy._hopsy.Data()

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

    gmm_skl = GaussianMixture(**kwargs)
    gmm_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(gmm_skl.score_samples(xy_sample))
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
        # only populate upper triangular part with subplots
        axs = [[fig.add_subplot(rows, cols, (i*rows)+j+1) for j in range(i, cols)] for i in range(rows)]
    else:
        rows, cols = len(axs), len(axs[0])

    # hack to save an if statement since int(True) == 1 and int(False) == 0
    colors = [matplotlib.cm.tab10(int((distinguish_chains) * i) % 10) for i in range(number_of_chains)]
    alpha = np.max([1./len(states[0]), 1./255]) # apperantly minimal valid alpha

    if "alpha" not in kwargs:
        kwargs["alpha"] = alpha

    for i in range(rows):
        for c in range(number_of_chains):
            for j in range(cols-i):
                axs[i][j].scatter(states[c,:,j+i+1], states[c,:,i], color=colors[c], **kwargs)

    return fig, axs


def densityplot(data, 
                fig = None, 
                axs = None, 
                distinguish_chains = False, 
                dims = None, 
                density_estimator = kde,
                cm = matplotlib.cm.tab10,
                **kwargs):
    states = np.array(data.states)
    dim = len(states[0,0]) if dims is None else len(dims)
    number_of_chains = len(states)

    if fig is None and axs is None:
        fig = plt.figure()

    if axs is None:
        rows, cols = dim-1, dim-1
        # only populate upper triangular part with subplots
        axs = [[fig.add_subplot(rows, cols, (i*rows)+j+1) for j in range(i, cols)] for i in range(rows)]
    else:
        rows, cols = len(axs), len(axs[0])

    # hack to save an if statement since int(True) == 1 and int(False) == 0
    colors = [cm(int((distinguish_chains) * i) % 10) for i in range(number_of_chains)]

    for i in range(rows):
        for c in range(number_of_chains):
            for j in range(cols-i):
                xx, yy, zz = density_estimator(states[c,:,i], states[c,:,j+i+1], **kwargs)
                axs[i][j].contour(xx, yy, zz, alpha = 1./number_of_chains, **kwargs)

    return fig, axs


def scatterdensityplot(data, 
                       fig = None, 
                       axs = None, 
                       distinguish_chains = False, 
                       dims = None, 
                       density_estimator = kde,
                       cm = matplotlib.cm.tab10,
                       **kwargs):
        fig, axs = scatterplot(data, fig, axs, distinguish_chains, dims, cm, **kwargs)
        fig, axs = densityplot(data, fig, axs, distinguish_chains, dims, density_estimator, cm, **kwargs)
        return fig, axs


def histplot(data,
              fig = None,
              axs = None,
              distinguish_chains = True,
              dims = None,
              cm = matplotlib.cm.tab10,
              **kwargs):
    states = np.array(data.states)
    dim = len(states[0,0]) if dims is None else len(dims)
    dims = range(dim) if dims is None else dims
    number_of_chains = len(states)

    if fig is None and axs is None:
        fig = plt.figure()

    if axs is None:
        rows, cols = int(np.ceil(np.sqrt(dim))), int(np.ceil(np.sqrt(dim)))
        axs = [[fig.add_subplot(rows, cols, i*cols + j + 1) for j in range(cols) if (i*cols+j) < dim] for i in range(rows)]
    else:
        rows, cols = len(axs), len(axs[0])

    # hack to save an if statement since int(True) == 1 and int(False) == 0
    colors = [cm(int((distinguish_chains) * i) % 10) for i in range(number_of_chains)]

    for i in range(rows):
        for j in range(cols):
            if j < len(axs[i]) and i*cols+j < dim:
                for c in range(number_of_chains):
                    axs[i][j].hist(states[c,:,dims[i*cols+j]], color=colors[c], alpha = 1./number_of_chains)

    return fig, axs

def jointplot(data,
              fig = None,
              axs = None,
              distinguish_chains = True,
              dims = None,
              diag_plot=histplot,
              off_diag_plot=scatterplot,
              cm = matplotlib.cm.tab10,
              **kwargs):
    states = np.array(data.states)
    dim = len(states[0,0]) if dims is None else len(dims)
    number_of_chains = len(states)

    if fig is None and axs is None:
        fig = plt.figure()

    if axs is None:
        rows, cols = dim, dim
        # only populate upper triangular part with subplots
        axs = [[fig.add_subplot(rows, cols, (i*rows)+j+1) for j in range(i, cols)] for i in range(rows)]

    diag_axs = []
    off_diag_axs = []
    for i in range(rows):
        off_diag_axs.append([])
        diag_axs.append(axs[i][0])

        for j in range(1, cols-i):
            off_diag_axs[i].append(axs[i][j])

    diag_plot(data, fig, [diag_axs], distinguish_chains, dims, cm, **kwargs)
    off_diag_plot(data, fig, off_diag_axs, distinguish_chains, dims, cm, **kwargs)

    return fig, axs

