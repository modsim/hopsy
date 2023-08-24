#!/usr/bin/env python
# coding: utf-8

# # Benchmarking Multiphase Monte Carlo Sampling

# # Motivation
#
# As a methods researcher one is often interested in testing new methods and benchmarking them together with state-of-the-art algorithms.
# hopsy provides the framework for conducting such comparisons, because it allows the methods researcher to focus on the algorithms, while the scaffolding, i.e., I/O, convergence diagnostics, and state-of-the-art implementations of existing algorithms are provided.
#
#
# We show in this notebook how one can adapt build a multiphase monte carlo sampling plugin for hopsy

# ## Background
# Instead of rounding before sampling, multiphase monte carlo sampling rounds adaptively based on the samples of the Markov chain during a run.
# It has been reported to be an efficient strategy, if the underlying Markov chain is efficient enough.
#
# The Idea has found previous adaption in the artificially centered coordinate hit-and-run and the optGP sampler.
# However, Coordinate Hit-and-Run Rounded showed greater and more stable performance (https://pubmed.ncbi.nlm.nih.gov/28158334/) than both of those algorithms.
#
# The idea is resurrected and shown to be efficient given a better Markov chain implementation.
# See https://drops.dagstuhl.de/opus/volltexte/2021/13820/pdf/LIPIcs-SoCG-2021-21.pdf

import time
from typing import List

import arviz
import numpy as np
import PolyRound
from matplotlib import pyplot as plt

import hopsy


def svd_rounding(samples, problem):
    # _samples have shape [n_chains, n_iterations, n_dims].
    # We concatenate them to Stack [n_dim, n_iterations] for rounding
    stacked_samples = np.concatenate(samples, axis=0)
    U, S, Vh = np.linalg.svd(stacked_samples)
    # Rescaling as mentioned in  https://drops.dagstuhl.de/opus/volltexte/2021/13820/pdf/LIPIcs-SoCG-2021-21.pdf
    s_ratio = np.max(S) / np.min(S)
    S = S / np.min(S)
    if np.max(S) >= 2.0:
        S[np.where(S > 2.0)] = 2.0
    # create rounding based on samples
    rounding_matrix = Vh.dot(np.diag(S))
    center = np.mean(stacked_samples, axis=0)
    print("center: ", center)

    # We do not want to use all transforms to transform the mcmc state space. Instead we use only the one from the current iteration
    sub_problem = hopsy.Problem(problem.A, problem.b, transformation=rounding_matrix)
    starting_points = hopsy.transform(
        sub_problem, [samples[i, -1, :] for i in range(samples.shape[0])]
    )

    # problem.shift = center if problem.shift is None else problem.shift + problem.transformation@center
    # problem.b = problem.b - (problem.A@problem.shift).flatten()
    problem.A = problem.A.dot(rounding_matrix)
    problem.transformation = (
        rounding_matrix
        if problem.transformation is None
        else problem.transformation.dot(rounding_matrix)
    )

    return s_ratio, starting_points, problem


def run_multiphase_sampling(
    proposal,
    problem: hopsy.Problem,
    seeds: List,
    target_ess: float,
    steps_per_phase: int,
    starting_points: List,
):
    limit_singular_ratio_value = 3  # from https://drops.dagstuhl.de/opus/volltexte/2021/13820/pdf/LIPIcs-SoCG-2021-21.pdf
    assert len(starting_points) == len(seeds)
    rngs = [hopsy.RandomNumberGenerator(s) for s in seeds]

    ess = 0
    iterations = 0
    s_ratio = limit_singular_ratio_value + 1
    while ess < target_ess:
        iterations += 1
        # print(starting_points[0].shape)
        markov_chains = [
            hopsy.MarkovChain(problem, proposal, starting_point=s)
            for s in starting_points
        ]

        acceptance_rate, _samples = hopsy.sample(
            markov_chains, rngs, n_samples=steps_per_phase, thinning=1
        )
        print("\titer:", iterations, "acc_rate", acceptance_rate)

        if s_ratio > limit_singular_ratio_value:
            s_ratio, starting_points, problem = svd_rounding(_samples, problem)
            print("\ts_ratio:", s_ratio)
            samples = _samples
        else:
            print("\tno rounding")
            samples = np.concatenate((samples, _samples), axis=1)
            starting_points = samples[:, -1, :]

        ess = np.min(hopsy.ess(samples))
        print("\tess", str(ess) + ",", "samples", samples.shape[1])

    return samples, iterations, ess


if __name__ == "__main__":
    # general parameters
    target_ess = 1000
    proposalTypes = {
        "Billiard walk": hopsy.BilliardWalkProposal,
        "Coordinate Hit-And-Run": hopsy.UniformCoordinateHitAndRunProposal,
    }

    bp = hopsy.BirkhoffPolytope(4)
    problem = hopsy.Problem(
        bp.A, bp.b, np.identity(bp.A.shape[1]), np.zeros(bp.A.shape[1])
    )
    seeds = [1, 2, 3, 4]
    steps_per_phase = (
        problem.A.shape[1] * 20
    )  # recommendation from https://drops.dagstuhl.de/opus/volltexte/2021/13820/pdf/LIPIcs-SoCG-2021-21.pdf
    cheby = hopsy.compute_chebyshev_center(problem).flatten()

    e_coli = PolyRound.api.StoichiometryParser.parse_sbml_cobrapy(
        "../extern/hops/resources/e_coli_core/e_coli_core.xml"
    )
    problem = hopsy.Problem(e_coli.A, e_coli.b)
    cheby = hopsy.compute_chebyshev_center(problem).flatten()
    steps_per_phase = (
        problem.A.shape[1] * 20
    )  # recommendation from https://drops.dagstuhl.de/opus/volltexte/2021/13820/pdf/LIPIcs-SoCG-2021-21.pdf

    samples = {}
    iterations = {}
    ess = {}
    times = {}
    ess_t = {}

    for name, p in proposalTypes.items():
        print("running benchmark for", name)
        # resets problem and starting points
        bp = hopsy.BirkhoffPolytope(5)
        problem = hopsy.Problem(bp.A, bp.b)
        cheby = hopsy.compute_chebyshev_center(problem).flatten()
        starting_points = [cheby for s in seeds]

        start = time.time()
        samples[name], iterations[name], ess[name] = run_multiphase_sampling(
            proposal=p,
            problem=problem,
            seeds=seeds,
            target_ess=target_ess,
            steps_per_phase=steps_per_phase,
            starting_points=starting_points,
        )
        end = time.time()
        times[name] = end - start
        ess_t[name] = ess[name] / times[name]

    print(ess_t)

    for name, p in proposalTypes.items():
        samples_full = bp.convert_to_full_space(
            samples[name].reshape((-1, problem.A.shape[1]))
        )
        arviz.plot_pair(
            arviz.convert_to_inference_data(
                {
                    f"x{i % bp.size}{i // bp.size}": samples_full[i, :]
                    for i in range(bp.size_squared)
                }
            )
        )
        plt.title(name)
        plt.show()
