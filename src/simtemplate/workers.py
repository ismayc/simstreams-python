"""The book's example workers (Chapter 4).

A worker has the signature ``worker(rg, n, n_reps, param_tuple)``. It draws
from ``rg``, the process's own generator, generates data for ``n_reps``
replications of size ``n``, computes the quantity of interest for each
replication, and returns the results.
"""

import numpy as np


def pi_worker(rg, n, n_reps, param_tuple):
    """Estimate pi by dart throwing.

    Each replication throws ``n`` uniform points at the square [-1, 1]^2 and
    estimates pi as four times the fraction that land inside the unit circle.
    ``param_tuple`` is unused.

    Returns an array of ``n_reps`` estimates.
    """
    x = 2 * rg.random((n, n_reps)) - 1
    y = 2 * rg.random((n, n_reps)) - 1

    inside = x * x + y * y <= 1
    pi_hat = 4 * inside.mean(axis=0)

    return pi_hat


def cauchy_worker(rg, n, n_reps, param_tuple):
    """Estimate the Cauchy scale by half the interquartile range.

    Each replication draws ``n`` observations from a Cauchy distribution with
    scale ``theta = param_tuple[0]`` by the quantile transformation and
    estimates the scale as half the sample interquartile range.

    Returns an array of ``n_reps`` scale estimates.
    """
    theta = param_tuple[0]

    U = rg.random((n, n_reps))
    X = theta * np.tan(np.pi * (U - 0.5))

    Q = np.quantile(X, [0.25, 0.75], axis=0)
    theta_hat = (Q[1] - Q[0]) / 2

    return theta_hat


__all__ = ["cauchy_worker", "pi_worker"]
