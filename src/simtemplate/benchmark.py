"""Timing helpers for speed-up experiments (Chapter 7).

Time the whole simulation call, including pool start-up, discard warm-up
runs, and summarize with the median. These are the functions behind the
timing tables in Chapters 4 and 5.
"""

import statistics
from functools import partial
from time import perf_counter

import numpy as np


def benchmark(func, repeats=10, warmup=1):
    """Benchmark a function that takes no arguments.

    Parameters
    ----------
    func : callable
        Function containing the complete simulation call.
    repeats : int
        Number of measured runs.
    warmup : int
        Number of unmeasured warm-up runs.

    Returns
    -------
    dict
        Summary statistics (``median``, ``mean``, ``stdev``, ``minimum``,
        ``maximum``, ``q1``, ``q3``) and all measured elapsed ``times`` in
        seconds.
    """
    # Warm-up runs are not included in the results.
    for _ in range(warmup):
        func()

    times = []

    for _ in range(repeats):
        start = perf_counter()
        func()
        elapsed = perf_counter() - start
        times.append(elapsed)

    return {
        "median": statistics.median(times),
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if repeats > 1 else 0.0,
        "minimum": min(times),
        "maximum": max(times),
        "q1": float(np.percentile(times, 25)),
        "q3": float(np.percentile(times, 75)),
        "times": times,
    }


def benchmark_cores(run_simulation, core_counts, repeats=10, warmup=1):
    """Benchmark a simulation over several core counts.

    Parameters
    ----------
    run_simulation : callable
        A function accepting ``n_cores`` as its only argument that runs the
        complete simulation. Typically a thin wrapper around
        :func:`simtemplate.sim_template` with every other argument fixed.
    core_counts : iterable
        Core counts to benchmark. The first one is the baseline for the
        speed-up.
    repeats : int
        Number of measured runs at each core count.
    warmup : int
        Number of warm-up runs at each core count.

    Returns
    -------
    list of dict
        One :func:`benchmark` summary per core count, with ``n_cores``,
        ``speedup`` (baseline median over this median), and ``efficiency``
        (``speedup / n_cores``) added.
    """
    results = []

    for n_cores in core_counts:
        summary = benchmark(
            partial(run_simulation, n_cores),
            repeats=repeats,
            warmup=warmup,
        )

        summary["n_cores"] = n_cores
        results.append(summary)

    serial_time = results[0]["median"]

    for result in results:
        result["speedup"] = serial_time / result["median"]
        result["efficiency"] = result["speedup"] / result["n_cores"]

    return results


__all__ = ["benchmark", "benchmark_cores"]
