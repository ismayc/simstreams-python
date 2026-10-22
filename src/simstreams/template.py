"""The general parallel simulation template.

The book's ``sim_template.py`` reads its worker with ``from sim_worker import
worker``. Here the worker is the first argument, and the two pieces of the
template that are useful on their own, dividing the replications and building
the per-process generators, are separate functions.
"""

from multiprocessing import Pool

from numpy.random import Generator, SeedSequence, default_rng


def split_reps(n_reps, n_cores):
    """Divide ``n_reps`` replications evenly across ``n_cores`` processes.

    Every process gets the same base share, and the remainder is handed out
    one replication at a time starting with the first process. This is the
    load-balancing step of the general parallel simulation algorithm.

    Returns a list of ``n_cores`` integers that sums to ``n_reps``.
    """
    n_reps = int(n_reps)
    n_cores = int(n_cores)
    if n_reps < 1:
        raise ValueError("n_reps must be a positive integer")
    if n_cores < 1:
        raise ValueError("n_cores must be a positive integer")

    base_reps = n_reps // n_cores
    remainder = n_reps % n_cores
    reps = [base_reps] * n_cores

    # load balance
    for i in range(remainder):
        reps[i] += 1

    return reps


def make_generators(seed, n_cores):
    """Create one independent PCG64 generator per process.

    ``SeedSequence(seed).spawn(n_cores)`` gives each process its own child
    seed, so the master ``seed`` fixes every process's stream and the streams
    do not overlap.

    Returns a list of ``n_cores`` ``numpy.random.Generator`` objects.
    """
    ss = SeedSequence(seed)
    child_seeds = ss.spawn(int(n_cores))
    return [default_rng(s) for s in child_seeds]


def make_set_up(seed, n, n_reps, n_cores, *args):
    """Build the argument tuple handed to each process.

    Element ``i`` is ``(rg, n, reps[i], args)``: the process's generator, the
    per-replication sample size, its share of the replications, and the model
    parameters as a tuple. A worker is called as ``worker(*set_up[i])``.
    """
    rg_list = make_generators(seed, n_cores)
    reps = split_reps(n_reps, n_cores)
    return [(rg_list[i], n, reps[i], args) for i in range(len(reps))]


def sim_template(worker, seed, n, n_reps, n_cores, *args, pool=None):
    """Run a simulation experiment in parallel.

    The general parallel simulation algorithm from the book: choose a master
    seed, give each process an independent PCG64 stream, divide the
    replications evenly, run the worker on every process, and return the
    combined results.

    Parameters
    ----------
    worker : callable
        ``worker(rg, n, n_reps, param_tuple)``. ``rg`` is the process's
        ``numpy.random.Generator``, ``n`` the per-replication sample
        size, ``n_reps`` the process's share of the replications, and
        ``param_tuple`` the model parameters passed as ``*args``. The worker
        must be importable by the child processes (defined at module level,
        not a lambda or a closure) and self-contained.
    seed : int
        Master seed.
    n : int
        Sample size for each replication.
    n_reps : int
        Total number of replications.
    n_cores : int
        Number of worker processes.
    *args
        Model parameters passed on to the worker.
    pool : multiprocessing.pool.Pool, optional
        An existing pool to reuse. When omitted, a pool of ``n_cores``
        processes is created for the call and closed afterwards.

    Returns
    -------
    list
        One element per process, in process order: whatever the worker
        returned for its replications.

    Notes
    -----
    Results are reproducible for a fixed ``seed`` **and** a fixed
    ``n_cores``. Changing the number of processes changes how the
    replications are divided and therefore which stream produces which draw.

    On macOS and Windows, child processes are spawned, so the calling script
    needs an ``if __name__ == "__main__":`` guard.
    """
    if not callable(worker):
        raise TypeError("worker must be a callable worker(rg, n, n_reps, param_tuple)")

    set_up = make_set_up(seed, n, n_reps, n_cores, *args)

    if pool is not None:
        return pool.starmap(worker, set_up)

    with Pool(processes=len(set_up)) as own_pool:
        result = own_pool.starmap(worker, set_up)

    return result


def sim_template_serial(worker, seed, n, n_reps, *args):
    """Run a simulation experiment in the current process.

    The serial counterpart of ``sim_template``: the same seed layout as a
    one-process run, executed without a pool. Useful for debugging a worker
    before running it in parallel, and for the one-core baseline of a timing
    experiment. The result equals
    ``sim_template(worker, seed, n, n_reps, 1, *args)[0]``.
    """
    if not callable(worker):
        raise TypeError("worker must be a callable worker(rg, n, n_reps, param_tuple)")

    (set_up,) = make_set_up(seed, n, n_reps, 1, *args)
    return worker(*set_up)


__all__ = [
    "Generator",
    "make_generators",
    "make_set_up",
    "sim_template",
    "sim_template_serial",
    "split_reps",
]
