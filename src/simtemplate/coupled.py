"""A template for workers that communicate (Chapter 6).

:func:`sim_template` keeps its workers independent. The epidemic case study
needs workers that exchange information while they run: each process is one
town, every town runs all ``n_reps`` replications, and migrants move between
towns at the end of each day. :func:`sim_template_coupled` is the book's
``sim_template_epidemic.py`` with the worker as an argument. It gives every
worker a shared barrier and one mailbox per worker, built with a
:class:`multiprocessing.Manager`.
"""

from multiprocessing import Manager, Pool

from .template import make_generators


def sim_template_coupled(worker, seed, n, n_reps, n_cores, *args):
    """Run a simulation whose workers communicate through mailboxes.

    Parameters
    ----------
    worker : callable
        ``worker(rg_and_id, n, n_reps, barrier, mailboxes, param_tuple)``.
        ``rg_and_id`` is the pair ``(rg, worker_id)`` with ``worker_id`` in
        ``range(n_cores)``, ``barrier`` is a shared
        :class:`threading.Barrier` for ``n_cores`` parties, and ``mailboxes``
        is a shared dict mapping each ``worker_id`` to a shared list.
    seed : int
        Master seed.
    n : int
        Per-replication size (days, for the epidemic).
    n_reps : int
        Number of replications **each** worker runs. Unlike
        :func:`sim_template`, the replications are not divided.
    n_cores : int
        Number of worker processes.
    *args
        Model parameters passed on to the worker.

    Returns
    -------
    list
        One element per process, in process order.
    """
    if not callable(worker):
        raise TypeError(
            "worker must be a callable "
            "worker(rg_and_id, n, n_reps, barrier, mailboxes, param_tuple)"
        )

    rg_list = [(rg, i) for i, rg in enumerate(make_generators(seed, n_cores))]

    with Manager() as manager:
        barrier = manager.Barrier(n_cores)

        mailboxes = manager.dict()
        for i in range(n_cores):
            mailboxes[i] = manager.list()

        set_up = []

        for i in range(n_cores):
            set_up.append([rg_list[i], n, n_reps, barrier, mailboxes, args])

        with Pool(processes=n_cores) as pool:
            result = pool.starmap(worker, set_up)

    return result


def _mailbox_worker(worker_id, mailboxes, barrier, use_barrier):
    if worker_id == 0:
        mailboxes[1][:] = [2, 3, 7]

    # wait until all workers reach this point
    if use_barrier:
        barrier.wait()

    if worker_id == 1:
        mailboxes[2][:] = mailboxes[1][:]

    if worker_id == 2:
        mailboxes[0].append(200)


def mailbox_demo(use_barrier=True):
    """The Chapter 6 inter-process communication demo.

    Three workers share three mailboxes and a barrier. Worker 0 fills worker
    1's mailbox, every worker waits at the barrier, then worker 1 copies its
    mailbox to worker 2 and worker 2 posts to worker 0. The result is
    ``{0: [200], 1: [2, 3, 7], 2: [2, 3, 7]}``.

    With ``use_barrier=False`` the workers skip the barrier, as in the book's
    listing, and worker 1 sometimes copies its mailbox before worker 0 has
    written to it, leaving mailbox 2 empty: the race condition the chapter
    discusses.

    Returns a plain dict of the final mailbox contents.
    """
    # create a manager to tend to shared memory objects
    with Manager() as manager:
        # shared dictionary with one mailbox per worker
        mailboxes = manager.dict(
            {0: manager.list(), 1: manager.list(), 2: manager.list()}
        )

        # shared barrier
        barrier = manager.Barrier(3)

        # information to be sent to each process
        set_up = [[i, mailboxes, barrier, use_barrier] for i in range(3)]

        # create the pool and send information to the workers
        with Pool(processes=3) as pool:
            pool.starmap(_mailbox_worker, set_up)

        final_mailboxes = {k: list(v) for k, v in mailboxes.items()}

    return final_mailboxes


__all__ = ["mailbox_demo", "sim_template_coupled"]
