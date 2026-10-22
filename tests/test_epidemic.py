"""Pin the Chapter 6 epidemic runs behind the book's figures.

The per-town [sum, max] of the daily infected counts come from the book
repository's verify/verify.py.
"""

import numpy as np
import pytest

from simstreams import epidemic_driver, sim_template_coupled
from simstreams.epidemic import initialize_population


def _sum_max(out):
    return [[int(np.array(h).sum()), int(np.array(h).max())] for h in out]


@pytest.mark.parametrize(
    "n_migrants, expected",
    [
        (0, [[1185, 32], [3992, 47], [15491, 111], [22678, 161]]),
        (10, [[1654, 34], [3538, 61], [15305, 99], [22449, 169]]),
    ],
)
def test_chapter_6_epidemic_per_town_sum_and_max(n_migrants, expected):
    out = epidemic_driver(
        1234, 30, 20, 4, 1000, 20, 0.05, 5, 3, n_migrants, 0.05, 0.15, 0.25, 0.30
    )
    assert len(out) == 4
    assert all(h.shape == (20, 30) for h in out)
    assert _sum_max(out) == expected


def test_epidemic_driver_needs_one_infection_prob_per_town():
    with pytest.raises(ValueError):
        epidemic_driver(1234, 5, 2, 4, 100, 5, 0.05, 5, 3, 0, 0.1, 0.2)


def test_initialize_population_marks_the_initial_cases():
    from numpy.random import default_rng

    population, recovery_time, death_day = initialize_population(
        default_rng(1), 50, 5, 0.0, 3
    )
    assert len(population) == 50
    assert population.count(1) == 5
    assert all(recovery_time[i] is not None for i in range(50) if population[i] == 1)
    assert all(d is None for d in death_day)


def id_worker(rg_and_id, n, n_reps, barrier, mailboxes, param_tuple):
    rg, worker_id = rg_and_id
    mailboxes[(worker_id + 1) % param_tuple[0]].append(worker_id)
    barrier.wait()
    return (worker_id, n, n_reps, list(mailboxes[worker_id]), rg.random())


def test_sim_template_coupled_gives_each_worker_an_id_and_a_mailbox():
    out = sim_template_coupled(id_worker, 1, 7, 3, 3, 3)
    assert [o[0] for o in out] == [0, 1, 2]
    assert all(o[1] == 7 and o[2] == 3 for o in out)
    assert [o[3] for o in out] == [[2], [0], [1]]
    assert len({o[4] for o in out}) == 3


def test_sim_template_coupled_rejects_a_non_callable():
    with pytest.raises(TypeError):
        sim_template_coupled("worker", 1, 7, 3, 3)
