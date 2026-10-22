"""Run the worker-side code in this process.

The parallel templates execute the workers in child processes, where line
coverage is not recorded. These tests call the same functions directly, with
one generator and, for the epidemic, a stand-in barrier and plain mailboxes.
"""

import warnings

import matplotlib
import numpy as np
import pytest
from numpy.random import default_rng

from simstreams import benchmark, cauchy_worker, pi_worker
from simstreams.coupled import _mailbox_worker
from simstreams.epidemic import (
    epidemic_worker,
    infect_subjects,
    initialize_population,
    migration,
    plot_epidemic,
    process_deaths,
    process_recoveries,
    select_exposures,
)

matplotlib.use("Agg")


class NoBarrier:
    """Stands in for a Manager barrier when there is only one party."""

    def __init__(self):
        self.waits = 0

    def wait(self):
        self.waits += 1


def test_pi_worker_estimates_pi_from_one_generator():
    pi_hat = pi_worker(default_rng(1), 1000, 50, ())
    assert pi_hat.shape == (50,)
    assert abs(pi_hat.mean() - np.pi) < 0.1


def test_cauchy_worker_estimates_the_scale():
    theta_hat = cauchy_worker(default_rng(1), 2000, 50, (5,))
    assert theta_hat.shape == (50,)
    assert abs(np.median(theta_hat) - 5) < 0.5


def test_benchmark_with_a_single_repeat_reports_zero_spread():
    calls = []
    out = benchmark(lambda: calls.append(1), repeats=1, warmup=1)
    assert len(calls) == 2
    assert out["stdev"] == 0.0
    assert out["times"] == [out["median"]]


def test_mailbox_worker_steps_with_and_without_the_barrier():
    mailboxes = {0: [], 1: [], 2: []}
    barrier = NoBarrier()
    for worker_id in range(3):
        _mailbox_worker(worker_id, mailboxes, barrier, True)
    assert mailboxes == {0: [200], 1: [2, 3, 7], 2: [2, 3, 7]}
    assert barrier.waits == 3

    mailboxes = {0: [], 1: [], 2: []}
    for worker_id in range(3):
        _mailbox_worker(worker_id, mailboxes, barrier, False)
    assert mailboxes == {0: [200], 1: [2, 3, 7], 2: [2, 3, 7]}
    assert barrier.waits == 3


def test_initialize_population_with_deaths_and_recoveries():
    rg = default_rng(3)
    population, recovery_time, death_day = initialize_population(rg, 200, 40, 0.5, 3)
    infected = [i for i, s in enumerate(population) if s in (1, 3)]
    assert len(infected) == 40
    dying = [i for i in infected if population[i] == 3]
    assert 0 < len(dying) < 40
    assert all(0 <= death_day[i] <= 3 for i in dying)
    assert all(1 <= recovery_time[i] <= 3 for i in infected if population[i] == 1)


def test_recoveries_deaths_exposures_and_infections():
    population = [1, 1, 0, 3, 0, 2]
    recovery_time = [2, 5, None, None, None, None]
    death_day = [None, None, None, 4, None, None]

    process_recoveries(population, recovery_time, day=3)
    assert population == [2, 1, 0, 3, 0, 2]

    process_deaths(population, recovery_time, death_day, day=4)
    assert population == [2, 1, 0, 0, 2]
    assert len(recovery_time) == 5 and len(death_day) == 5

    rg = default_rng(0)
    exposed = select_exposures(rg, population, contacts_per_day=3)
    assert len(exposed) == 3  # one infected person, three contacts
    assert all(0 <= i < 5 for i in exposed)

    # infection_prob 1 infects every susceptible contact; death_prob 1 marks
    # them as dying, death_prob 0 as recovering
    infect_subjects(rg, [2, 3], population, recovery_time, death_day, 1.0, 1.0, 3, 0)
    assert population[2] == 3 and population[3] == 3
    assert death_day[2] is not None
    population[2], population[3] = 0, 0
    infect_subjects(rg, [2, 3], population, recovery_time, death_day, 1.0, 0.0, 3, 0)
    assert population[2] == 1 and recovery_time[2] >= 1
    # infection_prob 0 and already-infected people are skipped
    before = list(population)
    infect_subjects(rg, [0, 4], population, recovery_time, death_day, 0.0, 0.0, 3, 0)
    assert population == before


def test_migration_round_trips_through_the_mailbox_of_a_single_town():
    rg = default_rng(5)
    population = [0, 1, 2, 3, 0]
    recovery_time = [None, 4, None, None, None]
    death_day = [None, None, None, 6, None]
    mailboxes = {0: []}
    barrier = NoBarrier()
    migration(rg, population, recovery_time, death_day, 2, 0, 1, barrier, mailboxes)
    # the two migrants left and came back, so the counts are unchanged
    assert sorted(population) == [0, 0, 1, 2, 3]
    assert len(recovery_time) == 5 and len(death_day) == 5
    assert mailboxes[0] == []
    assert barrier.waits == 2

    # no migrants: nothing happens, not even a barrier wait
    migration(rg, population, recovery_time, death_day, 0, 0, 1, barrier, mailboxes)
    assert barrier.waits == 2


@pytest.mark.parametrize("n_migrants", [0, 3])
def test_epidemic_worker_runs_one_town_in_process(n_migrants):
    history = epidemic_worker(
        (default_rng(1234), 0), 10, 3, NoBarrier(), {0: []},
        (200, 5, 0.05, 4, 3, n_migrants, (0.2,)),
    )
    assert history.shape == (3, 10)
    assert history.dtype.kind == "i"
    assert history.min() >= 0


def test_plot_epidemic_draws_one_town_or_every_town(tmp_path):
    one_town = np.array([[1, 2, 3], [2, 3, 4]])
    plot_epidemic(one_town, f_name=tmp_path / "one.png", show=False)
    assert (tmp_path / "one.png").exists()

    every_town = [one_town, one_town + 1]
    plot_epidemic(every_town, show=False)

    # show=True on the Agg backend only warns that it cannot show a window
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plot_epidemic(one_town, show=True)
