"""The SIRD epidemic case study with migration (Chapter 6).

Each worker process is one town. A town's population is a list of status
codes (0 susceptible, 1 infected, 2 recovered, 3 infected and going to die),
with parallel lists of recovery and death days. Every day each town processes
recoveries and deaths, exposes contacts, infects some of them, and then sends
``n_migrants`` people to the next town through the shared mailboxes.

The model functions are the book's ``epidemic_functions.py``, the worker is
``epidemic_worker.py``, and ``epidemic_driver`` is ``epidemic_driver.py``
run through ``simstreams.sim_template_coupled``.
"""

import numpy as np

from .coupled import sim_template_coupled


def initialize_population(rg, pop_size, initial_infected, death_prob, recovery_days):
    """Create a town with ``initial_infected`` infected people."""
    population = [0] * pop_size
    recovery_time = [None] * pop_size
    death_day = [None] * pop_size

    # Randomly choose initial_infected subjects to be infected
    initial_cases = rg.choice(range(pop_size), initial_infected, replace=False)

    # Determine if infected subjects will recover
    for i in initial_cases:
        population[i] = 1

        if rg.random() < death_prob:
            population[i] = 3

            # Randomly choose a death day
            death_day[i] = rg.integers(0, recovery_days + 1)

        else:
            # Randomly choose a recovery day
            recovery_time[i] = rg.integers(1, recovery_days + 1)

    return population, recovery_time, death_day


def process_recoveries(population, recovery_time, day):
    """Move infected people whose recovery day has arrived to recovered."""
    for i in range(len(population)):
        if population[i] == 1:
            if day >= recovery_time[i]:
                population[i] = 2


def process_deaths(population, recovery_time, death_day, day):
    """Remove people whose death day is today."""
    i = len(population) - 1

    while i >= 0:
        if population[i] == 3:
            if death_day[i] == day:
                population.pop(i)
                recovery_time.pop(i)
                death_day.pop(i)

        i -= 1


def select_exposures(rg, population, contacts_per_day):
    """List the people each infected person contacts today."""
    infected = [i for i, status in enumerate(population) if status == 1]

    exposed = []

    for _ in infected:
        n_contacts = min(contacts_per_day, len(population))

        contacts = rg.choice(range(len(population)), n_contacts, replace=False)

        exposed.extend(contacts)

    return exposed


def infect_subjects(
    rg,
    exposed,
    population,
    recovery_time,
    death_day,
    infection_prob,
    death_prob,
    recovery_days,
    day,
):
    """Infect exposed susceptible people with probability ``infection_prob``."""
    for i in exposed:
        if population[i] != 0:
            continue
        if rg.random() >= infection_prob:
            continue
        if rg.random() < death_prob:
            population[i] = 3
            death_day[i] = rg.integers(day, day + recovery_days + 1)
        else:
            population[i] = 1
            recovery_time[i] = day + rg.integers(1, recovery_days + 1)


def migration(
    rg,
    population,
    recovery_time,
    death_day,
    n_migrants,
    worker_id,
    n_cores,
    barrier,
    mailboxes,
):
    """Send ``n_migrants`` people to the next town and receive arrivals."""
    if n_migrants == 0:
        return

    n_send = min(n_migrants, len(population))
    dest = (worker_id + 1) % n_cores

    for _ in range(n_send):
        migrant = rg.integers(len(population))

        mailboxes[dest].append(
            [population[migrant], recovery_time[migrant], death_day[migrant]]
        )

        population.pop(migrant)
        recovery_time.pop(migrant)
        death_day.pop(migrant)

    barrier.wait()

    for person in mailboxes[worker_id]:
        population.append(person[0])
        recovery_time.append(person[1])
        death_day.append(person[2])

    # empty the mailbox
    mailboxes[worker_id][:] = []

    barrier.wait()


def epidemic_worker(rg_and_id, n, n_reps, barrier, mailboxes, param_tuple):
    """Simulate one town for ``n_reps`` replications of ``n`` days.

    ``param_tuple`` holds, in order: ``pop_size``, ``initial_infected``,
    ``death_prob``, ``contacts_per_day``, ``recovery_days``, ``n_migrants``,
    and a tuple of one ``infection_prob`` per town.

    Returns an ``(n_reps, n)`` integer array of daily infected counts.
    """
    rg, worker_id = rg_and_id

    pop_size = param_tuple[0]
    initial_infected = param_tuple[1]
    death_prob = param_tuple[2]
    contacts_per_day = param_tuple[3]
    recovery_days = param_tuple[4]
    n_migrants = param_tuple[5]
    infection_prob = param_tuple[6][worker_id]

    days = n

    n_cores = len(param_tuple[6])

    I_history = np.zeros((n_reps, days), dtype=int)

    for rep in range(n_reps):
        population, recovery_time, death_day = initialize_population(
            rg, pop_size, initial_infected, death_prob, recovery_days
        )

        for day in range(days):
            process_recoveries(population, recovery_time, day)

            process_deaths(population, recovery_time, death_day, day)

            exposed = select_exposures(rg, population, contacts_per_day)

            infect_subjects(
                rg,
                exposed,
                population,
                recovery_time,
                death_day,
                infection_prob,
                death_prob,
                recovery_days,
                day,
            )

            migration(
                rg,
                population,
                recovery_time,
                death_day,
                n_migrants,
                worker_id,
                n_cores,
                barrier,
                mailboxes,
            )

            I_history[rep, day] = population.count(1)

    return I_history


def epidemic_driver(
    seed,
    days,
    n_reps,
    n_cores,
    pop_size,
    initial_infected,
    death_prob,
    contacts_per_day,
    recovery_days,
    n_migrants,
    *infection_probs,
):
    """Run the epidemic across ``n_cores`` towns.

    ``infection_probs`` gives one infection probability per town, so it must
    have ``n_cores`` values. Returns a list with one ``(n_reps, days)`` array
    of daily infected counts per town.
    """
    infection_probs = tuple(infection_probs)
    if len(infection_probs) != n_cores:
        raise ValueError(
            f"expected {n_cores} infection probabilities (one per town), "
            f"got {len(infection_probs)}"
        )

    return sim_template_coupled(
        epidemic_worker,
        seed,
        days,
        n_reps,
        n_cores,
        pop_size,
        initial_infected,
        death_prob,
        contacts_per_day,
        recovery_days,
        n_migrants,
        infection_probs,
    )


def plot_epidemic(history, f_name=None, show=True):
    """Plot infected counts by day, one line per replication (Appendix A).

    ``history`` is one town's ``(n_reps, days)`` array, or the whole list
    returned by ``epidemic_driver``, in which case every town's
    replications are drawn. Requires matplotlib (``pip install
    simstreams[plot]``).
    """
    import matplotlib.pyplot as plt

    # convert list to array
    history = np.array(history)
    if history.ndim == 3:
        history = np.vstack(history)
    days = np.arange(history.shape[1])
    for i in range(history.shape[0]):
        plt.plot(days, history[i, :], alpha=0.2)

    plt.xlabel("Day")
    plt.ylabel("Infected")
    plt.title("Epidemic trajectories")
    if f_name is not None:
        plt.savefig(f_name, dpi=300)
    if show:
        plt.show()


__all__ = [
    "epidemic_driver",
    "epidemic_worker",
    "infect_subjects",
    "initialize_population",
    "migration",
    "plot_epidemic",
    "process_deaths",
    "process_recoveries",
    "select_exposures",
]
