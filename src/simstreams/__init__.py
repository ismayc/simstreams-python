"""simstreams: parallel simulation experiments on a personal computer.

The package generalizes the ``sim_template`` program developed in the book
*Parallel Simulation Experiments on a Personal Computer* by R. L. Eubank and
Chester Ismay (CRC Press). The book's template reads its worker with
``from sim_worker import worker``; here the worker is an argument, so any
number of workers can live side by side, and the pieces of the template are
available on their own.

Main functions:

- :func:`sim_template` runs a worker across ``n_cores`` processes with one
  independent PCG64 stream per process.
- :func:`sim_template_serial` runs the same experiment in the current process.
- :func:`make_generators`, :func:`split_reps`, and :func:`make_set_up` are the
  building blocks, for writing your own driver.
- :func:`pi_worker` and :func:`cauchy_worker` are the book's Chapter 4
  workers.
- :func:`sim_template_coupled` runs workers that communicate through shared
  mailboxes, and :mod:`simstreams.epidemic` is the Chapter 6 case study
  built on it.
- :func:`benchmark` and :func:`benchmark_cores` time a simulation across core
  counts.
- :class:`PCG8` is the toy generator of Appendix C.

The companion R package is ``simstreams`` on GitHub
(https://github.com/ismayc/simstreams). The book's code repository is
https://github.com/ismayc/parallel-simulation-book, and the web edition is
at https://parallel-simulation-book.netlify.app.
"""

from .benchmark import benchmark, benchmark_cores
from .coupled import mailbox_demo, sim_template_coupled
from .epidemic import epidemic_driver, epidemic_worker
from .pcg8 import PCG8, rotr8
from .template import (
    make_generators,
    make_set_up,
    sim_template,
    sim_template_serial,
    split_reps,
)
from .workers import cauchy_worker, pi_worker

__version__ = "0.1.0"

__all__ = [
    "PCG8",
    "__version__",
    "benchmark",
    "benchmark_cores",
    "cauchy_worker",
    "epidemic_driver",
    "epidemic_worker",
    "mailbox_demo",
    "make_generators",
    "make_set_up",
    "pi_worker",
    "rotr8",
    "sim_template",
    "sim_template_coupled",
    "sim_template_serial",
    "split_reps",
]
