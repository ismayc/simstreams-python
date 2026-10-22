# simstreams <img src="branding/logo.png" align="right" height="138" alt="simstreams hex sticker: four braided random-number streams, plum and gold" />

[![Tests](https://github.com/ismayc/simstreams-python/actions/workflows/test.yml/badge.svg)](https://github.com/ismayc/simstreams-python/actions/workflows/test.yml)

Run simulation experiments in parallel on a multicore personal computer.

`simstreams` generalizes the `sim_template` program developed in the book
*Parallel Simulation Experiments on a Personal Computer* by R. L. Eubank and
Chester Ismay (CRC Press). The book's template reads its worker with
`from sim_worker import worker`. In the package the worker is an argument, so
any number of workers can live side by side, and the pieces of the template
are available on their own:

- `sim_template()` runs a worker across `n_cores` processes, each with its own
  independent PCG64 stream from a spawned `SeedSequence`.
- `sim_template_serial()` runs the same experiment in the current process, for
  debugging a worker and for the one-core baseline of a timing study.
- `make_generators()`, `split_reps()`, and `make_set_up()` are the building
  blocks: the per-process generators, the even division of the replications,
  and the argument tuple each process receives.
- `pi_worker()` and `cauchy_worker()` are the book's example workers
  (Chapter 4).
- `sim_template_coupled()` runs workers that communicate through shared
  mailboxes and a barrier, and `simstreams.epidemic` is the Chapter 6 SIRD
  case study with migration built on it.
- `benchmark()` and `benchmark_cores()` time a simulation across core counts
  and report the speed-up (Chapter 7).
- `PCG8` is the toy 8-bit generator of Appendix C.

The companion R package is
[simstreams](https://github.com/ismayc/simstreams). The book's code
repository is
[parallel-simulation-book](https://github.com/ismayc/parallel-simulation-book),
and the web edition is at
<https://parallel-simulation-book.netlify.app>.

## Installation

```
pip install git+https://github.com/ismayc/simstreams-python
```

Plotting the epidemic needs matplotlib: add `[plot]`, as in
`pip install "simstreams[plot] @ git+https://github.com/ismayc/simstreams-python"`.

## Usage

The call mirrors the book: the seed, the sample size for each replication,
the number of replications, the number of processes, and then any model
parameters. The one difference is that the worker comes first.

```python
import numpy as np
from simstreams import sim_template, pi_worker, cauchy_worker

if __name__ == "__main__":
    # Chapter 4: estimate pi with 1,000,000 replications of 100 darts on 4 cores
    out = sim_template(pi_worker, 1234, 100, 1_000_000, 4)
    print(np.concatenate(out).mean())
    # 3.14141632

    # Chapter 4: the Cauchy scale estimator, with theta = 5 as a model parameter
    out = sim_template(cauchy_worker, 1234, 100, 20, 4, 5)
    print(np.round(out, 3))
    # [[5.621 5.955 5.26  4.612 5.325]
    #  ...
```

Those are the numbers printed in the book. The package's tests pin them, so
a change that alters a published result fails the test suite.

The `if __name__ == "__main__":` guard matters on macOS and Windows, where
child processes are spawned by importing the calling script.

### Writing your own worker

A worker is a function `worker(rg, n, n_reps, param_tuple)`. It draws from
`rg`, the process's own generator, generates the data for its `n_reps`
replications, computes the quantity of interest, and returns it. Define it at
module level so the child processes can import it, and keep it
self-contained.

```python
def mean_worker(rg, n, n_reps, param_tuple):
    mu = param_tuple[0]                      # model parameters arrive as a tuple
    x = rg.normal(loc=mu, size=(n, n_reps))  # replications in columns
    return x.mean(axis=0)                    # one estimate per replication

if __name__ == "__main__":
    out = sim_template(mean_worker, 2026, 50, 10_000, 4, 10.0)
    print(np.concatenate(out).std())
```

### The epidemic case study

```python
from simstreams import epidemic_driver

if __name__ == "__main__":
    # 4 towns, 30 days, 20 replications, 10 migrants a day, one infection
    # probability per town
    out = epidemic_driver(1234, 30, 20, 4, 1000, 20, 0.05, 5, 3, 10,
                          0.05, 0.15, 0.25, 0.30)
    print([h.max() for h in out])   # peak daily infections in each town
```

### Timing a speed-up experiment

```python
from simstreams import benchmark_cores

def run_pi(n_cores):
    return sim_template(pi_worker, 1234, 100, 1_000_000, n_cores)

if __name__ == "__main__":
    for r in benchmark_cores(run_pi, range(1, 9), repeats=10):
        print(r["n_cores"], round(r["median"], 3), round(r["speedup"], 2))
```

## Reproducibility

A fixed `seed` reproduces a run exactly on the same number of cores. Changing
`n_cores` changes how the replications are divided across the streams, so the
full quartet `seed`, `n`, `n_reps`, and `n_cores` is what reproduces a result.

## Development

```
uv sync --group dev
uv run pytest
```

The test suite covers every line, and `pytest` fails below 100% (the gate is
in `pyproject.toml`). The parallel templates run their workers in child
processes, where coverage is not recorded, so `tests/test_inprocess.py` calls
the same worker functions directly.

## License

MIT. See [LICENSE](LICENSE).
