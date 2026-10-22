import numpy as np
import pytest
from numpy.random import SeedSequence, default_rng

from simtemplate import (
    benchmark_cores,
    make_generators,
    make_set_up,
    pi_worker,
    sim_template,
    sim_template_serial,
    split_reps,
)


def mean_worker(rg, n, n_reps, param_tuple):
    mu = param_tuple[0]
    x = rg.normal(loc=mu, size=(n, n_reps))
    return x.mean(axis=0)


def test_split_reps_divides_evenly_and_hands_out_the_remainder_first():
    assert split_reps(8, 4) == [2, 2, 2, 2]
    assert split_reps(1_000_003, 8) == [125001] * 3 + [125000] * 5
    assert sum(split_reps(1_000_003, 8)) == 1_000_003
    assert split_reps(3, 5) == [1, 1, 1, 0, 0]
    with pytest.raises(ValueError):
        split_reps(0, 4)
    with pytest.raises(ValueError):
        split_reps(10, 0)


def test_make_generators_spawns_from_the_master_seed_like_the_book():
    rg_list = make_generators(1234, 4)
    book = [default_rng(s) for s in SeedSequence(1234).spawn(4)]
    assert len(rg_list) == 4
    for ours, theirs in zip(rg_list, book, strict=True):
        assert ours.random(5).tolist() == theirs.random(5).tolist()


def test_make_set_up_matches_the_book_layout():
    set_up = make_set_up(1234, 100, 10, 4, 5, "extra")
    assert len(set_up) == 4
    rg, n, reps, args = set_up[0]
    assert rg.random() == make_generators(1234, 4)[0].random()
    assert n == 100
    assert reps == 3
    assert args == (5, "extra")
    assert [s[2] for s in set_up] == [3, 3, 2, 2]


def test_serial_equals_a_one_process_parallel_run():
    serial = sim_template_serial(pi_worker, 1234, 100, 20)
    parallel_1 = sim_template(pi_worker, 1234, 100, 20, 1)
    assert len(parallel_1) == 1
    assert serial.tolist() == parallel_1[0].tolist()


def test_results_are_reproducible_for_a_fixed_seed_and_core_count():
    a = sim_template(pi_worker, 42, 50, 40, 2)
    b = sim_template(pi_worker, 42, 50, 40, 2)
    assert [x.tolist() for x in a] == [x.tolist() for x in b]
    assert [len(x) for x in a] == [20, 20]


def test_a_user_worker_receives_the_parameters():
    out = sim_template(mean_worker, 7, 30, 10, 2, 100.0)
    assert [len(x) for x in out] == [5, 5]
    assert abs(np.concatenate(out).mean() - 100.0) < 0.5


def test_an_existing_pool_can_be_reused():
    from multiprocessing import Pool

    with Pool(2) as pool:
        a = sim_template(pi_worker, 1234, 100, 20, 2, pool=pool)
        b = sim_template(pi_worker, 1234, 100, 20, 2, pool=pool)
    c = sim_template(pi_worker, 1234, 100, 20, 2)
    assert [x.tolist() for x in a] == [x.tolist() for x in b]
    assert [x.tolist() for x in a] == [x.tolist() for x in c]


def test_worker_must_be_callable():
    with pytest.raises(TypeError):
        sim_template("pi_worker", 1, 10, 4, 2)
    with pytest.raises(TypeError):
        sim_template_serial("pi_worker", 1, 10, 4)


def run_pi(n_cores):
    return sim_template(pi_worker, 1234, 100, 200, n_cores)


def test_benchmark_cores_reports_speedup_and_efficiency():
    results = benchmark_cores(run_pi, [1, 2], repeats=2, warmup=0)
    assert [r["n_cores"] for r in results] == [1, 2]
    assert results[0]["speedup"] == 1.0
    for r in results:
        assert r["efficiency"] == r["speedup"] / r["n_cores"]
        assert r["median"] > 0
        assert len(r["times"]) == 2
