"""Pin the package to the numbers printed in the book.

The values come from the book repository's verify/verify.py, so a change to
the template or a worker that alters a published result fails here.
"""

import numpy as np

from simstreams import cauchy_worker, mailbox_demo, pi_worker, sim_template


def test_chapter_4_cauchy_sim_template_1234_100_20_4_5():
    out = sim_template(cauchy_worker, 1234, 100, 20, 4, 5)
    assert len(out) == 4
    got = [[f"{v:.8f}" for v in row] for row in out]
    assert got == [
        ["5.62082898", "5.95482920", "5.25999085", "4.61180851", "5.32471512"],
        ["4.69824636", "5.99497178", "6.15112504", "5.02016374", "5.01611561"],
        ["5.35159213", "6.22737780", "4.56187580", "5.25657940", "5.83274694"],
        ["4.80614598", "5.43726919", "4.91259206", "5.42029505", "5.47352615"],
    ]
    # Appendix A: np.round(out, 3)
    assert np.round(out, 3).tolist() == [
        [5.621, 5.955, 5.26, 4.612, 5.325],
        [4.698, 5.995, 6.151, 5.02, 5.016],
        [5.352, 6.227, 4.562, 5.257, 5.833],
        [4.806, 5.437, 4.913, 5.42, 5.474],
    ]


def test_chapter_4_pi_sim_template_1234_100_1e6_4():
    out = sim_template(pi_worker, 1234, 100, 1_000_000, 4)
    assert [len(x) for x in out] == [250_000] * 4
    assert f"{np.concatenate(out).mean():.8f}" == "3.14141632"


def test_chapter_6_mailbox_demo():
    assert mailbox_demo() == {0: [200], 1: [2, 3, 7], 2: [2, 3, 7]}
