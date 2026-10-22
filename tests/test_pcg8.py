import pytest

from simtemplate import PCG8, rotr8


def test_rotr8_rotates_within_eight_bits():
    assert rotr8(0b00000001, 1) == 0b10000000
    assert rotr8(0b10010110, 4) == 0b01101001
    assert rotr8(0b10010110, 8) == 0b10010110
    assert rotr8(0b10010110, 0) == 0b10010110


def test_inc_must_be_odd():
    with pytest.raises(ValueError):
        PCG8(seed=1, inc=2)


def test_state_advances_by_the_congruential_rule():
    g = PCG8(seed=42, inc=77, mult=45)
    g.random()
    assert g.state == (45 * 42 + 77) % 256
    g.random()
    assert g.state == (45 * 175 + 77) % 256


def test_full_period_and_reproducibility():
    g = PCG8(seed=42, inc=77, mult=45)
    states = [g.state]
    for _ in range(255):
        g.random()
        states.append(g.state)
    # a full-period 8-bit LCG visits every state once before repeating
    assert sorted(states) == list(range(256))
    g.random()
    assert g.state == states[0]

    a = PCG8(seed=7, inc=1)
    b = PCG8(seed=7, inc=1)
    assert [a.random() for _ in range(20)] == [b.random() for _ in range(20)]
