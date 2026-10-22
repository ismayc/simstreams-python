"""A toy 8-bit permuted congruential generator (Appendix C).

PCG8 advances an 8-bit linear congruential state and then permutes the old
state with an xorshift and a rotation chosen by the state's top bits. It has
only 256 states, so its entire cycle can be enumerated, which is what makes
it useful for seeing how the full PCG64 generator works.
"""


def rotr8(x, r):
    """Rotate the 8-bit value ``x`` right by ``r`` bits."""
    r %= 8
    return ((x >> r) | (x << (8 - r))) & 0xFF


class PCG8:
    """Toy 8-bit PCG generator.

    Parameters
    ----------
    seed : int
        Initial state, reduced to 8 bits.
    inc : int
        Increment of the congruential step. Must be odd.
    mult : int
        Multiplier of the congruential step.
    """

    def __init__(self, seed=0, inc=1, mult=141):
        self.state = seed & 0xFF
        self.inc = inc & 0xFF
        self.mult = mult & 0xFF

        if self.inc % 2 == 0:
            raise ValueError("inc must be odd")

    def random(self):
        """Advance the state and return the next 8-bit output."""
        oldstate = self.state
        self.state = (self.mult * oldstate + self.inc) & 0xFF
        xorshifted = ((oldstate >> 3) ^ oldstate) & 0xFF
        rot = (oldstate >> 5) & 0x7
        return rotr8(xorshifted, rot)


__all__ = ["PCG8", "rotr8"]
