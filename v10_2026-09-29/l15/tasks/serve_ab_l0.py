"""SERVE-AB level L0: the same task text, but the pool never saturates (peak load at 100% B stays below the
autoscaler target at the quota on a +2.5 sd day).  The plain A/B answer is right: ship B everywhere.  The
bottom rung of the ladder - frontier models should pass it; it certifies that failures at L1/L2 come from
the mechanism, not from the task format."""
from .serve_ab import *            # noqa: F401,F403
from . import serve_ab as _b

LEVEL = "L0"


def sample_params(seed):
    return _b._sample(seed, LEVEL)


def pool_gate(seeds):
    return _b._pool([sample_params(s) for s in seeds])


class World(_b.World):
    NAME = "serve_ab_l0"


STRATEGIES = {"oracle": (_b.strat_oracle, "pass"),
              "ship_all": (_b.strat_ship_all, "pass"),
              "guard_safe": (_b.strat_guard_safe, "fail"),
              "no_ship": (_b.strat_no_ship, "fail"),
              "wide": (_b.strat_wide, "fail"),
              "nop": (_b.strat_nop, "fail"),
              "bad_format": (_b.strat_bad_format, "fail")}
SEARCH = []
PRINCIPLES = {"P2b_validated_fraction_is_not_the_launch": ("guard_safe", ["R1_schedule"]),
              "P4_partial_launch_beats_none": ("no_ship", ["R1_schedule"])}
