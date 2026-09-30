"""Lab backends.  A *lab* is a simulated service the agent can query: it defines its own knobs, its own
cost unit, its own observables and its own mechanism cards.  `scalelab.lab.Session` is generic and
dispatches every request to the backend named by `spec["lab"]` (default "pretrain").

A backend module must define:

  NAME        str                              value of spec["lab"]
  COST_UNIT   str                              e.g. "FLOPs", "eval credits", "GPU-hours"
  COST_TEXT   str                              one line: how a request's cost is computed
  BASE        dict                             every world parameter the backend reads, at neutral values
  full(p)     -> dict                          BASE updated with p
  check(sess, cfg)                             raise LabError on a combination the service refuses
  extras(sess, req) -> dict                    consume non-knob request fields (mutates req); {} if none
  cost(sess, cfg, extra) -> float
  execute(sess, cfg, seed, extra) -> dict      the result body ("config"/"status" included, no cost fields)
  MANUAL      str                              manual sections 1-2 (what is modelled, guarantees)
  CLI_HELP    str                              manual section 4 (how to drive the service)
"""
import importlib

_LOADED = {}
NAMES = ("pretrain", "evallab", "rllab", "servelab")


def backend(name):
    name = name or "pretrain"
    if name not in _LOADED:
        if name not in NAMES:
            raise KeyError("unknown lab %r (known: %s)" % (name, ", ".join(NAMES)))
        _LOADED[name] = importlib.import_module("scalelab.labs." + name)
    return _LOADED[name]
