# Task: an eviction policy for a key-value cache

You operate a shared in-memory cache in front of a slow key-value store. Many services use it at
the same time. Each object has size 1 and the cache holds `capacity` objects. Every request
that misses goes to the backend, so **we want as few misses as possible.** The cache currently
uses LRU. Replace it with a better policy.

## What you edit
In `solution.py`, implement

```python
class Policy:
    def __init__(self, capacity: int): ...
    def on_hit(self, key: str, t: int) -> None: ...
    def on_miss(self, key: str, t: int) -> str | None: ...
```

These are the rules. `simulator.py` uses the same rules as the grader.

- Requests arrive one at a time. `t` is the index of the request (0, 1, 2, ...).
- If `key` is in the cache, it is a hit and `on_hit(key, t)` is called.
- Otherwise it is a miss and `v = on_miss(key, t)` is called:
  - If the cache is not full, `key` is inserted and `v` is ignored.
  - If the cache is full, `v` must be one of these:
    - a key currently in the cache: it is evicted, and `key` is inserted;
    - `None`: bypass, meaning `key` is served but not cached.
  - Returning a key that is not in the cache is an error, and the setting then fails.
- Your policy only sees the requests up to the current one. The grader runs it in a separate
  process and sends request `t+1` only after it has received your answer to request `t`.

Only `solution.py` is graded. You may add helper files next to it (in the same directory).
Use plain Python, numpy, scipy, pandas or sklearn. No network. The total time limit is
**150 s per setting**; each setting is a trace of 150,000 requests. The grader's pipe adds
about 3 s of that. Memory use must stay reasonable (well under 2 GB).

## The traces
Keys follow the form `service:entity:field`, for example `user:48213:profile`,
`idx:shard7:page312`, or `sess:9f3a2c1b:cart`. An id can be a plain number, a number with a
prefix (`blk00042`), or a hex string. Each trace mixes the traffic of several clients whose
access patterns differ. The intensity and popularity of that traffic change over time.

- `data/dev_traces.pkl.gz` holds 3 dev traces. Load them with `gzip` + `pickle`. The file is a
  list of dicts with the keys `name`, `capacity`, `warm` and `keys` (a list of 150,000 strings).
- `baselines/` holds LRU (`solution.py` starts as a copy of it), LFU with dynamic aging, ARC and
  S3-FIFO, all written against the same interface.
- `eval_dev.py [file]` scores a policy on the dev traces. `simulator.py` is the local simulator.

## How it is graded
For each setting, the score is:

    score = (miss_LRU - miss_policy) / (miss_LRU - miss_OPT)

- The miss rates count only the requests after the first `warm` = 30,000. Your policy still
  runs during the warm-up.
- OPT is Belady's offline optimum with bypass. It knows the future, so it gives an upper bound.
- A score of 0 means you are as good as LRU, and 1 means you are optimal.

**Pass = score >= 0.35 on every setting.** The graded settings are the 3 dev traces plus
**4 hidden traces.** The hidden traces come from other deployments. Their service, entity and
field names differ from the dev traces, and so do their cache capacities (hundreds to a few
thousand) and their mix of client behaviours. A policy that recognises the dev key names will
not transfer.

Also write a short `NOTES.md`. Explain what structure your policy exploits, and why you expect
it to work on traces you have not seen.
