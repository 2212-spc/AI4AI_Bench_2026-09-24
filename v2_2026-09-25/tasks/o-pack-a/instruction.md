# Make the training-data packer fast without changing a single packed batch

`/app/pack_ref.py` is the reference sequence packer used by MiniStack's data loader. It is the
specification: whatever `pack(lengths, params)` returns for a given input is by definition the correct
packing. `/app/params.json` holds the parameters of this deployment. `/app/sample/` contains one small
input and the packing the reference produces for it, so you can sanity-check your work.

The reference is too slow: at production scale (hundreds of thousands of sequences per shard) it takes
minutes per shard and has become the bottleneck of the data pipeline.

Write `/app/solution.py` exposing

    def pack(lengths: list[int], params: dict) -> list[int]

that returns **exactly** the same bin id for every sequence as `pack_ref.py` does, for every input, and
that is fast enough to pack a 400,000-sequence shard well inside the time limit below.

Rules:
* `/app/solution.py` must be self-contained: it may import the Python standard library and numpy, but it
  must not import, exec, read or otherwise call `pack_ref.py` (a copy of it does not help - it is the slow
  implementation you are replacing).
* Your `pack` will be called in a fresh process, several times, with inputs you have not seen, including
  inputs specifically built around the corner cases of the reference rules.
* Grading is all-or-nothing: every hidden input must match the reference output element for element,
  and the 400,000-sequence shard must be packed in under 12 seconds (measured on a machine
  calibrated against the authoring host; the reference implementation needs roughly 47 s there).

Deliverable: `/app/solution.py`. Also write `/app/NOTES.md` listing the behavioural details of the
reference that your implementation had to preserve.


You have 5400 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
