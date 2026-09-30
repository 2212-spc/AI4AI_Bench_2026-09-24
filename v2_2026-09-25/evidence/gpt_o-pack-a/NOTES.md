# Reference behavior preserved

`solution.pack(lengths, params)` preserves these details from `pack_ref.py`:

- Bins receive IDs in creation order, starting at zero. The result contains the bin ID for each input position.
- A length `L` is eligible for ordinary packing exactly when `L <= capacity`. The first eligible open bin in creation order whose remaining capacity is at least `L` receives it. If none qualifies, a new bin is created and receives the sequence.
- After ordinary placement, the bin's remaining capacity is reduced by `L`. A bin is closed immediately when its remaining capacity is strictly less than `close_below`; equality stays open.
- If the number of open bins exceeds `max_open`, the open bin with the smallest remaining capacity is closed. Ties are resolved by the smaller bin ID, which is also the earlier creation order.
- At each input position whose one-based index is divisible by `flush_every`, if at least two bins remain open, the oldest open bin is closed. This check occurs after placement and the other closing rules.
- A length greater than `capacity` creates a bin containing only that sequence. Its remaining capacity is zero, it is never open for later packing, and it does not trigger either the `max_open` or periodic flush rule. It still advances the input position used by `flush_every`.
- A length equal to `capacity` is an ordinary placement and closes immediately because its remaining capacity is zero when the configured threshold is positive.
- The implementation keeps the reference's behavior for zero and negative lengths: they are ordinary eligible placements when `L <= capacity`, and subtracting them can increase a bin's remaining capacity.

For speed, the implementation stores the open bins' remaining capacities in a segment tree. Its root identifies whether any bin fits, and descending the tree finds the first fitting bin by bin ID. A lazy min-heap is used only after the open-bin limit has been exceeded; heap entries are checked against the current remaining capacity before use so updates and closures remain exact.
