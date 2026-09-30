# Design-doc claims about `/app/dataprep`

Six claims the data team wrote down about the pipeline. Each is either true of the code as shipped, or
false. A claim is false only if some input **inside its stated scope** violates it.

### V1

dedup.dedup: for any two documents of at least 300 and at most 4000 whitespace tokens whose 5-token-window sets have Jaccard similarity at least 0.80, dedup never keeps both.

### V2

decontam.filter_eval: no evaluation document returned by filter_eval shares a 13-token window (whitespace tokens, exact string match, no normalisation) with any training document.

### V3

shard.shard: for any n in [1, 10^6] and world_size in [1, 16] with drop_last=False, the assignments over all ranks contain every index in [0, n) exactly once.

### V4

pack.pack: no emitted batch has a total token count above max_tokens, where a document longer than max_tokens counts as max_tokens.

### V5

shuffle_buffer.shuffle_stream: the emitted sequence is a permutation of the input items for any buffer_size >= 1.

### V6

sampler.epoch_order: for any n in [1, 10^6], any seed and any epoch, the returned order contains every index in [0, n) exactly once.

## Witness schemas

A witness is a JSON file. One object per claim, exactly these fields:

* **V1** `{"docs": [str, str]}  - two documents, 300..4000 whitespace tokens each, printable ASCII`
* **V2** `{"train": [str, ...], "eval": [str, ...]}  - 1..20 documents per side, <=2000 tokens each`
* **V3** `{"n": int, "world_size": int, "drop_last": false}  - 1<=n<=10^6, 1<=world_size<=16`
* **V4** `{"lengths": [int, ...], "max_tokens": int}  - <=5000 lengths, each 0..10^6, max_tokens>=1`
* **V5** `{"items": [int, ...], "buffer_size": int, "seed": int}  - <=5000 items, buffer_size>=1`
* **V6** `{"n": int, "seed": int, "epoch": int}  - 1<=n<=10^6, 0<=seed<2^31, 0<=epoch<=1000`
