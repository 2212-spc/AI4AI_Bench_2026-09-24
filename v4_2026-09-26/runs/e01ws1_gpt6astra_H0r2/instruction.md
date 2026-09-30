# Task: Release gap: the denominator, the contamination, the error bar and the weighting of one eval report

You are a research engineer in a model-evaluation team.  The team's notebook (`/app/notebook/`) holds earlier runs and the team's notes.  You have a budget on the team's simulated evaluation service (`lab` command; read `/app/manual.md` first - it defines the service, its guarantees and how answers are graded).

Answer every question below by writing `/app/answers.json` (a template with the right keys is already there).  You may run any experiments within the budget.  Work carefully: the notes may be wrong, and some questions may not be fully answerable from anything the lab can measure - the manual says how to answer those.

## Questions

- **q1** Under the release convention (an item whose answer cannot be extracted counts wrong, over every item presented), `M_new`'s accuracy (a fraction in [0, 1]) on the items of the bank that do **not** occur in the pretraining corpus, format `mc_letter`, whole bank, noise-free expected value.  [unit: fraction] Answer format: `{"lo": "number", "hi": "number"}`
- **q2** The contamination premium of `M_new` under the release convention (an item whose answer cannot be extracted counts wrong, over every item presented): its accuracy over the whole bank minus its accuracy over the items that do not occur in the corpus (a difference of fractions, format `mc_letter`, noise-free expected values).  [unit: fraction] Answer format: `{"lo": "number", "hi": "number"}`
- **q3** The accuracy the **release report** will publish for `M_new` on the whole bank under the release convention (an item whose answer cannot be extracted counts wrong, over every item presented): the three slices aggregated with weight w0 on slice 0 and the remaining 1 - w0 split 2:1 between slices 1 and 2, format `mc_letter`.  Give the set of values this takes over the documented range of w0 (noise-free expected values).  [unit: fraction] Answer format: `{"lo": "number", "hi": "number"}`
- **q4** The standard deviation, across repeated calls, of the `acc` that one `svc=score model=M_new split=all fmt=mc_letter n=1500 reps=1` request reports.  [unit: fraction] Answer format: `{"lo": "number", "hi": "number"}`
- **q5** On the release report's own aggregate (weights as in q3), restricted to the items that do not occur in the pretraining corpus and under the release convention (an item whose answer cannot be extracted counts wrong, over every item presented), `M_new` scores higher than `M_ref`.  Answer format: `{"verdict": "supported | refuted | undetermined"}`

When you are done, make sure `/app/answers.json` is valid JSON with one entry per question id.
