# Task: Audit the release headline

You are a research engineer in a model-evaluation team.  The team's notebook (`/app/notebook/`) holds earlier runs and the team's notes.  You have a budget on the team's simulated evaluation service (`lab` command; read `/app/manual.md` first - it defines the service, its guarantees and how answers are graded).

`/app/analysis/` holds a teammate's analysis script.  It reads the logs in `/app/notebook/`, and the number it printed is the one quoted in the team's notes.

Answer every question below by writing `/app/answers.json` (a template with the right keys is already there).  You may run any experiments within the budget.  Work carefully: the notes may be wrong, and some questions may not be fully answerable from anything the lab can measure - the manual says how to answer those.

## Questions

- **a1** `analysis/headline.py` prints one headline number.  Exactly one step of it draws a wrong inference from the data it reads.  Name the defect from the taxonomy in Section 6 of the manual, give the file and line where that inference is made, and give the number the script should have printed instead - the same quantity (`M_new`'s accuracy over the uncontaminated items of the bank, aggregated over slices by item count, under the release convention: an item whose answer could not be extracted counts wrong, over every item presented), computed correctly from the same rows in `notebook/` and given as an interval no wider than 0.003.  Everything else in the script is right.  Answer format: `{"defect": "one defect id from the taxonomy in manual.md Section 6", "site": "file:line", "corrected": {"lo": "number", "hi": "number"}}`
- **a2** The same corrected quantity as the **release report** will aggregate it: weight w0 on slice 0 and the remaining 1 - w0 split 2:1 between slices 1 and 2, over the documented range of w0.  Give the set of values it takes.  Use the rows in `notebook/` (this is a question about the evidence the team already has, not a request to re-measure the bank).  [unit: fraction] Answer format: `{"lo": "number", "hi": "number"}`
- **a3** The release rule in the memo is met: `M_new`'s corrected clean accuracy, aggregated as the release report will aggregate it, is at least 0.26.  Answer format: `{"verdict": "supported | refuted | undetermined"}`

When you are done, make sure `/app/answers.json` is valid JSON with one entry per question id.
