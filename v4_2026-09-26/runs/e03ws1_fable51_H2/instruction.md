# Task: Audit the release headline

You are a research engineer in a model-evaluation team.  The team's notebook (`/app/notebook/`) holds earlier runs and the team's notes.  You have a budget on the team's simulated evaluation service (`lab` command; read `/app/manual.md` first - it defines the service, its guarantees and how answers are graded).

`/app/analysis/` holds a teammate's analysis script.  It reads the logs in `/app/notebook/`, and the number it printed is the one quoted in the team's notes.

Answer every question below by writing `/app/answers.json` (a template with the right keys is already there).  You may run any experiments within the budget.  Work carefully: the notes may be wrong, and some questions may not be fully answerable from anything the lab can measure - the manual says how to answer those.

## Questions

- **a1** `analysis/headline.py` prints one headline number.  Exactly one step of it draws a wrong inference from the data it reads.  Name the defect from the taxonomy in Section 6 of the manual, give the file and line where that inference is made, and give the number the script should have printed instead - the same quantity (`M_new`'s accuracy over the uncontaminated items of the bank, aggregated over slices by item count, under the release convention: an item whose answer could not be extracted counts wrong, over every item presented), computed correctly from the same rows in `notebook/` and given as an interval no wider than 0.003.  Everything else in the script is right.  Answer format: `{"defect": "one defect id from the taxonomy in manual.md Section 6", "site": "file:line", "corrected": {"lo": "number", "hi": "number"}}`
- **a2** The same corrected quantity as the **release report** will aggregate it: weight w0 on slice 0 and the remaining 1 - w0 split 2:1 between slices 1 and 2, over the documented range of w0.  Give the set of values it takes.  Use the rows in `notebook/` (this is a question about the evidence the team already has, not a request to re-measure the bank).  [unit: fraction] Answer format: `{"lo": "number", "hi": "number"}`
- **a3** The release rule in the memo is met: `M_new`'s corrected clean accuracy, aggregated as the release report will aggregate it, is at least 0.31.  Answer format: `{"verdict": "supported | refuted | undetermined"}`

When you are done, make sure `/app/answers.json` is valid JSON with one entry per question id.

## Hint (level H2)

The mechanisms active in this lab are exactly the following (all others are switched off):

- **Two-parameter item response**: `P(correct | model m, item j) = sigmoid(a_j*(theta_m + off_m[slice(j)] - b_j)); item information I_j = a_j^2*P*(1-P); SE(theta) = 1/sqrt(sum_j I_j)`
- **Format sensitivity and answer-extraction failure**: `per-format ability offset fmt[m][f]; extraction fails independently per item with rate fmt_ext[f]['by_slice'][slice(j)]; a failed extraction is scored wrong under one denominator convention and dropped under the other`
- **Contamination through duplicated items**: `item j occurs dup_j times in the corpus (dup_frac of the bank has dup_j ~ 1+Poisson(dup_lam)); a contaminated item gets a memorisation boost rising with dup_j and with the model's kappa`
- **Between-call variance and the resulting detectable effect size**: `reported accuracy = truth + noise, Var = (sigma_between^2 + sigma_within^2/K)/n_items with K repetitions; sigma_within^2 = mean p(1-p); sigma_between from an ability jitter sig_call applied per call`
- **Slice mixture and aggregation convention (Simpson)**: `reported = sum_g w_g * p_g; micro weights w_g proportional to slice size, macro weights uniform; the scored denominator may be attempted items or extracted items`
- **n-gram contamination detector with calibration error**: `detector score for item j = sigmoid(det_b + det_a*log(1+dup_j) + det_s*z_j), z_j fixed per item; an exact corpus scan returns dup_j itself at a much higher price`

The functional forms above are exact; their constants are unknown and must be measured.
