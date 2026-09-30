"""Taxonomy of *silent inferential defects* - the population the AUDIT item form draws from.

A defect is **silent** when it leaves the analysis running and its output plausible: no exception, no
NaN, no failing test, no anomalous log line.  That is the whole point of the form.  A test-oracle bench
(SWE-bench) cannot pose such a task, because a defect that changes no test outcome is invisible to it;
a crash-bug bench (DSDBench) cannot, because nothing crashes.  What is left is the thing a research
engineer actually does when reviewing a colleague's analysis: read the inference, not the traceback.

Every entry is a mistake documented in the evaluation-methodology literature, so an audited script is
wrong in a way a real team has been wrong.

Each defect is described **generically**: what the analyst did, never where it is.  The list is shipped
to the agent as manual section 6, and the answer must name one of these ids, the line the wrong
inference is made on, and the corrected number.  `sign` records which way the defect moves the reported
number, and is used at build time to check that the injected defect is detectable at all (a defect whose
effect is below the noise floor is not an item).
"""

DEFECTS = {
    "D1_selection_max_over_seeds": {
        "name": "best-of-several reported as an estimate",
        "text": "The script runs a measurement several times (seeds, repetitions, sampling temperatures) "
                "and reports the best of them as if it were an estimate of the underlying quantity. The "
                "expected maximum of k draws exceeds the mean by roughly the noise scale times a factor "
                "growing with k, so the reported number is biased upwards by an amount that depends on how "
                "many times it looked.",
        "sign": "+",
        "grounding": "winner's curse / selection bias; E[max of k standard normals] = 0.5642 (k=2), "
                     "0.8463 (k=3), 1.4236 (k=8), 1.7660 (k=16), 2.3193 (k=64)",
    },
    "D2_denominator_scored_only": {
        "name": "unparseable answers dropped from the denominator",
        "text": "Items whose answer could not be extracted are excluded from the denominator instead of "
                "being counted as failures (or reported separately). Accuracy is then computed over the "
                "items the extractor happened to succeed on, which is not the set the claim is about, and "
                "the size of the error differs by model and by answer format.",
        "sign": "+",
        "grounding": "answer-extraction sensitivity: extraction failures are model- and format-dependent, "
                     "and a denominator choice is an analyst's decision, not a property of the eval",
    },
    "D3_aggregation_weight_mismatch": {
        "name": "per-slice means averaged with the wrong weights",
        "text": "A number reported as an overall rate is computed as the unweighted mean of per-slice rates "
                "(macro) where the claim requires the item-weighted rate (micro), or the reverse. The two "
                "differ whenever the slices differ in size, and they can move in opposite directions when "
                "slice difficulty is correlated with slice size.",
        "sign": "+/-",
        "grounding": "micro vs macro averaging; Simpson-type reversal requires differential slice weights",
    },
    "D4_position_bias_uncorrected": {
        "name": "judge preferences pooled over one presentation order",
        "text": "Pairwise judgements are collected with the two answers in a fixed order, or in both orders "
                "but pooled without checking the order effect, and the resulting win rate is reported as a "
                "quality comparison. The judge's preference for whichever answer it sees first is then "
                "added to the quality signal.",
        "sign": "+/-",
        "grounding": "LLM-judge position bias: GPT-4 51.3% vs 23.8% consistency by order; ChatGPT 2.5% vs 82.5%",
    },
    "D5_paired_test_as_unpaired": {
        "name": "a paired design analysed as if unpaired",
        "text": "Two systems are measured on the *same* items, but the comparison is made with the "
                "unpaired standard error (or two independent confidence intervals are checked for overlap). "
                "Item difficulty is shared variance that a paired analysis removes, so the reported "
                "uncertainty is far too large and a real difference is called undetermined.",
        "sign": "width",
        "grounding": "paired vs unpaired comparison on a fixed item bank; between-item variance dominates",
    },
    "D6_ratio_of_means_vs_mean_of_ratios": {
        "name": "ratio of means reported as the mean of ratios",
        "text": "A per-unit quantity (cost per token, tokens per second, speed-up per request) is computed "
                "by dividing one total by another, where the claim is about the typical per-unit value; or "
                "the reverse. The two agree only when the denominators are equal across units, so the "
                "reported number is silently dominated by the largest unit.",
        "sign": "+/-",
        "grounding": "ratio estimator bias; harmonic vs arithmetic mean of rates",
    },
    "D7_contaminated_items_included": {
        "name": "items known to be contaminated left in the reported set",
        "text": "The analysis has the evidence that some items are in a model's training data (a detector "
                "score, an occurrence count) and reports the headline number over all items anyway, or "
                "filters with a threshold on a noisy detector and reports the result as clean. The excess "
                "on the affected items is then attributed to ability.",
        "sign": "+",
        "grounding": "benchmark contamination; a cheap overlap score is neither a sound nor a complete test "
                     "for contamination of any single item",
    },
    "D8_threshold_on_noisy_max": {
        "name": "a threshold crossing declared from a noisy statistic",
        "text": "The script declares that a quantity has crossed a threshold (an emergence point, a "
                "regression, a stability limit) from a single noisy measurement, or from the first of "
                "several measurements to cross it. The crossing point of a noisy series is biased towards "
                "wherever the series is noisiest.",
        "sign": "+/-",
        "grounding": "emergence as a metric artifact: >92% of claimed emergent jumps appear only under "
                     "discontinuous metrics; 34/39 BIG-Bench metrics show none",
    },
    "D9_survivorship_dropped_diverged": {
        "name": "failed runs dropped from a comparison",
        "text": "Runs that diverged, timed out, were refused or returned nothing are dropped from the "
                "aggregate, and the mean over the survivors is reported as the configuration's outcome. "
                "Failure is not independent of the setting under study, so the setting with the most "
                "failures looks best.",
        "sign": "+/-",
        "grounding": "survivorship bias in hyperparameter sweeps: divergence probability rises with the "
                     "learning rate being compared",
    },
}

SIGNS = ("+", "-", "+/-", "width")


def taxonomy_md():
    """Manual section 6: the closed list the audit answer must name an id from."""
    L = ["## 6. Defect taxonomy", "",
         "An audited analysis contains **exactly one** defect from this list. Every defect here is "
         "*silent*: the script runs to completion and prints a plausible number. Name the id, not a "
         "description of your own.", ""]
    for d in sorted(DEFECTS):
        L.append("- **`%s`** - %s. %s" % (d, DEFECTS[d]["name"], DEFECTS[d]["text"]))
    L += ["", "Anything else you dislike about the script (naming, redundant work, a hard-coded path, a "
          "loop that could be vectorised, a comment that contradicts the code) is not a defect unless "
          "fixing it changes the reported number.", ""]
    return "\n".join(L)
