"""Text shipped to the agent.

The manual is assembled per lab.  Sections 1-2 (what the service models, its guarantees) and section 4
(how to drive it) belong to the **lab backend**: they describe that service's abstraction boundary, and a
task built on RLLab must not ship pretraining prose.  Section 3 (how answers are graded) belongs here,
because grading semantics are a property of the *item forms*, not of the world; a block is emitted only
for the forms the task actually uses.  Section 5 (this lab's knobs, caps, known unknowns) is appended by
export.py, and section 6 (the defect taxonomy) only when the task contains an audit item.

Nothing in sections 1-4 depends on which cards are active: each backend's manual lists every effect its
service *can* contain and says that any given lab may switch some off.  The active card set appears only
in the hint ladder.
"""
from .cards import CARDS
from .defects import taxonomy_md
from . import labs

LAB_NAME = {"pretrain": "ScaleLab", "evallab": "EvalLab", "rllab": "RLLab", "servelab": "ServeLab"}

GRADE_HEAD = """## 3. How answers are graded

Every question is graded mechanically against this lab's hidden world.  Nothing you write outside the
answer file is read, and there is no credit for an explanation of a wrong answer.
"""

NUMERIC = r"""- **Numeric questions** take an interval `{"lo": x, "hi": y}`.  The interval must be the *set of values
  consistent with everything that can be known*: what the lab can measure within your budget, plus the
  guarantees and documented ranges in this manual.
  - If the lab pins a quantity down, answer a degenerate interval (`lo == hi` = your estimate).  Seed
    noise is handled by the grading tolerance, so **do not** widen for statistical uncertainty.
  - If the quantity depends on a known unknown (Section 5), `lo`/`hi` are the minimum and maximum over
    the values of that unknown that are consistent with its documented range **and** with what the
    lab can measure.
  - Each endpoint is compared with the true endpoint.  The tolerance is calibrated to how precisely a
    well-designed set of experiments within the budget can pin that endpoint down."""

VERDICT = r"""- **Verdicts** take `{"verdict": "supported" | "refuted" | "undetermined"}`.  *supported*: true in every
  world consistent with what can be known; *refuted*: false in every such world; *undetermined*: true
  in some and false in others (i.e. it hinges on a known unknown)."""

DECISION = r"""- **Decisions** take `{"choice": "<option>"}`: pick the option with the best outcome as defined in the question."""

CEX = r"""- **Counterexample questions** ask whether a stated claim is *forced* by the evidence, and take
  `{"verdict": "refutable" | "entailed", "witness": {...}}`.
  - The question lists a few **free parameters**, each with a range.  A *witness* gives a value for every
    one of them and for nothing else: the witness world is this lab's own world with exactly those
    parameters moved, and everything else left as it is.
  - Answer `refutable` **with a witness** if some assignment both (a) reproduces the evidence and (b)
    makes the claim false.  Answer `entailed`, with no witness, if no assignment can do both - that is,
    if the claim follows from the evidence no matter how the free parameters are set.
  - (a) is checked against **every notebook row and every run you have made**.  Each observable
    contributes a standardised deviation `z = (observed - predicted) / sd`; the witness passes iff every
    `|z| <= 3` and the sum of `z^2` is below the 0.999 quantile of chi-square with one degree of freedom
    per observable.  A quantity the world fixes *exactly* - a count, a deterministic detector reading -
    must match to six decimals.  Running more experiments therefore constrains the witness more, not less.
  - (b) is checked by recomputing the claim in the witness world.
  - There is no partial credit: a witness that reproduces the evidence but leaves the claim true fails,
    as does one that falsifies the claim but does not reproduce the evidence.  A value outside a declared
    range, a missing parameter or an extra parameter is rejected unread."""

PREREG = r"""- **Plan questions** ask for the experiment you *would* run and the rule you would apply to its result,
  and are graded on the **plan**, not on a number.  The answer for such a question is the plan object
  itself:

        {"runs": [{"label": "r1", ...a lab request...}, ...],
         "rule": {"expr": "r1.acc - r2.acc",
                  "cuts": [[">", 0.05, "A"], ["<", -0.05, "B"], ["else", "undetermined"]]}}

  - Each run is an ordinary lab request (the same fields `lab run` takes).  `label` is a name you choose
    for referring to that run's output in the rule; it is not a knob.  The plan must stay within the run
    count and budget the question states - that budget is separate from your own exploration budget, and
    a plan that exceeds it fails without being executed.
  - `expr` is arithmetic over `<label>.<field>`, where `<field>` is any numeric field of that run's
    result.  Repetition accuracies are `<label>.acc_r0`, `acc_r1`, ...; per-slice fields are
    `<label>.<field>_s0`, `_s1`, ...; per-order fields are `<label>.<field>_<order>`.  Allowed:
    `+ - * / **`, parentheses, numeric constants, and `abs min max sqrt log exp mean`.  Nothing else
    parses - no attribute chains, no comprehensions, no calls to anything unlisted.
  - `cuts` are tried in order and the first match wins; each is `[op, threshold, label]` with `op` one of
    `> >= < <=`.  The last cut must be `["else", label]`: the rule has to name a conclusion for every
    value the statistic could take.  Labels must come from the set the question lists.
  - **Grading.**  The plan is executed by the grader in *every world consistent with what was disclosed*
    - the true one and others that fit the notebook equally well - each under several independent noise
    salts.  It passes only if the rule returns that world's correct conclusion **every** time.  A rule
    that is right in the world you happen to be in and flips in another consistent world fails; so does
    one whose runs do not buy enough precision to separate the cases at the budget given.  Size the runs
    from the noise model in Section 1 before choosing the thresholds."""

AUDIT = r"""- **Audit questions** ship a teammate's analysis script under `/app/analysis/`.  It reads the logs in
  `/app/notebook/`, and the number it printed is the one quoted in the team's notes.  Exactly one
  **silent inferential defect** has been introduced: the script runs to completion and prints a
  plausible number, and the number is wrong.
  Answer `{"defect": "<id>", "site": "<file>:<line>", "corrected": {"lo": x, "hi": y}}`.
  - `defect` is one id from the taxonomy in Section 6.
  - `site` is the line where the wrong inference is made, not a line that merely prints or returns the
    result.  A small span around that line is accepted.
  - `corrected` is an interval containing what the analysis gives once the defect is fixed and nothing
    else is changed; the question states how wide it may be.  Recompute it from the shipped logs - the
    logs are the team's real data, and the lab is not needed for this part.
  - All three parts must be right.  The script also contains choices that are merely ugly, redundant or
    unusual; those are not the defect unless fixing them moves the reported number."""

BLOCKS = {"numeric": NUMERIC, "verdict": VERDICT, "decision": DECISION,
          "cex": CEX, "prereg": PREREG, "audit": AUDIT}
KIND_BLOCK = {"point": "numeric", "set": "numeric", "verdict": "verdict", "decision": "decision",
              "cex": "cex", "prereg": "prereg", "audit": "audit"}
ORDER = ("numeric", "verdict", "decision", "cex", "prereg", "audit")


def blocks_for(kinds=None):
    """Which section-3 blocks a task needs.  None -> all of them (used by the leakage scanner)."""
    if kinds is None:
        return list(ORDER)
    want = {KIND_BLOCK[k] for k in kinds if k in KIND_BLOCK}
    return [b for b in ORDER if b in want]


def grading_text(kinds=None):
    return "\n".join([GRADE_HEAD] + [BLOCKS[b] for b in blocks_for(kinds)]) + "\n"


def manual_core(lab, kinds=None):
    """Header + sections 1-4, assembled from the lab backend plus the grading blocks this task needs."""
    be = labs.backend(lab)
    return "\n".join(["# %s manual" % LAB_NAME.get(be.NAME, be.NAME), "",
                      be.MANUAL, grading_text(kinds), be.CLI_HELP])


def manual_tail(kinds=None):
    """Section 6, present only when the task contains an audit item."""
    return taxonomy_md() if (kinds is None or "audit" in set(kinds)) else ""


def hint_text(level, cards):
    if level == 0:
        return ""
    lines = ["## Hint (level H%d)" % level, "",
             "The mechanisms active in this lab are exactly the following (all others are switched off):", ""]
    for c in cards:
        cd = CARDS[c]
        lines.append("- **%s**" % cd["name"] + (": `%s`" % cd["form"] if level >= 2 else ""))
    if level >= 2:
        lines += ["", "The functional forms above are exact; their constants are unknown and must be measured."]
    return "\n".join(lines) + "\n"
