"""ScaleLab mechanism-card library.

A *card* is one grounded mechanism: a functional form taken from the literature, the parameters that
form needs, the value the literature reports ("textbook"), the range the world generator may draw from
("world"), and the *neutral* value that switches the card off.  Every task activates a subset of cards;
inactive cards sit at their neutral value so the single world function for that lab stays valid.

The world is deliberately NOT a replica of real training.  Forms are taken from published fits; the
parameters are redrawn inside ranges that bracket (and sometimes exceed) the published values, so a
model that answers from memorised numbers is wrong while a model that measures is right.  What the
agent is told about this boundary is in the per-lab manual (`scalelab/labs/<lab>.py: MANUAL`).

Cards are named by lab: C* = pretrain, E* = evallab, R* = rllab, S* = servelab.  A card's `params` keys
are the keys of that lab's parameter dict, so `common.draw_card` output can be merged straight in.

`textbook=None` marks a constant that the source reports **only as a figure**, or not at all.  Those
are never quoted to the agent and never used as a plausibility check: the range is chosen to bracket
the qualitative shape (sign, monotonicity, order of magnitude) that the source does establish.  This
distinction is load-bearing -- inventing a "published value" for a figure-only constant is the single
easiest way to ship a wrong item.

`DERIVATIONS` at the bottom holds regularities that are *not* world parameters: identities and
estimators the agent has to apply to get from what it can measure to what is asked.  A blueprint
declares which of them its answer path uses; the longest chain is the task's depth (gate G11).
"""

CARDS = {
    "C1": dict(
        name="Chinchilla-type loss surface",
        form="L = E + A*N^-alpha + Bc*Deff^-beta   (N = non-embedding params, Deff = effective tokens)",
        grounding=["Hoffmann et al. 2022 (arXiv:2203.15556): E=1.69 A=406.4 B=410.7 alpha=0.34 beta=0.28",
                   "Besiroglu et al. 2024 (arXiv:2404.10102) replication: E=1.82 A=482 B=2085 alpha=0.35 beta=0.37"],
        # A and Bc are not drawn independently of the exponents: common.draw_card sets them so that the
        # reducible loss at the reference scale lies in the band spanned by the published fits
        # (A*1e9^-alpha in [0.2, 0.6] nats; Bc*2e10^-beta in [0.2, 0.7]; Hoffmann 0.35/0.54, Besiroglu
        # 0.34/0.32).  The ranges below are the implied envelopes, used only to bound fits.
        params={"E": (1.69, (1.55, 2.05)), "A": (406.4, (50.0, 4000.0)), "alpha": (0.34, (0.27, 0.42)),
                "Bc": (410.7, (70.0, 6000.0)), "beta": (0.28, (0.25, 0.38))},
        neutral=None),
    "C2": dict(
        name="Learning-rate bowl with scale- and horizon-dependent optimum",
        form="P_lr = k*(ln(lr/eta*))^2 with k=k_lo below the optimum and k_hi above; "
             "eta* = eta0*(N/1e8)^-gN*(D/2e9)^-gD*(B/5e5)^gB",
        grounding=["Wortsman et al. 2023 (arXiv:2309.14322): LR sensitivity bowls, asymmetric",
                   "Bjorck et al. 2024 (arXiv:2409.19913): optimal LR decreases with token horizon",
                   "Everett et al. 2024 (arXiv:2407.05872); Lingle 2024 (arXiv:2404.05728): muP transfer is imperfect",
                   "DeepSeek LLM 2024 (arXiv:2401.02954): eta_opt and B_opt power laws in compute"],
        params={"eta0": (3e-3, (1.5e-3, 8e-3)), "gN": (0.0, (0.0, 0.15)), "gD": (0.2, (0.12, 0.35)),
                "gB": (0.5, (0.0, 0.5)), "k_lo": (0.04, (0.03, 0.06)), "k_hi": (0.09, (0.06, 0.14))},
        neutral={"k_lo": 0.0, "k_hi": 0.0}),
    "C3": dict(
        name="Critical batch size set by data horizon",
        form="Deff <- Deff/(1 + B/Bcrit), Bcrit = bc0*(D/2e9)^psi   (no dependence on N)",
        grounding=["McCandlish et al. 2018 (arXiv:1812.06162): data/steps trade-off 1+B/Bcrit",
                   "Zhang et al. 2024 (arXiv:2410.21676): critical batch scales with data size, not model size",
                   "Bergsma et al. 2025 Power Lines (arXiv:2505.13738): B_opt, B_crit power laws in D"],
        params={"bc0": (1.0e6, (3e5, 2e6)), "psi": (0.47, (0.3, 0.6))},
        neutral={"bc0": float("inf")}),
    "C4": dict(
        name="Divergence edge (attention-logit / output-logit instability)",
        form="run diverges iff lr > eta_max*jitter, eta_max = h0*(N/1e8)^-delta*(1+wu/w0)^omega*(Q if qk_norm)",
        grounding=["Wortsman et al. 2023 (arXiv:2309.14322): instabilities appear at smaller LR as models grow; "
                   "qk-layernorm and longer warmup widen the stable range",
                   "Dehghani et al. 2023 ViT-22B (arXiv:2302.05442): qk-norm needed for stability"],
        params={"h0": (0.03, (0.012, 0.05)), "delta": (0.35, (0.2, 0.6)), "w0": (0.02, (0.01, 0.05)),
                "omega": (0.35, (0.2, 0.6)), "Q": (8.0, (3.0, 30.0)), "c_wu": (0.02, (0.01, 0.04))},
        neutral={"h0": float("inf"), "c_wu": 0.0}),
    "C5": dict(
        name="AdamW EMA timescale",
        form="tau = B/(lr*wd*D);  P_wd = k_tau*clip(ln(tau/tau*),-3,3)^2;  tau* = tau0*((D/N)/20)^chi",
        grounding=["Wang & Aitchison 2024 (arXiv:2405.13698): AdamW as EMA, timescale 1/(eta*lambda)",
                   "Bergsma et al. 2025 Power Lines (arXiv:2505.13738): optimal tau is a power law in tokens/param"],
        params={"tau0": (0.2, (0.08, 0.4)), "chi": (-0.4, (-0.7, -0.15)), "k_tau": (0.02, (0.012, 0.04))},
        neutral={"k_tau": 0.0}),
    "C6": dict(
        name="Diminishing value of repeated tokens",
        form="R = D/U - 1;  D' = U*(1 + Rs*(1-exp(-R/Rs))) when D > U",
        grounding=["Muennighoff et al. 2023 (arXiv:2305.16264): R*_D ~ 15.4; ~4 epochs nearly free"],
        params={"Rs": (15.4, (4.0, 30.0)), "U0": (None, None)},
        neutral={"Rs": float("inf")}),
    "C7": dict(
        name="Quality filtering multiplies token value but shrinks the pool",
        form="m(q) = 1 + mu*q^nu;  pool U = U0*(1-q)",
        grounding=["Goyal et al. 2024 CVPR 'Scaling laws for data filtering - data curation cannot be compute agnostic'",
                   "Penedo et al. 2024 FineWeb-Edu (arXiv:2406.17557): aggressive filtering trades quantity for quality"],
        params={"mu": (0.6, (0.2, 1.4)), "nu": (1.5, (0.8, 3.0))},
        neutral={"mu": 0.0}),
    "C8": dict(
        name="Exact-match metric as a power of per-token probability",
        form="per-token answer loss l = et + ct*(L - E);  acc_k = exp(-l)^k;  aggregate = sum_k w_k acc_k",
        grounding=["Schaeffer et al. 2023 (arXiv:2304.15004): emergence as a metric artifact",
                   "Du et al. 2024 (arXiv:2403.15796): loss-threshold view of emergence"],
        params={"et": (0.02, (0.0, 0.06)), "ct": (0.9, (0.6, 1.6))},
        neutral=None),
    "C9": dict(
        name="Benchmark contamination through a mixture source",
        form="clean acc a = sigmoid((L50 - Lk)/sa), Lk = L - gb*b + hb*b^2; contaminated items: "
             "a + (1-a)*(1-exp(-km*(N/1e8)^xi*b*D/1e9)); aggregate mixes a fraction phi of contaminated items",
        grounding=["Carlini et al. 2022 (arXiv:2202.07646): memorisation grows log-linearly with model size and repetitions",
                   "Magar & Schwartz 2022 (arXiv:2203.08242): contamination inflates benchmark scores"],
        params={"phi": (0.1, (0.05, 0.3)), "km": (0.02, (0.005, 0.08)), "xi": (0.6, (0.3, 1.0)),
                "gb": (0.3, (0.1, 0.5)), "hb": (0.6, (0.3, 1.2)), "L50": (2.9, (2.6, 3.2)), "sa": (0.12, (0.08, 0.2)),
                "vb": (0.1, (0.05, 0.3))},
        neutral={"phi": 0.0, "gb": 0.0, "hb": 0.0, "vb": 0.0}),
    "C10": dict(
        name="Heteroscedastic seed noise",
        form="observed = mean + sigma0*(N/1e8)^-rho * z  (z depends deterministically on config and seed)",
        grounding=["Madaan et al. 2024 'Quantifying variance in evaluation benchmarks' (arXiv:2406.10229)",
                   "Picard 2021; seed variance shrinks with scale in practice"],
        params={"sigma0": (0.008, (0.004, 0.012)), "rho": (0.3, (0.15, 0.4))},
        neutral=None),
    "C11": dict(
        name="Schedule-dependent intermediate-checkpoint loss",
        form="checkpoint at fraction f of a D-token run: L(N, fD) + ca*r(f)^zeta, r = relative LR at f "
             "(cosine to rmin, WSD cooldown to 0, constant 1)",
        grounding=["Hagele et al. 2024 (arXiv:2405.18392): WSD + cooldown matches cosine, enables cheap scaling laws",
                   "Porian et al. 2024 (arXiv:2406.19146); Pearce & Song 2024 (arXiv:2406.12907): Kaplan vs Chinchilla "
                   "discrepancy from using intermediate checkpoints / non-matched schedules",
                   "Tissue et al. 2024 (arXiv:2408.11029): loss depends on the LR annealing area"],
        params={"ca": (0.15, (0.08, 0.3)), "zeta": (1.0, (0.7, 1.5)), "rmin": (0.1, (0.1, 0.1)), "cd": (0.2, (0.2, 0.2))},
        neutral={"ca": 0.0}),

    # ================================================================================= evallab
    "E1": dict(
        name="Two-parameter item response",
        form="P(correct | model m, item j) = sigmoid(a_j*(theta_m + off_m[slice(j)] - b_j)); "
             "item information I_j = a_j^2*P*(1-P); SE(theta) = 1/sqrt(sum_j I_j)",
        grounding=["Birnbaum 1968 / Lord 1980: the 2PL model and the information function",
                   "Lalor et al. 2019 (arXiv:1908.11421): latent item parameters fitted from model responses",
                   "Vania et al. 2021 (arXiv:2106.00840): IRT comparison of NLP test sets",
                   "Rodriguez et al. 2021 (ACL): most benchmark items carry almost no information; a "
                   "minority of high-discrimination items decide the ranking"],
        # median discrimination near 1.5 and difficulty spread of about one logit are what the NLP-IRT
        # fits report; the world draws around that, wider on both sides
        params={"a_mu": (1.49, (0.7, 2.2)), "a_sd": (0.30, (0.15, 0.55)),
                "b_mu": (0.0, (-0.6, 0.6)), "b_sd": (0.95, (0.5, 1.6))},
        neutral=None),
    "E2": dict(
        name="Format sensitivity and answer-extraction failure",
        form="per-format ability offset fmt[m][f]; extraction fails independently per item with rate "
             "fmt_ext[f]['by_slice'][slice(j)]; a failed extraction is scored wrong under one denominator "
             "convention and dropped under the other",
        grounding=["Sclar et al. 2024 FormatSpread (arXiv:2310.11324): semantically equivalent prompt "
                   "formats spread accuracy by up to 76 points on one model/task",
                   "Alzahrani et al. 2024 (arXiv:2402.01781): small, meaning-preserving benchmark "
                   "perturbations reorder leaderboards",
                   "Robinson et al. 2023 (arXiv:2210.12353): cloze vs letter-choice scoring disagree"],
        # extraction-failure rates are implementation-specific and unpublished as constants
        params={"ext_base": (None, (0.0, 0.14)), "ext_slice_sd": (None, (0.0, 0.08)),
                "ext_model_sd": (None, (0.0, 0.18)), "fmt_off_sd": (None, (0.0, 0.45))},
        derived={"ext_base": "fmt_ext[fmt]['base']",
                 "ext_slice_sd": "spread of fmt_ext[fmt]['by_slice'] around the base rate",
                 "ext_model_sd": "spread of models[m]['ext'], the per-model shift in extraction-failure "
                                 "rate (FormatSpread's core finding is that the same format hurts models "
                                 "by different amounts, so the rate cannot be a property of the format alone; "
                                 "a verbose model under a strict extractor can lose a fifth of its answers "
                                 "where a terse one loses a twentieth, which is why the range runs this wide)",
                 "fmt_off_sd": "spread of models[m]['fmt'][fmt], the per-model ability offset per format"},
        neutral={"ext_base": 0.0, "ext_slice_sd": 0.0, "ext_model_sd": 0.0, "fmt_off_sd": 0.0}),
    "E3": dict(
        name="Contamination through duplicated items",
        form="item j occurs dup_j times in the corpus (dup_frac of the bank has dup_j ~ 1+Poisson(dup_lam)); "
             "a contaminated item gets a memorisation boost rising with dup_j and with the model's kappa",
        grounding=["Carlini et al. 2023 (arXiv:2202.07646): memorisation grows log-linearly in model "
                   "size, in the number of duplicates, and in context length; reported slope about "
                   "19 percentage points per decade of parameters (R^2 = 0.998)",
                   "Magar & Schwartz 2022 (arXiv:2203.08242): contamination inflates benchmark scores",
                   "Lee et al. 2022 (arXiv:2107.06499): deduplication changes memorisation sharply"],
        params={"dup_frac": (None, (0.05, 0.35)), "dup_lam": (None, (2.0, 12.0)),
                "kappa": (None, (0.0, 1.4))},
        derived={"kappa": "models[m]['kappa'], the per-model memorisation strength"},
        neutral={"dup_frac": 0.0, "kappa": 0.0}),
    "E4": dict(
        name="LLM-judge position and style preference",
        form="P(judge prefers a) = sigmoid(cq*(q_a - q_b) + dpos*[a shown first] "
             "+ phi*tanh((len_a - len_b)/sigl) + psi*(style_a - style_b))",
        grounding=["Wang et al. 2023 (arXiv:2305.17926): swapping the presentation order flips GPT-4's "
                   "verdict on a large share of pairs (one pair went 51.3% vs 23.8% by order; ChatGPT "
                   "2.5% vs 82.5%), i.e. the bias is comparable in size to the quality signal",
                   "Zheng et al. 2023 MT-Bench (arXiv:2306.05685): position, verbosity and self-"
                   "enhancement biases in LLM judges",
                   "Dubois et al. 2024 (arXiv:2404.04475): a logistic model with a tanh length term; "
                   "length control moved two models from 22.9%/64.3% to 41.9%/51.6%, raised correlation "
                   "with human preference 0.94 -> 0.98 and cut gameability 26% -> 10%, but "
                   "OVER-corrects at style extremes (reported +24.35pp / -17.33pp residual)"],
        params={"cq": (2.2, (1.4, 3.2)), "dpos": (None, (-1.2, 1.2)), "phi": (None, (0.0, 1.1)),
                "psi": (None, (0.0, 0.8)), "sigl": (None, (60.0, 300.0))},
        derived={"cq": "judge['cq']", "dpos": "judge['dpos']", "phi": "judge['phi']",
                 "psi": "judge['psi']", "sigl": "judge['sigl']"},
        neutral={"dpos": 0.0, "phi": 0.0, "psi": 0.0}),
    "E5": dict(
        name="Pairwise-comparison identifiability (Ford's condition)",
        form="P(i beats j) = sigmoid(s_i - s_j); the maximum-likelihood score vector is finite and "
             "unique iff the directed 'who beat whom' graph is STRONGLY connected.  This lab has no "
             "score parameters: the win probability is the judge's mean preference over the bank, so "
             "what varies is the COMPARISON SCHEDULE the agent chooses, and the failure is a property "
             "of that schedule (see derivation X10_ford).  Ties are the one world parameter, because "
             "they decide whether an edge exists at all.",
        grounding=["Ford 1957: necessary and sufficient condition for a finite unique Bradley-Terry MLE",
                   "Chiang et al. 2024 Chatbot Arena (arXiv:2403.04132): BT fitting over a sparse, "
                   "non-uniform comparison graph",
                   "Bradley & Terry 1952"],
        # the trap: undirected connectivity is NOT enough.  A schedule where one model only ever wins
        # and another only ever loses leaves the scale unbounded no matter how many games were played.
        params={"tie_rate": (None, (0.0, 0.25))},
        neutral={"tie_rate": 0.0}),
    "E6": dict(
        name="Between-call variance and the resulting detectable effect size",
        form="reported accuracy = truth + noise, Var = (sigma_between^2 + sigma_within^2/K)/n_items "
             "with K repetitions; sigma_within^2 = mean p(1-p); sigma_between from an ability jitter "
             "sig_call applied per call",
        grounding=["Miller 2024 'Adding Error Bars to Evals' (arXiv:2411.00640): the "
                   "clustered-standard-error decomposition and MDE ~ 2.80*sqrt(2p(1-p)/n) for a "
                   "two-sided 0.05 test at 80% power",
                   "Madaan et al. 2024 (arXiv:2406.10229): seed changes flip 5-12% of decisions at "
                   "temperature 0 and 12-24% at temperature 1.0; three passes at temperature 0 spread "
                   "6.3 points on one benchmark",
                   "Picard 2021 (arXiv:2109.08203): seed choice alone spans a reportable range"],
        params={"sig_call": (None, (0.0, 0.22))},
        neutral={"sig_call": 0.0}),
    "E7": dict(
        name="Slice mixture and aggregation convention (Simpson)",
        form="reported = sum_g w_g * p_g; micro weights w_g proportional to slice size, macro weights "
             "uniform; the scored denominator may be attempted items or extracted items",
        grounding=["Simpson 1951; Blyth 1972: reversal requires the weights to differ ACROSS the "
                   "compared units, not merely to be unequal",
                   "measured on this lab's own predecessor: micro vs macro on an MMLU-shaped mixture "
                   "correlate 0.988, so aggregation alone is a weak lever, while changing the "
                   "answer-extraction convention moved ranks by up to 8 positions -- the strong lever "
                   "is the denominator, not the weighting"],
        params={"slice_b_sd": (None, (0.0, 0.7)), "slice_off_sd": (None, (0.0, 0.75))},
        derived={"slice_b_sd": "spread of the slice_b vector (per-slice difficulty offsets)",
                 "slice_off_sd": "spread of models[m]['slice_off'], the per-model per-slice ability "
                                 "offset.  Required by this card's own grounding: a reweighting can only "
                                 "change a comparison if the compared units differ ACROSS slices, so a "
                                 "slice mechanism with common difficulties only is not the mechanism."},
        neutral={"slice_b_sd": 0.0, "slice_off_sd": 0.0}),
    "E8": dict(
        name="n-gram contamination detector with calibration error",
        form="detector score for item j = sigmoid(det_b + det_a*log(1+dup_j) + det_s*z_j), z_j fixed per "
             "item; an exact corpus scan returns dup_j itself at a much higher price",
        grounding=["Shi et al. 2024 Min-K% Prob (arXiv:2310.16789): membership inference on "
                   "pretraining data is well above chance but far from exact",
                   "Oren et al. 2024 (arXiv:2310.17623): exchangeability tests detect test-set "
                   "contamination without corpus access",
                   "detector slope/intercept/noise are not published as constants for any real stack"],
        params={"det_a": (None, (0.6, 1.8)), "det_b": (None, (-3.2, -1.4)), "det_s": (None, (0.4, 1.3))},
        neutral={"det_a": 0.0, "det_s": 0.0}),

    # ================================================================================= rllab
    "R1": dict(
        name="Reward-model overoptimization",
        form="with d = sqrt(KL(policy || init)): gold score R_bon(d) = d*(alpha - beta*d) for "
             "best-of-n, R_rl(d) = d*(alpha - beta*log d) for policy gradient; the PROXY reward is "
             "monotone in d, so the peak is invisible in the proxy",
        grounding=["Gao, Schulman & Hilton 2023 (arXiv:2210.10760): both functional forms, fitted "
                   "against sqrt(KL) rather than KL; alpha_rl is roughly constant across reward-model "
                   "size while beta_rl falls about logarithmically; for best-of-n the proxy beta is "
                   "near zero, so proxy and gold diverge without bound",
                   "Coste et al. 2024 (arXiv:2310.02743): reward-model ensembles delay but do not "
                   "remove overoptimization"],
        # Gao's alpha/beta are read off figures, never tabulated; they are also specific to that
        # reward-model family.  Ranges here only preserve sign and the location of the peak inside the
        # reachable KL window -- blueprints must check that (see rllab.best_d).
        params={"alpha": (None, (0.6, 1.8)), "beta": (None, (0.05, 0.6)),
                "beta_bon": (None, (0.0, 0.25))},
        neutral={"beta": 0.0, "beta_bon": 0.0}),
    "R2": dict(
        name="KL budget as the invariant, not the knob that sets it",
        form="best-of-n: KL = log n - (n-1)/n exactly; policy gradient: KL(steps) rises and saturates, "
             "KL = kl_cap*(1 - exp(-steps/steps0)); a KL penalty acts as early stopping",
        grounding=["Beirami et al. 2024 (arXiv:2401.01879), Theorem 3.1: log n - (n-1)/n is EXACT only "
                   "for an absolutely continuous reward distribution and is an UPPER BOUND when ties "
                   "have positive probability",
                   "Stiennon et al. 2020 (arXiv:2009.01325): the KL-vs-steps trajectory is reported "
                   "empirically; no functional form is published, and the penalty coefficient beta is "
                   "not comparable across runs -- the achieved KL is",
                   "Gao et al. 2023: matching two methods requires matching KL, not hyperparameters"],
        params={"steps0": (None, (150.0, 1200.0)), "kl_cap": (None, (8.0, 400.0))},
        neutral={"kl_cap": float("inf")}),
    "R3": dict(
        name="Reward-model accuracy scales in comparison count, with a floor",
        form="acc(D) = chance below rm_floor comparisons, then acc_max - (acc_max - 0.5)*"
             "(rm_ref/D)^rm_b, i.e. roughly linear in log D; the asymptote acc_max is set by the "
             "reward model's size",
        grounding=["Gao, Schulman & Hilton 2023 (arXiv:2210.10760): reward-model accuracy is linear in "
                   "log(comparisons) and near chance below roughly 2,000 comparisons",
                   "Touvron et al. 2023 Llama 2 (arXiv:2307.09288): reward-model accuracy keeps "
                   "improving with data and with reward-model size; reported gains are of order one "
                   "to two points per doubling of preference data"],
        params={"rm_floor": (None, (800.0, 5000.0)), "rm_ref": (None, (3.0e4, 4.0e5)),
                "rm_b": (None, (0.15, 0.7))},
        neutral={"rm_b": 0.0}),
    "R4": dict(
        name="Length as the dominant reward channel",
        form="mean length grows with d; the proxy reward is wq*(quality term) + wl*tanh((len-len0)/sl), "
             "so a length-only policy can reproduce most of the proxy gain",
        grounding=["Singhal et al. 2024 (arXiv:2310.03716): length explains 70-90% of the reward "
                   "improvement from RLHF; a reward that only counts tokens reproduces the win rates "
                   "of full PPO (56%/64% against 58%/63%)",
                   "Yu et al. 2025 DAPO (arXiv:2503.14476): a piecewise length penalty, zero below a "
                   "safe length, linear through a cache window, then saturated at -1 (16384 / 4096 / "
                   "20480 tokens in their setup)"],
        params={"len0": (None, (180.0, 420.0)), "lam": (None, (0.0, 1.6)),
                "sl": (None, (80.0, 320.0)), "wl": (None, (0.0, 0.8))},
        neutral={"lam": 0.0, "wl": 0.0}),
    "R5": dict(
        name="Preference-label noise damages calibration, not ranking",
        form="rm accuracy loses only acc_noise_pen*noise; expected calibration error grows as "
             "ece0 + ece1*noise, and jumps further below the data floor",
        grounding=["Ouyang et al. 2022 InstructGPT (arXiv:2203.02155) and Stiennon et al. 2020: "
                   "inter-annotator agreement on preference pairs is only about 72-73%, so a large "
                   "noise floor is the normal case, not a pathology",
                   "reward-model data-quality audits report large flagged-noise fractions in public "
                   "preference sets (HH-RLHF about 39%, SHP about 48%, UltraFeedback about 6%)",
                   "flipping 30% of labels still retains 79-95% of peak pairwise ranking accuracy: the "
                   "ANTI-PRIOR here is that 'label noise caps achievable accuracy' is the wrong model; "
                   "what degrades is the reward scale's calibration, and therefore everything "
                   "downstream that compares reward MAGNITUDES across prompts"],
        params={"noise": (None, (0.0, 0.45)), "ece0": (None, (0.01, 0.05)),
                "ece1": (None, (0.4, 1.4)), "acc_noise_pen": (None, (0.02, 0.12))},
        neutral={"noise": 0.0}),
    "R6": dict(
        name="Entropy collapse bounds achievable reward",
        form="policy entropy H(d) = H0*exp(-d/dh); the reachable reward is bounded by "
             "R = -ent_a*exp(H) + ent_b, so spending KL buys reward only while entropy remains",
        grounding=["Cui et al. 2025 (arXiv:2505.22617): the reward ceiling is an exponential function "
                   "of policy entropy, R = -a*exp(H) + b, fitted across model families; entropy "
                   "collapses early in RL and the ceiling is reached before the compute budget is"],
        params={"H0": (None, (0.7, 1.6)), "dh": (None, (0.8, 3.0)),
                "ent_a": (None, (0.0, 0.5)), "ent_b": (None, (0.0, 1.2))},
        neutral={"ent_a": 0.0}),
    "R7": dict(
        name="RL compute-performance sigmoid with a recipe-invariant ceiling",
        form="pass rate R(C) = R_start + (A_ceil - R_start)/(1 + (C_mid/C)^B_exp) in log-compute",
        grounding=["'The Art of Scaling Reinforcement Learning Compute for LLMs' (2025): a sigmoidal "
                   "fit in log compute, with reported asymptotes 0.61 / 0.515 / 0.490 for three "
                   "recipes; the key claim is that recipe changes move the midpoint C_mid and the "
                   "exponent B_exp while leaving the ASYMPTOTE A_ceil almost unchanged, so early-run "
                   "comparisons systematically mis-rank recipes",
                   "Hu et al. 2025 and related open-recipe reports: early-training advantage does not "
                   "predict the final plateau"],
        params={"A_ceil": (None, (0.42, 0.72)), "B_exp": (None, (0.5, 2.2)),
                "C_mid": (None, (0.3, 12.0)), "R_start": (None, (0.02, 0.2))},
        neutral={"use_sigmoid": 0}),

    # ================================================================================= servelab
    "S1": dict(
        name="Roofline: arithmetic intensity and the critical batch for decoding",
        form="step time = max(bytes_moved/BW, FLOPs/P_peak); the batch at which decoding stops being "
             "memory-bound is B_crit = (P_peak/BW)*bytes_per_param",
        grounding=["Williams et al. 2009: the roofline model",
                   "Pope et al. 2023 (arXiv:2211.05102): prefill runs at 150-450 FLOP per byte while "
                   "decoding runs at 1-2, so the two phases sit on opposite sides of the roofline",
                   "hardware ratios: an H100 at BF16 gives P_peak/BW of roughly 118 FLOP/byte, i.e. "
                   "B_crit near 295 at 2 bytes per parameter; a TPU v5e near 240"],
        params={"eff": (None, (0.45, 0.85)), "prefill_flop_eff": (None, (0.35, 0.75))},
        neutral=None),
    "S2": dict(
        name="Precision as effective parameters",
        form="N_eff = N * prod_x (1 - exp(-P_x/g_x)) over weights/activations/KV; "
             "L = A*N_eff^-alpha + B*D^-beta + E + delta_ptq(bits)",
        grounding=["Kumar et al. 2025 (arXiv:2411.04330) 'Scaling Laws for Precision': the effective-"
                   "parameter form, and a compute-optimal training precision of roughly 7-8 bits",
                   "Dettmers & Zettlemoyer 2023 (arXiv:2212.09720): 4-bit is near-universally optimal "
                   "for inference at fixed model bytes",
                   "the sign of the 'more data makes quantization worse' effect is CONFOUNDED in the "
                   "published fits (data and post-training-quantization degradation co-vary), so a "
                   "task must not treat either direction as settled"],
        params={"alpha": (None, (0.05, 0.11)), "beta": (None, (0.07, 0.14)),
                "A_loss": (None, (1.2, 3.0)), "E_loss": (None, (1.45, 1.85)),
                "acc_k": (None, (1.6, 3.6))},
        neutral={"delta_ptq": None}),
    "S3": dict(
        name="Speculative decoding with positionally decaying acceptance",
        form="acceptance at draft position i is a_i = a*rho^i within the positions the deployment's own "
             "proposal length exercises, and continues as a_i = a*rho^(k-1)*rho_tail^(i-k+1) beyond "
             "position k (rho_tail = rho unless the lab states otherwise); "
             "expected accepted tokens per step E = sum_{i<=g} prod_{k<i} a_k; "
             "improvement factor IF = E/(g*c + 1)",
        grounding=["Leviathan et al. 2023 (arXiv:2211.17192): the geometric expectation "
                   "(1-a^(g+1))/(1-a), the cost ratio c, and walltime improvement "
                   "(1-a^(g+1))/((1-a)(gc+1)); reported acceptance rates 0.5-0.9, c near 0.128, "
                   "optimal draft length 3-4 and 2.0-2.5x speedups",
                   "Chen et al. 2023 (arXiv:2302.01318): the same identity for large models",
                   "positional decay is what breaks the single-acceptance-rate model: fitting one a "
                   "from a long draft window and extrapolating to a different window is biased",
                   "published acceptance profiles are measured at the draft length the authors ran, so "
                   "the decay beyond that length is an extrapolation and not a measurement"],
        params={"a0": (None, (0.45, 0.92)), "rho": (None, (0.75, 1.0)), "c": (None, (0.05, 0.3)),
                "rho_tail": (None, (0.85, 1.0))},
        derived={"a0": "drafts[name]['a'], acceptance at the first draft position",
                 "rho": "drafts[name]['rho'], the per-position decay",
                 "rho_tail": "drafts[name]['rho_tail'], the decay beyond drafts[name]['tail_from'] "
                             "positions; equal to rho when unset",
                 "c": "drafts[name]['c'], draft cost relative to one target step"},
        neutral={"rho": 1.0}),
    "S4": dict(
        name="Queueing: utilisation is determined, not chosen",
        form="rho = arrival_rate * E[S] where E[S] is the service time the BATCH implies; "
             "M/G/1 Pollaczek-Khinchine E[Wq] = rho*E[S]*(1+Cs^2)/(2*(1-rho)); E[T] = E[Wq] + E[S]; "
             "quantiles from an exponential sojourn time, so p99/p50 = ln(100)/ln(2) ~ 6.64",
        grounding=["Pollaczek 1930 / Khinchine 1932: the mean waiting time of M/G/1",
                   "Kwon et al. 2023 vLLM (arXiv:2309.06180): serving throughput is bounded by "
                   "KV-cache capacity, so batch size and service time are coupled",
                   "the utilisation-versus-tail-latency curve of any real serving stack is not "
                   "published in closed form; the trap is that raising the batch raises throughput "
                   "AND E[S], so the tail can worsen while utilisation looks healthier"],
        params={"cs2": (None, (0.4, 3.0))},
        neutral=None),
    "S5": dict(
        name="KV-cache capacity sets the feasible batch",
        form="per-token KV bytes = 2*L*H_kv*d_head*bytes; "
             "B_max = floor((M_gpu*mem_util - weights - activations)/(per_token*seq_len))",
        grounding=["Shazeer 2019 multi-query attention (arXiv:1911.02150) and Ainslie et al. 2023 GQA "
                   "(arXiv:2305.13245): H_kv, not H, sets the cache size",
                   "Kwon et al. 2023 (arXiv:2309.06180): fragmentation and reservation waste in "
                   "practice, which is why the usable fraction is below 1"],
        params={"mem_util": (None, (0.80, 0.94)), "kv_bytes": (None, (1.0, 2.0))},
        neutral=None),
    "S6": dict(
        name="Test-time compute: coverage, selection and the voting ceiling",
        form="coverage(k) = 1 - B(tt_alpha, tt_beta + k)/B(tt_alpha, tt_beta) (beta-binomial over "
             "per-problem success rates), hence 1 - pass@k ~ C*k^-tt_alpha; what a verifier can "
             "actually select is bounded by its true-positive to false-positive ratio, and majority "
             "voting is bounded by the modal-answer hit rate tt_mode",
        grounding=["Brown et al. 2024 'Large Language Monkeys' (arXiv:2407.21787): coverage follows a "
                   "power law in k over four orders of magnitude, while what can be SELECTED plateaus "
                   "beyond roughly 100 samples",
                   "Chen et al. 2021 (arXiv:2107.03374): the unbiased pass@k estimator "
                   "1 - C(n-c,k)/C(n,k)",
                   "Snell et al. 2024 (arXiv:2408.03314): the best allocation of test-time compute "
                   "depends on problem difficulty; the allocation rules themselves are figure-only",
                   "correlated samples: the effective sample count N/(1 + (N-1)*rho) saturates at "
                   "1/rho, which is a genuine identifiability trap against the modal-hit ceiling -- "
                   "both flatten the same curve and one run cannot separate them"],
        params={"tt_alpha": (None, (0.35, 1.1)), "tt_beta": (None, (0.6, 3.0)),
                "tt_mode": (None, (0.3, 0.8)), "ver_tp": (None, (0.6, 0.97)),
                "ver_fp": (None, (0.02, 0.35))},
        neutral={"ver_fp": 0.0}),
}

# --------------------------------------------------------------------------------------------------
# Regularities that are NOT world parameters: identities and estimators the answer path has to apply.
# A blueprint declares the subset it uses as its derivation DAG; the longest chain is the depth knob.
DERIVATIONS = {
    "X1_winners_curse": dict(
        name="Optimism of a maximum over k noisy candidates",
        form="E[max of k standard normals]: M_2 = 0.5642, M_3 = 0.8463, M_8 = 1.4236, M_16 = 1.7660, "
             "M_64 = 2.3193; the reported best of k runs overstates its true mean by about sigma*M_k",
        grounding=["order statistics of the normal distribution",
                   "Dodge et al. 2019 'Show Your Work' (arXiv:1909.03004): reported bests are "
                   "budget-dependent and not comparable across search budgets"]),
    "X2_adaptive_reuse": dict(
        name="How much a holdout degrades under k adaptive queries",
        form="worst-case bias grows like sqrt(k/n); the reusable-holdout guarantees give rates of "
             "order k^(1/4)/sqrt(n) and log(k)^(2/3)/n^(1/3); nothing useful is known for k > n^2",
        grounding=["Dwork et al. 2015 (Science; NeurIPS 'Generalization in Adaptive Data Analysis and "
                   "Holdout Reuse'): the reusable holdout and its rates",
                   "Blum & Hardt 2015 (arXiv:1502.04585) the Ladder mechanism"]),
    "X3_metric_artifact": dict(
        name="A sharp metric turns a smooth loss curve into an apparent jump",
        form="exact-match accuracy = exp(-l)^k for a k-token answer, so a linear improvement in "
             "per-token loss looks like an emergent jump; N_crit = c*(L/ln 2)^(1/alpha)",
        grounding=["Schaeffer et al. 2023 (arXiv:2304.15004): over 92% of claimed emergent abilities "
                   "appear only under multiple-choice grading or exact string match, and 34 of 39 "
                   "BIG-Bench metrics show no emergence at all",
                   "Du et al. 2024 (arXiv:2403.15796): the same curves are smooth in loss"]),
    "X4_denominator": dict(
        name="Which denominator a rate is taken over",
        form="scored-only vs attempted vs all-items; the three differ by exactly the extraction-failure "
             "mass, which is itself slice-dependent",
        grounding=["Sclar et al. 2024 (arXiv:2310.11324); Robinson et al. 2023 (arXiv:2210.12353)"]),
    "X5_paired_design": dict(
        name="Pairing removes the item-difficulty component of the variance",
        form="Var(difference) = Var_a + Var_b - 2*Cov; comparing two models on the SAME items makes "
             "Cov large, so an unpaired test on paired data is badly conservative",
        grounding=["Miller 2024 (arXiv:2411.00640): the paired-difference estimator for model "
                   "comparison on shared items"]),
    "X6_kl_matching": dict(
        name="Compare alignment methods at equal KL, not equal hyperparameters",
        form="best-of-n KL = log n - (n-1)/n; policy-gradient KL must be measured; only at matched KL "
             "are the gold-score curves comparable",
        grounding=["Gao et al. 2023 (arXiv:2210.10760); Beirami et al. 2024 (arXiv:2401.01879)"]),
    "X7_proxy_peak": dict(
        name="A monotone proxy cannot locate its own optimum",
        form="d* = argmax of the GOLD curve; for the policy-gradient form d* solves "
             "alpha - beta*(1 + log d) = 0, i.e. d* = exp(alpha/beta - 1); the proxy is increasing "
             "everywhere, so the peak must be found by gold evaluation or by a model of the gap",
        grounding=["Gao et al. 2023 (arXiv:2210.10760)"]),
    "X8_amdahl_serving": dict(
        name="A speedup applies only to the phase it touches",
        form="speculative decoding accelerates decode steps only; end-to-end gain is bounded by the "
             "decode fraction of the request, which depends on prompt-to-generation ratio",
        grounding=["Pope et al. 2023 (arXiv:2211.05102): prefill and decode have different roofline "
                   "positions and different scalings in sequence length"]),
    "X9_selection_ceiling": dict(
        name="Selection cannot exceed coverage, and is bounded by verifier precision",
        form="selected(k) <= coverage(k); with true-positive rate tp and false-positive rate fp the "
             "achievable selected fraction is bounded by tp/(tp+fp) applied to coverage",
        grounding=["Brown et al. 2024 (arXiv:2407.21787): coverage keeps rising while selected "
                   "accuracy plateaus",
                   "Cobbe et al. 2021 (arXiv:2110.14168): verifier-reranked sampling"]),
    "X10_ford": dict(
        name="Check strong connectivity before fitting a pairwise model",
        form="a Bradley-Terry fit on a comparison schedule whose win digraph is not strongly connected "
             "has no finite MLE; the symptom is scores running away with iteration count, not an error",
        grounding=["Ford 1957"]),
}

OBSTACLES = {
    "O1": "confounded notebook: two causes co-vary in every prior run",
    "O2": "masking: two effects cancel inside the observed range",
    "O3": "regime onset outside the observed range",
    "O4": "measurement artifact: the reported metric is not the quantity of interest",
    "O5": "non-identifiability: the answer depends on something with zero footprint in the runnable region",
    "O6": "interaction: the effect of one knob depends on another",
    "O7": "checkpoint/temporal confound",
    "O8": "winner's curse / selective reporting",
    "O9": "scale-transfer failure: a rule fitted on one axis is applied along another",
    "O10": "aggregation (Simpson): a pooled number hides a mixture",
    # ---- templates the three new item forms need
    "O11": "under-determination with a direction: the disclosed evidence leaves a slack that a "
           "second mechanism can occupy, so the headline conclusion is refutable rather than wrong "
           "(the counterexample form's engine)",
    "O12": "sufficient evidence: the disclosed rows happen to pin the quantity down, so the inviting "
           "counterexample does not exist and the honest answer is 'entailed' (the trap pair for O11)",
    "O13": "plan-dependence: which conclusion the data supports depends on a design choice that must "
           "be made before any number is seen (the pre-registration form's engine)",
    "O14": "silent defect: an analysis script computes a defensible-looking number by an inferentially "
           "wrong route, and no test, crash or outlier reveals it",
    "O15": "harmless red herring: a second suspicious-looking line whose correction moves the number "
           "by less than a quarter of the tolerance, so acting on it is a false positive",
    "O16": "denominator/aggregation swap that is invisible in the reported summary but changes the "
           "sign of the comparison",
    "O17": "unreachability: the queried configuration is outside the lab's permitted request set, so "
           "the quantity cannot be measured at all and the governing law has to be identified inside "
           "the reachable region and then extended.  Measured 2026-09-26 on e05's (q1, q5) matched pair: "
           "it does force the derivation - both frontier models wrote the approach law down and inverted "
           "the best-of-n identity - but both then answered the unreachable item exactly, at the same "
           "0.00 tolerances as the reachable one.  So this template changes the *route* and not the "
           "*difficulty* when the law is smooth, low-dimensional and reported without noise; pair it with "
           "an estimator that has to be designed under noise (see e05's design note)",
}

# Difficulty knobs, ablated one at a time (see build.gates G11-G13).
KNOBS = {
    "depth": "length of the longest chain in the declared DERIVATIONS DAG that the answer path needs",
    "anti_prior": "the truth of the queried quantity is on the opposite side of the literature prior; "
                  "measured as the gap between the answer and what the published constant would give",
    "nuisance": "number of active cards that are load-bearing for the measurement but not for the "
                "answer, so they must be modelled and then divided out",
}

