"""ScaleLab mechanism-card library.

A *card* is one grounded mechanism: a functional form taken from the literature, the parameters that
form needs, the value the literature reports ("textbook"), the range the world generator may draw from
("world"), and the *neutral* value that switches the card off.  Every task activates a subset of cards;
inactive cards sit at their neutral value so the single world function in world.py stays valid.

The world is deliberately NOT a replica of real training.  Forms are taken from published fits; the
parameters are redrawn inside ranges that bracket (and sometimes exceed) the published values, so a
model that answers from memorised numbers is wrong while a model that measures is right.  What the
agent is told about this boundary is in the task manual (export.py writes it from ABSTRACTION below).
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
}
