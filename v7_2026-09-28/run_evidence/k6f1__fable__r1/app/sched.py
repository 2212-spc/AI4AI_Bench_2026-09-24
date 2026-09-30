# Closed-loop batch-size scheduler for solver-7b.
#
# Model (fitted on the dev replica):  steps(seg, b) = S * (1 + B / b)
#   S  ~= 0.0258  (model property, same before and after the corpus switch)
#   B  ~= 47 before the switch (fixed ramp, same every run)
#   B  jumps at the per-run switch point and may drift afterwards -> estimated online.
# Per-step time is t_overhead + b/throughput, so the optimum is b* = sqrt(B * t_overhead * throughput).
import math

S_MODEL = 0.0258
B_PRE = 47.0
DETECT_RATIO = 1.30        # steps / predicted above this => corpus switch
EARLIEST_SWITCH = 10       # never declare a switch before this segment


def run(env):
    n = env.n_segments
    t0 = env.t_overhead
    thr = env.throughput
    lo = max(env.b_min, 64)
    hi = min(env.b_max, 4096)

    def b_opt(B):
        b = math.sqrt(max(B, 1.0) * t0 * thr)
        b = int(round(min(hi, max(lo, b))))
        # hard guarantee: never leave the env's legal range
        return int(min(env.b_max, max(env.b_min, b)))

    switched = False
    post = []          # (segment, implied B) after the switch
    pre_hist = []      # implied B before the switch (for a soft, late detection)
    b = b_opt(B_PRE)

    for i in range(n):
        r = env.run_segment(b)
        s = r["steps"]
        ratio = s / (S_MODEL * (1.0 + B_PRE / b))
        B_i = max(5.0, b * (s / S_MODEL - 1.0))

        if not switched:
            pre_hist.append(B_i)
            hard = i >= EARLIEST_SWITCH and ratio > DETECT_RATIO
            # soft detection: two consecutive mildly elevated segments
            soft = (i >= EARLIEST_SWITCH and len(pre_hist) >= 2
                    and min(pre_hist[-1], pre_hist[-2]) > 1.7 * B_PRE)
            if hard or soft:
                switched = True
                if soft and not hard:
                    post.append((i - 1, pre_hist[-2]))
                post.append((i, B_i))
        else:
            post.append((i, B_i))

        if switched:
            B_next = _predict_B(post, i + 1)
            b = b_opt(B_next)
        else:
            b = b_opt(B_PRE)


def _predict_B(post, k_next):
    """Predict B for the next segment from post-switch implied values.
    Exponentially-weighted mean plus a shrunk linear trend (B may grow after the switch)."""
    m = len(post)
    if m == 1:
        return post[0][1]
    # recent-weighted mean
    hl = 4.0
    w = [0.5 ** ((post[-1][0] - k) / hl) for k, _ in post]
    sw = sum(w)
    mean_B = sum(wi * B for wi, (_, B) in zip(w, post)) / sw
    mean_k = sum(wi * k for wi, (k, _) in zip(w, post)) / sw
    if m >= 4:
        sxx = sum(wi * (k - mean_k) ** 2 for wi, (k, _) in zip(w, post))
        sxy = sum(wi * (k - mean_k) * (B - mean_B) for wi, (k, B) in zip(w, post))
        slope = sxy / sxx if sxx > 0 else 0.0
        # shrink the slope: noise on implied B is ~10 %, so trust the trend only partially
        slope *= min(1.0, (m - 3) / 6.0)
        # cap the slope to +-15 % of the level per segment
        cap = 0.15 * mean_B
        slope = max(-cap, min(cap, slope))
        pred = mean_B + slope * (k_next - mean_k)
    else:
        pred = mean_B
    # never fall below the pre-switch level: the switch only adds noise
    return max(B_PRE, pred)
