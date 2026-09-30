# Current production policy: majority vote over 6 samples of template 0.
# Replace this with your own. See /app/docs/policy_api.md.


def run(env):
    out = {}
    per = env.budget // len(env.questions)
    for q in env.questions:
        counts = {}
        for _ in range(per):
            a, s = env.sample(q["id"], 0)
            counts[a] = counts.get(a, 0) + 1
        out[q["id"]] = max(counts, key=counts.get)
    env.submit(out)
