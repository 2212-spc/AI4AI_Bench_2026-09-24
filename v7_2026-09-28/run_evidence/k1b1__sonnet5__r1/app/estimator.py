# Current practice: report what the judge said.
# The eval team knows this is wrong and does not know by how much. Replace it.
# See /app/docs/estimator_api.md.


def estimate(env):
    rows = env.rows
    p = sum(1 for r in rows if r["judge_win"]) / float(len(rows))
    env.submit(p, p - 0.05, p + 0.05)
