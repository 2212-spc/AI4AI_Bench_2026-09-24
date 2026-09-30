"""Copied next to the agent's preprocess.py inside the sandbox; runs fit + transforms in one child process."""
import preprocess


def run(train_df, test_dfs):
    st = preprocess.fit_preprocess(train_df.copy())
    return preprocess.transform(st, train_df.copy()), [preprocess.transform(st, d.copy()) for d in test_dfs]
