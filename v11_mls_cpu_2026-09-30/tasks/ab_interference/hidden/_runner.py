"""Copied next to the agent's solution.py inside the sandbox; runs every replicate in one child process."""
import solution


def run(experiments):
    return [float(solution.estimate(u.copy())) for u in experiments]
