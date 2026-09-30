"""Decoy: 'heavily regularised is always safe' - always the most regularised boosting config."""
def select(X, y):
    return "gbm_lr0.03_leaf300"
