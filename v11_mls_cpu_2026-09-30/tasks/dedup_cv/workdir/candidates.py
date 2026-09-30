"""The fixed menu of model configurations. The grader fits the config you select on ALL training rows
and scores it on new data. Do not edit (the grader uses its own copy)."""
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def candidates():
    C = []
    for k in [1, 2, 4, 8, 16, 32, 64, 128]:
        C.append((f"knn_k{k}", lambda k=k: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k, weights="distance" if k == 1 else "uniform"))))
    for leaf in [1, 5, 20, 80, 250]:
        C.append((f"rf_leaf{leaf}", lambda leaf=leaf: RandomForestRegressor(n_estimators=80, min_samples_leaf=leaf, max_features=0.5, n_jobs=1, random_state=0)))
    for lr, leaf in [(0.1, 2), (0.1, 20), (0.05, 100), (0.03, 300)]:
        C.append((f"gbm_lr{lr}_leaf{leaf}", lambda lr=lr, leaf=leaf: HistGradientBoostingRegressor(learning_rate=lr, max_iter=300, min_samples_leaf=leaf, early_stopping=False, random_state=0)))
    return C
