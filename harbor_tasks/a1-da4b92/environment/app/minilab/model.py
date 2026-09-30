"""Two-layer ReLU MLP regressor with hand-written backprop."""
import numpy as np


def init_params(rng, d_in, hidden):
    W1 = rng.standard_normal((d_in, hidden)) * np.sqrt(2.0 / d_in)   # He init (ReLU layer)
    W2 = rng.standard_normal((hidden, 1)) * np.sqrt(1.0 / hidden)  # LeCun init (linear output)
    return {"W1": W1, "b1": np.zeros(hidden), "W2": W2, "b2": np.zeros(1)}


def forward(params, x):
    h = x @ params["W1"] + params["b1"]
    a = np.maximum(h, 0.0)
    out = a @ params["W2"] + params["b2"]
    return out[:, 0], (h, a)


def loss_and_grads(params, x, y, denom):
    """Squared-error loss 0.5*(pred-y)^2, averaged; returns (loss, grads)."""
    pred, (h, a) = forward(params, x)
    r = pred - y
    loss = 0.5 * float(np.sum(r * r)) / denom
    g_out = (r / denom)[:, None]
    g_h = (g_out @ params["W2"].T) * (h > 0)
    grads = {
        "W1": x.T @ g_h,
        "b1": g_h.sum(0),
        "W2": a.T @ g_out,
        "b2": g_out.sum(0),
    }
    return loss, grads


def mse(params, x, y):
    pred, _ = forward(params, x)
    return float(np.mean((pred - y) ** 2))
