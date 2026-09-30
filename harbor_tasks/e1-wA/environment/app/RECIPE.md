# MiniLab production recipe (current)

Model: 2-layer MLP (d_in=32 -> hidden 96 ReLU -> 1), He-normal first layer, output layer std 1/sqrt(hidden), zero biases.
Task: regression, mean-squared error. Training set 20,000 examples, fixed validation set 4,000 examples (both fixed; only the
training seed varies between runs).
Optimiser: SGD with momentum 0.9 (dampening 0), peak lr 0.03, linear warm-up 10 steps then
cosine decay to 0.05 x peak, 400 steps, batch size 32, no weight decay,
no gradient clipping.
The training seed controls initialisation and minibatch order.

Metric: validation MSE of the final weights (lower is better). The quantity that matters for shipping is the
**expected** validation MSE over training seeds. A run that diverges produces a useless model and is counted as
val MSE = 0.3466 (the MSE of predicting the mean).
