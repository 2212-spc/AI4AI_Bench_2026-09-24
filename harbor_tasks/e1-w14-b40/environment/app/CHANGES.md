# Proposed recipe changes

Each change is a small, independent edit to the recipe; any subset can be combined.

- **c1**: Raise the peak learning rate from 0.04 to 0.1.
- **c2**: Raise SGD momentum from 0.9 to 0.97.
- **c3**: Enable global gradient-norm clipping (max norm 0.3).
- **c4**: Widen the hidden layer from 96 to 256 units (same initialisation scheme).
- **c5**: Add L2 weight decay 0.001 (coupled, added to the gradient as in torch.optim.SGD).
- **c6**: Draw minibatch indices from a dedicated RNG stream instead of the RNG used for initialisation (decouples data order from init).
