# Proposed recipe changes

Each change is a small, independent edit to the recipe; any subset can be combined.

- **c1**: Raise the peak learning rate from 0.03 to 0.06.
- **c2**: Raise SGD momentum from 0.9 to 0.95.
- **c3**: Enable global gradient-norm clipping (max norm 1.0).
- **c4**: Widen the hidden layer from 96 to 160 units (same initialisation scheme).
- **c5**: Add L2 weight decay 0.0003 (coupled, added to the gradient as in torch.optim.SGD).
- **c6**: Draw minibatch indices from a dedicated RNG stream instead of the RNG used for initialisation (decouples data order from init).
