"""SGD with momentum and L2 weight decay."""
import numpy as np


class SGD:
    def __init__(self, params, lr, momentum=0.0, dampening=None, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        # dampening defaults to the momentum coefficient so that the buffer is a
        # properly normalised running average of recent gradients
        self.dampening = momentum if dampening is None else dampening
        self.weight_decay = weight_decay
        self.buf = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        for k, p in self.params.items():
            g = grads[k]
            if self.weight_decay:
                g = g + self.weight_decay * p
            b = self.buf[k]
            b *= self.momentum
            b += (1.0 - self.dampening) * g
            p -= self.lr * b
