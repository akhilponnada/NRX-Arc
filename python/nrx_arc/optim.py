"""Optimizers. Phase 1: SGD only."""

class SGD:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = float(lr)

    def step(self):
        for p in self.params:
            p.sgd_update_(self.lr)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
