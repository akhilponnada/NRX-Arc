"""Minimal nn module for Phase 1: Module, Parameter, Linear.

Parameter is a thin wrapper around Tensor that lets Module.__setattr__
auto-discover learnable parameters. Forward passes access the underlying
tensor via `.tensor`.
"""
import numpy as np
from . import _core as _c


class Parameter:
    """Marks a Tensor as a learnable parameter for Module discovery."""
    def __init__(self, tensor):
        self.tensor = tensor


class Module:
    def __init__(self):
        # Bypass our __setattr__ for the internal lists themselves.
        object.__setattr__(self, "_parameters", [])
        object.__setattr__(self, "_modules", [])

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters.append(value)
        elif isinstance(value, Module):
            self._modules.append(value)
        object.__setattr__(self, name, value)

    def parameters(self):
        out = [p.tensor for p in self._parameters]
        for m in self._modules:
            out.extend(m.parameters())
        return out

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    """y = x @ W + b. Kaiming/He init for W (sqrt(2/fan_in)), zero init for b."""
    def __init__(self, in_features, out_features, seed):
        super().__init__()
        rng = np.random.default_rng(seed)
        std = float(np.sqrt(2.0 / in_features))
        W_init = (rng.standard_normal(size=(in_features, out_features)) * std).astype(np.float32)
        b_init = np.zeros((out_features,), dtype=np.float32)
        self.W = Parameter(_c.from_numpy(W_init))
        self.b = Parameter(_c.from_numpy(b_init))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return x @ self.W.tensor + self.b.tensor
