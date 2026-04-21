# Dev log

## 2026-04-20 — Phase 1 complete: CPU autograd + 8 ops + MLP convergence

Shipped a from-scratch reverse-mode autograd in Rust with Python 
bindings, plus enough infrastructure (nn.Module, Parameter, Linear, 
optim.SGD) to train a 2-layer MLP on 3-blob classification to 95% 
test accuracy.

### Ops (8 total, all gradient-checked)

Rounds 1–5: add, mul, matmul, relu, sum  
Round 6 (utility): neg, sub (= add + neg), scalar_mul  
Plus row-vector broadcasting in add and mul for bias handling in Linear.

Every op verified with double-sided finite-difference checks at 
rtol=1e-3, fp64 reference. Chain test sum(relu(x @ W + b)) passes at 
~7e-8 relative error. Full MSE chain dL/dW of (1/B)·sum((x@W+b−y)²) 
FD-checked end-to-end.

### Infrastructure

python/nrx_arc/nn.py: Module base class with __setattr__-based 
auto-registration of Parameters and child Modules (same pattern as 
PyTorch), Parameter as a thin wrapper over Tensor, Linear with 
He/Kaiming init (std=sqrt(2/fan_in)) and zero bias.

python/nrx_arc/optim.py: SGD with step() doing in-place param updates 
via Tensor.sgd_update_(lr) — a Rust method that bypasses the tape 
entirely. Byte-exactness FD-verified.

### MLP integration test

2-layer MLP (10 → 32 → 3), 100 synthetic 3-Gaussian-blob points, 
MSE vs one-hot labels, SGD lr=0.05. Converges at epoch 110 with 
100% train acc, 95% test acc, 9.4ms wall clock.

### The debugging story worth remembering

First run hit only 85% test accuracy. Instead of tweaking 
hyperparameters, I cross-checked against sklearn on the same data 
split:

    sklearn MLP(hidden=32): 80%
    sklearn LogReg:         80%
    our framework:          85%

If sklearn is at 80%, the framework isn't the bottleneck. The data 
is. Random unit vectors in 10-dim space have pairwise cosine ~1/√10 
≈ 0.32, so my "random unit-vector blob centers scaled by 3.0" ended 
up only ~3.19 apart — barely more than the within-class radius of 
~3.16. Bayes accuracy on a 20-point test set was around 80%, full stop.

Fix: QR-orthogonalize the centers before scaling. Pairwise distance 
becomes 3√2 ≈ 4.24, task becomes cleanly separable. After fix: 95% 
test accuracy on the nose.

Lesson: when a test underperforms, check your reference before 
blaming your code.

### Non-obvious things I learned

- The FD reference has to run in fp64 even when the framework is 
  fp32. Otherwise fp32 roundoff in the reference eats the entire 
  1e-3 budget and you get false failures. Upcasting the reference is 
  the fix; loosening rtol is the mistake.
- ReLU at x=0 is undefined and FD-checking there is meaningless. 
  Perturbation margins (|x| > 0.05) around the kink solve this cleanly.
- The temptation to write "manual grad" tests that assert ones equals 
  ones is very strong and tests nothing. Pulling sum forward to Round 
  1 let every subsequent op have a real FD check from the start.
- Tape-bypassing in-place mutation is the single most error-prone 
  part of the optimizer. Implement it as a focused Rust method, not 
  a Python workaround.

### Fragile bits carried forward as tech debt

- Process-wide thread-local tape; every test has to call clear_tape().
- Backward consumes the tape; one-shot per graph. Panics with clear 
  message if violated.
- Grads accumulate lazily; zero_grad() is mandatory between steps.
- Module.__setattr__ auto-registration leaks if you reassign an 
  attribute mid-training. Fine for now, tighten if it bites.
- 1.82% of training epochs have tiny loss upticks (all <1e-6). 
  This is fp32 sum-reordering noise on the loss readout, not actual 
  divergence.

### Next

Phase 2: CUDA port. Start with elementwise ops, then matmul, then 
re-run the full FD suite on GPU tensors.
