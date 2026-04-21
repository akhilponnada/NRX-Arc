"""Phase 1 end-to-end: train a 2-layer MLP on synthetic data, assert convergence.

Loss = sum((pred - y)^2), expanded to avoid the missing sub op:
    L = sum(pred*pred) + sum((-2*y) * pred)   [+ const sum(y*y), dropped]
Gradient is identical to MSE; printed loss adds the constant back so it's
the actual MSE.

Optimizer is hand-rolled SGD; params live in numpy and are wrapped into
fresh nrx_arc tensors each iteration (we have no in-place sub or assign yet).
"""
import numpy as np
import nrx_arc as nx


def _he_init(rng, fan_in, fan_out):
    return (rng.standard_normal(size=(fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def _forward_loss(x_np, y_np, W1_np, W2_np):
    """Build the graph, return (loss_tensor, W1_tensor, W2_tensor)."""
    x = nx.from_numpy(x_np)
    y_neg2 = nx.from_numpy((-2.0 * y_np).astype(np.float32))
    W1 = nx.from_numpy(W1_np)
    W2 = nx.from_numpy(W2_np)
    h = (x @ W1).relu()
    pred = h @ W2
    loss = (pred * pred).sum() + (y_neg2 * pred).sum()
    return loss, W1, W2, pred


def test_mlp_converges_on_synthetic_regression():
    rng = np.random.default_rng(seed=0)
    B, D_in, H, D_out = 64, 4, 16, 1

    # Teacher: same architecture, fixed weights. Student must learn to match.
    W1_true = _he_init(rng, D_in, H)
    W2_true = _he_init(rng, H, D_out)
    x_np = rng.standard_normal(size=(B, D_in)).astype(np.float32)
    h_true = np.maximum(x_np @ W1_true, 0.0)
    y_np = (h_true @ W2_true).astype(np.float32)
    y_sq_const = float((y_np * y_np).sum())

    # Student: independent init.
    W1 = _he_init(np.random.default_rng(seed=1), D_in, H)
    W2 = _he_init(np.random.default_rng(seed=2), H, D_out)

    lr = 1e-3
    n_steps = 400

    # Initial loss
    nx.clear_tape()
    loss_t, _, _, pred_t = _forward_loss(x_np, y_np, W1, W2)
    init_loss = float(np.asarray(loss_t.to_numpy())) + y_sq_const
    nx.clear_tape()  # discard graph; we built it just to read the loss

    losses = []
    for step in range(n_steps):
        loss_t, W1_t, W2_t, _ = _forward_loss(x_np, y_np, W1, W2)
        loss_val = float(np.asarray(loss_t.to_numpy())) + y_sq_const
        loss_t.backward()
        g1 = np.asarray(W1_t.grad)
        g2 = np.asarray(W2_t.grad)
        W1 = (W1 - lr * g1).astype(np.float32)
        W2 = (W2 - lr * g2).astype(np.float32)
        losses.append(loss_val)
        if step in (0, 10, 50, 100, 200, 399):
            print(f"  step {step:4d}  mse={loss_val/B:.6f}")

    final_loss = losses[-1]
    init_mse = init_loss / B
    final_mse = final_loss / B
    drop_ratio = init_loss / max(final_loss, 1e-12)
    print(f"  init MSE={init_mse:.4f}  final MSE={final_mse:.4f}  drop={drop_ratio:.1f}x")

    # Sanity: loss must drop by at least 100x. With seed=0 and these
    # hyperparams it actually drops by several orders of magnitude.
    assert final_loss < init_loss, "loss did not decrease at all"
    assert drop_ratio > 100.0, f"loss only dropped {drop_ratio:.1f}x (need >100x)"
    assert final_mse < 1e-2, f"final MSE {final_mse:.4f} not below 1e-2"


def test_double_backward_raises():
    """Hardening check: calling backward twice on the same graph errors clearly."""
    nx.clear_tape()
    rng = np.random.default_rng(seed=0)
    a = nx.from_numpy(rng.standard_normal(size=(3,)).astype(np.float32))
    b = nx.from_numpy(rng.standard_normal(size=(3,)).astype(np.float32))
    loss = (a * b).sum()
    loss.backward()  # ok
    import pytest
    with pytest.raises(RuntimeError, match="tape is empty"):
        loss.backward()
