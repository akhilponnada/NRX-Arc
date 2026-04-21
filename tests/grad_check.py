"""Finite-difference gradient checks for nrx-arc Phase 1 ops.

Config: eps=1e-4, rtol=1e-3, double-sided FD: (f(x+eps) - f(x-eps)) / 2eps.
"""
import numpy as np
import pytest
import nrx_arc as nx

EPS  = 1e-4
RTOL = 1e-3
ATOL = 1e-5


def grad_check(make_loss, x_np, name=""):
    """make_loss(t: Tensor) -> Tensor (must be scalar). Compares analytic vs FD numeric grad of x."""
    nx.clear_tape()

    # Analytic
    x_t = nx.from_numpy(x_np.astype(np.float32))
    loss_t = make_loss(x_t)
    loss_t.backward()
    g = x_t.grad
    assert g is not None, f"{name}: x.grad is None after backward"
    analytic = np.asarray(g, dtype=np.float32).reshape(-1)

    # Numerical
    n = x_np.size
    numeric = np.zeros(n, dtype=np.float32)
    for i in range(n):
        x_p = x_np.astype(np.float32).copy(); x_p.flat[i] += EPS
        x_m = x_np.astype(np.float32).copy(); x_m.flat[i] -= EPS
        nx.clear_tape()
        f_p = float(make_loss(nx.from_numpy(x_p)).to_list()[0])
        nx.clear_tape()
        f_m = float(make_loss(nx.from_numpy(x_m)).to_list()[0])
        numeric[i] = (f_p - f_m) / (2 * EPS)

    diff = np.abs(analytic - numeric)
    denom = np.maximum(np.abs(analytic), np.abs(numeric))
    rel = diff / np.maximum(denom, 1e-8)
    max_rel, max_abs = float(rel.max()), float(diff.max())

    ok = bool(((rel < RTOL) | (diff < ATOL)).all())
    if not ok:
        raise AssertionError(
            f"{name}: FAILED max_rel={max_rel:.3e} max_abs={max_abs:.3e}\n"
            f"  analytic={analytic}\n  numeric={numeric}"
        )
    print(f"  {name}: PASS (max_rel={max_rel:.2e}, max_abs={max_abs:.2e})")


# ─── add ──────────────────────────────────────────────────────────────────

def test_add_via_manual_grad_same_shape():
    """Round 1 only: skip the loss-reduction layer; seed the output gradient
    manually and verify input gradients element-wise."""
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((3, 4)).astype(np.float32)
    b_np = rng.standard_normal((3, 4)).astype(np.float32)
    nx.clear_tape()
    a_t = nx.from_numpy(a_np)
    b_t = nx.from_numpy(b_np)
    out = a_t.add(b_t)
    # Seed upstream grad manually then run backward (loss=identity-of-out, sum-of-ones upstream).
    # We can't seed grad directly via Python in round 1, so use loss = sum of (out + neg) tricks?
    # Simpler: backward() seeds 1.0 into loss tensor regardless of shape (per tape.rs). Use out itself as "loss".
    out.backward()
    # For y = a + b, dy/da = 1, dy/db = 1. With upstream seeded as ones: grad_a = ones, grad_b = ones.
    ga = np.asarray(a_t.grad)
    gb = np.asarray(b_t.grad)
    assert ga.shape == a_np.shape and gb.shape == b_np.shape
    assert np.allclose(ga, np.ones_like(a_np), atol=ATOL), f"grad_a: {ga}"
    assert np.allclose(gb, np.ones_like(b_np), atol=ATOL), f"grad_b: {gb}"
    print("  add(same_shape) manual-grad: PASS")


def test_add_via_manual_grad_left_scalar():
    rng = np.random.default_rng(1)
    s_np = rng.standard_normal((1,)).astype(np.float32)
    b_np = rng.standard_normal((3, 4)).astype(np.float32)
    nx.clear_tape()
    s_t = nx.from_numpy(s_np)
    b_t = nx.from_numpy(b_np)
    out = s_t.add(b_t)
    out.backward()
    # y = s + b (broadcast). dy/ds at each elt = 1, summed over 12 elts => grad_s = [12].
    # dy/db = 1 elementwise.
    gs = np.asarray(s_t.grad)
    gb = np.asarray(b_t.grad)
    assert np.allclose(gs, np.array([12.0], dtype=np.float32), atol=ATOL), f"grad_s: {gs}"
    assert np.allclose(gb, np.ones_like(b_np), atol=ATOL), f"grad_b: {gb}"
    print("  add(left_scalar) manual-grad: PASS")


def test_add_via_manual_grad_right_scalar():
    rng = np.random.default_rng(2)
    a_np = rng.standard_normal((3, 4)).astype(np.float32)
    s_np = rng.standard_normal((1,)).astype(np.float32)
    nx.clear_tape()
    a_t = nx.from_numpy(a_np)
    s_t = nx.from_numpy(s_np)
    out = a_t.add(s_t)
    out.backward()
    ga = np.asarray(a_t.grad)
    gs = np.asarray(s_t.grad)
    assert np.allclose(ga, np.ones_like(a_np), atol=ATOL), f"grad_a: {ga}"
    assert np.allclose(gs, np.array([12.0], dtype=np.float32), atol=ATOL), f"grad_s: {gs}"
    print("  add(right_scalar) manual-grad: PASS")


# ─── mul ──────────────────────────────────────────────────────────────────

def test_mul_via_manual_grad_same_shape():
    rng = np.random.default_rng(10)
    a_np = rng.standard_normal((3, 4)).astype(np.float32)
    b_np = rng.standard_normal((3, 4)).astype(np.float32)
    nx.clear_tape()
    a_t = nx.from_numpy(a_np)
    b_t = nx.from_numpy(b_np)
    out = a_t.mul(b_t)
    out.backward()
    # upstream = ones; dL/da = b, dL/db = a
    ga = np.asarray(a_t.grad)
    gb = np.asarray(b_t.grad)
    assert np.allclose(ga, b_np, atol=1e-5), f"grad_a expected b, got {ga}"
    assert np.allclose(gb, a_np, atol=1e-5), f"grad_b expected a, got {gb}"
    print("  mul(same_shape) manual-grad: PASS")


def test_mul_via_manual_grad_left_scalar():
    rng = np.random.default_rng(11)
    s_np = rng.standard_normal((1,)).astype(np.float32)
    b_np = rng.standard_normal((3, 4)).astype(np.float32)
    nx.clear_tape()
    s_t = nx.from_numpy(s_np)
    b_t = nx.from_numpy(b_np)
    out = s_t.mul(b_t)
    out.backward()
    # y = s*b, upstream=ones. dL/ds = sum(b); dL/db = s_scalar broadcast
    gs = np.asarray(s_t.grad)
    gb = np.asarray(b_t.grad)
    assert np.allclose(gs, np.array([b_np.sum()], dtype=np.float32), atol=1e-4), f"grad_s: {gs}"
    assert np.allclose(gb, np.full_like(b_np, s_np[0]), atol=1e-5), f"grad_b: {gb}"
    print("  mul(left_scalar) manual-grad: PASS")


def test_mul_via_manual_grad_right_scalar():
    rng = np.random.default_rng(12)
    a_np = rng.standard_normal((3, 4)).astype(np.float32)
    s_np = rng.standard_normal((1,)).astype(np.float32)
    nx.clear_tape()
    a_t = nx.from_numpy(a_np)
    s_t = nx.from_numpy(s_np)
    out = a_t.mul(s_t)
    out.backward()
    ga = np.asarray(a_t.grad)
    gs = np.asarray(s_t.grad)
    assert np.allclose(ga, np.full_like(a_np, s_np[0]), atol=1e-5), f"grad_a: {ga}"
    assert np.allclose(gs, np.array([a_np.sum()], dtype=np.float32), atol=1e-4), f"grad_s: {gs}"
    print("  mul(right_scalar) manual-grad: PASS")


# ─── matmul ───────────────────────────────────────────────────────────────

def test_matmul_forward_and_grad():
    rng = np.random.default_rng(20)
    a_np = rng.standard_normal((3, 5)).astype(np.float32)
    b_np = rng.standard_normal((5, 4)).astype(np.float32)
    nx.clear_tape()
    a_t = nx.from_numpy(a_np)
    b_t = nx.from_numpy(b_np)
    out = a_t.matmul(b_t)
    # forward
    out_np = np.asarray(out.to_numpy())
    ref = a_np @ b_np
    fwd_diff = float(np.abs(out_np - ref).max())
    assert fwd_diff < 1e-4, f"matmul forward max diff {fwd_diff}"
    # backward (upstream = ones)
    out.backward()
    ga = np.asarray(a_t.grad)
    gb = np.asarray(b_t.grad)
    upstream = np.ones((3, 4), dtype=np.float32)
    expected_ga = upstream @ b_np.T
    expected_gb = a_np.T @ upstream
    da_diff = float(np.abs(ga - expected_ga).max())
    db_diff = float(np.abs(gb - expected_gb).max())
    assert da_diff < 1e-4, f"matmul dA max diff {da_diff}"
    assert db_diff < 1e-4, f"matmul dB max diff {db_diff}"
    print(f"  matmul: PASS (fwd_diff={fwd_diff:.2e}, dA_diff={da_diff:.2e}, dB_diff={db_diff:.2e})")


def test_matmul_then_add_chain():
    """Verify tape order: out = (A@B) + bias, backward through add then matmul."""
    rng = np.random.default_rng(21)
    a_np = rng.standard_normal((3, 5)).astype(np.float32)
    b_np = rng.standard_normal((5, 4)).astype(np.float32)
    bias_np = rng.standard_normal((3, 4)).astype(np.float32)
    nx.clear_tape()
    a_t = nx.from_numpy(a_np)
    b_t = nx.from_numpy(b_np)
    bias_t = nx.from_numpy(bias_np)
    out = a_t.matmul(b_t).add(bias_t)
    out.backward()
    upstream = np.ones((3, 4), dtype=np.float32)
    assert np.allclose(np.asarray(a_t.grad),    upstream @ b_np.T, atol=1e-4)
    assert np.allclose(np.asarray(b_t.grad),    a_np.T @ upstream, atol=1e-4)
    assert np.allclose(np.asarray(bias_t.grad), upstream,          atol=1e-5)
    print("  matmul + add chain: PASS")


def test_relu_manual_grad():
    import nrx_arc as nx
    nx.clear_tape()
    rng = np.random.default_rng(seed=7)
    # Avoid |x| < 0.05 so the subgradient at zero never matters
    raw = rng.standard_normal(size=(4, 5)).astype(np.float32)
    raw = np.where(np.abs(raw) < 0.05, np.sign(raw + 1e-6) * 0.05, raw).astype(np.float32)
    x = nx.from_numpy(raw)
    y = x.relu()
    # Seed upstream = ones across y, then check x.grad equals (raw > 0).astype(f32)
    y_np = np.asarray(y.to_numpy())
    expected_fwd = np.maximum(raw, 0.0)
    fwd_diff = float(np.max(np.abs(y_np - expected_fwd)))
    assert fwd_diff < 1e-6, f"relu forward diff {fwd_diff}"
    # backward with upstream = ones-of-shape(y)

 

    y.backward()
    g = np.asarray(x.grad)
    expected_g = (raw > 0.0).astype(np.float32)
    g_diff = float(np.max(np.abs(g - expected_g)))
    assert g_diff < 1e-6, f"relu grad diff {g_diff}"
    print(f"  relu manual-grad: PASS (fwd_diff={fwd_diff:.2e}, g_diff={g_diff:.2e})")


def test_sum_manual_grad():
    import nrx_arc as nx
    nx.clear_tape()
    rng = np.random.default_rng(seed=11)
    raw = rng.standard_normal(size=(3, 4)).astype(np.float32)
    x = nx.from_numpy(raw)
    s = x.sum()
    s_np = np.asarray(s.to_numpy())
    fwd_diff = float(np.abs(s_np - raw.sum()))
    assert fwd_diff < 1e-4, f"sum forward diff {fwd_diff}"
    s.backward()  # upstream scalar = 1
    g = np.asarray(x.grad)
    expected = np.ones_like(raw)
    g_diff = float(np.max(np.abs(g - expected)))
    assert g_diff < 1e-6, f"sum grad diff {g_diff}"
    print(f"  sum manual-grad: PASS (fwd_diff={fwd_diff:.2e}, g_diff={g_diff:.2e})")


def _fd_grad(make_loss, x_np, eps=1e-4):
    """Double-sided finite-difference grad of scalar loss.
    Computes the reference in fp64 (callers must use fp64 numpy ops inside make_loss)
    so FD roundoff doesn't eat into the 1e-3 rtol budget on N-element sums."""
    x64 = x_np.astype(np.float64)
    g = np.zeros_like(x64)
    flat = x64.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        f_plus = float(make_loss(x64))
        flat[i] = old - eps
        f_minus = float(make_loss(x64))
        flat[i] = old
        g.reshape(-1)[i] = (f_plus - f_minus) / (2.0 * eps)
    return g.astype(np.float32)


def _analytic_grad(make_tensor_loss, x_np):
    """Run the framework forward+backward, return x.grad as numpy."""
    import nrx_arc as nx
    nx.clear_tape()
    x = nx.from_numpy(x_np.copy())
    loss = make_tensor_loss(x)
    loss.backward()
    return np.asarray(x.grad).copy()


def test_fd_add_sum():
    """End-to-end FD: loss = sum(x + c), expected dL/dx = 1."""
    import nrx_arc as nx
    rng = np.random.default_rng(seed=21)
    x_np = rng.standard_normal(size=(2, 3)).astype(np.float32)
    c_np = rng.standard_normal(size=(2, 3)).astype(np.float32)
    c_np64 = c_np.astype(np.float64)
    def fd_loss(x):
        return (x + c_np64).sum()
    def tensor_loss(x):
        c = nx.from_numpy(c_np)
        return (x + c).sum()
    g_fd = _fd_grad(fd_loss, x_np)
    g_an = _analytic_grad(tensor_loss, x_np)
    diff = float(np.max(np.abs(g_fd - g_an)))
    rel = diff / max(1e-12, float(np.max(np.abs(g_fd))))
    assert rel < 1e-3, f"fd_add_sum rel={rel} diff={diff}"
    print(f"  FD add+sum: PASS (rel={rel:.2e})")


def test_fd_mul_sum():
    """loss = sum(x * c), expected dL/dx = c."""
    import nrx_arc as nx
    rng = np.random.default_rng(seed=22)
    x_np = rng.standard_normal(size=(3, 2)).astype(np.float32)
    c_np = rng.standard_normal(size=(3, 2)).astype(np.float32)
    c_np64 = c_np.astype(np.float64)
    def fd_loss(x):
        return (x * c_np64).sum()
    def tensor_loss(x):
        c = nx.from_numpy(c_np)
        return (x * c).sum()
    g_fd = _fd_grad(fd_loss, x_np)
    g_an = _analytic_grad(tensor_loss, x_np)
    diff = float(np.max(np.abs(g_fd - g_an)))
    rel = diff / max(1e-12, float(np.max(np.abs(g_fd))))
    assert rel < 1e-3, f"fd_mul_sum rel={rel} diff={diff}"
    print(f"  FD mul+sum: PASS (rel={rel:.2e})")


def test_fd_matmul_sum():
    """loss = sum(x @ B), expected dL/dx = ones(M,N) @ B^T."""
    import nrx_arc as nx
    rng = np.random.default_rng(seed=23)
    x_np = rng.standard_normal(size=(3, 4)).astype(np.float32)
    b_np = rng.standard_normal(size=(4, 5)).astype(np.float32)
    b_np64 = b_np.astype(np.float64)
    def fd_loss(x):
        return (x @ b_np64).sum()
    def tensor_loss(x):
        b = nx.from_numpy(b_np)
        return (x @ b).sum()
    g_fd = _fd_grad(fd_loss, x_np)
    g_an = _analytic_grad(tensor_loss, x_np)
    diff = float(np.max(np.abs(g_fd - g_an)))
    rel = diff / max(1e-12, float(np.max(np.abs(g_fd))))
    assert rel < 1e-3, f"fd_matmul_sum rel={rel} diff={diff}"
    print(f"  FD matmul+sum: PASS (rel={rel:.2e})")


def test_fd_relu_sum():
    """loss = sum(relu(x)), expected dL/dx = (x > 0)."""
    import nrx_arc as nx
    rng = np.random.default_rng(seed=24)
    x_np = rng.standard_normal(size=(4, 5)).astype(np.float32)
    # Avoid x near zero so FD doesn't straddle the kink
    x_np = np.where(np.abs(x_np) < 0.1, np.sign(x_np + 1e-6) * 0.1, x_np).astype(np.float32)
    def fd_loss(x):
        return np.maximum(x, 0.0).sum()
    def tensor_loss(x):
        return x.relu().sum()
    g_fd = _fd_grad(fd_loss, x_np)
    g_an = _analytic_grad(tensor_loss, x_np)
    diff = float(np.max(np.abs(g_fd - g_an)))
    rel = diff / max(1e-12, float(np.max(np.abs(g_fd))))
    assert rel < 1e-3, f"fd_relu_sum rel={rel} diff={diff}"
    print(f"  FD relu+sum: PASS (rel={rel:.2e})")


def test_fd_chain_matmul_relu_sum():
    """loss = sum(relu(x @ B)). Chains all 4 ops; the realistic Phase 1 pattern."""
    import nrx_arc as nx
    rng = np.random.default_rng(seed=25)
    x_np = rng.standard_normal(size=(3, 4)).astype(np.float32) * 0.5
    b_np = rng.standard_normal(size=(4, 5)).astype(np.float32) * 0.5
    b_np64 = b_np.astype(np.float64)
    def fd_loss(x):
        return np.maximum(x @ b_np64, 0.0).sum()
    def tensor_loss(x):
        b = nx.from_numpy(b_np)
        return (x @ b).relu().sum()
    g_fd = _fd_grad(fd_loss, x_np)
    g_an = _analytic_grad(tensor_loss, x_np)
    diff = float(np.max(np.abs(g_fd - g_an)))
    rel = diff / max(1e-12, float(np.max(np.abs(g_fd))))
    assert rel < 1e-2, f"fd_chain rel={rel} diff={diff}"
    print(f"  FD chain (matmul+relu+sum): PASS (rel={rel:.2e})")


# ─── round 6: neg, scalar_mul, row-vec broadcast add, sub composition ──

def test_fd_neg_sum():
    """loss = sum(neg(x) * c). Expected dL/dx = -c."""
    import nrx_arc as nx
    rng = np.random.default_rng(seed=31)
    x_np = rng.standard_normal(size=(3, 4)).astype(np.float32)
    c_np = rng.standard_normal(size=(3, 4)).astype(np.float32)
    c_np64 = c_np.astype(np.float64)
    def fd_loss(x):
        return ((-x) * c_np64).sum()
    def tensor_loss(x):
        c = nx.from_numpy(c_np)
        return ((-x) * c).sum()
    g_fd = _fd_grad(fd_loss, x_np)
    g_an = _analytic_grad(tensor_loss, x_np)
    diff = float(np.max(np.abs(g_fd - g_an)))
    rel = diff / max(1e-12, float(np.max(np.abs(g_fd))))
    assert rel < 1e-3, f"fd_neg rel={rel}"
    print(f"  FD neg: PASS (rel={rel:.2e})")


def test_fd_scalar_mul_sum():
    """loss = sum(scalar_mul(x, s)). Expected dL/dx = s."""
    import nrx_arc as nx
    rng = np.random.default_rng(seed=32)
    x_np = rng.standard_normal(size=(4, 3)).astype(np.float32)
    s_val = 0.37
    def fd_loss(x):
        return (s_val * x).sum()
    def tensor_loss(x):
        return x.scalar_mul(s_val).sum()
    g_fd = _fd_grad(fd_loss, x_np)
    g_an = _analytic_grad(tensor_loss, x_np)
    diff = float(np.max(np.abs(g_fd - g_an)))
    rel = diff / max(1e-12, float(np.max(np.abs(g_fd))))
    assert rel < 1e-3, f"fd_scalar_mul rel={rel}"
    print(f"  FD scalar_mul: PASS (rel={rel:.2e})")


def test_fd_sub_sum():
    """loss = sum((x - c) * w). Expected dL/dx = w (b's grad would be -w)."""
    import nrx_arc as nx
    rng = np.random.default_rng(seed=33)
    x_np = rng.standard_normal(size=(2, 5)).astype(np.float32)
    c_np = rng.standard_normal(size=(2, 5)).astype(np.float32)
    w_np = rng.standard_normal(size=(2, 5)).astype(np.float32)
    c_np64, w_np64 = c_np.astype(np.float64), w_np.astype(np.float64)
    def fd_loss(x):
        return ((x - c_np64) * w_np64).sum()
    def tensor_loss(x):
        c = nx.from_numpy(c_np)
        w = nx.from_numpy(w_np)
        return ((x - c) * w).sum()
    g_fd = _fd_grad(fd_loss, x_np)
    g_an = _analytic_grad(tensor_loss, x_np)
    diff = float(np.max(np.abs(g_fd - g_an)))
    rel = diff / max(1e-12, float(np.max(np.abs(g_fd))))
    assert rel < 1e-3, f"fd_sub rel={rel}"
    print(f"  FD sub: PASS (rel={rel:.2e})")


def test_fd_row_bias_add():
    """Row-vec broadcast add: loss = sum((x @ W + b) * c).
    Verifies x's grad and (separately) b's grad."""
    import nrx_arc as nx
    rng = np.random.default_rng(seed=34)
    B, D_in, D_out = 4, 3, 5
    x_np = rng.standard_normal(size=(B, D_in)).astype(np.float32)
    W_np = rng.standard_normal(size=(D_in, D_out)).astype(np.float32)
    b_np = rng.standard_normal(size=(D_out,)).astype(np.float32)
    c_np = rng.standard_normal(size=(B, D_out)).astype(np.float32)

    # Check d/dx
    W64, b64, c64 = W_np.astype(np.float64), b_np.astype(np.float64), c_np.astype(np.float64)
    def fd_loss_x(x):
        return (((x @ W64) + b64) * c64).sum()
    def tensor_loss_x(x):
        W = nx.from_numpy(W_np); b = nx.from_numpy(b_np); c = nx.from_numpy(c_np)
        return (((x @ W) + b) * c).sum()
    g_fd_x = _fd_grad(fd_loss_x, x_np)
    g_an_x = _analytic_grad(tensor_loss_x, x_np)
    rel_x = float(np.max(np.abs(g_fd_x - g_an_x))) / max(1e-12, float(np.max(np.abs(g_fd_x))))
    assert rel_x < 1e-3, f"row-bias dL/dx rel={rel_x}"

    # Check d/db (the broadcast direction)
    x64 = x_np.astype(np.float64)
    def fd_loss_b(b):
        return (((x64 @ W64) + b) * c64).sum()
    def tensor_loss_b(b):
        x = nx.from_numpy(x_np); W = nx.from_numpy(W_np); c = nx.from_numpy(c_np)
        return (((x @ W) + b) * c).sum()
    g_fd_b = _fd_grad(fd_loss_b, b_np)
    g_an_b = _analytic_grad(tensor_loss_b, b_np)
    rel_b = float(np.max(np.abs(g_fd_b - g_an_b))) / max(1e-12, float(np.max(np.abs(g_fd_b))))
    assert rel_b < 1e-3, f"row-bias dL/db rel={rel_b}"
    print(f"  FD row-bias add: PASS (dL/dx rel={rel_x:.2e}, dL/db rel={rel_b:.2e})")


def test_fd_mse_loss_chain():
    """Full MSE loss chain end-to-end:  L = (1/B) * sum((x @ W + b - y)^2).
    Exercises every op the MLP test will use, including in the same composition."""
    import nrx_arc as nx
    rng = np.random.default_rng(seed=35)
    B, D_in, D_out = 5, 3, 2
    x_np = rng.standard_normal(size=(B, D_in)).astype(np.float32)
    W_np = rng.standard_normal(size=(D_in, D_out)).astype(np.float32) * 0.3
    b_np = rng.standard_normal(size=(D_out,)).astype(np.float32) * 0.3
    y_np = rng.standard_normal(size=(B, D_out)).astype(np.float32)

    inv_B = 1.0 / B
    x64, b64, y64 = x_np.astype(np.float64), b_np.astype(np.float64), y_np.astype(np.float64)

    # Check d/dW (treat W as the FD variable)
    def fd_loss_W(W):
        diff = (x64 @ W) + b64 - y64
        return inv_B * (diff * diff).sum()
    def tensor_loss_W(W):
        x = nx.from_numpy(x_np); b = nx.from_numpy(b_np); y = nx.from_numpy(y_np)
        diff = (x @ W) + b - y
        return (diff * diff).sum().scalar_mul(inv_B)
    g_fd = _fd_grad(fd_loss_W, W_np)
    g_an = _analytic_grad(tensor_loss_W, W_np)
    rel = float(np.max(np.abs(g_fd - g_an))) / max(1e-12, float(np.max(np.abs(g_fd))))
    assert rel < 1e-3, f"mse chain dL/dW rel={rel}"
    print(f"  FD MSE chain (dL/dW): PASS (rel={rel:.2e})")


def test_sgd_update_inplace():
    """Sanity: sgd_update_ does data -= lr * grad in place, no tape activity."""
    import nrx_arc as nx
    nx.clear_tape()
    rng = np.random.default_rng(seed=36)
    raw = rng.standard_normal(size=(2, 3)).astype(np.float32)
    t = nx.from_numpy(raw.copy())
    # Manually inject a known grad via a tiny graph
    s = (t * t).sum()
    s.backward()  # grad of sum(t*t) is 2*t
    g_before = np.asarray(t.grad).copy()
    expected_g = 2.0 * raw
    assert np.allclose(g_before, expected_g, atol=1e-5), "sanity: grad should be 2*t"
    lr = 0.1
    t.sgd_update_(lr)  # in-place: t.data -= 0.1 * (2*raw)
    new_data = np.asarray(t.to_numpy())
    expected = raw - lr * expected_g
    diff = float(np.max(np.abs(new_data - expected)))
    assert diff < 1e-6, f"sgd_update_ produced wrong data, diff={diff}"
    # grad must remain (zero_grad is a separate call)
    assert t.grad is not None, "sgd_update_ should not clear .grad"
    print(f"  sgd_update_ in-place: PASS (max diff={diff:.2e})")
