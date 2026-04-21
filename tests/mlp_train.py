"""Phase 1 final deliverable: end-to-end MLP training on 3-blob synthetic data.

Architecture: Linear(10, 32) -> ReLU -> Linear(32, 3)
Init: He/Kaiming on W (std=sqrt(2/fan_in)), zeros on b
Loss: MSE against one-hot labels (no softmax/CE yet — Phase 3)
Optimizer: SGD lr=0.05, full-batch
Convergence: train acc 99% (early stop) or 500 epochs; require test acc >= 95%
"""
import time
import numpy as np
import pytest
import nrx_arc as nx
from nrx_arc import nn, optim


SEED = 42
N_POINTS = 100
N_DIM = 10
N_CLASSES = 3
TRAIN_FRAC = 0.8
LR = 0.05
MAX_EPOCHS = 500
EARLY_STOP_TRAIN_ACC = 0.99


def make_data(seed):
    rng = np.random.default_rng(seed)
    # Random unit-vector centers, scaled by 3.0. We QR-orthogonalize first:
    # in D=10, random unit vectors have pairwise cos~1/sqrt(D) ~= 0.32 which leaves
    # centers only ~3.5 apart vs a within-class radius of sqrt(D) ~= 3.16, putting
    # Bayes accuracy at only ~80%. Orthogonalizing forces center separation to
    # 3*sqrt(2) ~= 4.24, restoring a cleanly separable task.
    raw_centers = rng.standard_normal(size=(N_CLASSES, N_DIM)).astype(np.float32)
    Q, _ = np.linalg.qr(raw_centers.T)  # Q is (D, N_CLASSES) with orthonormal cols
    centers = (Q.T * 3.0).astype(np.float32)
    pts_per_class = N_POINTS // N_CLASSES
    leftover = N_POINTS - pts_per_class * N_CLASSES
    counts = [pts_per_class + (1 if i < leftover else 0) for i in range(N_CLASSES)]

    Xs, ys = [], []
    for c, n in enumerate(counts):
        Xs.append((rng.standard_normal(size=(n, N_DIM)) + centers[c]).astype(np.float32))
        ys.append(np.full((n,), c, dtype=np.int64))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    # Shuffle
    perm = rng.permutation(N_POINTS)
    X, y = X[perm], y[perm]

    n_train = int(TRAIN_FRAC * N_POINTS)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    one_hot_train = np.zeros((n_train, N_CLASSES), dtype=np.float32)
    one_hot_train[np.arange(n_train), y_train] = 1.0
    return X_train, one_hot_train, y_train, X_test, y_test


class MLP(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.fc1 = nn.Linear(N_DIM, 32, seed=seed)
        self.fc2 = nn.Linear(32, N_CLASSES, seed=seed + 1)

    def forward(self, x):
        return self.fc2(self.fc1(x).relu())


def predict_classes(model, X_np):
    nx.clear_tape()
    x = nx.from_numpy(X_np)
    pred = model(x)
    pred_np = np.asarray(pred.to_numpy()).reshape(-1, N_CLASSES)
    nx.clear_tape()  # discard graph; this was inference only
    return pred_np.argmax(axis=1)


def test_mlp_blobs_converges():
    np.random.seed(SEED)
    X_train, Y_train_oh, y_train, X_test, y_test = make_data(seed=SEED)
    B = X_train.shape[0]
    inv_B = 1.0 / B

    model = MLP(seed=SEED)
    opt = optim.SGD(model.parameters(), lr=LR)

    losses = []
    train_accs = []
    converged_epoch = None
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        opt.zero_grad()
        # Build constants fresh each epoch (avoids stale-grad ambiguity on
        # non-parameter inputs; constants don't need retained grad).
        x = nx.from_numpy(X_train)
        y_oh = nx.from_numpy(Y_train_oh)
        pred = model(x)
        diff = pred - y_oh
        loss = (diff * diff).sum().scalar_mul(inv_B)
        loss_val = float(np.asarray(loss.to_numpy()))
        loss.backward()
        opt.step()
        losses.append(loss_val)

        # Train acc check (cheap on full batch). Uses a fresh forward — the
        # graph from training was already consumed by backward.
        if epoch % 5 == 0 or epoch == MAX_EPOCHS - 1:
            preds = predict_classes(model, X_train)
            train_acc = float((preds == y_train).mean())
            train_accs.append((epoch, train_acc))
            if train_acc >= EARLY_STOP_TRAIN_ACC and converged_epoch is None:
                converged_epoch = epoch

        if epoch % 25 == 0:
            print(f"  epoch {epoch:4d}  loss={loss_val:.6f}")

        if converged_epoch is not None:
            print(f"  epoch {epoch:4d}  loss={loss_val:.6f}  (early stop @ train_acc>=99%)")
            break

    wall = time.perf_counter() - t0
    final_loss = losses[-1]
    final_epoch = len(losses) - 1

    # Final eval
    train_preds = predict_classes(model, X_train)
    test_preds = predict_classes(model, X_test)
    train_acc = float((train_preds == y_train).mean())
    test_acc = float((test_preds == y_test).mean())

    print()
    print(f"  ── REPORT ──────────────────────────────────────────")
    print(f"  epochs run        : {final_epoch + 1} (max {MAX_EPOCHS})")
    print(f"  early-stop epoch  : {converged_epoch}")
    print(f"  final train loss  : {final_loss:.6f}")
    print(f"  final train acc   : {train_acc * 100:.2f}%")
    print(f"  final test  acc   : {test_acc * 100:.2f}%")
    print(f"  wall-clock        : {wall * 1000:.1f} ms")
    print(f"  ─────────────────────────────────────────────────────")

    # Monotonic-ish loss check: tolerate tiny non-monotonicity from fp32 noise
    # (count violations >1e-6 above prior epoch; require <5% of epochs).
    increases = sum(1 for i in range(1, len(losses)) if losses[i] > losses[i-1] + 1e-6)
    monotonic_frac = increases / max(1, len(losses) - 1)
    print(f"  loss-increase fraction: {monotonic_frac * 100:.2f}% of epochs")

    assert test_acc >= 0.95, f"test acc {test_acc} below 0.95"
    assert monotonic_frac < 0.05, f"loss not monotonic: {monotonic_frac*100:.1f}% of epochs increased"
