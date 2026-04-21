"""Phase 2 step 1: device infrastructure tests.

Verifies that:
  1. CPU is the default device (back-compat with all Phase 1 code).
  2. A CUDA tensor reports its device as "cuda:0".
  3. .to() round-trips byte-identically through the GPU.
  4. Mixing devices in a binary op raises a clear RuntimeError naming both.
"""
import numpy as np
import pytest
import nrx_arc as nx


def test_default_device_is_cpu():
    t = nx.zeros([3])
    assert t.device == "cpu", f"expected 'cpu', got {t.device!r}"


def test_cuda_zeros_reports_cuda_zero():
    t = nx.zeros([3], device="cuda")
    assert t.device == "cuda:0", f"expected 'cuda:0', got {t.device!r}"
    # The data should be zeros even after the dtoh round-trip.
    np.testing.assert_array_equal(t.to_numpy(), np.zeros((3,), dtype=np.float32))


def test_to_cuda_to_cpu_round_trip_is_byte_identical():
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((4, 5)).astype(np.float32)
    t_cpu = nx.from_numpy(arr)
    assert t_cpu.device == "cpu"
    t_gpu = t_cpu.to("cuda")
    assert t_gpu.device == "cuda:0"
    t_back = t_gpu.to("cpu")
    assert t_back.device == "cpu"
    # Byte-identical means exact float equality, not allclose.
    np.testing.assert_array_equal(t_back.to_numpy(), arr)


def test_mixed_device_add_raises_runtime_error():
    a = nx.zeros([3])
    b = nx.zeros([3], device="cuda")
    with pytest.raises(RuntimeError) as excinfo:
        _ = a + b
    msg = str(excinfo.value)
    assert "cpu" in msg and "cuda" in msg, (
        f"error message should name both devices; got: {msg!r}"
    )
