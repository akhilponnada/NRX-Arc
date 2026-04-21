use std::sync::{Arc, OnceLock};
use cudarc::driver::{CudaContext, CudaSlice};
use pyo3::prelude::*;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Device {
    Cpu,
    Cuda,
}

pub fn device_str(d: Device) -> String {
    match d {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda => "cuda:0".to_string(),
    }
}

/// Parse the user-facing device string. Accepts "cpu", "cuda", or "cuda:0".
/// We're single-GPU only, so "cuda" and "cuda:0" are aliases.
pub fn parse_device(s: &str) -> PyResult<Device> {
    match s {
        "cpu" => Ok(Device::Cpu),
        "cuda" | "cuda:0" => Ok(Device::Cuda),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown device {:?}, expected 'cpu', 'cuda', or 'cuda:0'",
            other
        ))),
    }
}

/// Tensor-internal storage. CPU is a plain heap vec; CUDA is a device buffer
/// owned by the cudarc crate.
///
/// IMPORTANT: `Cuda` intentionally holds only a `CudaSlice<f32>` and NOT an
/// `Arc<CudaContext>`. The context is a process-wide singleton with 'static
/// lifetime (see `cuda_context` below), so the slice is always valid for the
/// program's lifetime. Adding an Arc here would cost one atomic refcount per
/// tensor for zero benefit. Do NOT "fix" this without reading the Phase 2
/// design notes in CLAUDE.md.
pub enum Storage {
    Cpu(Vec<f32>),
    Cuda(CudaSlice<f32>),
}

impl Storage {
    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            Storage::Cuda(_) => Device::Cuda,
        }
    }

    pub fn numel(&self) -> usize {
        match self {
            Storage::Cpu(v) => v.len(),
            Storage::Cuda(s) => s.len(),
        }
    }

    /// Borrow the CPU slice. Panics if storage is on CUDA — callers are
    /// expected to dispatch on `device()` first.
    pub fn cpu(&self) -> &[f32] {
        match self {
            Storage::Cpu(v) => v.as_slice(),
            Storage::Cuda(_) => {
                panic!("Storage::cpu() called on CUDA storage; dispatch on device() first")
            }
        }
    }

    /// Mutable borrow of the CPU vec. Same caveat as `cpu`.
    pub fn cpu_mut(&mut self) -> &mut Vec<f32> {
        match self {
            Storage::Cpu(v) => v,
            Storage::Cuda(_) => {
                panic!("Storage::cpu_mut() called on CUDA storage")
            }
        }
    }
}

static CUDA_CTX: OnceLock<Arc<CudaContext>> = OnceLock::new();

/// Lazily initialize and return the process-wide CUDA context for device 0.
/// All CUDA tensors share this single context. The OnceLock guarantees
/// init-once semantics and a 'static lifetime for the Arc.
pub fn cuda_context() -> PyResult<&'static Arc<CudaContext>> {
    if let Some(ctx) = CUDA_CTX.get() {
        return Ok(ctx);
    }
    let ctx = CudaContext::new(0).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "failed to initialize CUDA device 0: {}",
            e
        ))
    })?;
    let _ = CUDA_CTX.set(ctx);
    Ok(CUDA_CTX.get().expect("CUDA_CTX just set"))
}

/// Copy host -> a fresh CUDA buffer.
pub fn host_to_cuda(host: &[f32]) -> PyResult<CudaSlice<f32>> {
    let ctx = cuda_context()?;
    let stream = ctx.default_stream();
    stream.clone_htod(host).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("htod copy failed: {}", e))
    })
}

/// Copy CUDA -> a fresh host vec.
pub fn cuda_to_host(slice: &CudaSlice<f32>) -> PyResult<Vec<f32>> {
    let ctx = cuda_context()?;
    let stream = ctx.default_stream();
    stream.clone_dtoh(slice).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("dtoh copy failed: {}", e))
    })
}

/// Allocate a fresh zeroed CUDA buffer of `n` elements.
pub fn cuda_zeros(n: usize) -> PyResult<CudaSlice<f32>> {
    let ctx = cuda_context()?;
    let stream = ctx.default_stream();
    stream.alloc_zeros::<f32>(n).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("cuda alloc_zeros failed: {}", e))
    })
}

/// Clone an existing CUDA buffer into a fresh allocation on the same device.
pub fn cuda_clone(slice: &CudaSlice<f32>) -> PyResult<CudaSlice<f32>> {
    let ctx = cuda_context()?;
    let stream = ctx.default_stream();
    stream.clone_dtod(slice).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("dtod copy failed: {}", e))
    })
}
