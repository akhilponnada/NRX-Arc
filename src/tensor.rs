use pyo3::prelude::*;
use numpy::{PyReadonlyArrayDyn, IntoPyArray, PyArrayMethods, PyUntypedArrayMethods};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use crate::tape;
use crate::device::{self, Device, Storage};

#[pyclass]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub storage: Storage,
    pub strides: Vec<usize>,
    pub grad: Option<Storage>,
}

impl Tensor {
    /// Build a tensor from any storage. The numel must match `shape.product()`.
    pub fn from_storage(shape: Vec<usize>, storage: Storage) -> Self {
        debug_assert_eq!(storage.numel(), shape.iter().product::<usize>().max(1));
        let strides = row_major_strides(&shape);
        Tensor { shape, storage, strides, grad: None }
    }

    /// Convenience constructor for CPU tensors. Most ops take this path because
    /// every op currently runs on CPU; CUDA ops will be added in later commits.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self::from_storage(shape, Storage::Cpu(data))
    }

    pub fn numel(&self) -> usize { self.shape.iter().product::<usize>().max(1) }
}

#[pymethods]
impl Tensor {
    fn shape(&self) -> Vec<usize> { self.shape.clone() }
    fn strides(&self) -> Vec<usize> { self.strides.clone() }

    /// Returns a flat list of the tensor's data. If the tensor is on CUDA,
    /// this triggers a dtoh copy.
    fn to_list(&self) -> PyResult<Vec<f32>> {
        match &self.storage {
            Storage::Cpu(v) => Ok(v.clone()),
            Storage::Cuda(s) => device::cuda_to_host(s),
        }
    }

    /// Returns a numpy view of the tensor's data. CUDA storage is dtoh-copied first.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let host: Vec<f32> = match &self.storage {
            Storage::Cpu(v) => v.clone(),
            Storage::Cuda(s) => device::cuda_to_host(s)?,
        };
        let arr = host.into_pyarray(py);
        let reshaped = arr.reshape(self.shape.clone())?;
        Ok(reshaped.into_any())
    }

    #[getter]
    fn grad<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.grad {
            None => Ok(None),
            Some(g) => {
                let host: Vec<f32> = match g {
                    Storage::Cpu(v) => v.clone(),
                    Storage::Cuda(s) => device::cuda_to_host(s)?,
                };
                let arr = host.into_pyarray(py);
                let reshaped = arr.reshape(self.shape.clone())?;
                Ok(Some(reshaped.into_any()))
            }
        }
    }

    #[getter]
    fn device(&self) -> String {
        device::device_str(self.storage.device())
    }

    fn zero_grad(&mut self) { self.grad = None; }

    /// Backpropagate gradients from this tensor through the autograd graph.
    ///
    /// Only meaningful on scalar tensors (numel == 1); the upstream gradient
    /// is seeded as 1.
    ///
    /// Consumes the tape: cannot be called twice on the same graph. Re-run
    /// the forward pass to build a new graph before calling backward again.
    ///
    /// Accumulates into `.grad` on input tensors. Call `zero_grad()` between
    /// training steps to avoid summing gradients across iterations.
    fn backward(slf: Bound<'_, Self>, py: Python) -> PyResult<()> {
        tape::backward(py, slf.unbind())
    }

    /// Return a NEW tensor with storage moved to `device`. The returned tensor
    /// has no grad and is not connected to the autograd tape — `.to()` is a
    /// leaf operation. Same-device `.to()` produces a cloned tensor (still a
    /// fresh allocation), matching PyTorch's behavior.
    fn to(&self, device: &str, py: Python) -> PyResult<Py<Tensor>> {
        let target = device::parse_device(device)?;
        let new_storage = match (&self.storage, target) {
            (Storage::Cpu(v), Device::Cpu) => Storage::Cpu(v.clone()),
            (Storage::Cpu(v), Device::Cuda) => Storage::Cuda(device::host_to_cuda(v)?),
            (Storage::Cuda(s), Device::Cpu) => Storage::Cpu(device::cuda_to_host(s)?),
            (Storage::Cuda(s), Device::Cuda) => Storage::Cuda(device::cuda_clone(s)?),
        };
        Py::new(py, Tensor::from_storage(self.shape.clone(), new_storage))
    }

    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?}, device={})", self.shape, device::device_str(self.storage.device()))
    }

    fn add(slf: Bound<'_, Self>, other: Bound<'_, Self>, py: Python) -> PyResult<Py<Tensor>> {
        crate::ops::add::forward(py, slf, other)
    }
    fn __add__(slf: Bound<'_, Self>, other: Bound<'_, Self>, py: Python) -> PyResult<Py<Tensor>> {
        crate::ops::add::forward(py, slf, other)
    }

    fn mul(slf: Bound<'_, Self>, other: Bound<'_, Self>, py: Python) -> PyResult<Py<Tensor>> {
        crate::ops::mul::forward(py, slf, other)
    }
    fn __mul__(slf: Bound<'_, Self>, other: Bound<'_, Self>, py: Python) -> PyResult<Py<Tensor>> {
        crate::ops::mul::forward(py, slf, other)
    }

    fn matmul(slf: Bound<'_, Self>, other: Bound<'_, Self>, py: Python) -> PyResult<Py<Tensor>> {
        crate::ops::matmul::forward(py, slf, other)
    }
    fn __matmul__(slf: Bound<'_, Self>, other: Bound<'_, Self>, py: Python) -> PyResult<Py<Tensor>> {
        crate::ops::matmul::forward(py, slf, other)
    }
    fn relu(slf: Bound<'_, Self>, py: Python) -> PyResult<Py<Tensor>> {
        crate::ops::relu::forward(py, slf)
    }
    fn sum(slf: Bound<'_, Self>, py: Python) -> PyResult<Py<Tensor>> {
        crate::ops::sum::forward(py, slf)
    }
    fn neg(slf: Bound<'_, Self>, py: Python) -> PyResult<Py<Tensor>> {
        crate::ops::neg::forward(py, slf)
    }
    fn __neg__(slf: Bound<'_, Self>, py: Python) -> PyResult<Py<Tensor>> {
        crate::ops::neg::forward(py, slf)
    }
    fn __sub__(slf: Bound<'_, Self>, other: Bound<'_, Self>, py: Python) -> PyResult<Py<Tensor>> {
        // a - b = a + neg(b). Two tape records, one composition.
        let neg_b = crate::ops::neg::forward(py, other)?;
        let neg_b_bound = neg_b.bind(py).clone();
        crate::ops::add::forward(py, slf, neg_b_bound)
    }
    fn scalar_mul(slf: Bound<'_, Self>, s: f32, py: Python) -> PyResult<Py<Tensor>> {
        crate::ops::scalar_mul::forward(py, slf, s)
    }

    /// In-place SGD update: self.data -= lr * self.grad. No-op if grad is None.
    /// Bypasses the tape (this is an optimizer step, not part of the forward graph).
    /// CPU-only in Phase 2 step 1; CUDA optimizer step lands once we have CUDA
    /// elementwise ops in place.
    fn sgd_update_(&mut self, lr: f32) -> PyResult<()> {
        if self.storage.device() == Device::Cuda {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "cuda sgd_update_ not implemented yet (Phase 2 step 2)"
            ));
        }
        let grad_vec = match &self.grad {
            None => return Ok(()),
            Some(Storage::Cpu(v)) => v.clone(),
            Some(Storage::Cuda(_)) => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "cuda sgd_update_ not implemented yet (Phase 2 step 2)"
                ));
            }
        };
        let data_vec = self.storage.cpu_mut();
        for (d, gi) in data_vec.iter_mut().zip(grad_vec.iter()) {
            *d -= lr * gi;
        }
        Ok(())
    }
}

pub fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() { return vec![]; }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[pyfunction]
#[pyo3(signature = (shape, *, device="cpu"))]
pub fn zeros(py: Python, shape: Vec<usize>, device: &str) -> PyResult<Py<Tensor>> {
    let dev = device::parse_device(device)?;
    let n: usize = shape.iter().product::<usize>().max(1);
    let storage = match dev {
        Device::Cpu => Storage::Cpu(vec![0.0; n]),
        Device::Cuda => Storage::Cuda(device::cuda_zeros(n)?),
    };
    Py::new(py, Tensor::from_storage(shape, storage))
}

#[pyfunction]
#[pyo3(signature = (shape, *, device="cpu"))]
pub fn ones(py: Python, shape: Vec<usize>, device: &str) -> PyResult<Py<Tensor>> {
    let dev = device::parse_device(device)?;
    let n: usize = shape.iter().product::<usize>().max(1);
    let host = vec![1.0f32; n];
    let storage = match dev {
        Device::Cpu => Storage::Cpu(host),
        Device::Cuda => Storage::Cuda(device::host_to_cuda(&host)?),
    };
    Py::new(py, Tensor::from_storage(shape, storage))
}

#[pyfunction]
#[pyo3(signature = (shape, seed=None, *, device="cpu"))]
pub fn randn(py: Python, shape: Vec<usize>, seed: Option<u64>, device: &str) -> PyResult<Py<Tensor>> {
    let dev = device::parse_device(device)?;
    let n: usize = shape.iter().product::<usize>().max(1);
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let host: Vec<f32> = (0..n)
        .map(|_| { let v: f32 = StandardNormal.sample(&mut rng); v })
        .collect();
    let storage = match dev {
        Device::Cpu => Storage::Cpu(host),
        Device::Cuda => Storage::Cuda(device::host_to_cuda(&host)?),
    };
    Py::new(py, Tensor::from_storage(shape, storage))
}

#[pyfunction]
#[pyo3(signature = (arr, *, device="cpu"))]
pub fn from_numpy(py: Python, arr: PyReadonlyArrayDyn<'_, f32>, device: &str) -> PyResult<Py<Tensor>> {
    let dev = device::parse_device(device)?;
    let shape: Vec<usize> = arr.shape().to_vec();
    let host: Vec<f32> = arr.as_array().iter().copied().collect();
    let storage = match dev {
        Device::Cpu => Storage::Cpu(host),
        Device::Cuda => Storage::Cuda(device::host_to_cuda(&host)?),
    };
    Py::new(py, Tensor::from_storage(shape, storage))
}
