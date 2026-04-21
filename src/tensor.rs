use pyo3::prelude::*;
use numpy::{PyReadonlyArrayDyn, IntoPyArray, PyArrayMethods, PyUntypedArrayMethods};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use crate::tape;

#[pyclass]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub strides: Vec<usize>,
    pub grad: Option<Vec<f32>>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>().max(1));
        let strides = row_major_strides(&shape);
        Tensor { shape, data, strides, grad: None }
    }
    pub fn numel(&self) -> usize { self.shape.iter().product::<usize>().max(1) }
}

#[pymethods]
impl Tensor {
    fn shape(&self) -> Vec<usize> { self.shape.clone() }
    fn strides(&self) -> Vec<usize> { self.strides.clone() }
    fn to_list(&self) -> Vec<f32> { self.data.clone() }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let arr = self.data.clone().into_pyarray(py);
        let reshaped = arr.reshape(self.shape.clone())?;
        Ok(reshaped.into_any())
    }

    #[getter]
    fn grad<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.grad {
            None => Ok(None),
            Some(g) => {
                let arr = g.clone().into_pyarray(py);
                let reshaped = arr.reshape(self.shape.clone())?;
                Ok(Some(reshaped.into_any()))
            }
        }
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

    fn __repr__(&self) -> String { format!("Tensor(shape={:?})", self.shape) }

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
    fn sgd_update_(&mut self, lr: f32) {
        let g = match &self.grad {
            None => return,
            Some(v) => v.clone(),
        };
        for (d, gi) in self.data.iter_mut().zip(g.iter()) {
            *d -= lr * gi;
        }
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
pub fn zeros(py: Python, shape: Vec<usize>) -> Py<Tensor> {
    let n: usize = shape.iter().product::<usize>().max(1);
    Py::new(py, Tensor::new(shape, vec![0.0; n])).unwrap()
}

#[pyfunction]
pub fn ones(py: Python, shape: Vec<usize>) -> Py<Tensor> {
    let n: usize = shape.iter().product::<usize>().max(1);
    Py::new(py, Tensor::new(shape, vec![1.0; n])).unwrap()
}

#[pyfunction]
#[pyo3(signature = (shape, seed=None))]
pub fn randn(py: Python, shape: Vec<usize>, seed: Option<u64>) -> Py<Tensor> {
    let n: usize = shape.iter().product::<usize>().max(1);
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let data: Vec<f32> = (0..n)
        .map(|_| { let v: f32 = StandardNormal.sample(&mut rng); v })
        .collect();
    Py::new(py, Tensor::new(shape, data)).unwrap()
}

#[pyfunction]
pub fn from_numpy(py: Python, arr: PyReadonlyArrayDyn<'_, f32>) -> Py<Tensor> {
    let shape: Vec<usize> = arr.shape().to_vec();
    let data: Vec<f32> = arr.as_array().iter().copied().collect();
    Py::new(py, Tensor::new(shape, data)).unwrap()
}
