use pyo3::prelude::*;

// PHASE 1: single-threaded CPU is fine; revisit for multi-device later.

#[pyclass]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub strides: Vec<usize>,
    pub grad: Option<Vec<f32>>,
}

#[pymethods]
impl Tensor {
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn to_list(&self) -> Vec<f32> {
        self.data.clone()
    }

    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?})", self.shape)
    }
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[pyfunction]
pub fn zeros(shape: Vec<usize>) -> Tensor {
    let numel: usize = shape.iter().product();
    let strides = row_major_strides(&shape);
    Tensor { shape, data: vec![0.0; numel], strides, grad: None }
}
