use pyo3::prelude::*;
use crate::tensor::Tensor;
use crate::tape::{self, OpKind, OpRecord, accumulate_grad, output_grad_cpu, input_data_cpu};
use crate::device::{device_str, Device, Storage};

// 2D matmul: A (m, k) @ B (k, n) -> C (m, n).
//
// Derivation of backward (forward C = A @ B):
//   C[i,j] = sum_p A[i,p] * B[p,j]
//   dL/dA[i,p] = sum_j (dL/dC[i,j]) * B[p,j]   = (dL/dC @ B^T)[i,p]
//   dL/dB[p,j] = sum_i A[i,p] * (dL/dC[i,j])   = (A^T @ dL/dC)[p,j]
// Shapes:
//   dA = dC @ B^T   :  (m,n) @ (n,k) -> (m,k)
//   dB = A^T @ dC   :  (k,m) @ (m,n) -> (k,n)
// Row-major flat indexing: A[i*k + p], B[p*n + j], C[i*n + j].

pub fn forward(py: Python, a: Bound<'_, Tensor>, b: Bound<'_, Tensor>) -> PyResult<Py<Tensor>> {
    {
        let a_dev = a.borrow().storage.device();
        let b_dev = b.borrow().storage.device();
        match (a_dev, b_dev) {
            (Device::Cpu, Device::Cpu) => {}
            (Device::Cuda, Device::Cuda) => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "cuda matmul not implemented yet (Phase 2 step 2)"
                ));
            }
            (da, db) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "cannot matmul tensors on different devices: {} and {}",
                    device_str(da), device_str(db)
                )));
            }
        }
    }

    let (out_shape, out_data, m, k, n) = {
        let a_ref = a.borrow();
        let b_ref = b.borrow();
        if a_ref.shape.len() != 2 || b_ref.shape.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "matmul: Phase 1 requires 2D tensors"
            ));
        }
        let m = a_ref.shape[0];
        let k = a_ref.shape[1];
        if b_ref.shape[0] != k {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "matmul shape mismatch: A is {:?}, B is {:?}", a_ref.shape, b_ref.shape
            )));
        }
        let n = b_ref.shape[1];
        let a_data = a_ref.storage.cpu();
        let b_data = b_ref.storage.cpu();
        let mut data = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a_data[i * k + p] * b_data[p * n + j];
                }
                data[i * n + j] = acc;
            }
        }
        (vec![m, n], data, m, k, n)
    };
    let output = Py::new(py, Tensor::from_storage(out_shape, Storage::Cpu(out_data)))?;
    tape::record(OpRecord {
        kind: OpKind::Matmul { m, k, n },
        inputs: vec![a.unbind(), b.unbind()],
        output: output.clone_ref(py),
    });
    Ok(output)
}

pub fn backward(py: Python, rec: &OpRecord, m: usize, k: usize, n: usize) -> PyResult<()> {
    let upstream = output_grad_cpu(py, &rec.output, "matmul");           // (m,n)
    let a_data = input_data_cpu(py, &rec.inputs[0], "matmul");           // (m,k)
    let b_data = input_data_cpu(py, &rec.inputs[1], "matmul");           // (k,n)

    // dA[i,p] = sum_j upstream[i,j] * B[p,j]
    let mut da = vec![0.0f32; m * k];
    for i in 0..m {
        for p in 0..k {
            let mut acc = 0.0f32;
            for j in 0..n {
                acc += upstream[i * n + j] * b_data[p * n + j];
            }
            da[i * k + p] = acc;
        }
    }

    // dB[p,j] = sum_i A[i,p] * upstream[i,j]
    let mut db = vec![0.0f32; k * n];
    for p in 0..k {
        for j in 0..n {
            let mut acc = 0.0f32;
            for i in 0..m {
                acc += a_data[i * k + p] * upstream[i * n + j];
            }
            db[p * n + j] = acc;
        }
    }

    accumulate_grad(py, &rec.inputs[0], &da);
    accumulate_grad(py, &rec.inputs[1], &db);
    Ok(())
}
