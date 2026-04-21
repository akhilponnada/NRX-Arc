use pyo3::prelude::*;
use crate::tensor::Tensor;
use crate::tape::{self, OpKind, OpRecord, BroadcastKind, accumulate_grad};

pub fn forward(py: Python, a: Bound<'_, Tensor>, b: Bound<'_, Tensor>) -> PyResult<Py<Tensor>> {
    let (out_shape, out_data, broadcast) = {
        let a_ref = a.borrow();
        let b_ref = b.borrow();
        if a_ref.shape == b_ref.shape {
            let n = a_ref.numel();
            let data: Vec<f32> = (0..n).map(|i| a_ref.data[i] * b_ref.data[i]).collect();
            (a_ref.shape.clone(), data, BroadcastKind::SameShape)
        } else if a_ref.numel() == 1 {
            let s = a_ref.data[0];
            let data: Vec<f32> = b_ref.data.iter().map(|&x| s * x).collect();
            (b_ref.shape.clone(), data, BroadcastKind::LeftScalar)
        } else if b_ref.numel() == 1 {
            let s = b_ref.data[0];
            let data: Vec<f32> = a_ref.data.iter().map(|&x| x * s).collect();
            (a_ref.shape.clone(), data, BroadcastKind::RightScalar)
        } else if a_ref.shape.len() == 2 && b_ref.shape.len() == 1
            && a_ref.shape[1] == b_ref.shape[0]
        {
            let rows = a_ref.shape[0];
            let cols = a_ref.shape[1];
            let mut data = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    data[i * cols + j] = a_ref.data[i * cols + j] * b_ref.data[j];
                }
            }
            (a_ref.shape.clone(), data, BroadcastKind::RowVecRight { rows, cols })
        } else if a_ref.shape.len() == 1 && b_ref.shape.len() == 2
            && b_ref.shape[1] == a_ref.shape[0]
        {
            let rows = b_ref.shape[0];
            let cols = b_ref.shape[1];
            let mut data = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    data[i * cols + j] = a_ref.data[j] * b_ref.data[i * cols + j];
                }
            }
            (b_ref.shape.clone(), data, BroadcastKind::RowVecLeft { rows, cols })
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "mul: shape mismatch {:?} vs {:?}", a_ref.shape, b_ref.shape
            )));
        }
    };
    let output = Py::new(py, Tensor::new(out_shape, out_data))?;
    tape::record(OpRecord {
        kind: OpKind::Mul { broadcast },
        inputs: vec![a.unbind(), b.unbind()],
        output: output.clone_ref(py),
    });
    Ok(output)
}

pub fn backward(py: Python, rec: &OpRecord, broadcast: BroadcastKind) -> PyResult<()> {
    let upstream = rec.output.borrow(py).grad.clone()
        .expect("mul backward: missing output grad");
    let a_data = rec.inputs[0].borrow(py).data.clone();
    let b_data = rec.inputs[1].borrow(py).data.clone();
    match broadcast {
        BroadcastKind::SameShape => {
            let ga: Vec<f32> = upstream.iter().zip(b_data.iter()).map(|(u,bv)| u*bv).collect();
            let gb: Vec<f32> = upstream.iter().zip(a_data.iter()).map(|(u,av)| u*av).collect();
            accumulate_grad(py, &rec.inputs[0], &ga);
            accumulate_grad(py, &rec.inputs[1], &gb);
        }
        BroadcastKind::LeftScalar => {
            let ga_val: f32 = upstream.iter().zip(b_data.iter()).map(|(u,bv)| u*bv).sum();
            let gb: Vec<f32> = upstream.iter().map(|u| u * a_data[0]).collect();
            accumulate_grad(py, &rec.inputs[0], &[ga_val]);
            accumulate_grad(py, &rec.inputs[1], &gb);
        }
        BroadcastKind::RightScalar => {
            let gb_val: f32 = upstream.iter().zip(a_data.iter()).map(|(u,av)| u*av).sum();
            let ga: Vec<f32> = upstream.iter().map(|u| u * b_data[0]).collect();
            accumulate_grad(py, &rec.inputs[0], &ga);
            accumulate_grad(py, &rec.inputs[1], &[gb_val]);
        }
        BroadcastKind::RowVecRight { rows, cols } => {
            // y[i,j] = a[i,j] * b[j]
            // dL/da[i,j] = upstream[i,j] * b[j]
            // dL/db[j]   = sum_i upstream[i,j] * a[i,j]
            let mut ga = vec![0.0f32; rows * cols];
            let mut gb = vec![0.0f32; cols];
            for i in 0..rows {
                for j in 0..cols {
                    let u = upstream[i * cols + j];
                    ga[i * cols + j] = u * b_data[j];
                    gb[j] += u * a_data[i * cols + j];
                }
            }
            accumulate_grad(py, &rec.inputs[0], &ga);
            accumulate_grad(py, &rec.inputs[1], &gb);
        }
        BroadcastKind::RowVecLeft { rows, cols } => {
            let mut ga = vec![0.0f32; cols];
            let mut gb = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    let u = upstream[i * cols + j];
                    ga[j] += u * b_data[i * cols + j];
                    gb[i * cols + j] = u * a_data[j];
                }
            }
            accumulate_grad(py, &rec.inputs[0], &ga);
            accumulate_grad(py, &rec.inputs[1], &gb);
        }
    }
    Ok(())
}
