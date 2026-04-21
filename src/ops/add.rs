use pyo3::prelude::*;
use crate::tensor::Tensor;
use crate::tape::{self, OpKind, OpRecord, BroadcastKind, accumulate_grad};

pub fn forward(
    py: Python,
    a: Bound<'_, Tensor>,
    b: Bound<'_, Tensor>,
) -> PyResult<Py<Tensor>> {
    let (out_shape, out_data, broadcast) = {
        let a_ref = a.borrow();
        let b_ref = b.borrow();
        if a_ref.shape == b_ref.shape {
            let n = a_ref.numel();
            let data: Vec<f32> = (0..n).map(|i| a_ref.data[i] + b_ref.data[i]).collect();
            (a_ref.shape.clone(), data, BroadcastKind::SameShape)
        } else if a_ref.numel() == 1 {
            let s = a_ref.data[0];
            let data: Vec<f32> = b_ref.data.iter().map(|&x| s + x).collect();
            (b_ref.shape.clone(), data, BroadcastKind::LeftScalar)
        } else if b_ref.numel() == 1 {
            let s = b_ref.data[0];
            let data: Vec<f32> = a_ref.data.iter().map(|&x| x + s).collect();
            (a_ref.shape.clone(), data, BroadcastKind::RightScalar)
        } else if a_ref.shape.len() == 2 && b_ref.shape.len() == 1
            && a_ref.shape[1] == b_ref.shape[0]
        {
            let rows = a_ref.shape[0];
            let cols = a_ref.shape[1];
            let mut data = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    data[i * cols + j] = a_ref.data[i * cols + j] + b_ref.data[j];
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
                    data[i * cols + j] = a_ref.data[j] + b_ref.data[i * cols + j];
                }
            }
            (b_ref.shape.clone(), data, BroadcastKind::RowVecLeft { rows, cols })
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "add: shape mismatch {:?} vs {:?}; supported: same-shape, scalar broadcast, 2D+1D row-vec",
                a_ref.shape, b_ref.shape
            )));
        }
    };
    let output = Py::new(py, Tensor::new(out_shape, out_data))?;
    tape::record(OpRecord {
        kind: OpKind::Add { broadcast },
        inputs: vec![a.unbind(), b.unbind()],
        output: output.clone_ref(py),
    });
    Ok(output)
}

pub fn backward(py: Python, rec: &OpRecord, broadcast: BroadcastKind) -> PyResult<()> {
    let upstream = rec.output.borrow(py).grad.clone()
        .expect("add backward: missing output grad");
    match broadcast {
        BroadcastKind::SameShape => {
            accumulate_grad(py, &rec.inputs[0], &upstream);
            accumulate_grad(py, &rec.inputs[1], &upstream);
        }
        BroadcastKind::LeftScalar => {
            let s: f32 = upstream.iter().sum();
            accumulate_grad(py, &rec.inputs[0], &[s]);
            accumulate_grad(py, &rec.inputs[1], &upstream);
        }
        BroadcastKind::RightScalar => {
            let s: f32 = upstream.iter().sum();
            accumulate_grad(py, &rec.inputs[0], &upstream);
            accumulate_grad(py, &rec.inputs[1], &[s]);
        }
        BroadcastKind::RowVecRight { rows, cols } => {
            // dL/da = upstream (same shape as a). dL/db = sum_i upstream[i, j].
            accumulate_grad(py, &rec.inputs[0], &upstream);
            let mut gb = vec![0.0f32; cols];
            for i in 0..rows {
                for j in 0..cols {
                    gb[j] += upstream[i * cols + j];
                }
            }
            accumulate_grad(py, &rec.inputs[1], &gb);
        }
        BroadcastKind::RowVecLeft { rows, cols } => {
            let mut ga = vec![0.0f32; cols];
            for i in 0..rows {
                for j in 0..cols {
                    ga[j] += upstream[i * cols + j];
                }
            }
            accumulate_grad(py, &rec.inputs[0], &ga);
            accumulate_grad(py, &rec.inputs[1], &upstream);
        }
    }
    Ok(())
}
