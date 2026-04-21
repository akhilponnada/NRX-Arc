use pyo3::prelude::*;
use crate::tensor::Tensor;
use crate::tape::{self, OpKind, OpRecord, BroadcastKind, accumulate_grad, output_grad_cpu};
use crate::device::{device_str, Device, Storage};

pub fn forward(
    py: Python,
    a: Bound<'_, Tensor>,
    b: Bound<'_, Tensor>,
) -> PyResult<Py<Tensor>> {
    // Device dispatch: both inputs must agree. CUDA path lands in a later commit.
    {
        let a_dev = a.borrow().storage.device();
        let b_dev = b.borrow().storage.device();
        match (a_dev, b_dev) {
            (Device::Cpu, Device::Cpu) => {}
            (Device::Cuda, Device::Cuda) => unimplemented!("cuda add not yet"),
            (da, db) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "cannot add tensors on different devices: {} and {}",
                    device_str(da), device_str(db)
                )));
            }
        }
    }

    let (out_shape, out_data, broadcast) = {
        let a_ref = a.borrow();
        let b_ref = b.borrow();
        let a_data = a_ref.storage.cpu();
        let b_data = b_ref.storage.cpu();
        if a_ref.shape == b_ref.shape {
            let n = a_ref.numel();
            let data: Vec<f32> = (0..n).map(|i| a_data[i] + b_data[i]).collect();
            (a_ref.shape.clone(), data, BroadcastKind::SameShape)
        } else if a_ref.numel() == 1 {
            let s = a_data[0];
            let data: Vec<f32> = b_data.iter().map(|&x| s + x).collect();
            (b_ref.shape.clone(), data, BroadcastKind::LeftScalar)
        } else if b_ref.numel() == 1 {
            let s = b_data[0];
            let data: Vec<f32> = a_data.iter().map(|&x| x + s).collect();
            (a_ref.shape.clone(), data, BroadcastKind::RightScalar)
        } else if a_ref.shape.len() == 2 && b_ref.shape.len() == 1
            && a_ref.shape[1] == b_ref.shape[0]
        {
            let rows = a_ref.shape[0];
            let cols = a_ref.shape[1];
            let mut data = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    data[i * cols + j] = a_data[i * cols + j] + b_data[j];
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
                    data[i * cols + j] = a_data[j] + b_data[i * cols + j];
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
    let output = Py::new(py, Tensor::from_storage(out_shape, Storage::Cpu(out_data)))?;
    tape::record(OpRecord {
        kind: OpKind::Add { broadcast },
        inputs: vec![a.unbind(), b.unbind()],
        output: output.clone_ref(py),
    });
    Ok(output)
}

pub fn backward(py: Python, rec: &OpRecord, broadcast: BroadcastKind) -> PyResult<()> {
    let upstream = output_grad_cpu(py, &rec.output, "add");
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
