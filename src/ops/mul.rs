use pyo3::prelude::*;
use crate::tensor::Tensor;
use crate::tape::{self, OpKind, OpRecord, BroadcastKind, accumulate_grad, output_grad_cpu, input_data_cpu};
use crate::device::{device_str, Device, Storage};

pub fn forward(py: Python, a: Bound<'_, Tensor>, b: Bound<'_, Tensor>) -> PyResult<Py<Tensor>> {
    {
        let a_dev = a.borrow().storage.device();
        let b_dev = b.borrow().storage.device();
        match (a_dev, b_dev) {
            (Device::Cpu, Device::Cpu) => {}
            (Device::Cuda, Device::Cuda) => unimplemented!("cuda mul not yet"),
            (da, db) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "cannot multiply tensors on different devices: {} and {}",
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
            let data: Vec<f32> = (0..n).map(|i| a_data[i] * b_data[i]).collect();
            (a_ref.shape.clone(), data, BroadcastKind::SameShape)
        } else if a_ref.numel() == 1 {
            let s = a_data[0];
            let data: Vec<f32> = b_data.iter().map(|&x| s * x).collect();
            (b_ref.shape.clone(), data, BroadcastKind::LeftScalar)
        } else if b_ref.numel() == 1 {
            let s = b_data[0];
            let data: Vec<f32> = a_data.iter().map(|&x| x * s).collect();
            (a_ref.shape.clone(), data, BroadcastKind::RightScalar)
        } else if a_ref.shape.len() == 2 && b_ref.shape.len() == 1
            && a_ref.shape[1] == b_ref.shape[0]
        {
            let rows = a_ref.shape[0];
            let cols = a_ref.shape[1];
            let mut data = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    data[i * cols + j] = a_data[i * cols + j] * b_data[j];
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
                    data[i * cols + j] = a_data[j] * b_data[i * cols + j];
                }
            }
            (b_ref.shape.clone(), data, BroadcastKind::RowVecLeft { rows, cols })
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "mul: shape mismatch {:?} vs {:?}", a_ref.shape, b_ref.shape
            )));
        }
    };
    let output = Py::new(py, Tensor::from_storage(out_shape, Storage::Cpu(out_data)))?;
    tape::record(OpRecord {
        kind: OpKind::Mul { broadcast },
        inputs: vec![a.unbind(), b.unbind()],
        output: output.clone_ref(py),
    });
    Ok(output)
}

pub fn backward(py: Python, rec: &OpRecord, broadcast: BroadcastKind) -> PyResult<()> {
    let upstream = output_grad_cpu(py, &rec.output, "mul");
    let a_data = input_data_cpu(py, &rec.inputs[0], "mul");
    let b_data = input_data_cpu(py, &rec.inputs[1], "mul");
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
