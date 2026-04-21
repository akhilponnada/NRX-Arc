use pyo3::prelude::*;
use crate::tensor::Tensor;
use crate::tape::{self, OpKind, OpRecord, accumulate_grad, output_grad_cpu};
use crate::device::{Device, Storage};

pub fn forward(py: Python, a: Bound<'_, Tensor>, s: f32) -> PyResult<Py<Tensor>> {
    if a.borrow().storage.device() == Device::Cuda {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "cuda scalar_mul not implemented yet (Phase 2 step 2)"
        ));
    }
    let (out_shape, out_data) = {
        let a_ref = a.borrow();
        let a_data = a_ref.storage.cpu();
        let data: Vec<f32> = a_data.iter().map(|&x| s * x).collect();
        (a_ref.shape.clone(), data)
    };
    let output = Py::new(py, Tensor::from_storage(out_shape, Storage::Cpu(out_data)))?;
    tape::record(OpRecord {
        kind: OpKind::ScalarMul { s },
        inputs: vec![a.unbind()],
        output: output.clone_ref(py),
    });
    Ok(output)
}

pub fn backward(py: Python, rec: &OpRecord, s: f32) -> PyResult<()> {
    // y = s*x; dL/dx = s*upstream
    let upstream = output_grad_cpu(py, &rec.output, "scalar_mul");
    let grad: Vec<f32> = upstream.iter().map(|&u| s * u).collect();
    accumulate_grad(py, &rec.inputs[0], &grad);
    Ok(())
}
