use pyo3::prelude::*;
use crate::tensor::Tensor;
use crate::tape::{self, OpKind, OpRecord, accumulate_grad, output_grad_cpu};
use crate::device::{Device, Storage};

pub fn forward(py: Python, a: Bound<'_, Tensor>) -> PyResult<Py<Tensor>> {
    if a.borrow().storage.device() == Device::Cuda {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "cuda sum not implemented yet (Phase 2 step 2)"
        ));
    }
    let (input_numel, total) = {
        let a_ref = a.borrow();
        let a_data = a_ref.storage.cpu();
        let s: f32 = a_data.iter().sum();
        (a_ref.numel(), s)
    };
    // Scalar output: shape vec![] (0-dim), data = [total]
    let output = Py::new(py, Tensor::from_storage(vec![], Storage::Cpu(vec![total])))?;
    tape::record(OpRecord {
        kind: OpKind::Sum { input_numel },
        inputs: vec![a.unbind()],
        output: output.clone_ref(py),
    });
    Ok(output)
}

pub fn backward(py: Python, rec: &OpRecord, input_numel: usize) -> PyResult<()> {
    // y = sum(x); dL/dx[i] = dL/dy (scalar) broadcast to input shape
    let upstream = output_grad_cpu(py, &rec.output, "sum");
    debug_assert_eq!(upstream.len(), 1, "sum upstream must be scalar");
    let s = upstream[0];
    let grad: Vec<f32> = vec![s; input_numel];
    accumulate_grad(py, &rec.inputs[0], &grad);
    Ok(())
}
