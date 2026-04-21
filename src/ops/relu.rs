use pyo3::prelude::*;
use crate::tensor::Tensor;
use crate::tape::{self, OpKind, OpRecord, accumulate_grad, output_grad_cpu, input_data_cpu};
use crate::device::{Device, Storage};

pub fn forward(py: Python, a: Bound<'_, Tensor>) -> PyResult<Py<Tensor>> {
    if a.borrow().storage.device() == Device::Cuda {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "cuda relu not implemented yet (Phase 2 step 2)"
        ));
    }
    let (out_shape, out_data) = {
        let a_ref = a.borrow();
        let a_data = a_ref.storage.cpu();
        let data: Vec<f32> = a_data.iter().map(|&x| x.max(0.0)).collect();
        (a_ref.shape.clone(), data)
    };
    let output = Py::new(py, Tensor::from_storage(out_shape, Storage::Cpu(out_data)))?;
    tape::record(OpRecord {
        kind: OpKind::Relu,
        inputs: vec![a.unbind()],
        output: output.clone_ref(py),
    });
    Ok(output)
}

pub fn backward(py: Python, rec: &OpRecord) -> PyResult<()> {
    // dL/dx = upstream * (x > 0). Subgradient at x=0 is 0.
    let upstream = output_grad_cpu(py, &rec.output, "relu");
    let input_data = input_data_cpu(py, &rec.inputs[0], "relu");
    let grad: Vec<f32> = input_data.iter().zip(upstream.iter())
        .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
        .collect();
    accumulate_grad(py, &rec.inputs[0], &grad);
    Ok(())
}
