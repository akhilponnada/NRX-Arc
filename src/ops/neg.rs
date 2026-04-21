use pyo3::prelude::*;
use crate::tensor::Tensor;
use crate::tape::{self, OpKind, OpRecord, accumulate_grad};

pub fn forward(py: Python, a: Bound<'_, Tensor>) -> PyResult<Py<Tensor>> {
    let (out_shape, out_data) = {
        let a_ref = a.borrow();
        let data: Vec<f32> = a_ref.data.iter().map(|&x| -x).collect();
        (a_ref.shape.clone(), data)
    };
    let output = Py::new(py, Tensor::new(out_shape, out_data))?;
    tape::record(OpRecord {
        kind: OpKind::Neg,
        inputs: vec![a.unbind()],
        output: output.clone_ref(py),
    });
    Ok(output)
}

pub fn backward(py: Python, rec: &OpRecord) -> PyResult<()> {
    // y = -x; dL/dx = -upstream
    let upstream = rec.output.borrow(py).grad.clone()
        .expect("neg backward: missing output grad");
    let grad: Vec<f32> = upstream.iter().map(|&u| -u).collect();
    accumulate_grad(py, &rec.inputs[0], &grad);
    Ok(())
}
