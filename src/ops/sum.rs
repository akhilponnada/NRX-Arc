use pyo3::prelude::*;
use crate::tensor::Tensor;
use crate::tape::{self, OpKind, OpRecord, accumulate_grad};

pub fn forward(py: Python, a: Bound<'_, Tensor>) -> PyResult<Py<Tensor>> {
    let (input_numel, total) = {
        let a_ref = a.borrow();
        let s: f32 = a_ref.data.iter().sum();
        (a_ref.numel(), s)
    };
    // Scalar output: shape vec![] (0-dim), data = [total]
    let output = Py::new(py, Tensor::new(vec![], vec![total]))?;
    tape::record(OpRecord {
        kind: OpKind::Sum { input_numel },
        inputs: vec![a.unbind()],
        output: output.clone_ref(py),
    });
    Ok(output)
}

pub fn backward(py: Python, rec: &OpRecord, input_numel: usize) -> PyResult<()> {
    // y = sum(x); dL/dx[i] = dL/dy (scalar) broadcast to input shape
    let upstream = rec.output.borrow(py).grad.clone()
        .expect("sum backward: missing output grad");
    debug_assert_eq!(upstream.len(), 1, "sum upstream must be scalar");
    let s = upstream[0];
    let grad: Vec<f32> = vec![s; input_numel];
    accumulate_grad(py, &rec.inputs[0], &grad);
    Ok(())
}
