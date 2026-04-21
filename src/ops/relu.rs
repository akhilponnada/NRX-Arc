use pyo3::prelude::*;
use crate::tensor::Tensor;
use crate::tape::{self, OpKind, OpRecord, accumulate_grad};

pub fn forward(py: Python, a: Bound<'_, Tensor>) -> PyResult<Py<Tensor>> {
    let (out_shape, out_data) = {
        let a_ref = a.borrow();
        let data: Vec<f32> = a_ref.data.iter().map(|&x| x.max(0.0)).collect();
        (a_ref.shape.clone(), data)
    };
    let output = Py::new(py, Tensor::new(out_shape, out_data))?;
    tape::record(OpRecord {
        kind: OpKind::Relu,
        inputs: vec![a.unbind()],
        output: output.clone_ref(py),
    });
    Ok(output)
}

pub fn backward(py: Python, rec: &OpRecord) -> PyResult<()> {
    // dL/dx = upstream * (x > 0). Subgradient at x=0 is 0.
    let upstream = rec.output.borrow(py).grad.clone()
        .expect("relu backward: missing output grad");
    let input_data = rec.inputs[0].borrow(py).data.clone();
    let grad: Vec<f32> = input_data.iter().zip(upstream.iter())
        .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
        .collect();
    accumulate_grad(py, &rec.inputs[0], &grad);
    Ok(())
}
