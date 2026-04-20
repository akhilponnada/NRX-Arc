use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod tensor;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tensor::Tensor>()?;
    m.add_function(wrap_pyfunction!(tensor::zeros, m)?)?;
    Ok(())
}
