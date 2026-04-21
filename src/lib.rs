use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod tensor;
mod tape;
mod ops;
mod device;

#[pyfunction]
fn clear_tape() { tape::clear(); }

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tensor::Tensor>()?;
    m.add_function(wrap_pyfunction!(tensor::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::ones, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::randn, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(clear_tape, m)?)?;
    Ok(())
}
