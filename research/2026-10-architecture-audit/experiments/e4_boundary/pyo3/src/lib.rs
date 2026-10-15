use pyo3::prelude::*;
#[pyfunction]
fn noop() {}
#[pyfunction]
fn add3(a: f64, b: f64, c: f64) -> f64 { a + b + c }
#[pymodule]
fn pyo3_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(noop, m)?)?;
    m.add_function(wrap_pyfunction!(add3, m)?)?;
    Ok(())
}
