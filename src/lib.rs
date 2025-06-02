use pyo3::prelude::*;

// Submodules
#[allow(dead_code)]
pub mod autograd;
#[allow(dead_code)]
pub mod neuralnet;

#[pyfunction]
fn hello_from_bin() -> String {
    "Hello from teensygrad!".to_string()
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_bin, m)?)?;
    Ok(())
}
