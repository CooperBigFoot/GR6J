use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use pydrology_core::metrics;

#[pyfunction]
fn rust_nse(observed: PyReadonlyArray1<'_, f64>, simulated: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(metrics::nse(observed.as_slice()?, simulated.as_slice()?))
}

#[pyfunction]
fn rust_log_nse(observed: PyReadonlyArray1<'_, f64>, simulated: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(metrics::log_nse(observed.as_slice()?, simulated.as_slice()?))
}

#[pyfunction]
fn rust_kge(observed: PyReadonlyArray1<'_, f64>, simulated: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(metrics::kge(observed.as_slice()?, simulated.as_slice()?))
}

#[pyfunction]
fn rust_pbias(observed: PyReadonlyArray1<'_, f64>, simulated: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(metrics::pbias(observed.as_slice()?, simulated.as_slice()?))
}

#[pyfunction]
fn rust_rmse(observed: PyReadonlyArray1<'_, f64>, simulated: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(metrics::rmse(observed.as_slice()?, simulated.as_slice()?))
}

#[pyfunction]
fn rust_mae(observed: PyReadonlyArray1<'_, f64>, simulated: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(metrics::mae(observed.as_slice()?, simulated.as_slice()?))
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "metrics")?;
    m.add_function(wrap_pyfunction!(rust_nse, &m)?)?;
    m.add_function(wrap_pyfunction!(rust_log_nse, &m)?)?;
    m.add_function(wrap_pyfunction!(rust_kge, &m)?)?;
    m.add_function(wrap_pyfunction!(rust_pbias, &m)?)?;
    m.add_function(wrap_pyfunction!(rust_rmse, &m)?)?;
    m.add_function(wrap_pyfunction!(rust_mae, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
