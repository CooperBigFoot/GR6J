use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::convert::{checked_slice, contiguous_slice};

use pydrology_core::gr2m::params::Parameters;
use pydrology_core::gr2m::run;
use pydrology_core::gr2m::state::State;

// ---------------------------------------------------------------------------
// Typed pyclass result objects
// ---------------------------------------------------------------------------

define_timeseries_result! {
    /// GR2M run results with typed numpy array attributes.
    pub struct GR2MResult from pydrology_core::gr2m::fluxes::FluxesTimeseries {
        pet, precip, production_store, rainfall_excess, storage_fill,
        actual_et, percolation, routing_input, routing_store, exchange, streamflow,
    }
}

define_step_result! {
    /// GR2M single-timestep flux results.
    pub struct GR2MStepFluxes from pydrology_core::gr2m::fluxes::Fluxes {
        pet, precip, production_store, rainfall_excess, storage_fill,
        actual_et, percolation, routing_input, routing_store, exchange, streamflow,
    }
}

// ---------------------------------------------------------------------------
// Existing dict-returning functions (backward compatible)
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (params, precip, pet, initial_state=None))]
fn gr2m_run<'py>(
    py: Python<'py>,
    params: PyReadonlyArray1<'py, f64>,
    precip: PyReadonlyArray1<'py, f64>,
    pet: PyReadonlyArray1<'py, f64>,
    initial_state: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let p_slice = checked_slice(&params, 2, "params")?;
    let p = Parameters::new(p_slice[0], p_slice[1])
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let precip_slice = contiguous_slice(&precip)?;
    let pet_slice = contiguous_slice(&pet)?;

    let state = match &initial_state {
        Some(s) => {
            let s_slice = checked_slice(s, 2, "initial_state")?;
            Some(State {
                production_store: s_slice[0],
                routing_store: s_slice[1],
            })
        }
        None => None,
    };

    let result = run::run(&p, precip_slice, pet_slice, state.as_ref());

    let dict = timeseries_to_dict!(
        py, result,
        pet, precip, production_store, rainfall_excess, storage_fill,
        actual_et, percolation, routing_input, routing_store, exchange, streamflow,
    );
    Ok(dict)
}

#[pyfunction]
fn gr2m_step<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
    precip: f64,
    pet: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>)> {
    let p_slice = checked_slice(&params, 2, "params")?;
    let p = Parameters::new(p_slice[0], p_slice[1])
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let s_slice = checked_slice(&state, 2, "state")?;
    let s = State {
        production_store: s_slice[0],
        routing_store: s_slice[1],
    };

    let (new_state, fluxes) = run::step(&s, &p, precip, pet);

    let state_arr = PyArray1::from_vec(
        py,
        vec![new_state.production_store, new_state.routing_store],
    );

    let dict = fluxes_to_dict!(
        py, fluxes,
        pet, precip, production_store, rainfall_excess, storage_fill,
        actual_et, percolation, routing_input, routing_store, exchange, streamflow,
    );
    Ok((state_arr, dict))
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "gr2m")?;
    m.add_function(wrap_pyfunction!(gr2m_run, &m)?)?;
    m.add_function(wrap_pyfunction!(gr2m_step, &m)?)?;
    m.add_class::<GR2MResult>()?;
    m.add_class::<GR2MStepFluxes>()?;
    parent.add_submodule(&m)?;
    Ok(())
}
