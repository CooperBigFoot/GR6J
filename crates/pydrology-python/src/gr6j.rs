use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::convert::{checked_slice, checked_slice_min, contiguous_slice};

use pydrology_core::gr6j::constants::NH;
use pydrology_core::gr6j::params::Parameters;
use pydrology_core::gr6j::run;
use pydrology_core::gr6j::state::State;
use pydrology_core::gr6j::unit_hydrographs::compute_uh_ordinates;

// ---------------------------------------------------------------------------
// Typed pyclass result objects
// ---------------------------------------------------------------------------

define_timeseries_result! {
    /// GR6J run results with typed numpy array attributes.
    pub struct GR6JResult from pydrology_core::gr6j::fluxes::FluxesTimeseries {
        pet, precip, production_store, net_rainfall, storage_infiltration,
        actual_et, percolation, effective_rainfall, q9, q1, routing_store,
        exchange, actual_exchange_routing, actual_exchange_direct,
        actual_exchange_total, qr, qrexp, exponential_store, qd, streamflow,
    }
}

define_step_result! {
    /// GR6J single-timestep flux results.
    pub struct GR6JStepFluxes from pydrology_core::gr6j::fluxes::Fluxes {
        pet, precip, production_store, net_rainfall, storage_infiltration,
        actual_et, percolation, effective_rainfall, q9, q1, routing_store,
        exchange, actual_exchange_routing, actual_exchange_direct,
        actual_exchange_total, qr, qrexp, exponential_store, qd, streamflow,
    }
}

// ---------------------------------------------------------------------------
// Existing dict-returning functions (backward compatible)
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (params, precip, pet, initial_state=None))]
fn gr6j_run<'py>(
    py: Python<'py>,
    params: PyReadonlyArray1<'py, f64>,
    precip: PyReadonlyArray1<'py, f64>,
    pet: PyReadonlyArray1<'py, f64>,
    initial_state: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let p_slice = checked_slice_min(&params, 6, "params")?;
    let p = Parameters::new_unchecked(p_slice[0], p_slice[1], p_slice[2], p_slice[3], p_slice[4], p_slice[5]);

    let precip_slice = contiguous_slice(&precip)?;
    let pet_slice = contiguous_slice(&pet)?;

    let state = match &initial_state {
        Some(s) => {
            let s_slice = checked_slice(s, 63, "initial_state")?;
            let mut arr = [0.0f64; 63];
            arr.copy_from_slice(s_slice);
            Some(State::from_array(&arr))
        }
        None => None,
    };

    let result = run::run(&p, precip_slice, pet_slice, state.as_ref());

    let dict = timeseries_to_dict!(
        py, result,
        pet, precip, production_store, net_rainfall, storage_infiltration,
        actual_et, percolation, effective_rainfall, q9, q1, routing_store,
        exchange, actual_exchange_routing, actual_exchange_direct,
        actual_exchange_total, qr, qrexp, exponential_store, qd, streamflow,
    );
    Ok(dict)
}

#[pyfunction]
fn gr6j_step<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
    precip: f64,
    pet: f64,
    uh1_ordinates: PyReadonlyArray1<'py, f64>,
    uh2_ordinates: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>)> {
    let p_slice = checked_slice_min(&params, 6, "params")?;
    let p = Parameters::new_unchecked(p_slice[0], p_slice[1], p_slice[2], p_slice[3], p_slice[4], p_slice[5]);

    let s_slice = checked_slice(&state, 63, "state")?;
    let mut state_arr = [0.0f64; 63];
    state_arr.copy_from_slice(s_slice);
    let s = State::from_array(&state_arr);

    let uh1_slice = checked_slice(&uh1_ordinates, NH, "uh1_ordinates")?;
    let mut uh1_arr = [0.0f64; NH];
    uh1_arr.copy_from_slice(uh1_slice);

    let uh2_slice = checked_slice(&uh2_ordinates, 2 * NH, "uh2_ordinates")?;
    let mut uh2_arr = [0.0f64; 2 * NH];
    uh2_arr.copy_from_slice(uh2_slice);

    let (new_state, fluxes) = run::step(&s, &p, precip, pet, &uh1_arr, &uh2_arr);

    let new_state_arr = new_state.to_array();
    let state_out = PyArray1::from_vec(py, new_state_arr.to_vec());

    let dict = fluxes_to_dict!(
        py, fluxes,
        pet, precip, production_store, net_rainfall, storage_infiltration,
        actual_et, percolation, effective_rainfall, q9, q1, routing_store,
        exchange, actual_exchange_routing, actual_exchange_direct,
        actual_exchange_total, qr, qrexp, exponential_store, qd, streamflow,
    );
    Ok((state_out, dict))
}

#[pyfunction]
#[allow(clippy::type_complexity)]
fn gr6j_compute_uh_ordinates<'py>(
    py: Python<'py>,
    x4: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (uh1, uh2) = compute_uh_ordinates(x4);
    Ok((
        PyArray1::from_slice(py, &uh1),
        PyArray1::from_slice(py, &uh2),
    ))
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "gr6j")?;
    m.add_function(wrap_pyfunction!(gr6j_run, &m)?)?;
    m.add_function(wrap_pyfunction!(gr6j_step, &m)?)?;
    m.add_function(wrap_pyfunction!(gr6j_compute_uh_ordinates, &m)?)?;
    m.add_class::<GR6JResult>()?;
    m.add_class::<GR6JStepFluxes>()?;
    parent.add_submodule(&m)?;
    Ok(())
}
