use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::convert::{checked_slice, contiguous_slice};

use pydrology_core::hbv_light::params::Parameters;
use pydrology_core::hbv_light::routing::compute_triangular_weights;
use pydrology_core::hbv_light::run;
use pydrology_core::hbv_light::state::State;

// ---------------------------------------------------------------------------
// Typed pyclass result objects
// ---------------------------------------------------------------------------

define_timeseries_result! {
    /// HBV-Light run results with typed numpy array attributes.
    pub struct HBVResult from pydrology_core::hbv_light::fluxes::FluxesTimeseries {
        precip, temp, pet, precip_rain, precip_snow, snow_pack, snow_melt,
        liquid_water_in_snow, snow_input, soil_moisture, recharge, actual_et,
        upper_zone, lower_zone, q0, q1, q2, percolation, qgw, streamflow,
    }
}

define_step_result! {
    /// HBV-Light single-timestep flux results.
    pub struct HBVStepFluxes from pydrology_core::hbv_light::fluxes::Fluxes {
        precip, temp, pet, precip_rain, precip_snow, snow_pack, snow_melt,
        liquid_water_in_snow, snow_input, soil_moisture, recharge, actual_et,
        upper_zone, lower_zone, q0, q1, q2, percolation, qgw, streamflow,
    }
}

// ---------------------------------------------------------------------------
// Existing dict-returning functions (backward compatible)
// ---------------------------------------------------------------------------

/// Run HBV-Light over a timeseries.
///
/// Returns (fluxes_dict, optional_zone_outputs_dict).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    params,
    precip,
    pet,
    temp,
    initial_state=None,
    n_zones=1,
    zone_elevations=None,
    zone_fractions=None,
    input_elevation=None,
    temp_gradient=None,
    precip_gradient=None,
))]
fn hbv_run<'py>(
    py: Python<'py>,
    params: PyReadonlyArray1<'py, f64>,
    precip: PyReadonlyArray1<'py, f64>,
    pet: PyReadonlyArray1<'py, f64>,
    temp: PyReadonlyArray1<'py, f64>,
    initial_state: Option<PyReadonlyArray1<'py, f64>>,
    n_zones: usize,
    zone_elevations: Option<PyReadonlyArray1<'py, f64>>,
    zone_fractions: Option<PyReadonlyArray1<'py, f64>>,
    input_elevation: Option<f64>,
    temp_gradient: Option<f64>,
    precip_gradient: Option<f64>,
) -> PyResult<(Bound<'py, PyDict>, Option<Bound<'py, PyDict>>)> {
    let p_slice = checked_slice(&params, 14, "params")?;
    let p = Parameters::from_array(p_slice)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let precip_slice = contiguous_slice(&precip)?;
    let pet_slice = contiguous_slice(&pet)?;
    let temp_slice = contiguous_slice(&temp)?;

    let state = match &initial_state {
        Some(s) => {
            let expected_len = n_zones * 3 + 9;
            let s_slice = checked_slice(s, expected_len, "initial_state")?;
            Some(State::from_array(s_slice, n_zones))
        }
        None => None,
    };

    // Extract zone arrays
    let ze_vec: Vec<f64>;
    let zone_elevs = match &zone_elevations {
        Some(ze) => {
            ze_vec = ze.as_slice()?.to_vec();
            Some(ze_vec.as_slice())
        }
        None => None,
    };

    let zf_vec: Vec<f64>;
    let zone_fracs = match &zone_fractions {
        Some(zf) => {
            zf_vec = zf.as_slice()?.to_vec();
            Some(zf_vec.as_slice())
        }
        None => None,
    };

    let (result, zone_outputs) = run::run(
        &p,
        precip_slice,
        pet_slice,
        temp_slice,
        state.as_ref(),
        n_zones,
        zone_elevs,
        zone_fracs,
        input_elevation,
        temp_gradient,
        precip_gradient,
    );

    // Build fluxes dict
    let dict = timeseries_to_dict!(
        py, result,
        precip, temp, pet, precip_rain, precip_snow, snow_pack, snow_melt,
        liquid_water_in_snow, snow_input, soil_moisture, recharge, actual_et,
        upper_zone, lower_zone, q0, q1, q2, percolation, qgw, streamflow,
    );

    // Build zone outputs dict if present
    let zone_dict = match zone_outputs {
        Some(zo) => {
            let zd = PyDict::new(py);
            zd.set_item("zone_elevations", PyArray1::from_vec(py, zo.zone_elevations))?;
            zd.set_item("zone_fractions", PyArray1::from_vec(py, zo.zone_fractions))?;

            // Convert Vec<Vec<f64>> to 2D: flatten to 1D and store shape
            let n_timesteps = zo.zone_temp.len();
            let n_z = if n_timesteps > 0 { zo.zone_temp[0].len() } else { 0 };

            // Helper to flatten 2D Vec<Vec<f64>> to contiguous Vec<f64> (row-major)
            let flatten_2d = |data: &[Vec<f64>]| -> Vec<f64> {
                let mut flat = Vec::with_capacity(n_timesteps * n_z);
                for row in data {
                    debug_assert_eq!(row.len(), n_z, "zone row length mismatch");
                    flat.extend_from_slice(row);
                }
                flat
            };

            // Store as flat arrays with shape info for Python to reshape
            zd.set_item("n_timesteps", n_timesteps)?;
            zd.set_item("n_zones", n_z)?;
            zd.set_item("zone_temp", PyArray1::from_vec(py, flatten_2d(&zo.zone_temp)))?;
            zd.set_item("zone_precip", PyArray1::from_vec(py, flatten_2d(&zo.zone_precip)))?;
            zd.set_item("snow_pack", PyArray1::from_vec(py, flatten_2d(&zo.snow_pack)))?;
            zd.set_item("liquid_water_in_snow", PyArray1::from_vec(py, flatten_2d(&zo.liquid_water_in_snow)))?;
            zd.set_item("snow_melt", PyArray1::from_vec(py, flatten_2d(&zo.snow_melt)))?;
            zd.set_item("snow_input", PyArray1::from_vec(py, flatten_2d(&zo.snow_input)))?;
            zd.set_item("soil_moisture", PyArray1::from_vec(py, flatten_2d(&zo.soil_moisture)))?;
            zd.set_item("recharge", PyArray1::from_vec(py, flatten_2d(&zo.recharge)))?;
            zd.set_item("actual_et", PyArray1::from_vec(py, flatten_2d(&zo.actual_et)))?;

            Some(zd)
        }
        None => None,
    };

    Ok((dict, zone_dict))
}

/// Execute one timestep of HBV-Light.
///
/// Returns (new_state_array, fluxes_dict).
#[pyfunction]
fn hbv_step<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
    precip: f64,
    pet: f64,
    temp: f64,
    uh_weights: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>)> {
    let p_slice = checked_slice(&params, 14, "params")?;
    let p = Parameters::from_array(p_slice)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let s_slice = checked_slice(&state, 12, "state")?;
    let s = State::from_array(s_slice, 1);

    let uh_slice = uh_weights.as_slice()?;

    let (new_state, fluxes) = run::step(&s, &p, precip, pet, temp, uh_slice);

    let state_arr = PyArray1::from_vec(py, new_state.to_array());

    let dict = fluxes_to_dict!(
        py, fluxes,
        precip, temp, pet, precip_rain, precip_snow, snow_pack, snow_melt,
        liquid_water_in_snow, snow_input, soil_moisture, recharge, actual_et,
        upper_zone, lower_zone, q0, q1, q2, percolation, qgw, streamflow,
    );

    Ok((state_arr, dict))
}

/// Compute triangular unit hydrograph weights.
#[pyfunction]
fn hbv_triangular_weights<'py>(
    py: Python<'py>,
    maxbas: f64,
) -> Bound<'py, PyArray1<f64>> {
    let weights = compute_triangular_weights(maxbas);
    PyArray1::from_vec(py, weights)
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "hbv_light")?;
    m.add_function(wrap_pyfunction!(hbv_run, &m)?)?;
    m.add_function(wrap_pyfunction!(hbv_step, &m)?)?;
    m.add_function(wrap_pyfunction!(hbv_triangular_weights, &m)?)?;
    m.add_class::<HBVResult>()?;
    m.add_class::<HBVStepFluxes>()?;
    parent.add_submodule(&m)?;
    Ok(())
}
