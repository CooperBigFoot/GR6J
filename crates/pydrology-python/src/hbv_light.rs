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
/// Args:
///     params: 1D array of 14 parameters.
///     precip: 1D array of daily precipitation [mm/day].
///     pet: 1D array of daily potential evapotranspiration [mm/day].
///     temp: 1D array of daily mean temperature [C].
///     initial_state: Optional 1D state array.
///     n_zones: Number of elevation zones (default 1).
///     zone_elevations: Optional 1D array of zone elevations [m].
///     zone_fractions: Optional 1D array of zone area fractions [-].
///     input_elevation: Optional elevation of input measurement [m].
///     temp_gradient: Optional temperature lapse rate [C/100m].
///     precip_gradient: Optional precipitation gradient [m^-1].
///
/// Returns:
///     A single dict containing all flux timeseries. When multiple elevation
///     zones are used, zone-level outputs are added to the same dict with a
///     ``zone_`` prefix (e.g. ``zone_temp``, ``zone_snow_pack``).
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
    precip_gradient_type=None,
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
    precip_gradient_type: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
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
            ze_vec = contiguous_slice(ze)?.to_vec();
            Some(ze_vec.as_slice())
        }
        None => None,
    };

    let zf_vec: Vec<f64>;
    let zone_fracs = match &zone_fractions {
        Some(zf) => {
            zf_vec = contiguous_slice(zf)?.to_vec();
            Some(zf_vec.as_slice())
        }
        None => None,
    };

    let use_linear = match precip_gradient_type {
        Some("exponential") | None => false,
        Some("linear") => true,
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown precip_gradient_type '{}'. Expected 'exponential' or 'linear'.",
                other
            )));
        }
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
        Some(use_linear),
    );

    // Build fluxes dict
    let dict = timeseries_to_dict!(
        py, result,
        precip, temp, pet, precip_rain, precip_snow, snow_pack, snow_melt,
        liquid_water_in_snow, snow_input, soil_moisture, recharge, actual_et,
        upper_zone, lower_zone, q0, q1, q2, percolation, qgw, streamflow,
    );

    // Add zone outputs to the same dict if present
    if let Some(zo) = zone_outputs {
        dict.set_item("zone_elevations", PyArray1::from_vec(py, zo.zone_elevations))?;
        dict.set_item("zone_fractions", PyArray1::from_vec(py, zo.zone_fractions))?;
        dict.set_item("zone_n_timesteps", zo.n_timesteps)?;
        dict.set_item("zone_n_zones", zo.n_zones)?;
        dict.set_item("zone_temp", PyArray1::from_vec(py, zo.zone_temp))?;
        dict.set_item("zone_precip", PyArray1::from_vec(py, zo.zone_precip))?;
        dict.set_item("zone_snow_pack", PyArray1::from_vec(py, zo.snow_pack))?;
        dict.set_item("zone_liquid_water_in_snow", PyArray1::from_vec(py, zo.liquid_water_in_snow))?;
        dict.set_item("zone_snow_melt", PyArray1::from_vec(py, zo.snow_melt))?;
        dict.set_item("zone_snow_input", PyArray1::from_vec(py, zo.snow_input))?;
        dict.set_item("zone_soil_moisture", PyArray1::from_vec(py, zo.soil_moisture))?;
        dict.set_item("zone_recharge", PyArray1::from_vec(py, zo.recharge))?;
        dict.set_item("zone_actual_et", PyArray1::from_vec(py, zo.actual_et))?;
    }

    Ok(dict)
}

/// Execute one timestep of HBV-Light.
///
/// Args:
///     state: 1D state array.
///     params: 1D array of 14 parameters.
///     precip: Daily precipitation [mm/day].
///     pet: Daily potential evapotranspiration [mm/day].
///     temp: Daily mean temperature [C].
///     uh_weights: 1D array of unit hydrograph weights.
///
/// Returns:
///     Tuple of (new_state_array, fluxes_dict).
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

    let uh_slice = contiguous_slice(&uh_weights)?;

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
