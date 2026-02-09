use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::convert::{checked_slice, checked_slice_min, contiguous_slice};

use pydrology_core::cemaneige::constants::LAYER_STATE_SIZE;
use pydrology_core::cemaneige::coupled;
use pydrology_core::gr6j::constants::NH;
use pydrology_core::gr6j::params::Parameters as GR6JParameters;
use pydrology_core::gr6j::state::State as GR6JState;

// ---------------------------------------------------------------------------
// Typed pyclass result objects
// ---------------------------------------------------------------------------

define_timeseries_result! {
    /// CemaNeige snow run results with typed numpy array attributes.
    pub struct SnowResult from pydrology_core::cemaneige::coupled::SnowTimeseries {
        pliq, psol, snow_pack, thermal_state, gratio,
        pot_melt, melt, pliq_and_melt, temp, precip,
    }
}

define_step_result! {
    /// CemaNeige snow single-timestep flux results.
    pub struct SnowStepFluxes from pydrology_core::cemaneige::coupled::SnowFluxes {
        pliq, psol, snow_pack, thermal_state, gratio,
        pot_melt, melt, pliq_and_melt, temp, precip,
    }
}

// ---------------------------------------------------------------------------
// Existing dict-returning functions (backward compatible)
// ---------------------------------------------------------------------------

/// Run the coupled GR6J-CemaNeige model over a timeseries.
///
/// Returns (snow_dict, gr6j_dict, layer_dict).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    params,
    precip,
    pet,
    temp,
    initial_state=None,
    uh1_ordinates=None,
    uh2_ordinates=None,
    n_layers=1,
    layer_elevations=None,
    layer_fractions=None,
    input_elevation=None,
    temp_gradient=None,
    precip_gradient=None,
    mean_annual_solid_precip=0.0,
))]
fn gr6j_cemaneige_run<'py>(
    py: Python<'py>,
    params: PyReadonlyArray1<'py, f64>,
    precip: PyReadonlyArray1<'py, f64>,
    pet: PyReadonlyArray1<'py, f64>,
    temp: PyReadonlyArray1<'py, f64>,
    initial_state: Option<PyReadonlyArray1<'py, f64>>,
    uh1_ordinates: Option<PyReadonlyArray1<'py, f64>>,
    uh2_ordinates: Option<PyReadonlyArray1<'py, f64>>,
    n_layers: usize,
    layer_elevations: Option<PyReadonlyArray1<'py, f64>>,
    layer_fractions: Option<PyReadonlyArray1<'py, f64>>,
    input_elevation: Option<f64>,
    temp_gradient: Option<f64>,
    precip_gradient: Option<f64>,
    mean_annual_solid_precip: f64,
) -> PyResult<(
    Bound<'py, PyDict>,
    Bound<'py, PyDict>,
    Bound<'py, PyDict>,
)> {
    let p_slice = checked_slice_min(&params, 8, "params")?;

    let gr6j_params =
        GR6JParameters::new_unchecked(p_slice[0], p_slice[1], p_slice[2], p_slice[3], p_slice[4], p_slice[5]);
    let ctg = p_slice[6];
    let kf = p_slice[7];

    let precip_slice = contiguous_slice(&precip)?;
    let pet_slice = contiguous_slice(&pet)?;
    let temp_slice = contiguous_slice(&temp)?;

    // Build layer config
    let default_elevs = vec![0.0f64; n_layers];
    let default_fracs = vec![1.0 / n_layers as f64; n_layers];
    let layer_elevs: Vec<f64> = match &layer_elevations {
        Some(arr) => arr.as_slice()?.to_vec(),
        None => default_elevs,
    };
    let layer_fracs: Vec<f64> = match &layer_fractions {
        Some(arr) => arr.as_slice()?.to_vec(),
        None => default_fracs,
    };

    let input_elev = input_elevation.unwrap_or(f64::NAN);
    let t_grad = temp_gradient.unwrap_or(pydrology_core::cemaneige::constants::GRAD_T_DEFAULT);
    let p_grad = precip_gradient.unwrap_or(pydrology_core::cemaneige::constants::GRAD_P_DEFAULT);

    // Parse initial state
    let (init_gr6j, init_snow) = match &initial_state {
        Some(s) => {
            let s_slice = s.as_slice()?;
            let expected = 63 + n_layers * LAYER_STATE_SIZE;
            if s_slice.len() != expected {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "initial_state must have {} elements (63 GR6J + {} * {} snow), got {}",
                    expected,
                    n_layers,
                    LAYER_STATE_SIZE,
                    s_slice.len()
                )));
            }
            let mut gr6j_arr = [0.0f64; 63];
            gr6j_arr.copy_from_slice(&s_slice[..63]);
            let gr6j_state = GR6JState::from_array(&gr6j_arr);

            let mut snow_states = Vec::with_capacity(n_layers);
            for i in 0..n_layers {
                let base = 63 + i * LAYER_STATE_SIZE;
                let mut ls = [0.0f64; LAYER_STATE_SIZE];
                ls.copy_from_slice(&s_slice[base..base + LAYER_STATE_SIZE]);
                snow_states.push(ls);
            }
            (Some(gr6j_state), Some(snow_states))
        }
        None => (None, None),
    };

    // Handle pre-computed UH ordinates (ignore them -- coupled_run computes internally)
    let _ = uh1_ordinates;
    let _ = uh2_ordinates;

    let result = coupled::coupled_run(
        &gr6j_params,
        ctg,
        kf,
        precip_slice,
        pet_slice,
        temp_slice,
        init_gr6j.as_ref(),
        init_snow.as_deref(),
        n_layers,
        &layer_elevs,
        &layer_fracs,
        input_elev,
        t_grad,
        p_grad,
        mean_annual_solid_precip,
    );

    // Build snow output dict (uses prefixed keys for backward compat)
    let snow_dict = PyDict::new(py);
    snow_dict.set_item("snow_pliq", PyArray1::from_vec(py, result.snow.pliq))?;
    snow_dict.set_item("snow_psol", PyArray1::from_vec(py, result.snow.psol))?;
    snow_dict.set_item("snow_pack", PyArray1::from_vec(py, result.snow.snow_pack))?;
    snow_dict.set_item("snow_thermal_state", PyArray1::from_vec(py, result.snow.thermal_state))?;
    snow_dict.set_item("snow_gratio", PyArray1::from_vec(py, result.snow.gratio))?;
    snow_dict.set_item("snow_pot_melt", PyArray1::from_vec(py, result.snow.pot_melt))?;
    snow_dict.set_item("snow_melt", PyArray1::from_vec(py, result.snow.melt))?;
    snow_dict.set_item("snow_pliq_and_melt", PyArray1::from_vec(py, result.snow.pliq_and_melt))?;
    snow_dict.set_item("snow_temp", PyArray1::from_vec(py, result.snow.temp))?;

    // Build GR6J output dict
    let gr6j_dict = timeseries_to_dict!(
        py, result.gr6j,
        pet, precip, production_store, net_rainfall, storage_infiltration,
        actual_et, percolation, effective_rainfall, q9, q1, routing_store,
        exchange, actual_exchange_routing, actual_exchange_direct,
        actual_exchange_total, qr, qrexp, exponential_store, qd, streamflow,
    );

    // Build per-layer outputs dict
    let layer_dict = PyDict::new(py);
    let n_timesteps = precip_slice.len();

    // Build 2D arrays (n_timesteps x n_layers) for each layer field
    let mut layer_snow_pack = vec![0.0f64; n_timesteps * n_layers];
    let mut layer_thermal = vec![0.0f64; n_timesteps * n_layers];
    let mut layer_gratio = vec![0.0f64; n_timesteps * n_layers];
    let mut layer_melt = vec![0.0f64; n_timesteps * n_layers];
    let mut layer_pliq_and_melt = vec![0.0f64; n_timesteps * n_layers];
    let mut layer_temp = vec![0.0f64; n_timesteps * n_layers];
    let mut layer_precip_arr = vec![0.0f64; n_timesteps * n_layers];

    for t in 0..n_timesteps {
        debug_assert_eq!(result.layers[t].len(), n_layers, "layer count mismatch at timestep {t}");
        for l in 0..n_layers {
            let idx = t * n_layers + l;
            let lf = &result.layers[t][l];
            layer_snow_pack[idx] = lf.snow_pack;
            layer_thermal[idx] = lf.thermal_state;
            layer_gratio[idx] = lf.gratio;
            layer_melt[idx] = lf.melt;
            layer_pliq_and_melt[idx] = lf.pliq_and_melt;
            layer_temp[idx] = lf.temp;
            layer_precip_arr[idx] = lf.precip;
        }
    }

    layer_dict.set_item(
        "snow_pack",
        PyArray2::from_vec2(py, &to_2d(&layer_snow_pack, n_timesteps, n_layers))?,
    )?;
    layer_dict.set_item(
        "snow_thermal_state",
        PyArray2::from_vec2(py, &to_2d(&layer_thermal, n_timesteps, n_layers))?,
    )?;
    layer_dict.set_item(
        "snow_gratio",
        PyArray2::from_vec2(py, &to_2d(&layer_gratio, n_timesteps, n_layers))?,
    )?;
    layer_dict.set_item(
        "snow_melt",
        PyArray2::from_vec2(py, &to_2d(&layer_melt, n_timesteps, n_layers))?,
    )?;
    layer_dict.set_item(
        "snow_pliq_and_melt",
        PyArray2::from_vec2(py, &to_2d(&layer_pliq_and_melt, n_timesteps, n_layers))?,
    )?;
    layer_dict.set_item(
        "layer_temp",
        PyArray2::from_vec2(py, &to_2d(&layer_temp, n_timesteps, n_layers))?,
    )?;
    layer_dict.set_item(
        "layer_precip",
        PyArray2::from_vec2(py, &to_2d(&layer_precip_arr, n_timesteps, n_layers))?,
    )?;

    Ok((snow_dict, gr6j_dict, layer_dict))
}

/// Convert a flat 1D array to a Vec<Vec<f64>> for PyArray2::from_vec2.
fn to_2d(flat: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    debug_assert_eq!(flat.len(), rows * cols, "flat array length mismatch in to_2d");
    (0..rows)
        .map(|r| flat[r * cols..(r + 1) * cols].to_vec())
        .collect()
}

/// Execute one timestep of the coupled GR6J-CemaNeige model.
///
/// Returns (new_state, fluxes_dict).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    state,
    params,
    precip,
    pet,
    temp,
    uh1_ordinates,
    uh2_ordinates,
    layer_elevations,
    layer_fractions,
    input_elevation=None,
    temp_gradient=None,
    precip_gradient=None,
))]
fn gr6j_cemaneige_step<'py>(
    py: Python<'py>,
    state: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
    precip: f64,
    pet: f64,
    temp: f64,
    uh1_ordinates: PyReadonlyArray1<'py, f64>,
    uh2_ordinates: PyReadonlyArray1<'py, f64>,
    layer_elevations: PyReadonlyArray1<'py, f64>,
    layer_fractions: PyReadonlyArray1<'py, f64>,
    input_elevation: Option<f64>,
    temp_gradient: Option<f64>,
    precip_gradient: Option<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>)> {
    let p_slice = checked_slice_min(&params, 8, "params")?;

    let gr6j_params =
        GR6JParameters::new_unchecked(p_slice[0], p_slice[1], p_slice[2], p_slice[3], p_slice[4], p_slice[5]);
    let ctg = p_slice[6];
    let kf = p_slice[7];

    let elevs = contiguous_slice(&layer_elevations)?;
    let fracs = contiguous_slice(&layer_fractions)?;
    let n_layers = elevs.len();

    let expected_state_size = 63 + n_layers * LAYER_STATE_SIZE;
    let s_slice = checked_slice(&state, expected_state_size, "state")?;

    // Parse GR6J state
    let mut gr6j_arr = [0.0f64; 63];
    gr6j_arr.copy_from_slice(&s_slice[..63]);
    let gr6j_state = GR6JState::from_array(&gr6j_arr);

    // Parse snow layer states
    let mut snow_states = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let base = 63 + i * LAYER_STATE_SIZE;
        let mut ls = [0.0f64; LAYER_STATE_SIZE];
        ls.copy_from_slice(&s_slice[base..base + LAYER_STATE_SIZE]);
        snow_states.push(ls);
    }

    // Parse UH ordinates
    let uh1_slice = checked_slice(&uh1_ordinates, NH, "uh1_ordinates")?;
    let mut uh1_arr = [0.0f64; NH];
    uh1_arr.copy_from_slice(uh1_slice);

    let uh2_slice = checked_slice(&uh2_ordinates, 2 * NH, "uh2_ordinates")?;
    let mut uh2_arr = [0.0f64; 2 * NH];
    uh2_arr.copy_from_slice(uh2_slice);

    let input_elev = input_elevation.unwrap_or(f64::NAN);
    let t_grad = temp_gradient.unwrap_or(pydrology_core::cemaneige::constants::GRAD_T_DEFAULT);
    let p_grad = precip_gradient.unwrap_or(pydrology_core::cemaneige::constants::GRAD_P_DEFAULT);
    let skip_extrap = input_elev.is_nan();

    let (new_gr6j, new_snow, snow_fluxes, gr6j_fluxes, _per_layer) = coupled::coupled_step(
        &gr6j_state,
        &snow_states,
        &gr6j_params,
        ctg,
        kf,
        precip,
        pet,
        temp,
        &uh1_arr,
        &uh2_arr,
        elevs,
        fracs,
        input_elev,
        t_grad,
        p_grad,
        skip_extrap,
    );

    // Build new state array
    let gr6j_state_arr = new_gr6j.to_array();
    let mut state_out = Vec::with_capacity(expected_state_size);
    state_out.extend_from_slice(&gr6j_state_arr);
    for ls in &new_snow {
        state_out.extend_from_slice(ls);
    }

    // Build fluxes dict
    let dict = PyDict::new(py);
    // Snow fluxes
    dict.set_item("precip_raw", precip)?;
    dict.set_item("snow_pliq", snow_fluxes.pliq)?;
    dict.set_item("snow_psol", snow_fluxes.psol)?;
    dict.set_item("snow_pack", snow_fluxes.snow_pack)?;
    dict.set_item("snow_thermal_state", snow_fluxes.thermal_state)?;
    dict.set_item("snow_gratio", snow_fluxes.gratio)?;
    dict.set_item("snow_pot_melt", snow_fluxes.pot_melt)?;
    dict.set_item("snow_melt", snow_fluxes.melt)?;
    dict.set_item("snow_pliq_and_melt", snow_fluxes.pliq_and_melt)?;
    dict.set_item("snow_temp", snow_fluxes.temp)?;
    // GR6J fluxes
    dict.set_item("pet", gr6j_fluxes.pet)?;
    dict.set_item("precip", gr6j_fluxes.precip)?;
    dict.set_item("production_store", gr6j_fluxes.production_store)?;
    dict.set_item("net_rainfall", gr6j_fluxes.net_rainfall)?;
    dict.set_item("storage_infiltration", gr6j_fluxes.storage_infiltration)?;
    dict.set_item("actual_et", gr6j_fluxes.actual_et)?;
    dict.set_item("percolation", gr6j_fluxes.percolation)?;
    dict.set_item("effective_rainfall", gr6j_fluxes.effective_rainfall)?;
    dict.set_item("q9", gr6j_fluxes.q9)?;
    dict.set_item("q1", gr6j_fluxes.q1)?;
    dict.set_item("routing_store", gr6j_fluxes.routing_store)?;
    dict.set_item("exchange", gr6j_fluxes.exchange)?;
    dict.set_item("actual_exchange_routing", gr6j_fluxes.actual_exchange_routing)?;
    dict.set_item("actual_exchange_direct", gr6j_fluxes.actual_exchange_direct)?;
    dict.set_item("actual_exchange_total", gr6j_fluxes.actual_exchange_total)?;
    dict.set_item("qr", gr6j_fluxes.qr)?;
    dict.set_item("qrexp", gr6j_fluxes.qrexp)?;
    dict.set_item("exponential_store", gr6j_fluxes.exponential_store)?;
    dict.set_item("qd", gr6j_fluxes.qd)?;
    dict.set_item("streamflow", gr6j_fluxes.streamflow)?;

    Ok((PyArray1::from_vec(py, state_out), dict))
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "cemaneige")?;
    m.add_function(wrap_pyfunction!(gr6j_cemaneige_run, &m)?)?;
    m.add_function(wrap_pyfunction!(gr6j_cemaneige_step, &m)?)?;
    m.add_class::<SnowResult>()?;
    m.add_class::<SnowStepFluxes>()?;
    parent.add_submodule(&m)?;
    Ok(())
}
