//! Coupled GR6J-CemaNeige model.
//!
//! Runs CemaNeige snow processing per elevation layer, aggregates liquid output,
//! then routes it through GR6J.

use super::constants::LAYER_STATE_SIZE;
use super::processes;
use crate::elevation;
use crate::gr6j::constants::NH;
use crate::gr6j::fluxes::{Fluxes as GR6JFluxes, FluxesTimeseries as GR6JFluxesTimeseries};
use pydrology_macros::Fluxes;
use crate::gr6j::params::Parameters as GR6JParameters;
use crate::gr6j::run::step as gr6j_step;
use crate::gr6j::state::State as GR6JState;
use crate::gr6j::unit_hydrographs::compute_uh_ordinates;

/// Number of aggregated snow flux fields in coupled output.
pub const N_SNOW_FLUXES: usize = 10;

/// Number of per-layer snow flux fields in coupled output.
pub const N_LAYER_FLUXES: usize = 10;

/// Snow fluxes for a single timestep (aggregated across layers).
#[derive(Debug, Clone, Copy, Fluxes)]
#[fluxes(timeseries_name = "SnowTimeseries")]
pub struct SnowFluxes {
    pub pliq: f64,
    pub psol: f64,
    pub snow_pack: f64,
    pub thermal_state: f64,
    pub gratio: f64,
    pub pot_melt: f64,
    pub melt: f64,
    pub pliq_and_melt: f64,
    pub temp: f64,
    pub precip: f64,
}

/// Per-layer snow fluxes for a single timestep.
#[derive(Debug, Clone, Copy)]
pub struct LayerFluxes {
    pub pliq: f64,
    pub psol: f64,
    pub snow_pack: f64,
    pub thermal_state: f64,
    pub gratio: f64,
    pub pot_melt: f64,
    pub melt: f64,
    pub pliq_and_melt: f64,
    pub temp: f64,
    pub precip: f64,
}

/// Timeseries output from the coupled model.
#[derive(Debug)]
pub struct CoupledOutput {
    pub snow: SnowTimeseries,
    pub gr6j: GR6JFluxesTimeseries,
    pub layers: Vec<LayerFluxes>,
    pub n_layers: usize,
}

/// Run a single CemaNeige layer step (inline, no struct overhead).
///
/// Returns (new_layer_state, layer_fluxes).
fn cemaneige_layer_step(
    layer_state: &[f64; LAYER_STATE_SIZE],
    ctg: f64,
    kf: f64,
    precip: f64,
    temp: f64,
) -> ([f64; LAYER_STATE_SIZE], LayerFluxes) {
    let g = layer_state[0];
    let etg = layer_state[1];
    let gthreshold = layer_state[2];
    let glocalmax = layer_state[3];

    let solid_fraction = processes::compute_solid_fraction(temp);
    let (pliq, psol) = processes::partition_precipitation(precip, solid_fraction);
    let g_after_accum = g + psol;
    let new_etg = processes::update_thermal_state(etg, temp, ctg);
    let pot_melt = processes::compute_potential_melt(new_etg, temp, kf, g_after_accum);
    let gratio_for_melt = processes::compute_gratio(g_after_accum, gthreshold);
    let melt = processes::compute_actual_melt(pot_melt, gratio_for_melt);
    let new_g = g_after_accum - melt;
    let gratio_output = processes::compute_gratio(new_g, gthreshold);
    let pliq_and_melt = pliq + melt;

    let new_state = [new_g, new_etg, gthreshold, glocalmax];
    let fluxes = LayerFluxes {
        pliq,
        psol,
        snow_pack: new_g,
        thermal_state: new_etg,
        gratio: gratio_output,
        pot_melt,
        melt,
        pliq_and_melt,
        temp,
        precip,
    };

    (new_state, fluxes)
}

/// Execute one timestep of the coupled GR6J-CemaNeige model.
///
/// Processes snow for each layer, aggregates liquid output, then runs GR6J.
#[allow(clippy::too_many_arguments)]
pub fn coupled_step(
    gr6j_state: &GR6JState,
    snow_layer_states: &[[f64; LAYER_STATE_SIZE]],
    gr6j_params: &GR6JParameters,
    ctg: f64,
    kf: f64,
    precip: f64,
    pet: f64,
    temp: f64,
    uh1_ordinates: &[f64; NH],
    uh2_ordinates: &[f64; 2 * NH],
    layer_elevations: &[f64],
    layer_fractions: &[f64],
    input_elevation: f64,
    temp_gradient: f64,
    precip_gradient: f64,
    skip_extrapolation: bool,
) -> (
    GR6JState,
    Vec<[f64; LAYER_STATE_SIZE]>,
    SnowFluxes,
    GR6JFluxes,
    Vec<LayerFluxes>,
) {
    let n_layers = snow_layer_states.len();
    let mut new_snow_states = Vec::with_capacity(n_layers);
    let mut per_layer_fluxes = Vec::with_capacity(n_layers);

    // Aggregation accumulators
    let mut agg = SnowFluxes {
        pliq: 0.0,
        psol: 0.0,
        snow_pack: 0.0,
        thermal_state: 0.0,
        gratio: 0.0,
        pot_melt: 0.0,
        melt: 0.0,
        pliq_and_melt: 0.0,
        temp: 0.0,
        precip: 0.0,
    };

    for i in 0..n_layers {
        let (layer_temp, layer_precip) = if skip_extrapolation {
            (temp, precip)
        } else {
            (
                elevation::extrapolate_temp(temp, input_elevation, layer_elevations[i], temp_gradient),
                elevation::extrapolate_precip(precip, input_elevation, layer_elevations[i], precip_gradient),
            )
        };

        let (new_ls, fluxes) =
            cemaneige_layer_step(&snow_layer_states[i], ctg, kf, layer_precip, layer_temp);

        new_snow_states.push(new_ls);

        let frac = layer_fractions[i];
        agg.pliq += fluxes.pliq * frac;
        agg.psol += fluxes.psol * frac;
        agg.snow_pack += fluxes.snow_pack * frac;
        agg.thermal_state += fluxes.thermal_state * frac;
        agg.gratio += fluxes.gratio * frac;
        agg.pot_melt += fluxes.pot_melt * frac;
        agg.melt += fluxes.melt * frac;
        agg.pliq_and_melt += fluxes.pliq_and_melt * frac;
        agg.temp += fluxes.temp * frac;
        agg.precip += fluxes.precip * frac;

        per_layer_fluxes.push(fluxes);
    }

    // Feed pliq_and_melt as precipitation input to GR6J
    let (new_gr6j_state, gr6j_fluxes) = gr6j_step(
        gr6j_state,
        gr6j_params,
        agg.pliq_and_melt,
        pet,
        uh1_ordinates,
        uh2_ordinates,
    );

    (
        new_gr6j_state,
        new_snow_states,
        agg,
        gr6j_fluxes,
        per_layer_fluxes,
    )
}

/// Run the coupled GR6J-CemaNeige model over a timeseries.
///
/// Inlines the step logic to eliminate per-timestep heap allocations.
/// `coupled_step()` is preserved separately for single-step PyO3 use.
#[allow(clippy::too_many_arguments)]
pub fn coupled_run(
    gr6j_params: &GR6JParameters,
    ctg: f64,
    kf: f64,
    precip: &[f64],
    pet: &[f64],
    temp: &[f64],
    initial_gr6j_state: Option<&GR6JState>,
    initial_snow_states: Option<&[[f64; LAYER_STATE_SIZE]]>,
    n_layers: usize,
    layer_elevations: &[f64],
    layer_fractions: &[f64],
    input_elevation: f64,
    temp_gradient: f64,
    precip_gradient: f64,
    mean_annual_solid_precip: f64,
) -> CoupledOutput {
    let n = precip.len();
    assert_eq!(
        precip.len(),
        pet.len(),
        "precip and pet must have the same length"
    );
    assert_eq!(
        precip.len(),
        temp.len(),
        "precip and temp must have the same length"
    );

    let (uh1_ordinates, uh2_ordinates) = compute_uh_ordinates(gr6j_params.x4);
    let skip_extrapolation = input_elevation.is_nan();

    // Initialize GR6J state
    let mut gr6j_state = match initial_gr6j_state {
        Some(s) => s.clone(),
        None => GR6JState::initialize(gr6j_params),
    };

    // Initialize snow layer states — owned, mutated in place
    let mut snow_states: Vec<[f64; LAYER_STATE_SIZE]> = match initial_snow_states {
        Some(states) => states.to_vec(),
        None => {
            let gthreshold = super::constants::GTHRESHOLD_FACTOR * mean_annual_solid_precip;
            vec![[0.0, 0.0, gthreshold, gthreshold]; n_layers]
        }
    };

    // Pre-allocate outputs
    let mut snow_ts = SnowTimeseries::with_len(n);
    let mut gr6j_ts = GR6JFluxesTimeseries::with_len(n);
    // Flat allocation: n_layers per timestep, n timesteps total
    let mut all_layer_fluxes: Vec<LayerFluxes> = Vec::with_capacity(n * n_layers);

    for t in 0..n {

        // Aggregation accumulators
        let mut agg = SnowFluxes {
            pliq: 0.0,
            psol: 0.0,
            snow_pack: 0.0,
            thermal_state: 0.0,
            gratio: 0.0,
            pot_melt: 0.0,
            melt: 0.0,
            pliq_and_melt: 0.0,
            temp: 0.0,
            precip: 0.0,
        };

        for i in 0..n_layers {
            let (layer_temp, layer_precip) = if skip_extrapolation {
                (temp[t], precip[t])
            } else {
                (
                    elevation::extrapolate_temp(
                        temp[t],
                        input_elevation,
                        layer_elevations[i],
                        temp_gradient,
                    ),
                    elevation::extrapolate_precip(
                        precip[t],
                        input_elevation,
                        layer_elevations[i],
                        precip_gradient,
                    ),
                )
            };

            let (new_ls, fluxes) =
                cemaneige_layer_step(&snow_states[i], ctg, kf, layer_precip, layer_temp);
            snow_states[i] = new_ls;

            let frac = layer_fractions[i];
            agg.pliq += fluxes.pliq * frac;
            agg.psol += fluxes.psol * frac;
            agg.snow_pack += fluxes.snow_pack * frac;
            agg.thermal_state += fluxes.thermal_state * frac;
            agg.gratio += fluxes.gratio * frac;
            agg.pot_melt += fluxes.pot_melt * frac;
            agg.melt += fluxes.melt * frac;
            agg.pliq_and_melt += fluxes.pliq_and_melt * frac;
            agg.temp += fluxes.temp * frac;
            agg.precip += fluxes.precip * frac;

            all_layer_fluxes.push(fluxes);
        }

        // Feed aggregated liquid to GR6J
        let (new_gr6j_state, gr6j_fluxes) = gr6j_step(
            &gr6j_state,
            gr6j_params,
            agg.pliq_and_melt,
            pet[t],
            &uh1_ordinates,
            &uh2_ordinates,
        );

        gr6j_state = new_gr6j_state;
        unsafe {
            snow_ts.write_unchecked(t, &agg);
            gr6j_ts.write_unchecked(t, &gr6j_fluxes);
        }
    }

    CoupledOutput {
        snow: snow_ts,
        gr6j: gr6j_ts,
        layers: all_layer_fluxes,
        n_layers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cemaneige::constants::GTHRESHOLD_FACTOR;

    fn test_gr6j_params() -> GR6JParameters {
        GR6JParameters::new(350.0, 0.0, 90.0, 1.7, 0.0, 5.0).unwrap()
    }

    fn test_snow_states(n_layers: usize, masp: f64) -> Vec<[f64; LAYER_STATE_SIZE]> {
        let gth = GTHRESHOLD_FACTOR * masp;
        vec![[0.0, 0.0, gth, gth]; n_layers]
    }

    // -- coupled_step tests --

    #[test]
    fn coupled_step_returns_finite() {
        let gp = test_gr6j_params();
        let gs = GR6JState::initialize(&gp);
        let ss = test_snow_states(1, 150.0);
        let (uh1, uh2) = compute_uh_ordinates(gp.x4);

        let (new_gs, new_ss, snow_f, gr6j_f, _) = coupled_step(
            &gs, &ss, &gp, 0.97, 2.5, 10.0, 3.0, 2.0,
            &uh1, &uh2, &[0.0], &[1.0], f64::NAN, 0.6, 0.00041, true,
        );

        assert!(new_gs.production_store.is_finite());
        assert!(new_ss[0][0].is_finite());
        assert!(snow_f.pliq_and_melt.is_finite());
        assert!(gr6j_f.streamflow.is_finite());
        assert!(gr6j_f.streamflow >= 0.0);
    }

    #[test]
    fn coupled_step_snow_feeds_gr6j() {
        let gp = test_gr6j_params();
        let gs = GR6JState::initialize(&gp);
        let ss = test_snow_states(1, 150.0);
        let (uh1, uh2) = compute_uh_ordinates(gp.x4);

        let (_, _, snow_f, gr6j_f, _) = coupled_step(
            &gs, &ss, &gp, 0.97, 2.5, 10.0, 3.0, 5.0,
            &uh1, &uh2, &[0.0], &[1.0], f64::NAN, 0.6, 0.00041, true,
        );

        // GR6J precip input should equal snow pliq_and_melt
        assert!((gr6j_f.precip - snow_f.pliq_and_melt).abs() < 1e-10);
    }

    // -- coupled_run tests --

    #[test]
    fn coupled_run_output_length() {
        let gp = test_gr6j_params();
        let precip = [10.0, 5.0, 0.0, 15.0, 2.0];
        let pet = [3.0, 4.0, 5.0, 2.0, 3.5];
        let temp = [2.0, -1.0, -5.0, 0.0, 3.0];

        let result = coupled_run(
            &gp, 0.97, 2.5, &precip, &pet, &temp,
            None, None, 1,
            &[0.0], &[1.0], f64::NAN, 0.6, 0.00041, 150.0,
        );

        assert_eq!(result.snow.pliq.len(), 5);
        assert_eq!(result.gr6j.streamflow.len(), 5);
        assert_eq!(result.layers.len(), 5 * 1); // 5 timesteps * 1 layer
        assert_eq!(result.n_layers, 1);
    }

    #[test]
    fn coupled_run_all_finite() {
        let gp = test_gr6j_params();
        let precip = [10.0, 5.0, 0.0, 15.0, 2.0, 8.0];
        let pet = [3.0, 4.0, 5.0, 2.0, 3.5, 4.0];
        let temp = [2.0, -1.0, -5.0, 0.0, 3.0, 1.0];

        let result = coupled_run(
            &gp, 0.97, 2.5, &precip, &pet, &temp,
            None, None, 1,
            &[0.0], &[1.0], f64::NAN, 0.6, 0.00041, 150.0,
        );

        for t in 0..6 {
            assert!(result.gr6j.streamflow[t].is_finite(), "non-finite streamflow at t={t}");
            assert!(result.gr6j.streamflow[t] >= 0.0, "negative streamflow at t={t}");
            assert!(result.snow.pliq_and_melt[t].is_finite());
        }
    }

    #[test]
    fn coupled_run_multilayer() {
        let gp = test_gr6j_params();
        let precip = [10.0, 5.0, 15.0];
        let pet = [3.0, 4.0, 2.0];
        let temp = [2.0, -1.0, 3.0];

        let result = coupled_run(
            &gp, 0.97, 2.5, &precip, &pet, &temp,
            None, None, 3,
            &[200.0, 600.0, 1000.0], &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            500.0, 0.6, 0.00041, 150.0,
        );

        assert_eq!(result.gr6j.streamflow.len(), 3);
        // Flat layers: n_timesteps * n_layers
        assert_eq!(result.layers.len(), 3 * 3);
        assert_eq!(result.n_layers, 3);
    }

    #[test]
    fn coupled_run_snow_accumulates_in_cold() {
        let gp = test_gr6j_params();
        // All cold, all precip should become snow
        let precip = [10.0; 5];
        let pet = [1.0; 5];
        let temp = [-10.0; 5];

        let result = coupled_run(
            &gp, 0.97, 2.5, &precip, &pet, &temp,
            None, None, 1,
            &[0.0], &[1.0], f64::NAN, 0.6, 0.00041, 150.0,
        );

        // All precip should be solid
        for t in 0..5 {
            assert!((result.snow.psol[t] - 10.0).abs() < 1e-10);
            assert!(result.snow.pliq[t].abs() < 1e-10);
        }
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn coupled_run_panics_mismatched_lengths() {
        let gp = test_gr6j_params();
        coupled_run(
            &gp, 0.97, 2.5, &[10.0, 5.0], &[3.0], &[2.0, 1.0],
            None, None, 1,
            &[0.0], &[1.0], f64::NAN, 0.6, 0.00041, 150.0,
        );
    }
}
