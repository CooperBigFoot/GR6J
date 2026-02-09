/// HBV-Light model orchestration functions.
///
/// - `step()`: Execute a single timestep (single-zone)
/// - `run()`: Execute over a timeseries with optional multi-zone support
use smallvec::smallvec;

use super::fluxes::{Fluxes, FluxesTimeseries, ZoneOutputs};
use super::params::Parameters;
use super::processes;
use super::routing;
use super::state::State;
use crate::elevation;
use crate::traits::HydrologicalModel;

/// Execute one timestep of HBV-Light for a single zone.
///
/// Returns (new_state, fluxes).
pub fn step(
    state: &State,
    params: &Parameters,
    precip: f64,
    pet: f64,
    temp: f64,
    uh_weights: &[f64],
) -> (State, Fluxes) {
    assert_eq!(
        state.n_zones(),
        1,
        "step() only supports single-zone state"
    );

    let sp = state.zone_states[0][0];
    let lw = state.zone_states[0][1];
    let sm = state.zone_states[0][2];

    // 1. Snow routine
    let (p_rain, p_snow) = processes::partition_precipitation(precip, temp, params.tt, params.sfcf);
    let melt = processes::compute_melt(temp, params.tt, params.cfmax, sp);
    let refreeze = processes::compute_refreezing(temp, params.tt, params.cfmax, params.cfr, lw);
    let (new_sp, new_lw, snow_outflow) =
        processes::update_snow_pack(sp, lw, p_snow, melt, refreeze, params.cwh);

    let snow_input = p_rain + snow_outflow;

    // 2. Soil routine
    let recharge = processes::compute_recharge(snow_input, sm, params.fc, params.beta);
    let et_act = processes::compute_actual_et(pet, sm, params.fc, params.lp);
    let new_sm = processes::update_soil_moisture(sm, snow_input, recharge, et_act, params.fc);

    // 3. Response routine
    let (q0, q1) = routing::upper_zone_outflows(state.upper_zone, params.k0, params.k1, params.uzl);
    let perc = routing::compute_percolation(state.upper_zone, params.perc);
    let new_suz = routing::update_upper_zone(state.upper_zone, recharge, q0, q1, perc);

    let q2 = routing::lower_zone_outflow(state.lower_zone, params.k2);
    let new_slz = routing::update_lower_zone(state.lower_zone, perc, q2);

    let qgw = q0 + q1 + q2;

    // 4. Routing
    let (new_buffer, qsim) =
        routing::convolve_triangular(qgw, &state.routing_buffer, uh_weights);

    let new_state = State {
        zone_states: smallvec![[new_sp, new_lw, new_sm]],
        upper_zone: new_suz,
        lower_zone: new_slz,
        routing_buffer: new_buffer,
    };

    let fluxes = Fluxes {
        precip,
        temp,
        pet,
        precip_rain: p_rain,
        precip_snow: p_snow,
        snow_pack: new_sp,
        snow_melt: melt,
        liquid_water_in_snow: new_lw,
        snow_input,
        soil_moisture: new_sm,
        recharge,
        actual_et: et_act,
        upper_zone: new_suz,
        lower_zone: new_slz,
        q0,
        q1,
        q2,
        percolation: perc,
        qgw,
        streamflow: qsim,
    };

    (new_state, fluxes)
}

/// Run HBV-Light over a timeseries with optional multi-zone support.
///
/// When `zone_elevations` and `zone_fractions` are provided with more than one zone,
/// snow and soil routines are computed independently per zone. Recharge is aggregated
/// before the response routine.
///
/// Returns (FluxesTimeseries, optional ZoneOutputs).
#[allow(clippy::too_many_arguments)]
pub fn run(
    params: &Parameters,
    precip: &[f64],
    pet: &[f64],
    temp: &[f64],
    initial_state: Option<&State>,
    n_zones: usize,
    zone_elevations: Option<&[f64]>,
    zone_fractions: Option<&[f64]>,
    input_elevation: Option<f64>,
    temp_gradient: Option<f64>,
    precip_gradient: Option<f64>,
) -> (FluxesTimeseries, Option<ZoneOutputs>) {
    let n_timesteps = precip.len();
    assert_eq!(precip.len(), pet.len(), "precip and pet must have the same length");
    assert_eq!(precip.len(), temp.len(), "precip and temp must have the same length");

    // Defaults
    let default_elevations: Vec<f64>;
    let zone_elevs = match zone_elevations {
        Some(ze) => ze,
        None => {
            default_elevations = vec![0.0; n_zones];
            &default_elevations
        }
    };

    let default_fractions: Vec<f64>;
    let zone_fracs = match zone_fractions {
        Some(zf) => zf,
        None => {
            default_fractions = vec![1.0 / n_zones as f64; n_zones];
            &default_fractions
        }
    };

    let t_grad = temp_gradient.unwrap_or(elevation::GRAD_T_DEFAULT);
    let p_grad = precip_gradient.unwrap_or(elevation::GRAD_P_DEFAULT);
    let skip_extrapolation = input_elevation.is_none();

    // Initialize state
    let default_state = State::initialize(params, n_zones);
    let mut state = match initial_state {
        Some(s) => s.clone(),
        None => default_state,
    };

    // Compute UH weights once
    let uh_weights = routing::compute_triangular_weights(params.maxbas);

    // Allocate outputs
    let mut outputs = FluxesTimeseries::with_len(n_timesteps);
    let multi_zone = n_zones > 1;
    let mut zone_outputs = if multi_zone {
        let mut zo = ZoneOutputs::with_capacity(n_timesteps, n_zones);
        zo.zone_elevations = zone_elevs.to_vec();
        zo.zone_fractions = zone_fracs.to_vec();
        Some(zo)
    } else {
        None
    };

    // Main simulation loop
    for t in 0..n_timesteps {
        let p = precip[t];
        let pe = pet[t];
        let te = temp[t];

        // Single-zone fast path: skip accumulation overhead
        if n_zones == 1 {
            let (zone_temp, zone_precip) = if skip_extrapolation {
                (te, p)
            } else {
                let ie = input_elevation.unwrap();
                (
                    elevation::extrapolate_temp(te, ie, zone_elevs[0], t_grad),
                    elevation::extrapolate_precip_with_cap(p, ie, zone_elevs[0], p_grad, elevation::ELEV_CAP_PRECIP),
                )
            };

            let sp = state.zone_states[0][0];
            let lw = state.zone_states[0][1];
            let sm = state.zone_states[0][2];

            let (p_rain, p_snow) =
                processes::partition_precipitation(zone_precip, zone_temp, params.tt, params.sfcf);
            let melt = processes::compute_melt(zone_temp, params.tt, params.cfmax, sp);
            let refreeze =
                processes::compute_refreezing(zone_temp, params.tt, params.cfmax, params.cfr, lw);
            let (new_sp, new_lw, snow_outflow) =
                processes::update_snow_pack(sp, lw, p_snow, melt, refreeze, params.cwh);

            let snow_input = p_rain + snow_outflow;

            let recharge =
                processes::compute_recharge(snow_input, sm, params.fc, params.beta);
            let et_act = processes::compute_actual_et(pe, sm, params.fc, params.lp);
            let new_sm =
                processes::update_soil_moisture(sm, snow_input, recharge, et_act, params.fc);

            state.zone_states[0] = [new_sp, new_lw, new_sm];

            // Response routine
            let (q0, q1) =
                routing::upper_zone_outflows(state.upper_zone, params.k0, params.k1, params.uzl);
            let perc = routing::compute_percolation(state.upper_zone, params.perc);
            let new_suz = routing::update_upper_zone(state.upper_zone, recharge, q0, q1, perc);

            let q2 = routing::lower_zone_outflow(state.lower_zone, params.k2);
            let new_slz = routing::update_lower_zone(state.lower_zone, perc, q2);

            let qgw = q0 + q1 + q2;

            let (new_buffer, qsim) =
                routing::convolve_triangular(qgw, &state.routing_buffer, &uh_weights);

            state.upper_zone = new_suz;
            state.lower_zone = new_slz;
            state.routing_buffer = new_buffer;

            unsafe {
                outputs.write_unchecked(t, &Fluxes {
                    precip: p,
                    temp: te,
                    pet: pe,
                    precip_rain: p_rain,
                    precip_snow: p_snow,
                    snow_pack: new_sp,
                    snow_melt: melt,
                    liquid_water_in_snow: new_lw,
                    snow_input,
                    soil_moisture: new_sm,
                    recharge,
                    actual_et: et_act,
                    upper_zone: new_suz,
                    lower_zone: new_slz,
                    q0,
                    q1,
                    q2,
                    percolation: perc,
                    qgw,
                    streamflow: qsim,
                });
            }

            continue;
        }

        // Multi-zone path
        // Aggregation accumulators
        let mut agg_p_rain = 0.0;
        let mut agg_p_snow = 0.0;
        let mut agg_melt = 0.0;
        let mut agg_snow_input = 0.0;
        let mut agg_recharge = 0.0;
        let mut agg_et_act = 0.0;
        let mut agg_sm = 0.0;
        let mut agg_sp = 0.0;
        let mut agg_lw = 0.0;

        // Process each zone
        for zone_idx in 0..n_zones {
            let fraction = zone_fracs[zone_idx];

            // Extrapolate forcing
            let (zone_temp, zone_precip) = if skip_extrapolation {
                (te, p)
            } else {
                let ie = input_elevation.unwrap();
                (
                    elevation::extrapolate_temp(te, ie, zone_elevs[zone_idx], t_grad),
                    elevation::extrapolate_precip_with_cap(p, ie, zone_elevs[zone_idx], p_grad, elevation::ELEV_CAP_PRECIP),
                )
            };

            let sp = state.zone_states[zone_idx][0];
            let lw = state.zone_states[zone_idx][1];
            let sm = state.zone_states[zone_idx][2];

            // Snow routine
            let (p_rain, p_snow) =
                processes::partition_precipitation(zone_precip, zone_temp, params.tt, params.sfcf);
            let melt = processes::compute_melt(zone_temp, params.tt, params.cfmax, sp);
            let refreeze =
                processes::compute_refreezing(zone_temp, params.tt, params.cfmax, params.cfr, lw);
            let (new_sp, new_lw, snow_outflow) =
                processes::update_snow_pack(sp, lw, p_snow, melt, refreeze, params.cwh);

            let snow_input = p_rain + snow_outflow;

            // Soil routine
            let recharge =
                processes::compute_recharge(snow_input, sm, params.fc, params.beta);
            let et_act = processes::compute_actual_et(pe, sm, params.fc, params.lp);
            let new_sm =
                processes::update_soil_moisture(sm, snow_input, recharge, et_act, params.fc);

            // Update zone state
            state.zone_states[zone_idx] = [new_sp, new_lw, new_sm];

            // Store per-zone outputs
            if let Some(ref mut zo) = zone_outputs {
                zo.zone_temp[t][zone_idx] = zone_temp;
                zo.zone_precip[t][zone_idx] = zone_precip;
                zo.snow_pack[t][zone_idx] = new_sp;
                zo.liquid_water_in_snow[t][zone_idx] = new_lw;
                zo.snow_melt[t][zone_idx] = melt;
                zo.snow_input[t][zone_idx] = snow_input;
                zo.soil_moisture[t][zone_idx] = new_sm;
                zo.recharge[t][zone_idx] = recharge;
                zo.actual_et[t][zone_idx] = et_act;
            }

            // Aggregate (area-weighted)
            agg_p_rain += p_rain * fraction;
            agg_p_snow += p_snow * fraction;
            agg_melt += melt * fraction;
            agg_snow_input += snow_input * fraction;
            agg_recharge += recharge * fraction;
            agg_et_act += et_act * fraction;
            agg_sm += new_sm * fraction;
            agg_sp += new_sp * fraction;
            agg_lw += new_lw * fraction;
        }

        // Response routine (lumped)
        let (q0, q1) =
            routing::upper_zone_outflows(state.upper_zone, params.k0, params.k1, params.uzl);
        let perc = routing::compute_percolation(state.upper_zone, params.perc);
        let new_suz = routing::update_upper_zone(state.upper_zone, agg_recharge, q0, q1, perc);

        let q2 = routing::lower_zone_outflow(state.lower_zone, params.k2);
        let new_slz = routing::update_lower_zone(state.lower_zone, perc, q2);

        let qgw = q0 + q1 + q2;

        // Routing
        let (new_buffer, qsim) =
            routing::convolve_triangular(qgw, &state.routing_buffer, &uh_weights);

        state.upper_zone = new_suz;
        state.lower_zone = new_slz;
        state.routing_buffer = new_buffer;

        unsafe {
            outputs.write_unchecked(t, &Fluxes {
                precip: p,
                temp: te,
                pet: pe,
                precip_rain: agg_p_rain,
                precip_snow: agg_p_snow,
                snow_pack: agg_sp,
                snow_melt: agg_melt,
                liquid_water_in_snow: agg_lw,
                snow_input: agg_snow_input,
                soil_moisture: agg_sm,
                recharge: agg_recharge,
                actual_et: agg_et_act,
                upper_zone: new_suz,
                lower_zone: new_slz,
                q0,
                q1,
                q2,
                percolation: perc,
                qgw,
                streamflow: qsim,
            });
        }
    }

    (outputs, zone_outputs)
}

/// Forcing input for HBV-Light: precipitation, PET, and temperature.
#[derive(Debug, Clone, Copy)]
pub struct HBVForcing {
    pub precip: f64,
    pub pet: f64,
    pub temp: f64,
}

/// Precomputed context for HBV-Light: unit hydrograph weights.
pub struct HBVContext {
    pub uh_weights: Vec<f64>,
}

/// Marker type for single-zone HBV-Light trait implementation.
pub struct HBVLight;

impl HydrologicalModel for HBVLight {
    const NAME: &'static str = "HBV-Light";
    type Params = Parameters;
    type State = State;
    type Forcing = HBVForcing;
    type Fluxes = Fluxes;
    type FluxesTimeseries = super::fluxes::FluxesTimeseries;
    type Context = HBVContext;

    fn prepare(params: &Self::Params) -> Self::Context {
        let uh_weights = routing::compute_triangular_weights(params.maxbas);
        HBVContext { uh_weights }
    }

    fn initialize_state(params: &Self::Params) -> Self::State {
        State::initialize(params, 1)
    }

    fn step(
        state: &Self::State,
        params: &Self::Params,
        forcing: &Self::Forcing,
        context: &Self::Context,
    ) -> (Self::State, Self::Fluxes) {
        step(state, params, forcing.precip, forcing.pet, forcing.temp, &context.uh_weights)
    }
}

/// Run single-zone HBV-Light from separate arrays (convenience for PyO3 bindings).
pub fn run_from_slices(
    params: &Parameters,
    precip: &[f64],
    pet: &[f64],
    temp: &[f64],
    initial_state: Option<&State>,
) -> FluxesTimeseries {
    assert_eq!(precip.len(), pet.len(), "precip and pet must have the same length");
    assert_eq!(precip.len(), temp.len(), "precip and temp must have the same length");
    let forcing: Vec<HBVForcing> = precip.iter().zip(pet).zip(temp)
        .map(|((&p, &e), &t)| HBVForcing { precip: p, pet: e, temp: t })
        .collect();
    HBVLight::run(params, &forcing, initial_state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> Parameters {
        Parameters::new(
            0.0, 3.0, 1.0, 0.1, 0.05, 250.0, 0.9, 2.0, 0.4, 0.1, 0.01, 1.0, 20.0, 2.5,
        )
        .unwrap()
    }

    // -- step() tests --

    #[test]
    fn step_returns_finite_values() {
        let p = test_params();
        let s = State::initialize(&p, 1);
        let uh = routing::compute_triangular_weights(p.maxbas);
        let (new_state, fluxes) = step(&s, &p, 10.0, 3.0, 5.0, &uh);

        assert!(new_state.zone_states[0][2].is_finite());
        assert!(new_state.upper_zone.is_finite());
        assert!(fluxes.streamflow.is_finite());
    }

    #[test]
    fn step_does_not_mutate_input_state() {
        let p = test_params();
        let s = State::initialize(&p, 1);
        let original_sm = s.zone_states[0][2];
        let uh = routing::compute_triangular_weights(p.maxbas);

        let (_new_state, _fluxes) = step(&s, &p, 10.0, 3.0, 5.0, &uh);

        assert_eq!(s.zone_states[0][2], original_sm);
    }

    #[test]
    fn step_non_negative_streamflow() {
        let p = test_params();
        let s = State::initialize(&p, 1);
        let uh = routing::compute_triangular_weights(p.maxbas);
        let (_new_state, fluxes) = step(&s, &p, 0.0, 100.0, 10.0, &uh);
        assert!(fluxes.streamflow >= 0.0);
    }

    // -- run() tests --

    #[test]
    fn run_output_length_matches_input() {
        let p = test_params();
        let precip = [10.0, 5.0, 0.0, 15.0, 2.0];
        let pet = [3.0, 4.0, 5.0, 2.0, 3.5];
        let temp = [5.0, 2.0, -5.0, 3.0, 1.0];

        let (result, _) = run(&p, &precip, &pet, &temp, None, 1, None, None, None, None, None);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn run_all_outputs_finite() {
        let p = test_params();
        let precip = [10.0, 5.0, 0.0, 15.0, 2.0, 8.0];
        let pet = [3.0, 4.0, 5.0, 2.0, 3.5, 4.0];
        let temp = [5.0, 2.0, -5.0, 3.0, 1.0, 10.0];

        let (result, _) = run(&p, &precip, &pet, &temp, None, 1, None, None, None, None, None);

        for t in 0..result.len() {
            assert!(result.streamflow[t].is_finite(), "non-finite at t={t}");
            assert!(result.soil_moisture[t].is_finite());
        }
    }

    #[test]
    fn run_custom_initial_state() {
        let p = test_params();
        let precip = [10.0; 5];
        let pet = [3.0; 5];
        let temp = [5.0; 5];

        let (default_result, _) =
            run(&p, &precip, &pet, &temp, None, 1, None, None, None, None, None);

        let mut custom = State::initialize(&p, 1);
        custom.zone_states[0][2] = 200.0;
        custom.upper_zone = 50.0;

        let (custom_result, _) =
            run(&p, &precip, &pet, &temp, Some(&custom), 1, None, None, None, None, None);

        // Different initial state -> different qgw
        assert_ne!(default_result.qgw[0], custom_result.qgw[0]);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn run_panics_on_mismatched_lengths() {
        let p = test_params();
        run(&p, &[10.0, 5.0], &[3.0], &[5.0, 2.0], None, 1, None, None, None, None, None);
    }

    // -- Multi-zone tests --

    #[test]
    fn run_multi_zone_produces_zone_outputs() {
        let p = test_params();
        let precip = [10.0; 5];
        let pet = [3.0; 5];
        let temp = [5.0; 5];
        let elevs = [200.0, 600.0, 1000.0];
        let fracs = [0.3, 0.4, 0.3];

        let (result, zone_out) = run(
            &p, &precip, &pet, &temp, None, 3,
            Some(&elevs), Some(&fracs), Some(500.0), None, None,
        );

        assert_eq!(result.len(), 5);
        assert!(zone_out.is_some());
        let zo = zone_out.unwrap();
        assert_eq!(zo.zone_elevations.len(), 3);
        assert_eq!(zo.zone_temp.len(), 5);
        assert_eq!(zo.zone_temp[0].len(), 3);
    }

    #[test]
    fn multi_zone_higher_zones_colder() {
        let p = test_params();
        let precip = [10.0; 3];
        let pet = [3.0; 3];
        let temp = [5.0; 3];
        let elevs = [200.0, 800.0, 1400.0];
        let fracs = [1.0 / 3.0; 3];

        let (_, zone_out) = run(
            &p, &precip, &pet, &temp, None, 3,
            Some(&elevs), Some(&fracs), Some(500.0), None, None,
        );

        let zo = zone_out.unwrap();
        for t in 0..3 {
            assert!(
                zo.zone_temp[t][0] > zo.zone_temp[t][1],
                "zone 0 should be warmer than zone 1 at t={t}"
            );
            assert!(
                zo.zone_temp[t][1] > zo.zone_temp[t][2],
                "zone 1 should be warmer than zone 2 at t={t}"
            );
        }
    }

    #[test]
    fn single_zone_no_zone_outputs() {
        let p = test_params();
        let precip = [10.0; 3];
        let pet = [3.0; 3];
        let temp = [5.0; 3];

        let (_, zone_out) = run(&p, &precip, &pet, &temp, None, 1, None, None, None, None, None);
        assert!(zone_out.is_none());
    }

    #[test]
    fn run_non_negative_streamflow() {
        let p = test_params();
        let precip = [10.0, 0.0, 20.0, 0.0, 5.0, 0.0, 0.0, 15.0, 0.0, 0.0];
        let pet = [3.0, 4.0, 5.0, 2.0, 3.5, 4.0, 5.0, 2.0, 3.0, 4.0];
        let temp = [5.0, 2.0, -5.0, 3.0, 1.0, 10.0, -3.0, 7.0, 0.0, 15.0];

        let (result, _) = run(&p, &precip, &pet, &temp, None, 1, None, None, None, None, None);

        for t in 0..result.len() {
            assert!(result.streamflow[t] >= 0.0, "negative streamflow at t={t}");
        }
    }

    // -- Extrapolation tests --

    #[test]
    fn extrapolate_temp_correct() {
        // 500m input, 1000m zone, 0.6 C/100m gradient
        let t = elevation::extrapolate_temp(10.0, 500.0, 1000.0, 0.6);
        assert!((t - 7.0).abs() < 1e-10); // 10 - 0.6 * 500/100 = 7.0
    }

    #[test]
    fn extrapolate_precip_correct() {
        let p = elevation::extrapolate_precip_with_cap(10.0, 500.0, 500.0, 0.00041, 4000.0);
        assert!((p - 10.0).abs() < 1e-10); // same elevation, no change
    }
}
