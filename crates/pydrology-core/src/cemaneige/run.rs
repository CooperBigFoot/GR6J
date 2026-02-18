/// CemaNeige model orchestration functions.
///
/// - `step()`: Execute a single timestep for one layer
/// - `multi_layer_step()`: Execute with elevation extrapolation
use super::params::Parameters;
use super::processes;
use super::state::{LayerState, State};
use crate::elevation;

/// Fluxes output from a single CemaNeige timestep.
#[derive(Debug, Clone)]
pub struct Fluxes {
    pub pliq: f64,
    pub psol: f64,
    pub snow_pack: f64,
    pub thermal_state: f64,
    pub gratio: f64,
    pub pot_melt: f64,
    pub melt: f64,
    pub pliq_and_melt: f64,
    pub temp: f64,
    pub gthreshold: f64,
    pub glocalmax: f64,
}

/// Execute one timestep of CemaNeige for a single layer.
///
/// Returns (new_layer_state, fluxes).
pub fn step(
    layer_state: &LayerState,
    params: &Parameters,
    precip: f64,
    temp: f64,
) -> (LayerState, Fluxes) {
    let g = layer_state[0];
    let etg = layer_state[1];
    let gthreshold = layer_state[2];
    let glocalmax = layer_state[3];

    // 1. Solid fraction
    let solid_fraction = processes::compute_solid_fraction(temp);

    // 2. Partition precipitation
    let (pliq, psol) = processes::partition_precipitation(precip, solid_fraction);

    // 3. Accumulate snow
    let g_after_accum = g + psol;

    // 4. Update thermal state
    let new_etg = processes::update_thermal_state(etg, temp, params.ctg);

    // 5. Potential melt
    let pot_melt = processes::compute_potential_melt(new_etg, temp, params.kf, g_after_accum);

    // 6. Gratio before melt (for melt calculation)
    let gratio_for_melt = processes::compute_gratio(g_after_accum, gthreshold);

    // 7. Actual melt
    let melt = processes::compute_actual_melt(pot_melt, gratio_for_melt);

    // 8. Update snow pack
    let new_g = g_after_accum - melt;

    // 9. Output gratio (after melt)
    let gratio_output = processes::compute_gratio(new_g, gthreshold);

    // 10. Total liquid output
    let pliq_and_melt = pliq + melt;

    let new_state = [new_g, new_etg, gthreshold, glocalmax];

    let fluxes = Fluxes {
        pliq,
        psol,
        snow_pack: new_g,
        thermal_state: new_etg,
        gratio: gratio_output,
        pot_melt,
        melt,
        pliq_and_melt,
        temp,
        gthreshold,
        glocalmax,
    };

    (new_state, fluxes)
}

/// Execute one timestep of multi-layer CemaNeige with elevation extrapolation.
///
/// Returns (new_state, aggregated_fluxes, per_layer_fluxes).
#[allow(clippy::too_many_arguments)]
pub fn multi_layer_step(
    state: &State,
    params: &Parameters,
    precip: f64,
    temp: f64,
    layer_elevations: &[f64],
    layer_fractions: &[f64],
    input_elevation: f64,
    temp_gradient: Option<f64>,
    precip_gradient: Option<f64>,
    use_linear_precip_gradient: Option<bool>,
) -> (State, Fluxes, Vec<Fluxes>) {
    let n_layers = state.n_layers();
    let t_grad = temp_gradient.unwrap_or(elevation::GRAD_T_DEFAULT);
    let p_grad = precip_gradient.unwrap_or(elevation::GRAD_P_DEFAULT);
    let use_linear = use_linear_precip_gradient.unwrap_or(false);

    let mut new_layer_states = Vec::with_capacity(n_layers);
    let mut per_layer_fluxes = Vec::with_capacity(n_layers);

    // Aggregation accumulators
    let mut agg_pliq = 0.0;
    let mut agg_psol = 0.0;
    let mut agg_snow_pack = 0.0;
    let mut agg_thermal = 0.0;
    let mut agg_gratio = 0.0;
    let mut agg_pot_melt = 0.0;
    let mut agg_melt = 0.0;
    let mut agg_pliq_and_melt = 0.0;
    let mut agg_temp = 0.0;
    let mut agg_gthreshold = 0.0;
    let mut agg_glocalmax = 0.0;

    for i in 0..n_layers {
        let fraction = layer_fractions[i];

        // Extrapolate forcing
        let layer_temp = elevation::extrapolate_temp(temp, input_elevation, layer_elevations[i], t_grad);
        let layer_precip = if use_linear {
            elevation::extrapolate_precip_linear(precip, input_elevation, layer_elevations[i], p_grad)
        } else {
            elevation::extrapolate_precip(precip, input_elevation, layer_elevations[i], p_grad)
        };

        // Run single-layer step
        let (new_ls, fluxes) = step(&state.layer_states[i], params, layer_precip, layer_temp);

        new_layer_states.push(new_ls);

        // Aggregate
        agg_pliq += fluxes.pliq * fraction;
        agg_psol += fluxes.psol * fraction;
        agg_snow_pack += fluxes.snow_pack * fraction;
        agg_thermal += fluxes.thermal_state * fraction;
        agg_gratio += fluxes.gratio * fraction;
        agg_pot_melt += fluxes.pot_melt * fraction;
        agg_melt += fluxes.melt * fraction;
        agg_pliq_and_melt += fluxes.pliq_and_melt * fraction;
        agg_temp += fluxes.temp * fraction;
        agg_gthreshold += fluxes.gthreshold * fraction;
        agg_glocalmax += fluxes.glocalmax * fraction;

        per_layer_fluxes.push(fluxes);
    }

    let new_state = State {
        layer_states: new_layer_states,
    };

    let aggregated = Fluxes {
        pliq: agg_pliq,
        psol: agg_psol,
        snow_pack: agg_snow_pack,
        thermal_state: agg_thermal,
        gratio: agg_gratio,
        pot_melt: agg_pot_melt,
        melt: agg_melt,
        pliq_and_melt: agg_pliq_and_melt,
        temp: agg_temp,
        gthreshold: agg_gthreshold,
        glocalmax: agg_glocalmax,
    };

    (new_state, aggregated, per_layer_fluxes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> Parameters {
        Parameters::new(0.97, 2.5)
    }

    // -- step() tests --

    #[test]
    fn step_returns_finite_values() {
        let state = [0.0, 0.0, 135.0, 135.0];
        let params = test_params();
        let (new_state, fluxes) = step(&state, &params, 10.0, 2.0);

        assert!(new_state[0].is_finite());
        assert!(fluxes.pliq_and_melt.is_finite());
    }

    #[test]
    fn step_snow_accumulation_cold() {
        let state = [0.0, 0.0, 135.0, 135.0];
        let params = test_params();
        let (new_state, fluxes) = step(&state, &params, 10.0, -5.0);

        // All snow, no rain
        assert_eq!(fluxes.pliq, 0.0);
        assert_eq!(fluxes.psol, 10.0);
        assert!(new_state[0] > 0.0); // snow accumulated
    }

    #[test]
    fn step_all_rain_warm() {
        let state = [0.0, 0.0, 135.0, 135.0];
        let params = test_params();
        let (_, fluxes) = step(&state, &params, 10.0, 10.0);

        // All rain, no snow
        assert_eq!(fluxes.psol, 0.0);
        assert!((fluxes.pliq - 10.0).abs() < 1e-10);
        assert!((fluxes.pliq_and_melt - 10.0).abs() < 1e-10);
    }

    #[test]
    fn step_melt_with_existing_snow() {
        // Snow exists, thermal state at 0, warm temp -> melt should occur
        let state = [100.0, 0.0, 135.0, 135.0];
        let params = test_params();
        let (_, fluxes) = step(&state, &params, 0.0, 5.0);

        assert!(fluxes.melt > 0.0);
        assert!(fluxes.pliq_and_melt > 0.0);
    }

    #[test]
    fn step_no_melt_cold_snow() {
        // Snow exists but thermal state is cold -> no melt
        let state = [100.0, -5.0, 135.0, 135.0];
        let params = test_params();
        let (_, fluxes) = step(&state, &params, 0.0, 5.0);

        assert_eq!(fluxes.melt, 0.0);
    }

    #[test]
    fn step_conserves_water() {
        let state = [50.0, 0.0, 135.0, 135.0];
        let params = test_params();
        let precip = 10.0;
        let (new_state, fluxes) = step(&state, &params, precip, 2.0);

        // snow_pack_new = g_old + psol - melt
        let expected_snow = state[0] + fluxes.psol - fluxes.melt;
        assert!((new_state[0] - expected_snow).abs() < 1e-10);
    }

    // -- multi_layer_step() tests --

    #[test]
    fn multi_layer_higher_layers_colder() {
        let s = State::initialize(3, 150.0);
        let params = test_params();
        let elevs = [200.0, 600.0, 1000.0];
        let fracs = [1.0 / 3.0; 3];

        let (_, _, per_layer) = multi_layer_step(
            &s, &params, 10.0, 5.0, &elevs, &fracs, 500.0, None, None, None,
        );

        // Higher elevation -> colder -> more snow
        assert!(per_layer[0].psol < per_layer[2].psol);
    }

    #[test]
    fn multi_layer_aggregation_works() {
        let s = State::initialize(2, 100.0);
        let params = test_params();
        let elevs = [300.0, 700.0];
        let fracs = [0.5, 0.5];

        let (_, agg, per_layer) = multi_layer_step(
            &s, &params, 10.0, 2.0, &elevs, &fracs, 500.0, None, None, None,
        );

        // Aggregated pliq_and_melt should be average of two layers
        let expected = 0.5 * per_layer[0].pliq_and_melt + 0.5 * per_layer[1].pliq_and_melt;
        assert!((agg.pliq_and_melt - expected).abs() < 1e-10);
    }

    #[test]
    fn multi_layer_linear_gradient_differs() {
        let s = State::initialize(2, 100.0);
        let params = test_params();
        let elevs = [300.0, 1300.0];
        let fracs = [0.5, 0.5];

        let (_, agg_exp, _) = multi_layer_step(
            &s, &params, 10.0, 2.0, &elevs, &fracs, 500.0, None, None, Some(false),
        );
        let (_, agg_lin, _) = multi_layer_step(
            &s, &params, 10.0, 2.0, &elevs, &fracs, 500.0, None, None, Some(true),
        );

        // Different gradient types should produce different aggregated precip
        assert!((agg_exp.pliq_and_melt - agg_lin.pliq_and_melt).abs() > 1e-6);
    }

    #[test]
    fn multi_layer_linear_false_matches_default() {
        let s = State::initialize(2, 100.0);
        let params = test_params();
        let elevs = [300.0, 700.0];
        let fracs = [0.5, 0.5];

        let (_, agg_none, _) = multi_layer_step(
            &s, &params, 10.0, 2.0, &elevs, &fracs, 500.0, None, None, None,
        );
        let (_, agg_false, _) = multi_layer_step(
            &s, &params, 10.0, 2.0, &elevs, &fracs, 500.0, None, None, Some(false),
        );

        assert!((agg_none.pliq_and_melt - agg_false.pliq_and_melt).abs() < 1e-10);
    }
}
