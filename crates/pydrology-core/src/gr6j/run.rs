/// GR6J model orchestration functions.
///
/// - `step()`: Execute a single timestep → (State, Fluxes)
/// - `run()`: Execute over a timeseries → FluxesTimeseries
use super::constants::{B, C, NH};
use super::fluxes::{Fluxes, FluxesTimeseries};
use super::params::Parameters;
use super::processes;
use super::state::State;
use super::unit_hydrographs::{compute_uh_ordinates, convolve_uh};
use crate::traits::HydrologicalModel;

/// Execute one timestep of the GR6J model.
///
/// Takes current state + forcing + precomputed UH ordinates, returns new state + all fluxes.
pub fn step(
    state: &State,
    params: &Parameters,
    precip: f64,
    pet: f64,
    uh1_ordinates: &[f64; NH],
    uh2_ordinates: &[f64; 2 * NH],
) -> (State, Fluxes) {
    // 1. Production store update
    let (prod_store_after_ps, actual_et, net_rainfall_pn, effective_rainfall_pr) =
        processes::production_store_update(precip, pet, state.production_store, params.x1);

    // Compute storage infiltration for output
    let storage_infiltration = if precip >= pet {
        net_rainfall_pn - effective_rainfall_pr
    } else {
        0.0
    };

    // 2. Percolation
    let (prod_store_after_perc, percolation_amount) =
        processes::percolation(prod_store_after_ps, params.x1);

    // Add percolation to effective rainfall
    let total_effective_rainfall = effective_rainfall_pr + percolation_amount;

    // 3. Split effective rainfall to unit hydrographs
    let uh1_input = B * total_effective_rainfall;
    let uh2_input = (1.0 - B) * total_effective_rainfall;

    // 4. Convolve through unit hydrographs
    let mut uh1_states = state.uh1_states;
    let mut uh2_states = state.uh2_states;
    let q9 = convolve_uh(&mut uh1_states, uh1_ordinates, uh1_input);
    let q1 = convolve_uh(&mut uh2_states, uh2_ordinates, uh2_input);

    // 5. Groundwater exchange
    let exchange_f =
        processes::groundwater_exchange(state.routing_store, params.x2, params.x3, params.x5);

    // 6. Update routing store — receives (1-C) * q9
    let routing_input = (1.0 - C) * q9;
    let (new_routing_store, qr, actual_exchange_routing) =
        processes::routing_store_update(state.routing_store, routing_input, exchange_f, params.x3);

    // 7. Update exponential store — receives C * q9
    let exp_input = C * q9;
    let (new_exp_store, qrexp) = processes::exponential_store_update(
        state.exponential_store,
        exp_input,
        exchange_f,
        params.x6,
    );

    // 8. Direct branch
    let (qd, actual_exchange_direct) = processes::direct_branch(q1, exchange_f);

    // 9. Total streamflow
    let streamflow = (qr + qrexp + qd).max(0.0);

    // Total actual exchange
    let actual_exchange_total = actual_exchange_routing + actual_exchange_direct;

    let new_state = State {
        production_store: prod_store_after_perc,
        routing_store: new_routing_store,
        exponential_store: new_exp_store,
        uh1_states,
        uh2_states,
    };

    let fluxes = Fluxes {
        pet,
        precip,
        production_store: prod_store_after_perc,
        net_rainfall: net_rainfall_pn,
        storage_infiltration,
        actual_et,
        percolation: percolation_amount,
        effective_rainfall: total_effective_rainfall,
        q9,
        q1,
        routing_store: new_routing_store,
        exchange: exchange_f,
        actual_exchange_routing,
        actual_exchange_direct,
        actual_exchange_total,
        qr,
        qrexp,
        exponential_store: new_exp_store,
        qd,
        streamflow,
    };

    (new_state, fluxes)
}

/// Run the GR6J model over a timeseries.
///
/// Takes slices of precip and pet arrays. Computes UH ordinates once from params.x4.
/// If no initial state is provided, uses State::initialize(params).
pub fn run(
    params: &Parameters,
    precip: &[f64],
    pet: &[f64],
    initial_state: Option<&State>,
) -> FluxesTimeseries {
    assert_eq!(
        precip.len(),
        pet.len(),
        "precip and pet must have the same length"
    );

    let n = precip.len();

    // Compute UH ordinates once
    let (uh1_ordinates, uh2_ordinates) = compute_uh_ordinates(params.x4);

    // Initialize state
    let mut state = match initial_state {
        Some(s) => s.clone(),
        None => State::initialize(params),
    };

    // Pre-allocate output
    let mut outputs = FluxesTimeseries::with_capacity(n);

    // Main simulation loop
    for t in 0..n {
        let (new_state, fluxes) =
            step(&state, params, precip[t], pet[t], &uh1_ordinates, &uh2_ordinates);
        outputs.push(&fluxes);
        state = new_state;
    }

    outputs
}

/// Forcing input for GR6J: precipitation and PET.
#[derive(Debug, Clone, Copy)]
pub struct GR6JForcing {
    pub precip: f64,
    pub pet: f64,
}

/// Precomputed context for GR6J: unit hydrograph ordinates.
pub struct GR6JContext {
    pub uh1: [f64; NH],
    pub uh2: [f64; 2 * NH],
}

/// Marker type for the GR6J model's trait implementation.
pub struct GR6J;

impl HydrologicalModel for GR6J {
    type Params = Parameters;
    type State = State;
    type Forcing = GR6JForcing;
    type Fluxes = Fluxes;
    type FluxesTimeseries = super::fluxes::FluxesTimeseries;
    type Context = GR6JContext;

    fn prepare(params: &Self::Params) -> Self::Context {
        let (uh1, uh2) = compute_uh_ordinates(params.x4);
        GR6JContext { uh1, uh2 }
    }

    fn initialize_state(params: &Self::Params) -> Self::State {
        State::initialize(params)
    }

    fn step(
        state: &Self::State,
        params: &Self::Params,
        forcing: &Self::Forcing,
        context: &Self::Context,
    ) -> (Self::State, Self::Fluxes) {
        step(state, params, forcing.precip, forcing.pet, &context.uh1, &context.uh2)
    }
}

/// Run GR6J from separate precip/pet slices (convenience for PyO3 bindings).
pub fn run_from_slices(
    params: &Parameters,
    precip: &[f64],
    pet: &[f64],
    initial_state: Option<&State>,
) -> FluxesTimeseries {
    assert_eq!(
        precip.len(),
        pet.len(),
        "precip and pet must have the same length"
    );
    let forcing: Vec<GR6JForcing> = precip
        .iter()
        .zip(pet)
        .map(|(&p, &e)| GR6JForcing { precip: p, pet: e })
        .collect();
    GR6J::run(params, &forcing, initial_state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> Parameters {
        Parameters::new(350.0, 0.0, 90.0, 1.7, 0.0, 5.0).unwrap()
    }

    // -- step() tests --

    #[test]
    fn step_returns_finite_values() {
        let p = test_params();
        let s = State::initialize(&p);
        let (uh1, uh2) = compute_uh_ordinates(p.x4);
        let (new_state, fluxes) = step(&s, &p, 10.0, 3.0, &uh1, &uh2);

        assert!(new_state.production_store.is_finite());
        assert!(new_state.routing_store.is_finite());
        assert!(new_state.exponential_store.is_finite());
        assert!(fluxes.streamflow.is_finite());
    }

    #[test]
    fn step_does_not_mutate_input_state() {
        let p = test_params();
        let s = State::initialize(&p);
        let original_prod = s.production_store;
        let original_rout = s.routing_store;
        let original_exp = s.exponential_store;
        let (uh1, uh2) = compute_uh_ordinates(p.x4);

        let (_new_state, _fluxes) = step(&s, &p, 10.0, 3.0, &uh1, &uh2);

        assert_eq!(s.production_store, original_prod);
        assert_eq!(s.routing_store, original_rout);
        assert_eq!(s.exponential_store, original_exp);
    }

    #[test]
    fn step_non_negative_streamflow() {
        let p = test_params();
        let s = State::initialize(&p);
        let (uh1, uh2) = compute_uh_ordinates(p.x4);
        let (_new_state, fluxes) = step(&s, &p, 0.0, 100.0, &uh1, &uh2);
        assert!(fluxes.streamflow >= 0.0);
    }

    #[test]
    fn step_zero_input() {
        let p = test_params();
        let s = State::initialize(&p);
        let (uh1, uh2) = compute_uh_ordinates(p.x4);
        let (new_state, fluxes) = step(&s, &p, 0.0, 0.0, &uh1, &uh2);
        assert!(new_state.production_store.is_finite());
        assert!(fluxes.actual_et >= 0.0);
    }

    // -- run() tests --

    #[test]
    fn run_output_length_matches_input() {
        let p = test_params();
        let precip = [10.0, 5.0, 0.0, 15.0, 2.0];
        let pet = [3.0, 4.0, 5.0, 2.0, 3.5];

        let result = run(&p, &precip, &pet, None);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn run_all_outputs_finite() {
        let p = test_params();
        let precip = [10.0, 5.0, 0.0, 15.0, 2.0, 8.0];
        let pet = [3.0, 4.0, 5.0, 2.0, 3.5, 4.0];

        let result = run(&p, &precip, &pet, None);

        for t in 0..result.len() {
            assert!(result.streamflow[t].is_finite(), "non-finite at t={t}");
            assert!(result.production_store[t].is_finite());
            assert!(result.routing_store[t].is_finite());
            assert!(result.exponential_store[t].is_finite());
        }
    }

    #[test]
    fn run_custom_initial_state() {
        let p = test_params();
        let precip = [10.0; 5];
        let pet = [3.0; 5];

        let default_result = run(&p, &precip, &pet, None);

        let custom = State {
            production_store: 200.0,
            routing_store: 60.0,
            exponential_store: 10.0,
            uh1_states: [0.0; NH],
            uh2_states: [0.0; 2 * NH],
        };
        let custom_result = run(&p, &precip, &pet, Some(&custom));

        // Different initial state -> different streamflow
        assert_ne!(default_result.streamflow[0], custom_result.streamflow[0]);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn run_panics_on_mismatched_lengths() {
        let p = test_params();
        run(&p, &[10.0, 5.0], &[3.0], None);
    }

    #[test]
    fn run_streamflow_non_negative() {
        let p = test_params();
        let precip = [10.0, 0.0, 5.0, 0.0, 20.0, 0.0, 0.0, 0.0, 15.0, 3.0];
        let pet = [3.0, 4.0, 5.0, 6.0, 2.0, 5.0, 4.0, 3.0, 2.0, 4.0];

        let result = run(&p, &precip, &pet, None);

        for t in 0..result.len() {
            assert!(
                result.streamflow[t] >= 0.0,
                "negative streamflow at t={t}: {}",
                result.streamflow[t]
            );
        }
    }

    #[test]
    fn run_production_store_within_bounds() {
        let p = test_params();
        let precip = [10.0, 0.0, 5.0, 0.0, 20.0, 0.0, 0.0, 0.0, 15.0, 3.0];
        let pet = [3.0, 4.0, 5.0, 6.0, 2.0, 5.0, 4.0, 3.0, 2.0, 4.0];

        let result = run(&p, &precip, &pet, None);

        for t in 0..result.len() {
            assert!(
                result.production_store[t] >= 0.0,
                "negative production store at t={t}"
            );
            assert!(
                result.production_store[t] <= p.x1,
                "production store exceeds x1 at t={t}"
            );
        }
    }

    #[test]
    fn actual_exchange_total_equals_sum_of_components() {
        let p = test_params();
        let s = State::initialize(&p);
        let (uh1, uh2) = compute_uh_ordinates(p.x4);
        let (_new_state, fluxes) = step(&s, &p, 10.0, 3.0, &uh1, &uh2);

        let expected = fluxes.actual_exchange_routing + fluxes.actual_exchange_direct;
        assert!(
            (fluxes.actual_exchange_total - expected).abs() < 1e-10,
            "actual_exchange_total ({}) != routing ({}) + direct ({})",
            fluxes.actual_exchange_total,
            fluxes.actual_exchange_routing,
            fluxes.actual_exchange_direct,
        );
    }

    #[test]
    fn actual_exchange_total_with_nonzero_exchange() {
        // Use x2 != 0 to get non-zero exchange
        let p = Parameters::new(350.0, 1.5, 90.0, 1.7, 0.5, 5.0).unwrap();
        let s = State::initialize(&p);
        let (uh1, uh2) = compute_uh_ordinates(p.x4);

        let precip = [10.0, 5.0, 20.0, 0.0, 15.0, 8.0];
        let pet = [3.0, 4.0, 2.0, 5.0, 3.0, 4.0];

        let mut state = s;
        for t in 0..precip.len() {
            let (new_state, fluxes) = step(&state, &p, precip[t], pet[t], &uh1, &uh2);

            let expected = fluxes.actual_exchange_routing + fluxes.actual_exchange_direct;
            assert!(
                (fluxes.actual_exchange_total - expected).abs() < 1e-10,
                "Invariant violated at t={}: total ({}) != routing ({}) + direct ({})",
                t,
                fluxes.actual_exchange_total,
                fluxes.actual_exchange_routing,
                fluxes.actual_exchange_direct,
            );

            state = new_state;
        }
    }
}
