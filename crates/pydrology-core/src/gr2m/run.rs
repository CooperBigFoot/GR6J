/// GR2M model orchestration functions.
///
/// - `step()`: Execute a single timestep -> (State, Fluxes)
/// - `run()`: Execute over a timeseries -> FluxesTimeseries
use super::fluxes::{Fluxes, FluxesTimeseries};
use super::params::Parameters;
use super::processes;
use super::state::State;
use crate::traits::HydrologicalModel;

/// Execute one timestep of the GR2M model.
///
/// Takes current state + forcing, returns new state + all fluxes.
/// This is the pure, readable version — no mutation of input state.
pub fn step(state: &State, params: &Parameters, precip: f64, pet: f64) -> (State, Fluxes) {
    // Step 1: Production store — rainfall neutralization
    let (s1, p1, ps) =
        processes::production_store_rainfall(precip, state.production_store, params.x1);

    // Step 2: Production store — evaporation
    let (s2, ae) = processes::production_store_evaporation(pet, s1, params.x1);

    // Step 3: Percolation
    let (s_final, p2) = processes::percolation(s2, params.x1);

    // Step 4: Total water to routing
    let p3 = p1 + p2;

    // Step 5: Routing store update with exchange
    let (r2, aexch) = processes::routing_store_update(state.routing_store, p3, params.x2);

    // Step 6: Streamflow
    let (r_final, q) = processes::compute_streamflow(r2);

    let new_state = State {
        production_store: s_final,
        routing_store: r_final,
    };

    let fluxes = Fluxes {
        pet,
        precip,
        production_store: s_final,
        rainfall_excess: p1,
        storage_fill: ps,
        actual_et: ae,
        percolation: p2,
        routing_input: p3,
        routing_store: r_final,
        exchange: aexch,
        streamflow: q,
    };

    (new_state, fluxes)
}

/// Run the GR2M model over a timeseries.
///
/// Takes slices of precip and pet arrays.
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

    // Initialize state
    let mut state = match initial_state {
        Some(s) => *s,
        None => State::initialize(params),
    };

    // Pre-allocate output
    let mut outputs = FluxesTimeseries::with_len(n);

    // Main simulation loop
    for t in 0..n {
        let (new_state, fluxes) = step(&state, params, precip[t], pet[t]);
        unsafe { outputs.write_unchecked(t, &fluxes); }
        state = new_state;
    }

    outputs
}

/// Simple forcing input for GR2M: precipitation and PET.
#[derive(Debug, Clone, Copy)]
pub struct GR2MForcing {
    pub precip: f64,
    pub pet: f64,
}

/// Marker type for the GR2M model's trait implementation.
pub struct GR2M;

impl HydrologicalModel for GR2M {
    const NAME: &'static str = "GR2M";
    type Params = Parameters;
    type State = State;
    type Forcing = GR2MForcing;
    type Fluxes = Fluxes;
    type FluxesTimeseries = super::fluxes::FluxesTimeseries;
    type Context = ();

    fn prepare(_params: &Self::Params) -> Self::Context {}

    fn initialize_state(params: &Self::Params) -> Self::State {
        State::initialize(params)
    }

    fn step(
        state: &Self::State,
        params: &Self::Params,
        forcing: &Self::Forcing,
        _context: &Self::Context,
    ) -> (Self::State, Self::Fluxes) {
        step(state, params, forcing.precip, forcing.pet)
    }
}

/// Run GR2M from separate precip/pet slices (convenience for PyO3 bindings).
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
    let forcing: Vec<GR2MForcing> = precip
        .iter()
        .zip(pet)
        .map(|(&p, &e)| GR2MForcing { precip: p, pet: e })
        .collect();
    GR2M::run(params, &forcing, initial_state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> Parameters {
        Parameters::new(500.0, 1.0).unwrap()
    }

    // -- step() tests --

    #[test]
    fn step_returns_finite_values() {
        let p = test_params();
        let s = State::initialize(&p);
        let (new_state, fluxes) = step(&s, &p, 80.0, 25.0);

        assert!(new_state.production_store.is_finite());
        assert!(new_state.routing_store.is_finite());
        assert!(fluxes.streamflow.is_finite());
    }

    #[test]
    fn step_does_not_mutate_input_state() {
        let p = test_params();
        let s = State::initialize(&p);
        let original_prod = s.production_store;
        let original_rout = s.routing_store;

        let (_new_state, _fluxes) = step(&s, &p, 80.0, 25.0);

        // Original state is untouched (borrowing guarantees this)
        assert_eq!(s.production_store, original_prod);
        assert_eq!(s.routing_store, original_rout);
    }

    #[test]
    fn step_non_negative_streamflow() {
        let p = test_params();
        let s = State::initialize(&p);
        let (_new_state, fluxes) = step(&s, &p, 0.0, 100.0);
        assert!(fluxes.streamflow >= 0.0);
    }

    #[test]
    fn step_zero_input() {
        let p = test_params();
        let s = State::initialize(&p);
        let (new_state, fluxes) = step(&s, &p, 0.0, 0.0);
        assert!(new_state.production_store.is_finite());
        assert!(fluxes.actual_et >= 0.0);
    }

    // -- run() tests --

    #[test]
    fn run_output_length_matches_input() {
        let p = test_params();
        let precip = [80.0, 70.0, 60.0];
        let pet = [20.0, 25.0, 30.0];

        let result = run(&p, &precip, &pet, None);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn run_all_outputs_finite() {
        let p = test_params();
        let precip = [80.0, 70.0, 60.0, 40.0, 20.0, 10.0];
        let pet = [20.0, 25.0, 40.0, 55.0, 60.0, 65.0];

        let result = run(&p, &precip, &pet, None);

        for t in 0..result.len() {
            assert!(result.streamflow[t].is_finite(), "non-finite at t={t}");
            assert!(result.production_store[t].is_finite());
            assert!(result.routing_store[t].is_finite());
        }
    }

    #[test]
    fn run_custom_initial_state() {
        let p = test_params();
        let precip = [80.0; 3];
        let pet = [25.0; 3];

        let default_result = run(&p, &precip, &pet, None);

        let custom = State {
            production_store: 400.0,
            routing_store: 200.0,
        };
        let custom_result = run(&p, &precip, &pet, Some(&custom));

        // Different initial state → different streamflow
        assert_ne!(default_result.streamflow[0], custom_result.streamflow[0]);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn run_panics_on_mismatched_lengths() {
        let p = test_params();
        run(&p, &[80.0, 70.0], &[20.0], None);
    }
}
