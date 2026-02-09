use std::fmt::Debug;

/// Trait for model parameter types.
///
/// Provides uniform array serialization, validation, and metadata.
pub trait ModelParams: Clone + Copy + Debug {
    /// Number of parameters.
    const N_PARAMS: usize;
    /// Parameter names in canonical order.
    const PARAM_NAMES: &'static [&'static str];
    /// Parameter bounds as (min, max) tuples.
    const PARAM_BOUNDS: &'static [(f64, f64)];

    /// Construct from a slice, validating length and bounds.
    fn from_array(arr: &[f64]) -> Result<Self, String>;
    /// Serialize to a Vec.
    fn to_array(&self) -> Vec<f64>;
}

/// Trait for model state types.
///
/// Provides uniform serialization/deserialization.
pub trait ModelState: Clone + Debug {
    /// Serialize state to a flat Vec.
    fn to_vec(&self) -> Vec<f64>;
    /// Deserialize state from a flat slice. Never panics.
    fn from_slice(arr: &[f64]) -> Result<Self, String>;
    /// Length of the serialized array.
    fn array_len(&self) -> usize;
}

/// Core trait for lumped hydrological models.
///
/// Defines the interface all single-zone models implement: prepare context,
/// initialize state, step, and run over a timeseries.
pub trait HydrologicalModel {
    const NAME: &'static str;
    type Params;
    type State: Clone;
    type Forcing: Copy;
    type Fluxes;
    type FluxesTimeseries: FluxesTimeseriesOps<Self::Fluxes>;
    /// Precomputed context derived from params, constant for a given run.
    type Context;

    /// Precompute any run-constant data from parameters (e.g., UH ordinates).
    fn prepare(params: &Self::Params) -> Self::Context;

    /// Create a default initial state from parameters.
    fn initialize_state(params: &Self::Params) -> Self::State;

    /// Execute one timestep: given state, params, forcing, and context,
    /// return the new state and fluxes.
    fn step(
        state: &Self::State,
        params: &Self::Params,
        forcing: &Self::Forcing,
        context: &Self::Context,
    ) -> (Self::State, Self::Fluxes);

    /// Run the model over a forcing timeseries.
    ///
    /// Default implementation: prepare context, initialize/use provided state,
    /// loop over forcing calling step.
    fn run(
        params: &Self::Params,
        forcing: &[Self::Forcing],
        initial_state: Option<&Self::State>,
    ) -> Self::FluxesTimeseries {
        let context = Self::prepare(params);
        let mut state = match initial_state {
            Some(s) => s.clone(),
            None => Self::initialize_state(params),
        };

        let n = forcing.len();
        let mut outputs = Self::FluxesTimeseries::with_len(n);

        for (t, f) in forcing.iter().enumerate() {
            let (new_state, fluxes) = Self::step(&state, params, f, &context);
            unsafe { outputs.write_unchecked(t, &fluxes); }
            state = new_state;
        }

        outputs
    }
}

/// Operations required on the timeseries collection type.
pub trait FluxesTimeseriesOps<F> {
    fn with_capacity(n: usize) -> Self;
    fn push(&mut self, f: &F);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn with_len(n: usize) -> Self;
    unsafe fn write_unchecked(&mut self, t: usize, f: &F);
}
