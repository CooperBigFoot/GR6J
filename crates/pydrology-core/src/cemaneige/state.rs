/// CemaNeige model state variables.
///
/// Per-layer state for snow pack tracking.
use super::constants::{GTHRESHOLD_FACTOR, LAYER_STATE_SIZE};
use crate::traits::ModelState;

/// Single-layer snow state: [snow_pack, thermal_state, gthreshold, glocalmax].
pub type LayerState = [f64; LAYER_STATE_SIZE];

/// Multi-layer CemaNeige state.
#[derive(Debug, Clone)]
pub struct State {
    /// Per-layer states. Each element is [g, etg, gthreshold, glocalmax].
    pub layer_states: Vec<LayerState>,
}

impl State {
    /// Create initial state for given number of layers.
    pub fn initialize(n_layers: usize, mean_annual_solid_precip: f64) -> Self {
        let gthreshold = GTHRESHOLD_FACTOR * mean_annual_solid_precip;
        let layer_states = vec![[0.0, 0.0, gthreshold, gthreshold]; n_layers];
        Self { layer_states }
    }

    /// Create initial state with per-band gthresholds from elevation data.
    ///
    /// Each band's gthreshold is scaled by the same precipitation gradient
    /// applied to forcing, ensuring Gratio evolves proportionally.
    pub fn initialize_per_band(
        n_layers: usize,
        mean_annual_solid_precip: f64,
        layer_elevations: &[f64],
        input_elevation: f64,
        precip_gradient: f64,
        use_linear: bool,
    ) -> Self {
        let layer_states = (0..n_layers)
            .map(|i| {
                let gt = crate::elevation::compute_band_gthreshold(
                    GTHRESHOLD_FACTOR,
                    mean_annual_solid_precip,
                    input_elevation,
                    layer_elevations[i],
                    precip_gradient,
                    use_linear,
                );
                [0.0, 0.0, gt, gt]
            })
            .collect();
        Self { layer_states }
    }

    pub fn n_layers(&self) -> usize {
        self.layer_states.len()
    }

    /// Serialize to a flat array. Layout: layer_states flattened.
    pub fn to_array(&self) -> Vec<f64> {
        let n = self.n_layers();
        let mut arr = Vec::with_capacity(n * LAYER_STATE_SIZE);
        for ls in &self.layer_states {
            arr.extend_from_slice(ls);
        }
        arr
    }

    /// Deserialize from a flat array.
    pub fn from_array(arr: &[f64], n_layers: usize) -> Self {
        let expected = n_layers * LAYER_STATE_SIZE;
        assert_eq!(
            arr.len(), expected,
            "state array length {} does not match expected {} for {} layers",
            arr.len(), expected, n_layers
        );

        let mut layer_states = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let base = i * LAYER_STATE_SIZE;
            layer_states.push([arr[base], arr[base + 1], arr[base + 2], arr[base + 3]]);
        }
        Self { layer_states }
    }
}

impl ModelState for State {
    fn to_vec(&self) -> Vec<f64> {
        self.to_array()
    }

    fn from_slice(arr: &[f64]) -> Result<Self, String> {
        if arr.is_empty() {
            return Err("state array is empty".to_string());
        }
        if arr.len() % LAYER_STATE_SIZE != 0 {
            return Err(format!(
                "state array length {} is not a multiple of layer state size {}",
                arr.len(),
                LAYER_STATE_SIZE
            ));
        }
        let n_layers = arr.len() / LAYER_STATE_SIZE;

        let mut layer_states = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let base = i * LAYER_STATE_SIZE;
            layer_states.push([arr[base], arr[base + 1], arr[base + 2], arr[base + 3]]);
        }
        Ok(Self { layer_states })
    }

    fn array_len(&self) -> usize {
        self.n_layers() * LAYER_STATE_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialize_single_layer() {
        let s = State::initialize(1, 150.0);
        assert_eq!(s.n_layers(), 1);
        assert_eq!(s.layer_states[0][0], 0.0); // g
        assert_eq!(s.layer_states[0][1], 0.0); // etg
        assert_eq!(s.layer_states[0][2], 135.0); // 0.9 * 150
        assert_eq!(s.layer_states[0][3], 135.0); // glocalmax
    }

    #[test]
    fn initialize_multi_layer() {
        let s = State::initialize(3, 200.0);
        assert_eq!(s.n_layers(), 3);
        for ls in &s.layer_states {
            assert_eq!(ls[2], 180.0); // 0.9 * 200
        }
    }

    #[test]
    fn to_array_from_array_roundtrip() {
        let mut s = State::initialize(2, 100.0);
        s.layer_states[0][0] = 50.0; // some snow
        s.layer_states[1][1] = -2.0; // cold

        let arr = s.to_array();
        let s2 = State::from_array(&arr, 2);

        assert_eq!(s2.n_layers(), 2);
        assert_eq!(s2.layer_states[0][0], 50.0);
        assert_eq!(s2.layer_states[1][1], -2.0);
        assert_eq!(s2.layer_states[0][2], 90.0);
    }

    #[test]
    fn model_state_roundtrip() {
        use crate::traits::ModelState;

        let mut s = State::initialize(2, 100.0);
        s.layer_states[0][0] = 50.0;
        let v = s.to_vec();
        let s2 = State::from_slice(&v).unwrap();
        assert_eq!(s2.n_layers(), 2);
        assert_eq!(s2.layer_states[0][0], 50.0);
    }

    #[test]
    fn model_state_wrong_length() {
        use crate::traits::ModelState;

        assert!(State::from_slice(&[]).is_err());
        assert!(State::from_slice(&[1.0, 2.0, 3.0]).is_err()); // not multiple of 4
    }

    #[test]
    fn per_band_higher_elevation_higher_gthreshold() {
        let elevs = [200.0, 600.0, 1000.0];
        let s = State::initialize_per_band(3, 200.0, &elevs, 500.0, 0.00041, false);
        assert!(s.layer_states[0][2] < s.layer_states[2][2],
            "higher elevation band should have higher gthreshold");
    }

    #[test]
    fn per_band_matches_uniform_at_same_elevation() {
        let elevs = [500.0, 500.0, 500.0];
        let s_per_band = State::initialize_per_band(3, 200.0, &elevs, 500.0, 0.00041, false);
        let s_uniform = State::initialize(3, 200.0);
        for i in 0..3 {
            assert!((s_per_band.layer_states[i][2] - s_uniform.layer_states[i][2]).abs() < 1e-10);
        }
    }

    #[test]
    fn per_band_respects_cap() {
        let elevs = [3000.0, 5000.0]; // 5000 capped at 4000
        let s = State::initialize_per_band(2, 200.0, &elevs, 3000.0, 0.00041, false);
        let gt_3000 = s.layer_states[0][2];
        let gt_5000 = s.layer_states[1][2]; // capped at 4000
        assert!(gt_5000 > gt_3000);
        // With 4000-cap, the elevation diff for band 1 is only 1000m
        let expected = 0.9 * 200.0 * (0.00041 * 1000.0_f64).exp();
        assert!((gt_5000 - expected).abs() < 1e-6);
    }

    #[test]
    fn per_band_linear_vs_exponential() {
        let elevs = [200.0, 1500.0];
        let s_exp = State::initialize_per_band(2, 200.0, &elevs, 500.0, 0.0004, false);
        let s_lin = State::initialize_per_band(2, 200.0, &elevs, 500.0, 0.0004, true);
        // Exponential and linear give different results for different elevations
        assert!((s_exp.layer_states[1][2] - s_lin.layer_states[1][2]).abs() > 0.001);
        // Both should be > base for the higher band
        let base = 0.9 * 200.0;
        assert!(s_exp.layer_states[1][2] > base);
        assert!(s_lin.layer_states[1][2] > base);
    }
}
