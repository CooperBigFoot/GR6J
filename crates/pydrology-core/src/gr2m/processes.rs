/// GR2M core process functions.
///
/// Pure functions implementing the mathematical equations for each GR2M step.
/// All inputs and outputs are f64. Mirrors the airGR Fortran implementation.
use super::constants::{MAX_TANH_ARG, ROUTING_DENOMINATOR};

/// Step 1: Update production store for rainfall neutralization.
///
/// Rainfall is absorbed into the production store using a tanh function.
/// Returns (s1, p1, ps):
/// - s1: store level after rainfall [mm]
/// - p1: rainfall excess [mm/month]
/// - ps: storage fill [mm/month]
#[inline]
pub fn production_store_rainfall(precip: f64, store: f64, x1: f64) -> (f64, f64, f64) {
    // Scaled precipitation with numerical safeguard
    let ws = (precip / x1).min(MAX_TANH_ARG);
    let tws = ws.tanh();

    // Store ratio
    let sr = store / x1;

    // New store level after rainfall neutralization
    let s1 = (store + x1 * tws) / (1.0 + sr * tws);

    // Rainfall excess (water not stored)
    let p1 = precip + store - s1;

    // Storage fill (water entering store)
    let ps = precip - p1;

    (s1, p1, ps)
}

/// Step 2: Extract evapotranspiration from production store.
///
/// Returns (s2, ae):
/// - s2: store level after evaporation [mm]
/// - ae: actual evapotranspiration [mm/month]
#[inline]
pub fn production_store_evaporation(pet: f64, s1: f64, x1: f64) -> (f64, f64) {
    // Scaled evapotranspiration with numerical safeguard
    let ws = (pet / x1).min(MAX_TANH_ARG);
    let tws = ws.tanh();

    // Store ratio
    let sr = s1 / x1;

    // Store level after evaporation
    let s2 = s1 * (1.0 - tws) / (1.0 + (1.0 - sr) * tws);

    // Actual evapotranspiration
    let ae = s1 - s2;

    (s2, ae)
}

/// Step 3: Compute percolation from production store.
///
/// Uses a cube-root relationship.
/// Returns (s_final, p2):
/// - s_final: store level after percolation [mm]
/// - p2: percolation amount [mm/month]
#[inline]
pub fn percolation(s2: f64, x1: f64) -> (f64, f64) {
    // Ensure non-negative store
    let store = s2.max(0.0);

    // Store ratio cubed
    let sr = store / x1;
    let sr_cubed = sr * sr * sr;

    // Percolation using cube root formula
    let s_final = store / (1.0 + sr_cubed).cbrt();

    // Percolation amount
    let p2 = store - s_final;

    (s_final, p2)
}

/// Step 4: Update routing store with inflow and groundwater exchange.
///
/// Returns (r2, aexch):
/// - r2: routing store after exchange [mm]
/// - aexch: actual groundwater exchange [mm/month]
#[inline]
pub fn routing_store_update(routing_store: f64, p3: f64, x2: f64) -> (f64, f64) {
    // Routing store after inflow
    let r1 = routing_store + p3;

    // Apply groundwater exchange coefficient
    let r2 = x2 * r1;

    // Actual exchange (can be positive or negative)
    let aexch = r2 - r1;

    (r2, aexch)
}

/// Step 5: Compute streamflow from routing store.
///
/// Uses quadratic reservoir routing: Q = R² / (R + 60).
/// Returns (r_final, q):
/// - r_final: routing store after streamflow [mm]
/// - q: simulated streamflow [mm/month]
#[inline]
pub fn compute_streamflow(r2: f64) -> (f64, f64) {
    // Ensure non-negative store
    let store = r2.max(0.0);

    // Quadratic routing
    let q = if store > 0.0 {
        (store * store) / (store + ROUTING_DENOMINATOR)
    } else {
        0.0
    };

    // Update routing store
    let r_final = store - q;

    (r_final, q)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert two f64 values are close (like pytest.approx).
    fn assert_approx(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() < tol,
            "expected {expected} ± {tol}, got {actual}"
        );
    }

    // -- Step 1: Rainfall neutralization --

    #[test]
    fn rainfall_increases_store() {
        let (s1, _p1, _ps) = production_store_rainfall(80.0, 150.0, 500.0);
        assert!(s1 > 150.0, "store should increase with rainfall");
    }

    #[test]
    fn rainfall_conservation() {
        // P = P1 + PS (rainfall = excess + stored)
        let precip = 80.0;
        let store = 150.0;
        let (_s1, p1, ps) = production_store_rainfall(precip, store, 500.0);
        assert_approx(p1 + ps, precip, 1e-10);
    }

    #[test]
    fn zero_rainfall_no_change() {
        let (s1, p1, ps) = production_store_rainfall(0.0, 150.0, 500.0);
        assert_approx(s1, 150.0, 1e-10);
        assert_approx(p1, 0.0, 1e-10);
        assert_approx(ps, 0.0, 1e-10);
    }

    // -- Step 2: Evaporation --

    #[test]
    fn evaporation_decreases_store() {
        let (s2, ae) = production_store_evaporation(25.0, 280.0, 500.0);
        assert!(s2 < 280.0, "store should decrease with evaporation");
        assert!(ae > 0.0, "actual ET should be positive");
    }

    #[test]
    fn evaporation_conservation() {
        // AE = S1 - S2
        let s1 = 280.0;
        let (s2, ae) = production_store_evaporation(25.0, s1, 500.0);
        assert_approx(ae, s1 - s2, 1e-10);
    }

    #[test]
    fn zero_pet_no_evaporation() {
        let (s2, ae) = production_store_evaporation(0.0, 280.0, 500.0);
        assert_approx(s2, 280.0, 1e-10);
        assert_approx(ae, 0.0, 1e-10);
    }

    // -- Step 3: Percolation --

    #[test]
    fn percolation_decreases_store() {
        let (s_final, p2) = percolation(240.0, 500.0);
        assert!(s_final < 240.0);
        assert!(p2 > 0.0);
    }

    #[test]
    fn percolation_conservation() {
        let s2 = 240.0;
        let (s_final, p2) = percolation(s2, 500.0);
        assert_approx(s_final + p2, s2, 1e-10);
    }

    #[test]
    fn percolation_handles_negative_store() {
        let (s_final, p2) = percolation(-10.0, 500.0);
        assert_approx(s_final, 0.0, 1e-10);
        assert_approx(p2, 0.0, 1e-10);
    }

    // -- Step 4: Routing --

    #[test]
    fn routing_no_exchange_when_x2_is_one() {
        let (r2, aexch) = routing_store_update(100.0, 50.0, 1.0);
        assert_approx(r2, 150.0, 1e-10);
        assert_approx(aexch, 0.0, 1e-10);
    }

    #[test]
    fn routing_gain_when_x2_above_one() {
        let (_r2, aexch) = routing_store_update(100.0, 50.0, 1.5);
        assert!(aexch > 0.0, "x2 > 1 should produce water gain");
    }

    #[test]
    fn routing_loss_when_x2_below_one() {
        let (_r2, aexch) = routing_store_update(100.0, 50.0, 0.5);
        assert!(aexch < 0.0, "x2 < 1 should produce water loss");
    }

    // -- Step 5: Streamflow --

    #[test]
    fn streamflow_quadratic_formula() {
        // Q = R² / (R + 60) for R = 100
        let (r_final, q) = compute_streamflow(100.0);
        assert_approx(q, 10000.0 / 160.0, 1e-10); // 62.5
        assert_approx(r_final, 100.0 - q, 1e-10);
    }

    #[test]
    fn streamflow_zero_for_empty_store() {
        let (r_final, q) = compute_streamflow(0.0);
        assert_approx(q, 0.0, 1e-10);
        assert_approx(r_final, 0.0, 1e-10);
    }

    #[test]
    fn streamflow_handles_negative_store() {
        let (r_final, q) = compute_streamflow(-50.0);
        assert_approx(q, 0.0, 1e-10);
        assert_approx(r_final, 0.0, 1e-10);
    }
}
