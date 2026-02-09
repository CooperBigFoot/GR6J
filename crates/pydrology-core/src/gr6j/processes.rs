/// GR6J core process functions.
///
/// Pure functions implementing the mathematical equations for each GR6J step.
/// All inputs and outputs are f64. Mirrors the airGR Fortran implementation.
use super::constants::{EXP_BRANCH_THRESHOLD, MAX_EXP_ARG, MAX_TANH_ARG, PERC_CONSTANT};

/// Update the production store based on precipitation and evapotranspiration.
///
/// Handles two cases:
/// - Case 1: P < E (evapotranspiration dominant)
/// - Case 2: P >= E (rainfall dominant)
///
/// Returns (new_store, actual_et, net_rainfall_pn, effective_rainfall_pr).
#[inline]
pub fn production_store_update(
    precip: f64,
    pet: f64,
    production_store: f64,
    x1: f64,
) -> (f64, f64, f64, f64) {
    let store_ratio = production_store / x1;

    if precip < pet {
        // Case 1: Evapotranspiration dominant (P < E)
        let net_evap = pet - precip;
        let scaled_evap = (net_evap / x1).min(MAX_TANH_ARG);
        let tanh_ws = scaled_evap.tanh();

        // Evaporation from production store
        let numerator = (2.0 - store_ratio) * tanh_ws;
        let denominator = 1.0 + (1.0 - store_ratio) * tanh_ws;
        let evap_from_store = production_store * numerator / denominator;

        // Actual evapotranspiration = evaporation from store + precipitation
        let actual_et = evap_from_store + precip;

        // Update store
        let new_store = production_store - evap_from_store;

        // No effective rainfall when evapotranspiration dominant
        (new_store, actual_et, 0.0, 0.0)
    } else {
        // Case 2: Rainfall dominant (P >= E)
        let net_rainfall_pn = precip - pet;
        let actual_et = pet;

        let scaled_precip = (net_rainfall_pn / x1).min(MAX_TANH_ARG);
        let tanh_ws = scaled_precip.tanh();

        // Storage infiltration
        let numerator = (1.0 - store_ratio * store_ratio) * tanh_ws;
        let denominator = 1.0 + store_ratio * tanh_ws;
        let storage_infiltration = x1 * numerator / denominator;

        // Rainfall excess (effective rainfall before percolation)
        let effective_rainfall_pr = net_rainfall_pn - storage_infiltration;

        // Update store
        let new_store = production_store + storage_infiltration;

        (new_store, actual_et, net_rainfall_pn, effective_rainfall_pr)
    }
}

/// Compute percolation from the production store.
///
/// Perc = S * (1 - (1 + (S/X1)^4 / PERC_CONSTANT)^(-0.25))
///
/// Returns (new_store, percolation_amount).
#[inline]
pub fn percolation(production_store: f64, x1: f64) -> (f64, f64) {
    let store = production_store.max(0.0);

    let store_ratio = store / x1;
    let store_ratio_4 = store_ratio * store_ratio * store_ratio * store_ratio;
    let percolation_amount = store * (1.0 - (1.0 + store_ratio_4 / PERC_CONSTANT).powf(-0.25));

    let new_store = store - percolation_amount;

    (new_store, percolation_amount)
}

/// Compute potential groundwater exchange.
///
/// F = X2 * (R/X3 - X5)
#[inline]
pub fn groundwater_exchange(routing_store: f64, x2: f64, x3: f64, x5: f64) -> f64 {
    x2 * (routing_store / x3 - x5)
}

/// Update the routing store and compute outflow.
///
/// Receives (1-C) * uh1_output + exchange, applies non-negativity constraint,
/// and computes non-linear outflow: QR = R * (1 - 1/(1 + (R/X3)^4)^0.25).
///
/// Returns (new_store, outflow_qr, actual_exchange).
#[inline]
pub fn routing_store_update(
    routing_store: f64,
    uh1_output: f64,
    exchange: f64,
    x3: f64,
) -> (f64, f64, f64) {
    let store_after_inflow = routing_store + uh1_output + exchange;

    let (actual_exchange, store) = if store_after_inflow >= 0.0 {
        (exchange, store_after_inflow)
    } else {
        (-(routing_store + uh1_output), 0.0)
    };

    let outflow_qr = if store > 0.0 {
        let store_ratio = store / x3;
        let store_ratio_4 = store_ratio * store_ratio * store_ratio * store_ratio;
        store * (1.0 - (1.0 + store_ratio_4).powf(-0.25))
    } else {
        0.0
    };

    let new_store = store - outflow_qr;

    (new_store, outflow_qr, actual_exchange)
}

/// Update the exponential store and compute outflow.
///
/// Receives C * uh1_output + exchange. Note that exp_store can be negative.
/// Uses softplus-like function with branch equations for numerical stability.
///
/// Returns (new_store, outflow_qrexp).
#[inline]
pub fn exponential_store_update(
    exp_store: f64,
    uh1_output: f64,
    exchange: f64,
    x6: f64,
) -> (f64, f64) {
    let store = exp_store + uh1_output + exchange;

    // Scaled store level with numerical safeguard
    let ar = (store / x6).clamp(-MAX_EXP_ARG, MAX_EXP_ARG);

    let outflow_qrexp = if ar > EXP_BRANCH_THRESHOLD {
        // Large positive AR: QRExp = Exp + X6/exp(AR)
        store + x6 / ar.exp()
    } else if ar < -EXP_BRANCH_THRESHOLD {
        // Large negative AR: QRExp = X6 * exp(AR)
        x6 * ar.exp()
    } else {
        // Normal range: softplus function QRExp = X6 * ln(exp(AR) + 1)
        x6 * (ar.exp() + 1.0).ln()
    };

    let new_store = store - outflow_qrexp;

    (new_store, outflow_qrexp)
}

/// Compute direct branch outflow.
///
/// Applies non-negativity constraint: QD = max(uh2_output + exchange, 0).
///
/// Returns (outflow_qd, actual_exchange).
#[inline]
pub fn direct_branch(uh2_output: f64, exchange: f64) -> (f64, f64) {
    let combined = uh2_output + exchange;

    if combined >= 0.0 {
        (combined, exchange)
    } else {
        (0.0, -uh2_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() < tol,
            "expected {expected} ± {tol}, got {actual}"
        );
    }

    // -- Production store update --

    #[test]
    fn rainfall_dominant_increases_store() {
        let (new_store, _et, _pn, _pr) = production_store_update(10.0, 3.0, 105.0, 350.0);
        assert!(new_store > 105.0, "store should increase with rainfall > PET");
    }

    #[test]
    fn evap_dominant_decreases_store() {
        let (new_store, _et, _pn, _pr) = production_store_update(2.0, 5.0, 105.0, 350.0);
        assert!(new_store < 105.0, "store should decrease when PET > rainfall");
    }

    #[test]
    fn rainfall_dominant_net_rainfall_positive() {
        let (_new_store, _et, pn, _pr) = production_store_update(10.0, 3.0, 105.0, 350.0);
        assert_approx(pn, 7.0, 1e-10);
    }

    #[test]
    fn evap_dominant_net_rainfall_zero() {
        let (_new_store, _et, pn, pr) = production_store_update(2.0, 5.0, 105.0, 350.0);
        assert_approx(pn, 0.0, 1e-10);
        assert_approx(pr, 0.0, 1e-10);
    }

    #[test]
    fn zero_inputs_no_change() {
        let (new_store, et, pn, pr) = production_store_update(0.0, 0.0, 105.0, 350.0);
        assert_approx(new_store, 105.0, 1e-10);
        assert_approx(et, 0.0, 1e-10);
        assert_approx(pn, 0.0, 1e-10);
        assert_approx(pr, 0.0, 1e-10);
    }

    #[test]
    fn actual_et_equals_pet_when_rainfall_dominant() {
        let (_new_store, et, _pn, _pr) = production_store_update(10.0, 3.0, 105.0, 350.0);
        assert_approx(et, 3.0, 1e-10);
    }

    // -- Percolation --

    #[test]
    fn percolation_decreases_store() {
        let (new_store, perc) = percolation(105.0, 350.0);
        assert!(new_store < 105.0);
        assert!(perc > 0.0);
    }

    #[test]
    fn percolation_conservation() {
        let store = 105.0;
        let (new_store, perc) = percolation(store, 350.0);
        assert_approx(new_store + perc, store, 1e-10);
    }

    #[test]
    fn percolation_handles_negative_store() {
        let (new_store, perc) = percolation(-10.0, 350.0);
        assert_approx(new_store, 0.0, 1e-10);
        assert_approx(perc, 0.0, 1e-10);
    }

    // -- Groundwater exchange --

    #[test]
    fn exchange_zero_when_x2_zero() {
        let f = groundwater_exchange(45.0, 0.0, 90.0, 0.0);
        assert_approx(f, 0.0, 1e-10);
    }

    #[test]
    fn exchange_positive_import() {
        // R/X3 = 45/90 = 0.5, x5=0 → F = 1.0 * 0.5 = 0.5
        let f = groundwater_exchange(45.0, 1.0, 90.0, 0.0);
        assert_approx(f, 0.5, 1e-10);
    }

    #[test]
    fn exchange_negative_export() {
        // R/X3 = 45/90 = 0.5, x5=1.0 → F = 1.0 * (0.5 - 1.0) = -0.5
        let f = groundwater_exchange(45.0, 1.0, 90.0, 1.0);
        assert_approx(f, -0.5, 1e-10);
    }

    // -- Routing store update --

    #[test]
    fn routing_store_increases_with_inflow() {
        let (new_store, _qr, _aexch) = routing_store_update(45.0, 5.0, 0.0, 90.0);
        assert!(new_store > 45.0 - 5.0, "store should have some remaining after outflow");
    }

    #[test]
    fn routing_store_non_negative() {
        // Large negative exchange should floor at zero
        let (new_store, _qr, _aexch) = routing_store_update(5.0, 1.0, -100.0, 90.0);
        assert!(new_store >= 0.0);
    }

    #[test]
    fn routing_store_exchange_limited_when_negative() {
        let (new_store, qr, actual_exchange) = routing_store_update(5.0, 1.0, -100.0, 90.0);
        assert_approx(new_store, 0.0, 1e-10);
        assert_approx(qr, 0.0, 1e-10);
        assert_approx(actual_exchange, -6.0, 1e-10); // -(5.0 + 1.0)
    }

    // -- Exponential store update --

    #[test]
    fn exponential_store_positive_outflow() {
        let (new_store, qrexp) = exponential_store_update(0.0, 5.0, 0.0, 5.0);
        assert!(qrexp > 0.0);
        assert!(new_store < 5.0);
    }

    #[test]
    fn exponential_store_handles_negative() {
        let (new_store, qrexp) = exponential_store_update(-10.0, 1.0, 0.0, 5.0);
        assert!(qrexp.is_finite());
        assert!(new_store.is_finite());
    }

    #[test]
    fn exponential_store_large_positive() {
        // Test the large positive AR branch
        let (new_store, qrexp) = exponential_store_update(100.0, 5.0, 0.0, 5.0);
        assert!(qrexp > 0.0);
        assert!(new_store.is_finite());
    }

    #[test]
    fn exponential_store_large_negative() {
        // Test the large negative AR branch
        let (new_store, qrexp) = exponential_store_update(-100.0, 5.0, 0.0, 5.0);
        assert!(qrexp >= 0.0);
        assert!(new_store.is_finite());
    }

    // -- Direct branch --

    #[test]
    fn direct_branch_positive() {
        let (qd, aexch) = direct_branch(5.0, 1.0);
        assert_approx(qd, 6.0, 1e-10);
        assert_approx(aexch, 1.0, 1e-10);
    }

    #[test]
    fn direct_branch_zero_exchange() {
        let (qd, aexch) = direct_branch(5.0, 0.0);
        assert_approx(qd, 5.0, 1e-10);
        assert_approx(aexch, 0.0, 1e-10);
    }

    #[test]
    fn direct_branch_negative_limited() {
        let (qd, aexch) = direct_branch(3.0, -5.0);
        assert_approx(qd, 0.0, 1e-10);
        assert_approx(aexch, -3.0, 1e-10);
    }
}
