//! Shared elevation extrapolation utilities.
//!
//! Used by HBV-Light and CemaNeige for multi-zone/multi-layer runs.

/// Default temperature gradient [C/100m].
pub const GRAD_T_DEFAULT: f64 = 0.6;

/// Default precipitation gradient [m^-1].
pub const GRAD_P_DEFAULT: f64 = 0.00041;

/// Elevation cap for precipitation extrapolation [m].
pub const ELEV_CAP_PRECIP: f64 = 4000.0;

/// Default linear precipitation gradient [m^-1].
pub const GRAD_P_LINEAR_DEFAULT: f64 = 0.0004;

/// Extrapolate temperature to a different elevation using lapse rate.
#[inline]
pub fn extrapolate_temp(temp: f64, input_elev: f64, target_elev: f64, gradient: f64) -> f64 {
    temp - gradient * (target_elev - input_elev) / 100.0
}

/// Extrapolate precipitation to a different elevation using exponential gradient.
#[inline]
pub fn extrapolate_precip(precip: f64, input_elev: f64, target_elev: f64, gradient: f64) -> f64 {
    let eff_in = input_elev.min(ELEV_CAP_PRECIP);
    let eff_target = target_elev.min(ELEV_CAP_PRECIP);
    precip * (gradient * (eff_target - eff_in)).exp()
}

/// Extrapolate precipitation with a custom elevation cap.
#[inline]
pub fn extrapolate_precip_with_cap(
    precip: f64,
    input_elev: f64,
    target_elev: f64,
    gradient: f64,
    cap: f64,
) -> f64 {
    let eff_in = input_elev.min(cap);
    let eff_target = target_elev.min(cap);
    precip * (gradient * (eff_target - eff_in)).exp()
}

/// Extrapolate precipitation using a linear gradient.
/// Result clamped to >= 0.0 (linear can go negative for large negative ΔZ).
#[inline]
pub fn extrapolate_precip_linear(precip: f64, input_elev: f64, target_elev: f64, gradient: f64) -> f64 {
    let eff_in = input_elev.min(ELEV_CAP_PRECIP);
    let eff_target = target_elev.min(ELEV_CAP_PRECIP);
    (precip * (1.0 + gradient * (eff_target - eff_in))).max(0.0)
}

/// Compute per-band gthreshold scaled by precipitation gradient.
/// Reuses extrapolation logic so gthreshold scaling always matches forcing.
/// Falls back to uniform `factor * masp` when `input_elev` is NaN.
#[inline]
pub fn compute_band_gthreshold(
    factor: f64,
    masp: f64,
    input_elev: f64,
    band_elev: f64,
    gradient: f64,
    use_linear: bool,
) -> f64 {
    let base = factor * masp;
    if input_elev.is_nan() {
        return base;
    }
    if use_linear {
        extrapolate_precip_linear(base, input_elev, band_elev, gradient)
    } else {
        extrapolate_precip(base, input_elev, band_elev, gradient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_same_elevation() {
        assert!((extrapolate_temp(10.0, 500.0, 500.0, 0.6) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn temp_higher_is_colder() {
        let t = extrapolate_temp(10.0, 500.0, 1000.0, 0.6);
        assert!((t - 7.0).abs() < 1e-10);
    }

    #[test]
    fn precip_same_elevation() {
        assert!((extrapolate_precip(10.0, 500.0, 500.0, 0.00041) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn precip_with_cap_same_elevation() {
        assert!(
            (extrapolate_precip_with_cap(10.0, 500.0, 500.0, 0.00041, 4000.0) - 10.0).abs()
                < 1e-10
        );
    }

    #[test]
    fn linear_same_elevation() {
        let result = extrapolate_precip_linear(10.0, 500.0, 500.0, 0.0004);
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn linear_higher() {
        let result = extrapolate_precip_linear(10.0, 500.0, 1500.0, 0.0004);
        // 10 * (1 + 0.0004 * 1000) = 10 * 1.4 = 14.0
        assert!((result - 14.0).abs() < 1e-10);
    }

    #[test]
    fn linear_lower() {
        let result = extrapolate_precip_linear(10.0, 1000.0, 500.0, 0.0004);
        // 10 * (1 + 0.0004 * (-500)) = 10 * 0.8 = 8.0
        assert!((result - 8.0).abs() < 1e-10);
    }

    #[test]
    fn linear_clamps_to_zero() {
        // Very large negative ΔZ
        let result = extrapolate_precip_linear(10.0, 4000.0, 0.0, 0.001);
        // 10 * (1 + 0.001 * (-4000)) = 10 * (-3.0) = -30 -> clamped to 0
        assert_eq!(result, 0.0);
    }

    #[test]
    fn linear_with_cap() {
        // Input at 5000 -> capped to 4000; target at 3000
        let result = extrapolate_precip_linear(10.0, 5000.0, 3000.0, 0.0004);
        // 10 * (1 + 0.0004 * (3000 - 4000)) = 10 * 0.6 = 6.0
        assert!((result - 6.0).abs() < 1e-10);
    }

    #[test]
    fn band_gthreshold_same_elev() {
        let gt = compute_band_gthreshold(0.9, 200.0, 500.0, 500.0, 0.00041, false);
        assert!((gt - 180.0).abs() < 1e-10); // 0.9 * 200 = 180
    }

    #[test]
    fn band_gthreshold_higher_band() {
        let gt_low = compute_band_gthreshold(0.9, 200.0, 500.0, 500.0, 0.00041, false);
        let gt_high = compute_band_gthreshold(0.9, 200.0, 500.0, 1500.0, 0.00041, false);
        assert!(gt_high > gt_low, "higher band should have higher gthreshold");
    }

    #[test]
    fn band_gthreshold_nan_fallback() {
        let gt = compute_band_gthreshold(0.9, 200.0, f64::NAN, 1500.0, 0.00041, false);
        assert!((gt - 180.0).abs() < 1e-10); // fallback to base = 0.9 * 200
    }

    #[test]
    fn linear_gradient_default_constant() {
        assert!((GRAD_P_LINEAR_DEFAULT - 0.0004).abs() < 1e-10);
    }

    #[test]
    fn band_gthreshold_linear_mode() {
        let gt = compute_band_gthreshold(0.9, 200.0, 500.0, 1500.0, 0.0004, true);
        // base = 180.0, linear: 180 * (1 + 0.0004 * 1000) = 180 * 1.4 = 252.0
        assert!((gt - 252.0).abs() < 1e-10);
    }

    #[test]
    fn band_gthreshold_exponential_vs_linear_diverge() {
        let gt_exp = compute_band_gthreshold(0.9, 200.0, 500.0, 1500.0, 0.0004, false);
        let gt_lin = compute_band_gthreshold(0.9, 200.0, 500.0, 1500.0, 0.0004, true);
        // Both > base, but should differ
        assert!(gt_exp > 180.0);
        assert!(gt_lin > 180.0);
        assert!((gt_exp - gt_lin).abs() > 0.01);
    }

    #[test]
    fn linear_zero_precip_stays_zero() {
        assert_eq!(extrapolate_precip_linear(0.0, 500.0, 2000.0, 0.0004), 0.0);
    }
}
