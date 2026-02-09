/// CemaNeige snow module process functions.
///
/// Pure functions implementing the mathematical equations for each model component.
use super::constants::{MIN_SPEED, T_MELT, T_RAIN, T_SNOW};

/// Compute fraction of precipitation falling as snow using USACE formula.
///
/// Linear interpolation between T_SNOW (-1C) and T_RAIN (3C).
#[inline]
pub fn compute_solid_fraction(temp: f64) -> f64 {
    if temp <= T_SNOW {
        1.0
    } else if temp >= T_RAIN {
        0.0
    } else {
        (T_RAIN - temp) / (T_RAIN - T_SNOW)
    }
}

/// Split precipitation into liquid (rain) and solid (snow) components.
///
/// Returns (pliq, psol).
#[inline]
pub fn partition_precipitation(precip: f64, solid_fraction: f64) -> (f64, f64) {
    let pliq = (1.0 - solid_fraction) * precip;
    let psol = solid_fraction * precip;
    (pliq, psol)
}

/// Update snow pack thermal state using exponential smoothing.
///
/// The thermal state is capped at 0C (snow cannot be warmer than melting point).
#[inline]
pub fn update_thermal_state(etg: f64, temp: f64, ctg: f64) -> f64 {
    let new_etg = ctg * etg + (1.0 - ctg) * temp;
    new_etg.min(0.0)
}

/// Compute potential snow melt using degree-day method.
///
/// Melt only occurs when etg == 0 (snow at melting point) AND temp > T_MELT.
/// Capped at available snow pack.
#[inline]
pub fn compute_potential_melt(etg: f64, temp: f64, kf: f64, snow_pack: f64) -> f64 {
    if etg == 0.0 && temp > T_MELT {
        (kf * temp).min(snow_pack)
    } else {
        0.0
    }
}

/// Compute snow cover fraction (Gratio).
///
/// - 1.0 when snow_pack >= gthreshold
/// - snow_pack / gthreshold when below
/// - 0.0 if gthreshold is 0
#[inline]
pub fn compute_gratio(snow_pack: f64, gthreshold: f64) -> f64 {
    if gthreshold == 0.0 {
        0.0
    } else if snow_pack >= gthreshold {
        1.0
    } else {
        snow_pack / gthreshold
    }
}

/// Compute actual snow melt modulated by snow cover fraction.
///
/// Even with minimal cover, melt proceeds at MIN_SPEED (10%) of potential.
#[inline]
pub fn compute_actual_melt(potential_melt: f64, gratio: f64) -> f64 {
    ((1.0 - MIN_SPEED) * gratio + MIN_SPEED) * potential_melt
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- compute_solid_fraction --

    #[test]
    fn all_snow_below_threshold() {
        assert_eq!(compute_solid_fraction(-5.0), 1.0);
        assert_eq!(compute_solid_fraction(-1.0), 1.0);
    }

    #[test]
    fn all_rain_above_threshold() {
        assert_eq!(compute_solid_fraction(3.0), 0.0);
        assert_eq!(compute_solid_fraction(10.0), 0.0);
    }

    #[test]
    fn mixed_at_midpoint() {
        let sf = compute_solid_fraction(1.0);
        assert!((sf - 0.5).abs() < 1e-10);
    }

    // -- partition_precipitation --

    #[test]
    fn partition_all_snow() {
        let (pliq, psol) = partition_precipitation(10.0, 1.0);
        assert_eq!(pliq, 0.0);
        assert_eq!(psol, 10.0);
    }

    #[test]
    fn partition_all_rain() {
        let (pliq, psol) = partition_precipitation(10.0, 0.0);
        assert_eq!(pliq, 10.0);
        assert_eq!(psol, 0.0);
    }

    #[test]
    fn partition_conserves_mass() {
        let (pliq, psol) = partition_precipitation(10.0, 0.4);
        assert!((pliq + psol - 10.0).abs() < 1e-10);
    }

    // -- update_thermal_state --

    #[test]
    fn thermal_state_capped_at_zero() {
        // Warm temp should push state toward positive, but cap at 0
        let etg = update_thermal_state(-1.0, 10.0, 0.5);
        assert!(etg <= 0.0);
    }

    #[test]
    fn thermal_state_tracks_cold() {
        let etg = update_thermal_state(0.0, -10.0, 0.5);
        assert!(etg < 0.0);
        assert!((etg - (-5.0)).abs() < 1e-10);
    }

    // -- compute_potential_melt --

    #[test]
    fn no_melt_when_cold() {
        // etg < 0 means snow is still cold
        assert_eq!(compute_potential_melt(-1.0, 5.0, 3.0, 100.0), 0.0);
    }

    #[test]
    fn no_melt_when_temp_below_threshold() {
        assert_eq!(compute_potential_melt(0.0, -1.0, 3.0, 100.0), 0.0);
    }

    #[test]
    fn melt_when_conditions_met() {
        let melt = compute_potential_melt(0.0, 5.0, 3.0, 100.0);
        assert!((melt - 15.0).abs() < 1e-10); // kf * temp
    }

    #[test]
    fn melt_capped_by_snow_pack() {
        let melt = compute_potential_melt(0.0, 5.0, 3.0, 10.0);
        assert_eq!(melt, 10.0); // limited by snow_pack
    }

    // -- compute_gratio --

    #[test]
    fn gratio_full_cover() {
        assert_eq!(compute_gratio(200.0, 100.0), 1.0);
        assert_eq!(compute_gratio(100.0, 100.0), 1.0);
    }

    #[test]
    fn gratio_partial_cover() {
        assert!((compute_gratio(50.0, 100.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn gratio_zero_threshold() {
        assert_eq!(compute_gratio(50.0, 0.0), 0.0);
    }

    // -- compute_actual_melt --

    #[test]
    fn actual_melt_full_cover() {
        let melt = compute_actual_melt(10.0, 1.0);
        assert_eq!(melt, 10.0); // (0.9 * 1.0 + 0.1) * 10 = 10
    }

    #[test]
    fn actual_melt_no_cover() {
        let melt = compute_actual_melt(10.0, 0.0);
        assert!((melt - 1.0).abs() < 1e-10); // MIN_SPEED * 10 = 1.0
    }

    #[test]
    fn actual_melt_partial_cover() {
        let melt = compute_actual_melt(10.0, 0.5);
        // (0.9 * 0.5 + 0.1) * 10 = (0.45 + 0.1) * 10 = 5.5
        assert!((melt - 5.5).abs() < 1e-10);
    }
}
