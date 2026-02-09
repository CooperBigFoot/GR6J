//! HBV-Light snow and soil moisture process functions.
//!
//! Pure functions implementing the snow routine (precipitation partitioning,
//! accumulation, melt, refreezing) and soil moisture routine (recharge, ET).

/// Partition precipitation into rain and snow.
///
/// Returns (p_rain, p_snow).
#[inline]
pub fn partition_precipitation(precip: f64, temp: f64, tt: f64, sfcf: f64) -> (f64, f64) {
    if temp > tt {
        (precip, 0.0)
    } else {
        (0.0, sfcf * precip)
    }
}

/// Compute snowmelt using degree-day method.
///
/// Melt is limited by available snow pack.
#[inline]
pub fn compute_melt(temp: f64, tt: f64, cfmax: f64, snow_pack: f64) -> f64 {
    if temp > tt {
        let melt = cfmax * (temp - tt);
        melt.min(snow_pack)
    } else {
        0.0
    }
}

/// Compute refreezing of liquid water in snowpack.
///
/// Refreezing is limited by available liquid water.
#[inline]
pub fn compute_refreezing(temp: f64, tt: f64, cfmax: f64, cfr: f64, liquid_water: f64) -> f64 {
    if temp < tt {
        let refreeze = cfr * cfmax * (tt - temp);
        refreeze.min(liquid_water)
    } else {
        0.0
    }
}

/// Update snow pack and liquid water states.
///
/// Returns (new_sp, new_lw, outflow).
#[inline]
pub fn update_snow_pack(
    sp: f64,
    lw: f64,
    p_snow: f64,
    melt: f64,
    refreeze: f64,
    cwh: f64,
) -> (f64, f64, f64) {
    let mut new_sp = sp + p_snow - melt + refreeze;
    let mut new_lw = lw + melt - refreeze;

    let lw_max = cwh * new_sp;

    let outflow = if new_lw > lw_max {
        let excess = new_lw - lw_max;
        new_lw = lw_max;
        excess
    } else {
        0.0
    };

    new_sp = new_sp.max(0.0);
    new_lw = new_lw.max(0.0);

    (new_sp, new_lw, outflow)
}

/// Compute groundwater recharge from soil.
///
/// Uses the HBV soil moisture / runoff relationship: Recharge/Input = (SM/FC)^BETA.
#[inline]
pub fn compute_recharge(soil_input: f64, sm: f64, fc: f64, beta: f64) -> f64 {
    if fc <= 0.0 || soil_input <= 0.0 {
        return 0.0;
    }
    let sr = (sm / fc).clamp(0.0, 1.0);
    soil_input * sr.powf(beta)
}

/// Compute actual evapotranspiration from soil.
///
/// ET is reduced linearly when SM < LP * FC.
#[inline]
pub fn compute_actual_et(pet: f64, sm: f64, fc: f64, lp: f64) -> f64 {
    if fc <= 0.0 || lp <= 0.0 {
        return 0.0;
    }

    let lp_threshold = lp * fc;

    let et_act = if sm >= lp_threshold {
        pet
    } else {
        pet * sm / lp_threshold
    };

    et_act.min(sm.max(0.0))
}

/// Update soil moisture state.
///
/// Returns updated soil moisture, bounded to [0, FC].
#[inline]
pub fn update_soil_moisture(
    sm: f64,
    soil_input: f64,
    recharge: f64,
    et_act: f64,
    fc: f64,
) -> f64 {
    let infiltration = soil_input - recharge;
    let new_sm = sm + infiltration - et_act;
    new_sm.clamp(0.0, fc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() < tol,
            "expected {expected} +/- {tol}, got {actual}"
        );
    }

    // -- Precipitation partitioning --

    #[test]
    fn rain_above_threshold() {
        let (p_rain, p_snow) = partition_precipitation(10.0, 5.0, 0.0, 1.0);
        assert_eq!(p_rain, 10.0);
        assert_eq!(p_snow, 0.0);
    }

    #[test]
    fn snow_below_threshold() {
        let (p_rain, p_snow) = partition_precipitation(10.0, -5.0, 0.0, 1.0);
        assert_eq!(p_rain, 0.0);
        assert_eq!(p_snow, 10.0);
    }

    #[test]
    fn sfcf_correction() {
        let (p_rain, p_snow) = partition_precipitation(10.0, -5.0, 0.0, 0.8);
        assert_eq!(p_rain, 0.0);
        assert_eq!(p_snow, 8.0);
    }

    // -- Snowmelt --

    #[test]
    fn melt_above_threshold() {
        let melt = compute_melt(5.0, 0.0, 3.0, 100.0);
        assert_approx(melt, 15.0, 1e-10);
    }

    #[test]
    fn no_melt_below_threshold() {
        let melt = compute_melt(-5.0, 0.0, 3.0, 100.0);
        assert_eq!(melt, 0.0);
    }

    #[test]
    fn melt_limited_by_snow() {
        let melt = compute_melt(10.0, 0.0, 3.0, 5.0);
        assert_eq!(melt, 5.0);
    }

    // -- Refreezing --

    #[test]
    fn refreeze_below_threshold() {
        let refreeze = compute_refreezing(-5.0, 0.0, 3.0, 0.05, 10.0);
        assert_approx(refreeze, 0.75, 1e-10);
    }

    #[test]
    fn no_refreeze_above_threshold() {
        let refreeze = compute_refreezing(5.0, 0.0, 3.0, 0.05, 10.0);
        assert_eq!(refreeze, 0.0);
    }

    #[test]
    fn refreeze_limited_by_liquid() {
        let refreeze = compute_refreezing(-10.0, 0.0, 3.0, 0.5, 2.0);
        assert_eq!(refreeze, 2.0);
    }

    // -- Snow pack update --

    #[test]
    fn snow_pack_outflow_when_lw_exceeds_capacity() {
        let (new_sp, new_lw, outflow) = update_snow_pack(10.0, 0.0, 0.0, 5.0, 0.0, 0.1);
        // new_sp = 10 + 0 - 5 + 0 = 5
        // new_lw = 0 + 5 - 0 = 5
        // lw_max = 0.1 * 5 = 0.5
        // outflow = 5 - 0.5 = 4.5
        assert_approx(new_sp, 5.0, 1e-10);
        assert_approx(new_lw, 0.5, 1e-10);
        assert_approx(outflow, 4.5, 1e-10);
    }

    // -- Recharge --

    #[test]
    fn recharge_increases_with_moisture() {
        let low_sm = compute_recharge(10.0, 50.0, 250.0, 2.0);
        let high_sm = compute_recharge(10.0, 200.0, 250.0, 2.0);
        assert!(high_sm > low_sm);
    }

    #[test]
    fn recharge_at_saturation() {
        let recharge = compute_recharge(10.0, 250.0, 250.0, 2.0);
        assert_approx(recharge, 10.0, 1e-10);
    }

    #[test]
    fn no_recharge_when_no_input() {
        let recharge = compute_recharge(0.0, 125.0, 250.0, 2.0);
        assert_eq!(recharge, 0.0);
    }

    // -- Actual ET --

    #[test]
    fn full_et_above_lp_threshold() {
        let et = compute_actual_et(5.0, 225.0, 250.0, 0.9);
        assert_approx(et, 5.0, 1e-10);
    }

    #[test]
    fn reduced_et_below_lp_threshold() {
        let et = compute_actual_et(5.0, 112.5, 250.0, 0.9);
        assert_approx(et, 2.5, 1e-10);
    }

    #[test]
    fn et_limited_by_available() {
        let et = compute_actual_et(300.0, 230.0, 250.0, 0.9);
        assert_approx(et, 230.0, 1e-10);
    }

    // -- Soil moisture update --

    #[test]
    fn soil_moisture_bounded_by_fc() {
        let new_sm = update_soil_moisture(240.0, 100.0, 10.0, 5.0, 250.0);
        assert!(new_sm <= 250.0);
        assert!(new_sm >= 0.0);
    }

    #[test]
    fn soil_moisture_non_negative() {
        let new_sm = update_soil_moisture(5.0, 0.0, 0.0, 100.0, 250.0);
        assert_eq!(new_sm, 0.0);
    }
}
