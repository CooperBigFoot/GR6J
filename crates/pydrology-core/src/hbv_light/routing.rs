/// HBV-Light response and routing process functions.
///
/// Functions for the groundwater response routine (upper and lower zone
/// reservoirs) and triangular unit hydrograph routing.
use super::constants::ROUTING_BUFFER_SIZE;

/// Compute outflows from upper groundwater zone.
///
/// Q0 (surface flow) is threshold-activated when SUZ > UZL.
/// Q1 (interflow) is always active.
/// Returns (q0, q1).
#[inline]
pub fn upper_zone_outflows(suz: f64, k0: f64, k1: f64, uzl: f64) -> (f64, f64) {
    let q0 = if suz > uzl { k0 * (suz - uzl) } else { 0.0 };
    let q1 = k1 * suz;
    (q0, q1)
}

/// Compute percolation from upper to lower zone.
///
/// Limited by available storage and maximum percolation rate.
#[inline]
pub fn compute_percolation(suz: f64, perc_max: f64) -> f64 {
    perc_max.min(suz.max(0.0))
}

/// Update upper zone storage.
#[inline]
pub fn update_upper_zone(suz: f64, recharge: f64, q0: f64, q1: f64, perc: f64) -> f64 {
    let new_suz = suz + recharge - q0 - q1 - perc;
    new_suz.max(0.0)
}

/// Compute baseflow from lower groundwater zone.
#[inline]
pub fn lower_zone_outflow(slz: f64, k2: f64) -> f64 {
    k2 * slz
}

/// Update lower zone storage.
#[inline]
pub fn update_lower_zone(slz: f64, perc: f64, q2: f64) -> f64 {
    let new_slz = slz + perc - q2;
    new_slz.max(0.0)
}

/// Compute triangular unit hydrograph weights.
///
/// Uses analytical integration of the triangular function to compute
/// weights that sum to 1.0. Supports fractional MAXBAS values.
#[inline]
pub fn compute_triangular_weights(maxbas: f64) -> Vec<f64> {
    let n = (maxbas.ceil() as usize).max(1);
    let mut weights = vec![0.0; n];
    let half = maxbas / 2.0;

    for (i, weight) in weights.iter_mut().enumerate().take(n) {
        let t_start = i as f64;
        let t_end = ((i + 1) as f64).min(maxbas);

        if t_end <= t_start {
            continue;
        }

        let mut w = 0.0;

        // Rising limb portion
        if t_start < half {
            let t_r_end = t_end.min(half);
            w += (t_r_end * t_r_end - t_start * t_start) / (maxbas * maxbas);
        }

        // Falling limb portion
        if t_end > half {
            let t_f_start = t_start.max(half);
            w += 2.0 * (t_end - t_f_start) / maxbas
                - (t_end * t_end - t_f_start * t_f_start) / (maxbas * maxbas);
        }

        *weight = w;
    }

    // Normalize
    let total: f64 = weights.iter().sum();
    if total > 0.0 {
        for w in &mut weights {
            *w /= total;
        }
    }

    weights
}

/// Convolve groundwater runoff with triangular unit hydrograph.
///
/// Returns (new_buffer, qsim).
#[inline]
pub fn convolve_triangular(
    qgw: f64,
    buffer: &[f64; ROUTING_BUFFER_SIZE],
    weights: &[f64],
) -> ([f64; ROUTING_BUFFER_SIZE], f64) {
    // Output is first buffer element
    let qsim = buffer[0];

    // Shift buffer left
    let mut new_buffer = [0.0; ROUTING_BUFFER_SIZE];
    new_buffer[..(ROUTING_BUFFER_SIZE - 1)].copy_from_slice(&buffer[1..ROUTING_BUFFER_SIZE]);

    // Add current input contribution using weights
    let n_weights = weights.len().min(ROUTING_BUFFER_SIZE);
    for i in 0..n_weights {
        new_buffer[i] += qgw * weights[i];
    }

    (new_buffer, qsim)
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

    // -- Upper zone outflows --

    #[test]
    fn q0_above_uzl() {
        let (q0, q1) = upper_zone_outflows(50.0, 0.4, 0.1, 20.0);
        assert_approx(q0, 12.0, 1e-10);
        assert_approx(q1, 5.0, 1e-10);
    }

    #[test]
    fn no_q0_below_uzl() {
        let (q0, q1) = upper_zone_outflows(15.0, 0.4, 0.1, 20.0);
        assert_eq!(q0, 0.0);
        assert_approx(q1, 1.5, 1e-10);
    }

    // -- Percolation --

    #[test]
    fn percolation_limited_by_max() {
        let perc = compute_percolation(100.0, 2.0);
        assert_eq!(perc, 2.0);
    }

    #[test]
    fn percolation_limited_by_storage() {
        let perc = compute_percolation(1.0, 2.0);
        assert_eq!(perc, 1.0);
    }

    #[test]
    fn percolation_handles_negative() {
        let perc = compute_percolation(-5.0, 2.0);
        assert_eq!(perc, 0.0);
    }

    // -- Lower zone --

    #[test]
    fn baseflow_linear() {
        let q2 = lower_zone_outflow(100.0, 0.01);
        assert_approx(q2, 1.0, 1e-10);
    }

    #[test]
    fn lower_zone_update_non_negative() {
        let new_slz = update_lower_zone(1.0, 0.0, 5.0);
        assert_eq!(new_slz, 0.0);
    }

    // -- Triangular weights --

    #[test]
    fn weights_sum_to_one() {
        for maxbas in [1.0, 2.0, 2.5, 3.0, 5.5, 7.0] {
            let weights = compute_triangular_weights(maxbas);
            let total: f64 = weights.iter().sum();
            assert_approx(total, 1.0, 1e-10);
        }
    }

    #[test]
    fn weights_length() {
        assert_eq!(compute_triangular_weights(1.0).len(), 1);
        assert_eq!(compute_triangular_weights(2.0).len(), 2);
        assert_eq!(compute_triangular_weights(2.5).len(), 3);
        assert_eq!(compute_triangular_weights(7.0).len(), 7);
    }

    #[test]
    fn integer_maxbas_symmetric() {
        let weights = compute_triangular_weights(4.0);
        assert_approx(weights[0], weights[3], 1e-10);
        assert_approx(weights[1], weights[2], 1e-10);
    }

    // -- Convolution --

    #[test]
    fn convolution_conserves_mass() {
        let weights = compute_triangular_weights(3.0);
        let mut buffer = [0.0; ROUTING_BUFFER_SIZE];

        let mut total_input = 0.0;
        let mut total_output = 0.0;

        // Add impulse
        let (new_buf, qsim) = convolve_triangular(100.0, &buffer, &weights);
        buffer = new_buf;
        total_input += 100.0;
        total_output += qsim;

        // Flush buffer
        for _ in 0..10 {
            let (new_buf, qsim) = convolve_triangular(0.0, &buffer, &weights);
            buffer = new_buf;
            total_output += qsim;
        }

        assert_approx(total_output, total_input, 1e-10);
    }

    #[test]
    fn convolution_first_step_zero_from_empty_buffer() {
        let weights = compute_triangular_weights(2.0);
        let buffer = [0.0; ROUTING_BUFFER_SIZE];

        let (_new_buf, qsim) = convolve_triangular(50.0, &buffer, &weights);
        assert_eq!(qsim, 0.0);
    }
}
