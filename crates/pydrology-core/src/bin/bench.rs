/// Pure Rust core benchmarks for all pydrology models.
///
/// Uses std::time::Instant for timing, a deterministic LCG PRNG for data generation,
/// and std::hint::black_box to prevent dead-code elimination.
use std::hint::black_box;
use std::time::{Duration, Instant};

use pydrology_core::cemaneige::coupled::coupled_run;
use pydrology_core::gr2m::params::Parameters as GR2MParameters;
use pydrology_core::gr2m::run as gr2m_run;
use pydrology_core::gr6j::params::Parameters as GR6JParameters;
use pydrology_core::gr6j::run as gr6j_run;
use pydrology_core::hbv_light::params::Parameters as HBVParameters;
use pydrology_core::hbv_light::run as hbv_run;

const REPEATS: usize = 7;

/// Simple LCG PRNG for deterministic data generation.
fn make_data(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut state = seed;
    let mut next_f64 = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as f64 / (1u64 << 31) as f64
    };

    let precip: Vec<f64> = (0..n).map(|_| next_f64() * 10.0).collect();
    let pet: Vec<f64> = (0..n).map(|_| 0.5 + next_f64() * 4.5).collect();
    let temp: Vec<f64> = (0..n).map(|_| -10.0 + next_f64() * 30.0).collect();
    (precip, pet, temp)
}

/// Run a closure `REPEATS` times, return the median duration.
fn median_time<F: FnMut()>(mut f: F) -> Duration {
    let mut times: Vec<Duration> = (0..REPEATS)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .collect();
    times.sort();
    times[REPEATS / 2]
}

fn bench_gr2m(sizes: &[usize]) -> Vec<(&'static str, usize, Duration)> {
    let params = GR2MParameters::new(500.0, 1.0);
    let mut results = Vec::new();

    for &n in sizes {
        let (precip, pet, _) = make_data(n, 42);

        // Warmup
        black_box(gr2m_run::run(&params, &precip, &pet, None));

        let dur = median_time(|| {
            black_box(gr2m_run::run(&params, &precip, &pet, None));
        });
        results.push(("gr2m", n, dur));
    }
    results
}

fn bench_gr6j(sizes: &[usize]) -> Vec<(&'static str, usize, Duration)> {
    let params = GR6JParameters::new(350.0, 0.0, 90.0, 1.7, 0.0, 5.0);
    let mut results = Vec::new();

    for &n in sizes {
        let (precip, pet, _) = make_data(n, 42);

        // Warmup
        black_box(gr6j_run::run(&params, &precip, &pet, None));

        let dur = median_time(|| {
            black_box(gr6j_run::run(&params, &precip, &pet, None));
        });
        results.push(("gr6j", n, dur));
    }
    results
}

fn bench_hbv(sizes: &[usize]) -> Vec<(&'static str, usize, Duration)> {
    let params = HBVParameters::new(
        0.0, 3.0, 1.0, 0.1, 0.05, 250.0, 0.9, 2.0, 0.4, 0.1, 0.01, 1.0, 20.0, 2.5,
    )
    .unwrap();
    let mut results = Vec::new();

    for &n in sizes {
        let (precip, pet, temp) = make_data(n, 42);

        // Warmup
        black_box(hbv_run::run(
            &params, &precip, &pet, &temp, None, 1, None, None, None, None, None, None,
        ));

        let dur = median_time(|| {
            black_box(hbv_run::run(
                &params, &precip, &pet, &temp, None, 1, None, None, None, None, None, None,
            ));
        });
        results.push(("hbv", n, dur));
    }
    results
}

fn bench_cemaneige(sizes: &[usize]) -> Vec<(&'static str, usize, Duration)> {
    let gr6j_params = GR6JParameters::new(350.0, 0.0, 90.0, 1.7, 0.0, 5.0);
    let ctg = 0.97;
    let kf = 2.5;
    let layer_elevations = vec![0.0];
    let layer_fractions = vec![1.0];
    let mut results = Vec::new();

    for &n in sizes {
        let (precip, pet, temp) = make_data(n, 42);

        // Warmup
        black_box(coupled_run(
            &gr6j_params,
            ctg,
            kf,
            &precip,
            &pet,
            &temp,
            None,
            None,
            1,
            &layer_elevations,
            &layer_fractions,
            f64::NAN,
            0.6,
            0.00041,
            150.0,
            false,
        ));

        let dur = median_time(|| {
            black_box(coupled_run(
                &gr6j_params,
                ctg,
                kf,
                &precip,
                &pet,
                &temp,
                None,
                None,
                1,
                &layer_elevations,
                &layer_fractions,
                f64::NAN,
                0.6,
                0.00041,
                150.0,
                false,
            ));
        });
        results.push(("cemaneige", n, dur));
    }
    results
}

fn main() {
    println!("Pure Rust Core Benchmarks");
    println!("============================================================");
    println!("{:<18} {:>6}   {:>12}", "Model", "N", "Median (ms)");
    println!("--------------------------------------------");

    let mut all_results: Vec<(&str, usize, Duration)> = Vec::new();

    all_results.extend(bench_gr2m(&[120, 1200, 12000]));
    all_results.extend(bench_gr6j(&[3650, 36500]));
    all_results.extend(bench_hbv(&[3650, 36500]));
    all_results.extend(bench_cemaneige(&[3650, 36500]));

    for (model, n, dur) in &all_results {
        let ms = dur.as_secs_f64() * 1000.0;
        println!("{:<18} {:>6}      {:>8.2}", model, n, ms);
    }

    println!("============================================================");
}
