// SPDX-License-Identifier: MIT
//! Command line entry point for the experiment.
//!
//! Usage:
//!   experiment phase0   run the generator qualification battery
//!   experiment phase1   run the MLP comparison and write results
//!   experiment analyse  read the results and apply the hypothesis tests

mod analysis;
mod dataset;
mod mlp;
mod runner;

use chaos_rng::{stats, ChaChaRng, LorenzRng, RngKind};
use dataset::{make_moons, train_test_split};
use mlp::Config;
use runner::{run_condition, RunRecord};
use serde::{Deserialize, Serialize};
use std::io::Write;

/// Seed for dataset generation, fixed and separate from the training seeds so
/// that both conditions see identical data.
const DATASET_SEED: u64 = 20_260_813;
/// Seed for the train and validation partition, likewise fixed.
const SPLIT_SEED: u64 = 20_260_814;
/// Samples in the synthetic dataset.
const N_SAMPLES: usize = 2_000;
/// Standard deviation of the noise added to the moons.
const NOISE: f64 = 0.20;
/// Fraction of the data used for training.
const TRAIN_FRACTION: f64 = 0.75;
/// Runs per condition.
const N_RUNS: usize = 10;
/// Samples drawn for the Phase 0 battery.
const BATTERY_SAMPLES: usize = 1_000_000;
/// Bins used by the chi-squared uniformity test.
const BATTERY_BINS: usize = 1_000;

/// Serialisable form of a Phase 0 battery result.
#[derive(Debug, Serialize, Deserialize)]
struct BatterySummary {
    rng: String,
    samples: usize,
    bins: usize,
    chi_squared: f64,
    degrees_of_freedom: usize,
    chi_p_value: f64,
    autocorrelation_lag_1_to_10: Vec<f64>,
    mean: f64,
    expected_mean: f64,
    variance: f64,
    expected_variance: f64,
    passes: bool,
}

fn phase0() -> std::io::Result<()> {
    println!("Phase 0: generator qualification, {BATTERY_SAMPLES} samples, {BATTERY_BINS} bins");
    let mut summaries = Vec::new();

    for kind in [RngKind::Lorenz, RngKind::ChaCha] {
        let samples: Vec<f64> = match kind {
            RngKind::Lorenz => {
                let mut r = LorenzRng::from_seed(DATASET_SEED);
                (0..BATTERY_SAMPLES).map(|_| r.next_f64()).collect()
            }
            RngKind::ChaCha => {
                let mut r = ChaChaRng::from_seed(DATASET_SEED);
                (0..BATTERY_SAMPLES).map(|_| r.next_f64()).collect()
            }
        };
        let report = stats::run_battery(&samples, BATTERY_BINS);
        println!(
            "  {:<8} chi2 = {:>10.3}  p = {:.4}  mean = {:.6}  var = {:.6}  max |acf| = {:.5}  {}",
            kind.as_str(),
            report.chi.statistic,
            report.chi.p_value,
            report.mean,
            report.variance,
            report
                .autocorrelations
                .iter()
                .fold(0.0f64, |m, v| m.max(v.abs())),
            if report.passes() { "PASS" } else { "FAIL" }
        );
        summaries.push(BatterySummary {
            rng: kind.as_str().to_string(),
            samples: report.n,
            bins: BATTERY_BINS,
            chi_squared: report.chi.statistic,
            degrees_of_freedom: report.chi.degrees_of_freedom,
            chi_p_value: report.chi.p_value,
            autocorrelation_lag_1_to_10: report.autocorrelations.clone(),
            mean: report.mean,
            expected_mean: 0.5,
            variance: report.variance,
            expected_variance: 1.0 / 12.0,
            passes: report.passes(),
        });
    }

    std::fs::create_dir_all("results")?;
    let json = serde_json::to_string_pretty(&summaries).expect("summaries are serialisable");
    std::fs::write("results/phase0_battery.json", json)?;
    println!("  written to results/phase0_battery.json");

    if !summaries.iter().all(|s| s.passes) {
        eprintln!("Phase 0 gate failed: a generator did not qualify, later phases are blocked");
        std::process::exit(1);
    }
    Ok(())
}

fn phase1() -> std::io::Result<()> {
    let cfg = Config::default();
    println!("Phase 1: MLP on two moons");
    println!(
        "  dataset: {N_SAMPLES} samples, noise {NOISE}, train fraction {TRAIN_FRACTION}, seeds {DATASET_SEED} and {SPLIT_SEED}"
    );
    println!(
        "  network: hidden {:?}, dropout {}, learning rate {}, batch {}, epochs {}",
        cfg.hidden, cfg.dropout, cfg.learning_rate, cfg.batch_size, cfg.epochs
    );
    println!("  runs per condition: {N_RUNS}");

    let data = make_moons(N_SAMPLES, NOISE, DATASET_SEED);
    let (train, val) = train_test_split(&data, TRAIN_FRACTION, SPLIT_SEED);
    println!(
        "  split: {} training, {} validation",
        train.len(),
        val.len()
    );

    // The same seeds are given to both conditions.
    let seeds: Vec<u64> = (0..N_RUNS as u64).map(|i| 1_000 + i).collect();

    let mut all: Vec<RunRecord> = Vec::new();
    for kind in [RngKind::Lorenz, RngKind::ChaCha] {
        print!("  running {:<8} ", kind.as_str());
        std::io::stdout().flush()?;
        let started = std::time::Instant::now();
        let records = run_condition(kind, &seeds, &train, &val, cfg);
        println!(
            "done in {:.1} s, mean validation loss {:.4}",
            started.elapsed().as_secs_f64(),
            records.iter().map(|r| r.final_val_loss).sum::<f64>() / records.len() as f64
        );
        all.extend(records);
    }

    std::fs::create_dir_all("results")?;
    let json = serde_json::to_string_pretty(&all).expect("records are serialisable");
    std::fs::write("results/phase1_runs.json", json)?;

    let mut w = csv::Writer::from_path("results/phase1_runs.csv")?;
    w.write_record([
        "rng",
        "seed",
        "final_val_loss",
        "final_val_accuracy",
        "final_train_loss",
        "generalisation_gap",
        "epochs_to_threshold",
        "wall_clock_seconds",
        "weight_hash",
    ])?;
    for r in all.iter() {
        w.write_record([
            r.rng.clone(),
            r.seed.to_string(),
            format!("{:.10}", r.final_val_loss),
            format!("{:.10}", r.final_val_accuracy),
            format!("{:.10}", r.final_train_loss),
            format!("{:.10}", r.generalisation_gap),
            r.epochs_to_threshold
                .map(|v| v.to_string())
                .unwrap_or_else(|| "never".to_string()),
            format!("{:.4}", r.wall_clock_seconds),
            r.weight_hash.clone(),
        ])?;
    }
    w.flush()?;
    println!("  written to results/phase1_runs.json and results/phase1_runs.csv");

    // Reproducibility check, part of the protocol rather than an afterthought.
    println!("  reproducibility: repeating the first configuration of each condition");
    for kind in [RngKind::Lorenz, RngKind::ChaCha] {
        let again = runner::run_once(kind, seeds[0], &train, &val, cfg);
        let original = all
            .iter()
            .find(|r| r.rng == kind.as_str() && r.seed == seeds[0])
            .expect("the first run of each condition was recorded");
        let identical = again.weight_hash == original.weight_hash
            && again.final_val_loss.to_bits() == original.final_val_loss.to_bits();
        println!(
            "    {:<8} {}",
            kind.as_str(),
            if identical {
                "identical, bit for bit"
            } else {
                "DIFFERENT, reproducibility is broken"
            }
        );
        if !identical {
            std::process::exit(1);
        }
    }
    Ok(())
}

/// Reads the recorded runs, applies the hypothesis tests and writes the figures.
fn analyse() -> std::io::Result<()> {
    let raw = std::fs::read_to_string("results/phase1_runs.json")?;
    let records: Vec<RunRecord> = serde_json::from_str(&raw).expect("results file is well formed");

    let pick = |rng: &str, f: fn(&RunRecord) -> f64| -> Vec<f64> {
        records.iter().filter(|r| r.rng == rng).map(f).collect()
    };

    /// A named metric and the accessor that extracts it from a run.
    type Metric = (&'static str, fn(&RunRecord) -> f64);

    let metrics: Vec<Metric> = vec![
        ("final_val_loss", |r| r.final_val_loss),
        ("final_val_accuracy", |r| r.final_val_accuracy),
        ("generalisation_gap", |r| r.generalisation_gap),
        ("epochs_to_threshold", |r| {
            r.epochs_to_threshold.map(|v| v as f64).unwrap_or(f64::NAN)
        }),
        ("wall_clock_seconds", |r| r.wall_clock_seconds),
        // Post-hoc metric, added after observing that the pre-registered
        // convergence threshold of 0.35 on validation loss was reached by every
        // run in its first epoch and therefore carried no information. This
        // replacement is computed from the training curves already recorded,
        // and is reported as exploratory rather than confirmatory.
        ("epochs_to_train_loss_below_0.10_post_hoc", |r| {
            r.train_loss_per_epoch
                .iter()
                .position(|&v| v < 0.10)
                .map(|i| (i + 1) as f64)
                .unwrap_or(f64::NAN)
        }),
    ];

    let mut summaries = Vec::new();
    let mut comparisons = Vec::new();

    println!(
        "Analysis of {} runs, alpha = {}",
        records.len(),
        analysis::ALPHA
    );
    println!();
    for (name, f) in metrics.iter() {
        let l = pick("lorenz", *f);
        let c = pick("chacha8", *f);
        if l.iter().chain(c.iter()).any(|v| v.is_nan()) {
            println!("  {name}: skipped, at least one run never reached the threshold");
            continue;
        }
        let sl = analysis::summarise(
            &records
                .iter()
                .filter(|r| r.rng == "lorenz")
                .cloned()
                .collect::<Vec<_>>(),
            "lorenz",
            name,
            &l,
        );
        let sc = analysis::summarise(
            &records
                .iter()
                .filter(|r| r.rng == "chacha8")
                .cloned()
                .collect::<Vec<_>>(),
            "chacha8",
            name,
            &c,
        );
        let cmp = analysis::compare(name, &l, &c);

        println!("  {name}");
        println!(
            "    lorenz   mean {:>12.6}  sd {:>10.6}  range [{:.6}, {:.6}]",
            sl.mean, sl.std_dev, sl.min, sl.max
        );
        println!(
            "    chacha8  mean {:>12.6}  sd {:>10.6}  range [{:.6}, {:.6}]",
            sc.mean, sc.std_dev, sc.min, sc.max
        );
        println!("    {}", cmp.test_used);
        println!(
            "    statistic {:.6}, p = {:.6}, Cohen's d = {:.4}  ->  {}",
            cmp.statistic,
            cmp.p_value,
            cmp.cohens_d,
            if cmp.rejects_h0 {
                "H0 REJECTED"
            } else {
                "H0 not rejected"
            }
        );
        println!();
        summaries.push(sl);
        summaries.push(sc);
        comparisons.push(cmp);
    }

    std::fs::create_dir_all("results")?;
    std::fs::write(
        "results/phase1_summaries.json",
        serde_json::to_string_pretty(&summaries).expect("serialisable"),
    )?;
    std::fs::write(
        "results/phase1_comparisons.json",
        serde_json::to_string_pretty(&comparisons).expect("serialisable"),
    )?;

    std::fs::create_dir_all("figures")?;
    analysis::plot_loss_curves(&records, "figures/phase1_loss_curves.png")
        .expect("loss curve figure");
    analysis::plot_final_losses(&records, "figures/phase1_final_losses.png")
        .expect("final loss figure");
    println!("  written results/phase1_summaries.json, results/phase1_comparisons.json");
    println!("  written figures/phase1_loss_curves.png, figures/phase1_final_losses.png");
    Ok(())
}

fn main() -> std::io::Result<()> {
    let arg = std::env::args().nth(1).unwrap_or_default();
    match arg.as_str() {
        "phase0" => phase0(),
        "phase1" => phase1(),
        "analyse" => analyse(),
        other => {
            eprintln!("unknown command {other:?}; expected phase0, phase1 or analyse");
            std::process::exit(2);
        }
    }
}
