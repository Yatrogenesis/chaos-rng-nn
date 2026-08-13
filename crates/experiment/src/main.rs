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
mod phdim;
mod runner;
mod topology;

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

/// Phase 0.5: does either generator leave a topological fingerprint?
fn phase05() -> std::io::Result<()> {
    use nalgebra::DMatrix;
    use topology::*;

    // The series length only has to be long enough for the embedding search;
    // the clouds themselves are subsampled to CLOUD_POINTS.
    const SERIES: usize = 20_000;

    println!("Phase 0.5: topological fingerprint");
    println!("  cloud points {CLOUD_POINTS}, null resamples {NULL_RESAMPLES}");

    let lorenz_series: Vec<f64> = {
        let mut r = LorenzRng::from_seed(DATASET_SEED);
        (0..SERIES).map(|_| r.next_f64()).collect()
    };
    let chacha_series: Vec<f64> = {
        let mut r = ChaChaRng::from_seed(DATASET_SEED);
        (0..SERIES).map(|_| r.next_f64()).collect()
    };

    // Embedding parameters are chosen from the Lorenz stream and then applied
    // unchanged to every condition, so the comparison is not confounded by
    // different embeddings.
    let params = choose_embedding(&lorenz_series, 12, 8);
    println!(
        "  embedding from AMI and FNN on the Lorenz stream: delay {}, dimension {}",
        params.delay, params.dimension
    );
    println!(
        "    AMI at delays 1..: {:?}",
        params
            .ami_curve
            .iter()
            .map(|v| (v * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );
    println!(
        "    FNN fraction at dimensions 1..: {:?}",
        params
            .fnn_curve
            .iter()
            .map(|v| (v * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );

    let mut rows: Vec<TopologyRow> = Vec::new();

    // Positive control: the raw attractor. Its geometry must dominate.
    let raw: DMatrix<f64> = {
        let mut r = LorenzRng::from_seed(DATASET_SEED);
        let mut data = Vec::with_capacity(CLOUD_POINTS * 3);
        for _ in 0..CLOUD_POINTS {
            let s = r.next_raw_state();
            data.push(s.x);
            data.push(s.y);
            data.push(s.z);
        }
        DMatrix::from_row_slice(CLOUD_POINTS, 3, &data)
    };
    let raw_radius = adaptive_radius(&raw);
    let raw_h1 = total_h1_persistence(&raw, raw_radius);
    println!("  positive control, raw attractor states: total H1 = {raw_h1:.4}");
    rows.push(TopologyRow {
        label: "lorenz_raw_states_xyz".into(),
        points: CLOUD_POINTS,
        dimension: 3,
        delay: 0,
        radius: raw_radius,
        total_h1: raw_h1,
        p_value: None,
    });

    // The extraction, stage by stage.
    let stage_scaled: Vec<f64> = {
        let mut r = LorenzRng::from_seed(DATASET_SEED);
        (0..SERIES).map(|_| r.next_stage_scaled()).collect()
    };
    let stage_fraction: Vec<f64> = {
        let mut r = LorenzRng::from_seed(DATASET_SEED);
        (0..SERIES).map(|_| r.next_stage_fraction()).collect()
    };

    let measure = |label: &str, series: &[f64], rows: &mut Vec<TopologyRow>| -> f64 {
        let cloud = takens_embedding(series, params.dimension, params.delay, CLOUD_POINTS);
        let radius = adaptive_radius(&cloud);
        let h1 = total_h1_persistence(&cloud, radius);
        println!("  {label:<34} total H1 = {h1:.4}");
        rows.push(TopologyRow {
            label: label.into(),
            points: CLOUD_POINTS,
            dimension: params.dimension,
            delay: params.delay,
            radius,
            total_h1: h1,
            p_value: None,
        });
        h1
    };

    measure("stage 1, scaled coordinate", &stage_scaled, &mut rows);
    measure("stage 2, fractional part", &stage_fraction, &mut rows);
    let lorenz_h1 = measure("stage 3, after SplitMix64", &lorenz_series, &mut rows);
    let chacha_h1 = measure("chacha8 control", &chacha_series, &mut rows);

    // Null distribution from uniform noise of the same shape.
    print!("  building the null from {NULL_RESAMPLES} uniform clouds ");
    std::io::stdout().flush()?;
    let mut null = Vec::with_capacity(NULL_RESAMPLES);
    for k in 0..NULL_RESAMPLES {
        let mut r = ChaChaRng::from_seed(900_000 + k as u64);
        let n = CLOUD_POINTS * params.dimension;
        let data: Vec<f64> = (0..n).map(|_| r.next_f64()).collect();
        let cloud = DMatrix::from_row_slice(CLOUD_POINTS, params.dimension, &data);
        let radius = adaptive_radius(&cloud);
        null.push(total_h1_persistence(&cloud, radius));
        print!(".");
        std::io::stdout().flush()?;
    }
    println!();
    let null_mean = xstats::mean(&null);
    let null_sd = xstats::std_dev(&null);
    println!("  null: mean {null_mean:.4}, sd {null_sd:.4}");

    let p_lorenz = empirical_p_value(lorenz_h1, &null);
    let p_chacha = empirical_p_value(chacha_h1, &null);
    let p_raw = empirical_p_value(raw_h1, &null);
    println!("  empirical p-values against the uniform null:");
    println!("    raw attractor  H1 = {raw_h1:8.4}  p = {p_raw:.4}");
    println!("    lorenz stream  H1 = {lorenz_h1:8.4}  p = {p_lorenz:.4}");
    println!("    chacha8 stream H1 = {chacha_h1:8.4}  p = {p_chacha:.4}");

    for row in rows.iter_mut() {
        row.p_value = Some(empirical_p_value(row.total_h1, &null));
    }

    #[derive(serde::Serialize)]
    struct Phase05 {
        embedding: EmbeddingParams,
        rows: Vec<TopologyRow>,
        null_values: Vec<f64>,
        null_mean: f64,
        null_std_dev: f64,
    }
    std::fs::create_dir_all("results")?;
    std::fs::write(
        "results/phase05_topology.json",
        serde_json::to_string_pretty(&Phase05 {
            embedding: params,
            rows,
            null_values: null,
            null_mean,
            null_std_dev: null_sd,
        })
        .expect("serialisable"),
    )?;
    println!("  written results/phase05_topology.json");
    Ok(())
}

/// Phase 3: fractal dimension of the optimisation trajectory.
fn phase3() -> std::io::Result<()> {
    use chaos_rng::Rng;

    let cfg = Config::default();
    let data = make_moons(N_SAMPLES, NOISE, DATASET_SEED);
    let (train, val) = train_test_split(&data, TRAIN_FRACTION, SPLIT_SEED);
    let seeds: Vec<u64> = (0..N_RUNS as u64).map(|i| 1_000 + i).collect();

    println!("Phase 3: persistent-homology dimension of the training trajectory");

    // The published Phase 1 records, used to prove the re-run is the same run.
    let published: Vec<RunRecord> = {
        let raw = std::fs::read_to_string("results/phase1_runs.json")?;
        serde_json::from_str(&raw).expect("phase 1 results are present and well formed")
    };

    // Calibration at the sample size actually available. The estimator was
    // validated on clouds of 800 points; a trajectory of 60 epochs gives 60.
    // Rather than assume the estimate degrades gracefully, measure it.
    {
        let mut rng = Rng::new(chaos_rng::RngKind::ChaCha, 4242);
        let square: Vec<Vec<f64>> = (0..60)
            .map(|_| vec![rng.next_f64(), rng.next_f64()])
            .collect();
        let line: Vec<Vec<f64>> = {
            let mut r = Rng::new(chaos_rng::RngKind::ChaCha, 99);
            (0..60).map(|_| vec![r.next_f64()]).collect()
        };
        let sizes = [15usize, 20, 30, 40, 50, 60];
        let mut r1 = Rng::new(chaos_rng::RngKind::ChaCha, 7);
        let sq = phdim::estimate(&square, &sizes, 20, &mut r1);
        let mut r2 = Rng::new(chaos_rng::RngKind::ChaCha, 7);
        let ln = phdim::estimate(&line, &sizes, 20, &mut r2);
        println!(
            "  calibration at 60 points: uniform square gives {:.3} (true 2), uniform line gives {:.3} (true 1)",
            sq.dimension, ln.dimension
        );
        println!(
            "    fit quality r2: square {:.4}, line {:.4}",
            sq.r_squared, ln.r_squared
        );
    }

    let sizes = [15usize, 20, 30, 40, 50, 60];
    let mut rows = Vec::new();
    let mut mismatch = false;

    for kind in [RngKind::Lorenz, RngKind::ChaCha] {
        for &seed in seeds.iter() {
            let (record, snapshots) =
                runner::run_once_with_snapshots(kind, seed, &train, &val, cfg);
            let prior = published
                .iter()
                .find(|r| r.rng == kind.as_str() && r.seed == seed)
                .expect("every published run is found");
            // The parameter hash is the authoritative check: it covers every
            // weight and bias exactly, so identical hashes mean the two runs
            // followed the same trajectory. The loss is compared with a
            // tolerance rather than by bit pattern because it has been through
            // JSON, and a round trip through a decimal representation can move
            // the last unit in the last place. That is a property of the file
            // format, not of the computation, and treating it as a determinism
            // failure would be wrong.
            let loss_gap = (record.final_val_loss - prior.final_val_loss).abs();
            if record.weight_hash != prior.weight_hash
                || loss_gap > 1e-15 * prior.final_val_loss.abs().max(1.0)
            {
                eprintln!(
                    "  determinism broken for {} seed {seed}: hashes {} and {}, loss gap {:.3e}",
                    kind.as_str(),
                    &record.weight_hash[..12],
                    &prior.weight_hash[..12],
                    loss_gap
                );
                mismatch = true;
            }
            let mut est_rng = Rng::new(chaos_rng::RngKind::ChaCha, 31_337);
            let est = phdim::estimate(&snapshots, &sizes, 20, &mut est_rng);
            rows.push((
                kind.as_str().to_string(),
                seed,
                est,
                prior.generalisation_gap,
                prior.final_val_loss,
            ));
        }
        println!("  {} done", kind.as_str());
    }

    if mismatch {
        eprintln!("  aborting: the re-run does not reproduce the published Phase 1 numbers");
        std::process::exit(1);
    }
    println!("  all 20 runs reproduce the published Phase 1 parameter hashes exactly");

    let dims = |rng: &str| -> Vec<f64> {
        rows.iter()
            .filter(|r| r.0 == rng)
            .map(|r| r.2.dimension)
            .collect()
    };
    let l = dims("lorenz");
    let c = dims("chacha8");
    let cmp = analysis::compare("ph_dimension", &l, &c);

    println!();
    println!("  PH-dim by condition");
    println!(
        "    lorenz   mean {:.4}  sd {:.4}",
        xstats::mean(&l),
        xstats::std_dev(&l)
    );
    println!(
        "    chacha8  mean {:.4}  sd {:.4}",
        xstats::mean(&c),
        xstats::std_dev(&c)
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

    let all_dims: Vec<f64> = rows.iter().map(|r| r.2.dimension).collect();
    let all_gaps: Vec<f64> = rows.iter().map(|r| r.3).collect();
    let r_p = phdim::pearson(&all_dims, &all_gaps);
    let r_s = phdim::spearman(&all_dims, &all_gaps);
    let n = all_dims.len();
    println!();
    println!("  Correlation of PH-dim with the Phase 1 generalisation gap, all {n} runs pooled");
    println!(
        "    Pearson  r = {:.4}, p = {:.4}",
        r_p,
        phdim::correlation_p_value(r_p, n)
    );
    println!(
        "    Spearman rho = {:.4}, p = {:.4}",
        r_s,
        phdim::correlation_p_value(r_s, n)
    );
    let mean_r2 = rows.iter().map(|r| r.2.r_squared).sum::<f64>() / n as f64;
    println!("    mean r2 of the log-log fits: {mean_r2:.4}");

    #[derive(serde::Serialize)]
    struct Row {
        rng: String,
        seed: u64,
        ph_dimension: f64,
        slope: f64,
        r_squared: f64,
        sample_sizes: Vec<usize>,
        mst_weights: Vec<f64>,
        generalisation_gap: f64,
        final_val_loss: f64,
    }
    let out: Vec<Row> = rows
        .iter()
        .map(|(rng, seed, est, gap, loss)| Row {
            rng: rng.clone(),
            seed: *seed,
            ph_dimension: est.dimension,
            slope: est.slope,
            r_squared: est.r_squared,
            sample_sizes: est.sample_sizes.clone(),
            mst_weights: est.mst_weights.clone(),
            generalisation_gap: *gap,
            final_val_loss: *loss,
        })
        .collect();
    std::fs::create_dir_all("results")?;
    std::fs::write(
        "results/phase3_phdim.json",
        serde_json::to_string_pretty(&out).expect("serialisable"),
    )?;
    std::fs::write(
        "results/phase3_comparison.json",
        serde_json::to_string_pretty(&cmp).expect("serialisable"),
    )?;
    println!("  written results/phase3_phdim.json and results/phase3_comparison.json");
    Ok(())
}

fn main() -> std::io::Result<()> {
    let arg = std::env::args().nth(1).unwrap_or_default();
    match arg.as_str() {
        "phase0" => phase0(),
        "phase1" => phase1(),
        "analyse" => analyse(),
        "phase05" => phase05(),
        "phase3" => phase3(),
        other => {
            eprintln!("unknown command {other:?}; expected phase0, phase1 or analyse");
            std::process::exit(2);
        }
    }
}
