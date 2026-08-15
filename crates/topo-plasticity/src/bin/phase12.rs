// SPDX-License-Identifier: MIT
//! Runs Phase 12 and writes the results.

use experiment::dataset::{make_moons, train_test_split};
use predictive_coding::network::Config;
use serde::Serialize;
use topo_plasticity::plasticity::{Gate, FLOORS, SWEEP};
use topo_plasticity::run::{minimum_detectable_effect, train, Condition, ALPHA, SEEDS};

/// The three metrics of one floor setting, in seed order, alongside the floor
/// that produced them.
struct FloorRuns {
    floor: f64,
    losses: Vec<f64>,
    accuracies: Vec<f64>,
    gaps: Vec<f64>,
}

#[derive(Serialize)]
struct Cell {
    gate: String,
    floor: f64,
    multipliers: Vec<f64>,
    smallest_multiplier: f64,
    mean_val_loss: f64,
    mean_accuracy: f64,
    mean_gap: f64,
    /// Effect against the uniform baseline, the quantity Phase 10 reported.
    d_against_baseline: f64,
    p_against_baseline: f64,
    /// Paired against the same gate at floor zero, which is Phase 10's own
    /// condition run on the same seeds.
    d_paired_against_floor_zero: f64,
    p_paired_against_floor_zero: f64,
    p_holm_paired: f64,
    paired_test: String,
}

#[derive(Serialize)]
struct Phase12Report {
    seeds: usize,
    floors: Vec<f64>,
    baseline_mean_val_loss: f64,
    minimum_detectable_effect: f64,
    cells: Vec<Cell>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Config::default();
    println!("Phase 12: a non-zero floor on the graded plasticity gate");
    println!("  hypothesis motivated by a rejected hypothesis elsewhere, not by a rescue observed");

    let data = make_moons(2_000, 0.20, 20_260_813);
    let (train_set, val) = train_test_split(&data, 0.75, 20_260_814);
    let seeds: Vec<u64> = (0..SEEDS).map(|s| 700 + s as u64).collect();

    let uniform = Condition {
        label: "baseline",
        weights: None,
        gate: None,
        shuffled: false,
    };
    let baseline: Vec<f64> = seeds
        .iter()
        .map(|s| train(&uniform, *s, &cfg, &train_set, &val).final_val_loss)
        .collect();
    let baseline_mean = xstats::mean(&baseline);
    println!("\n  uniform baseline, mean validation loss {baseline_mean:.4}");

    let mut cells = Vec::new();
    for gate in SWEEP {
        println!("\n  gate {}", gate.label);
        println!("    floor   multipliers                       min      val loss   d vs base   paired d   paired p   holm");

        // Floor zero first: this is Phase 10's own condition, and every later
        // row is paired against it on the same seeds.
        let mut per_floor: Vec<FloorRuns> = Vec::new();
        for floor in FLOORS {
            let g = Gate { floor, ..gate };
            let condition = Condition {
                label: "graded-plasticity",
                weights: None,
                gate: Some(g),
                shuffled: false,
            };
            let records: Vec<_> = seeds
                .iter()
                .map(|s| train(&condition, *s, &cfg, &train_set, &val))
                .collect();
            per_floor.push(FloorRuns {
                floor,
                losses: records.iter().map(|r| r.final_val_loss).collect(),
                accuracies: records.iter().map(|r| r.final_val_accuracy).collect(),
                gaps: records.iter().map(|r| r.generalisation_gap).collect(),
            });
        }

        let reference = per_floor[0].losses.clone();
        let mut raw_p = Vec::new();
        for run in per_floor.iter().skip(1) {
            let losses = &run.losses;
            let diff: Vec<f64> = losses
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| a - b)
                .collect();
            let p = if xstats::shapiro_wilk(&diff).p_value > ALPHA {
                xstats::paired_t_test(losses, &reference).p_value
            } else {
                xstats::wilcoxon_signed_rank(losses, &reference).p_value
            };
            raw_p.push(p);
        }
        let holm = xstats::holm_adjust(&raw_p);

        for (index, run) in per_floor.iter().enumerate() {
            let (floor, losses, accs, gaps) = (&run.floor, &run.losses, &run.accuracies, &run.gaps);
            let g = Gate {
                floor: *floor,
                ..gate
            };
            let m = g.multipliers(cfg.layers.len() - 1);
            let smallest = m.iter().cloned().fold(f64::INFINITY, f64::min);
            let diff: Vec<f64> = losses
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| a - b)
                .collect();
            let (paired_d, paired_p, paired_test, p_holm) = if index == 0 {
                (0.0, f64::NAN, "reference".to_string(), f64::NAN)
            } else {
                let sd = xstats::std_dev(&diff);
                let d = if sd > 0.0 {
                    xstats::mean(&diff) / sd
                } else {
                    0.0
                };
                let normal = xstats::shapiro_wilk(&diff).p_value > ALPHA;
                (
                    d,
                    raw_p[index - 1],
                    if normal { "paired t" } else { "Wilcoxon" }.to_string(),
                    holm[index - 1],
                )
            };
            let against_baseline = xstats::welch_t_test(&baseline, losses);
            println!(
                "    {:>5.2}   {:<32}  {:>6.4}   {:>9.4}   {:>9.3}   {:>8.3}   {:>8.4}   {:>6.4}",
                floor,
                format!("{:.3?}", m),
                smallest,
                xstats::mean(losses),
                xstats::cohens_d(&baseline, losses),
                paired_d,
                paired_p,
                p_holm
            );
            cells.push(Cell {
                gate: gate.label.to_string(),
                floor: *floor,
                multipliers: m,
                smallest_multiplier: smallest,
                mean_val_loss: xstats::mean(losses),
                mean_accuracy: xstats::mean(accs),
                mean_gap: xstats::mean(gaps),
                d_against_baseline: xstats::cohens_d(&baseline, losses),
                p_against_baseline: against_baseline.p_value,
                d_paired_against_floor_zero: paired_d,
                p_paired_against_floor_zero: paired_p,
                p_holm_paired: p_holm,
                paired_test,
            });
        }
    }

    let report = Phase12Report {
        seeds: SEEDS,
        floors: FLOORS.to_vec(),
        baseline_mean_val_loss: baseline_mean,
        minimum_detectable_effect: minimum_detectable_effect(SEEDS, ALPHA),
        cells,
    };
    std::fs::create_dir_all("results")?;
    std::fs::write(
        "results/phase12_plasticity_floor.json",
        serde_json::to_string_pretty(&report)?,
    )?;
    println!("\n  written results/phase12_plasticity_floor.json");
    Ok(())
}
