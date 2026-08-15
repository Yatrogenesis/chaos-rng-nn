// SPDX-License-Identifier: MIT
//! Runs Phase 11 and writes the results.

use experiment::dataset::{make_moons, train_test_split};
use predictive_coding::network::Config;
use predictive_coding::run::{
    analyse, ConditionResult, MetricAnalysis, DATASET_SEED, NOISE, N_SAMPLES, SEEDS, SPLIT_SEED,
    TRAIN_FRACTION,
};
use predictive_coding::trajectory::{check_mechanism, conditions, OffsetCheck, DELTAS};
use serde::Serialize;

#[derive(Serialize)]
struct Phase11Report {
    deltas: Vec<usize>,
    seeds: usize,
    mechanism_checks: Vec<OffsetCheck>,
    conditions: Vec<Vec<ConditionResult>>,
    analyses: Vec<MetricAnalysis>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Config::default();
    println!("Phase 11: precision from one shared trajectory, read at a per-level offset");
    println!("  everything from Phase 9 is unchanged except where the numbers come from");

    println!("\n11a. Does the mechanism do what it says?");
    let mut checks = Vec::new();
    println!("  delta  generator        peak lag   peak corr   median L0   median L2   ok");
    for delta in DELTAS {
        for row in check_mechanism(delta, cfg.layers.len() - 1) {
            println!(
                "  {:>5}  {:<15} {:>9}   {:>9.4}   {:>9.4}   {:>9.4}   {}",
                row.delta,
                row.condition,
                row.peak_lag,
                row.peak_value,
                row.median_level_0,
                row.median_deepest,
                row.peak_at_delta && row.medians_hold > 0.5
            );
            checks.push(row);
        }
    }
    if !checks
        .iter()
        .all(|r| r.peak_at_delta && r.medians_hold > 0.5)
    {
        eprintln!("\n  The mechanism check failed. Comparing conditions would measure");
        eprintln!("  something other than what is described, so 11b is not run.");
        std::process::exit(1);
    }
    println!("  mechanism verified at every offset");

    println!("\n11b. Same comparison as Phase 9, new sampling scheme.");
    let data = make_moons(N_SAMPLES, NOISE, DATASET_SEED);
    let (train_set, val) = train_test_split(&data, TRAIN_FRACTION, SPLIT_SEED);

    let mut blocks = Vec::new();
    let mut analyses = Vec::new();
    for delta in DELTAS {
        println!("\n  offset delta = {delta}");
        let labels: Vec<&'static str> = conditions(0, delta).iter().map(|c| c.label()).collect();
        let results: Vec<ConditionResult> = labels
            .iter()
            .enumerate()
            .map(|(index, label)| ConditionResult {
                condition: (*label).to_string(),
                runs: (0..SEEDS)
                    .map(|s| {
                        let seed = 700 + s as u64;
                        let mut built = conditions(seed, delta);
                        let c = built.get_mut(index).expect("condition list is stable");
                        predictive_coding::run::train(c.as_mut(), seed, &cfg, &train_set, &val)
                    })
                    .collect(),
            })
            .collect();

        println!("    condition            val loss    accuracy         gap");
        for c in &results {
            let n = c.runs.len() as f64;
            println!(
                "    {:<18} {:>10.4}  {:>10.4}  {:>10.4}",
                c.condition,
                c.runs.iter().map(|r| r.final_val_loss).sum::<f64>() / n,
                c.runs.iter().map(|r| r.final_val_accuracy).sum::<f64>() / n,
                c.runs.iter().map(|r| r.generalisation_gap).sum::<f64>() / n
            );
        }
        for metric in ["final_val_loss", "final_val_accuracy", "generalisation_gap"] {
            let a = analyse(&results, metric);
            println!(
                "    {} omnibus {}: {:.4}, p = {:.4}, any significant: {}",
                a.metric, a.omnibus_test, a.omnibus_statistic, a.omnibus_p, a.any_significant
            );
            let worst = a.comparisons.iter().max_by(|x, y| {
                x.effect_size
                    .abs()
                    .partial_cmp(&y.effect_size.abs())
                    .unwrap()
            });
            if let Some(w) = worst {
                println!(
                    "      largest effect: {} d = {:.3}, holm p = {:.4}; control d = {:.3}; detectable d = {:.3}",
                    w.condition, w.effect_size, w.p_holm, a.negative_control_effect, a.minimum_detectable_effect
                );
            }
            analyses.push(a);
        }
        blocks.push(results);
    }

    let report = Phase11Report {
        deltas: DELTAS.to_vec(),
        seeds: SEEDS,
        mechanism_checks: checks,
        conditions: blocks,
        analyses,
    };
    std::fs::create_dir_all("results")?;
    std::fs::write(
        "results/phase11_shared_trajectory.json",
        serde_json::to_string_pretty(&report)?,
    )?;
    println!("\n  written results/phase11_shared_trajectory.json");
    Ok(())
}
