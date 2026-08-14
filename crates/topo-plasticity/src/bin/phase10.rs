// SPDX-License-Identifier: MIT
//! Runs Phase 10 and writes the results.

use topo_plasticity::plasticity;
use topo_plasticity::resilience;
use topo_plasticity::run::{analyse, run_conditions, Phase10Report, RECOMPUTE_EVERY, SEEDS};

/// The hyperparameter pairs the comparison is repeated at.
///
/// One primary setting and three that vary a single choice each, so the
/// sensitivity of the conclusion to the threshold and to the gate can be read
/// off rather than assumed.
///
/// The `loops-and-voids` weighting is **not** among them, and the reason is
/// cost rather than choice. Carrying a dimension-two term into training means
/// building the Rips complex one dimension higher, which on thirty-two nodes
/// takes 3.49 seconds per evaluation against 47 milliseconds one dimension
/// lower: a seventy-five-fold wall that puts the sweep at roughly five hours.
/// The reduction in the persistence implementation scans all previous columns
/// inside a loop over columns, so it is quadratic in the simplex count, and
/// dimension three over thirty-two points is near thirty-six thousand
/// tetrahedra.
///
/// The dimension-two term is still evaluated, in the Phase 10a calibration,
/// where twenty evaluations are affordable. What is untested is whether the
/// Phase 10c conclusion would change if the signal counted voids as well as
/// loops. Two things bound that gap and both are in the report: the topological
/// condition shows no effect at either threshold, with effect sizes at or below
/// the shuffled control; and in the calibration the dimension-two term barely
/// moves, a single bar clearing the threshold in one of four synthetic cases.
const SETTINGS: [(usize, usize); 4] = [(1, 0), (4, 0), (1, 1), (1, 2)];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Phase 10: topological resilience and graded plasticity as design hypotheses");
    println!("  neither is a validated result; both are hypotheses of this project");

    println!("\n10a. Does T(S) respond to structure at all?");
    let calibration = resilience::calibrate();
    println!("  weighting          modular      ring   mod-ring     noise   loops>noise");
    for r in &calibration.rows {
        println!(
            "  {:<16} {:>9.4} {:>9.4} {:>10.4} {:>9.4}   {}",
            r.weights, r.modular, r.ring, r.modular_ring, r.noise, r.loops_exceed_noise
        );
    }
    if !calibration.passes {
        eprintln!("\n  Calibration failed. T(S) does not separate loops from noise, so it is");
        eprintln!("  not measuring what its name says and 10c is not run.");
        std::process::exit(1);
    }
    println!("  calibration passed for every weighting carried forward");

    println!("\n10c. Conditions against the Phase 9 baseline, {SEEDS} seeds each.");
    println!("  signal recomputed every {RECOMPUTE_EVERY} weight updates");
    let mut blocks = Vec::new();
    let mut analyses = Vec::new();
    for (wi, gi) in SETTINGS {
        let weights = resilience::SWEEP[wi];
        let gate = plasticity::SWEEP[gi];
        println!(
            "\n  setting: weighting {}, gate {}",
            weights.label, gate.label
        );
        let results = run_conditions(weights, gate, SEEDS);
        println!("    condition            val loss    accuracy         gap");
        for c in &results {
            let n = c.runs.len() as f64;
            let loss = c.runs.iter().map(|r| r.final_val_loss).sum::<f64>() / n;
            let acc = c.runs.iter().map(|r| r.final_val_accuracy).sum::<f64>() / n;
            let gap = c.runs.iter().map(|r| r.generalisation_gap).sum::<f64>() / n;
            println!(
                "    {:<18} {:>10.4}  {:>10.4}  {:>10.4}",
                c.condition, loss, acc, gap
            );
        }
        for metric in ["final_val_loss", "generalisation_gap"] {
            let a = analyse(&results, metric, &weights, &gate);
            println!(
                "    {} omnibus {}: {:.4}, p = {:.4}, any significant: {}",
                a.metric, a.omnibus_test, a.omnibus_statistic, a.omnibus_p, a.any_significant
            );
            for c in &a.comparisons {
                println!(
                    "      {:<18} {:>8.4} vs {:>8.4}  {:<14} p {:>6.4}  holm {:>6.4}  d {:>6.3}",
                    c.condition,
                    c.baseline_mean,
                    c.condition_mean,
                    c.test,
                    c.p_value,
                    c.p_holm,
                    c.effect_size
                );
            }
            println!(
                "      detectable d = {:.3}; shuffled control d = {:.3}",
                a.minimum_detectable_effect, a.control_effect
            );
            analyses.push(a);
        }
        blocks.push(results);
    }

    let report = Phase10Report {
        seeds: SEEDS,
        recompute_every: RECOMPUTE_EVERY,
        calibration,
        conditions: blocks,
        analyses,
    };
    std::fs::create_dir_all("results")?;
    std::fs::write(
        "results/phase10_topo_plasticity.json",
        serde_json::to_string_pretty(&report)?,
    )?;
    println!("\n  written results/phase10_topo_plasticity.json");
    Ok(())
}
