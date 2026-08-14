// SPDX-License-Identifier: MIT
//! Runs Phase 9 and writes the results.

use predictive_coding::network::Config;
use predictive_coding::run::{analyse, gate, run_conditions, Phase9Report, GATE_THRESHOLD, SEEDS};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Config::default();
    println!("Phase 9: predictive coding, local Hebbian updates, no backpropagation");
    println!(
        "  layers {:?}, {} relaxation steps, inference rate {}, learning rate {}, {} epochs",
        cfg.layers, cfg.inference_steps, cfg.inference_rate, cfg.learning_rate, cfg.epochs
    );

    println!("\n9a. Does the local update approximate the backpropagation gradient?");
    let g = gate();
    println!("  relaxation steps   correlation   cosine");
    for r in &g.rows {
        println!(
            "  {:>16}   {:>11.4}   {:>6.4}",
            r.inference_steps, r.correlation, r.cosine
        );
    }
    println!(
        "  improves as the inference settles: {}",
        g.improves_with_relaxation
    );
    println!(
        "  settled correlation {:.4} against a threshold of {GATE_THRESHOLD}: {}",
        g.best_correlation, g.clears_threshold
    );
    if !g.passes {
        eprintln!("\n  Phase 9a failed. Comparing generators on a network that does not");
        eprintln!("  approximate what it should would be meaningless, so 9b is not run.");
        std::process::exit(1);
    }
    println!("  gate passed");

    println!("\n9b. Precision modulated by each generator, against constant precision.");
    let conditions = run_conditions(SEEDS, &cfg);
    println!("  condition            val loss    accuracy         gap");
    for c in &conditions {
        let n = c.runs.len() as f64;
        let loss = c.runs.iter().map(|r| r.final_val_loss).sum::<f64>() / n;
        let acc = c.runs.iter().map(|r| r.final_val_accuracy).sum::<f64>() / n;
        let gap = c.runs.iter().map(|r| r.generalisation_gap).sum::<f64>() / n;
        println!(
            "  {:<18} {:>10.4}  {:>10.4}  {:>10.4}",
            c.condition, loss, acc, gap
        );
    }

    let analyses = vec![
        analyse(&conditions, "final_val_loss"),
        analyse(&conditions, "generalisation_gap"),
    ];
    for a in &analyses {
        println!("\n  {} (all samples normal: {})", a.metric, a.all_normal);
        println!(
            "    omnibus {}: statistic {:.4}, p = {:.4}",
            a.omnibus_test, a.omnibus_statistic, a.omnibus_p
        );
        println!(
            "    condition          baseline    condition   test              p     p_holm       d"
        );
        for c in &a.comparisons {
            println!(
                "    {:<17} {:>8.4}   {:>10.4}   {:<14} {:>6.4}   {:>7.4}   {:>6.3}",
                c.condition,
                c.baseline_mean,
                c.condition_mean,
                c.test,
                c.p_value,
                c.p_holm,
                c.effect_size
            );
        }
        println!("    any significant after Holm: {}", a.any_significant);
        println!(
            "    smallest detectable effect at 80 percent power: d = {:.3}",
            a.minimum_detectable_effect
        );
        println!(
            "    chacha8 against the chacha12 control, the noise scale: d = {:.3}",
            a.negative_control_effect
        );
    }

    let report = Phase9Report {
        layers: cfg.layers.clone(),
        inference_steps: cfg.inference_steps,
        inference_rate: cfg.inference_rate,
        learning_rate: cfg.learning_rate,
        epochs: cfg.epochs,
        seeds: SEEDS,
        gate: g,
        conditions,
        analyses,
    };
    std::fs::create_dir_all("results")?;
    std::fs::write(
        "results/phase9_predictive_coding.json",
        serde_json::to_string_pretty(&report)?,
    )?;
    println!("\n  written results/phase9_predictive_coding.json");
    Ok(())
}
