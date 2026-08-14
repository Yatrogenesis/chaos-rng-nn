// SPDX-License-Identifier: MIT
//! Runs Phase 8 and writes the results.

use reservoir::run::{
    analyse, calibrate, run_conditions, Phase8Report, INSTANCES, N, SPECTRAL_RADIUS,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Phase 8: reservoir computing, ridge readout, no gradient descent");
    println!("  reservoir size {N}, spectral radius {SPECTRAL_RADIUS}, {INSTANCES} instances per condition");

    println!("\n8a. Calibration of the canonical reservoir, before any generator is used.");
    let calibration = calibrate(INSTANCES);
    println!("  spectral radius   memory capacity   MC/N");
    for p in &calibration.sweep {
        println!(
            "  {:>15.2}   {:>15.3}   {:>4.3}",
            p.spectral_radius, p.mean_memory_capacity, p.mean_normalised
        );
    }
    println!(
        "  NARMA-10 NRMSE of the standard reservoir: {:.4}",
        calibration.standard_narma_nrmse
    );
    println!(
        "  ceiling MC <= N respected:        {}",
        calibration.ceiling_respected
    );
    println!(
        "  memory grows with radius:         {}",
        calibration.memory_increases_with_radius
    );
    println!(
        "  NARMA-10 inside published band:   {}",
        calibration.narma_within_published_band
    );
    if !calibration.passes {
        eprintln!(
            "\n  Calibration failed. The later sub-phases are not meaningful and are not run."
        );
        std::process::exit(1);
    }
    println!("  calibration passed");

    println!("\n8b. Echo state property, every condition, before anything is measured.");
    let conditions = run_conditions(INSTANCES);
    println!("  condition        raw radius   worst final separation   holds");
    for c in &conditions {
        let raw = c.raw_spectral_radius.iter().sum::<f64>() / c.raw_spectral_radius.len() as f64;
        let worst = c
            .echo_state
            .iter()
            .map(|e| e.final_separation)
            .fold(0.0, f64::max);
        println!(
            "  {:<15} {:>10.3}   {:>22.3e}   {}",
            c.condition, raw, worst, c.echo_state_holds_everywhere
        );
    }
    let excluded: Vec<&str> = conditions
        .iter()
        .filter(|c| !c.echo_state_holds_everywhere)
        .map(|c| c.condition.as_str())
        .collect();
    if !excluded.is_empty() {
        println!(
            "  conditions without the echo state property: {}",
            excluded.join(", ")
        );
    }

    println!("\n8c. Comparison against the standard reservoir.");
    let analyses = vec![
        analyse(&conditions, "memory_capacity"),
        analyse(&conditions, "narma_nrmse"),
    ];
    for a in &analyses {
        println!("\n  {} (all samples normal: {})", a.metric, a.all_normal);
        println!(
            "    omnibus {}: statistic {:.4}, p = {:.4}",
            a.omnibus_test, a.omnibus_statistic, a.omnibus_p
        );
        println!("    condition        standard     condition   test              p       p_holm       d");
        for c in &a.comparisons {
            println!(
                "    {:<15} {:>8.4}   {:>11.4}   {:<14} {:>6.4}   {:>8.4}   {:>5.3}",
                c.condition,
                c.standard_mean,
                c.condition_mean,
                c.test,
                c.p_value,
                c.p_holm,
                c.effect_size
            );
        }
        println!("    any significant after Holm: {}", a.any_significant);
        println!(
            "    smallest detectable effect at 80 percent power: d = {:.3} uncorrected, {:.3} at the Holm level",
            a.minimum_detectable_effect, a.minimum_detectable_effect_holm
        );
        println!(
            "    chacha8 negative control: d = {:.3}, ranked {} of 4 by magnitude",
            a.noise_floor_effect, a.negative_control_rank
        );
    }

    let report = Phase8Report {
        n: N,
        spectral_radius: SPECTRAL_RADIUS,
        instances: INSTANCES,
        calibration,
        conditions,
        analyses,
    };
    std::fs::create_dir_all("results")?;
    std::fs::write(
        "results/phase8_reservoir.json",
        serde_json::to_string_pretty(&report)?,
    )?;
    println!("\n  written results/phase8_reservoir.json");
    Ok(())
}
