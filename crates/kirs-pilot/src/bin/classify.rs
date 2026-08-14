// SPDX-License-Identifier: MIT
//! Prints the applicability table and writes it as JSON.

fn main() -> std::io::Result<()> {
    let rows = kirs_pilot::classify();
    println!("Phase 7: applicability of Pesin-type formulas, decided by Horn resolution");
    println!("  engine: pirs-kirs at kirs commit 18a7276, consumed read-only");
    println!("  step budget per query: {}", kirs_pilot::STEP_BUDGET);
    println!();
    println!(
        "  {:<14} {:>10} {:>10} {:>12} {:>12}",
        "generator", "classical", "Liu random", "neither", "exhausted"
    );
    for r in rows.iter() {
        println!(
            "  {:<14} {:>10} {:>10} {:>12} {:>12}",
            r.generator,
            if r.classical { "yes" } else { "no" },
            if r.random_liu { "yes" } else { "no" },
            if r.neither { "yes" } else { "no" },
            if r.budget_exhausted { "YES" } else { "no" }
        );
    }
    let json: Vec<String> = rows
        .iter()
        .map(|r| {
            format!(
                "  {{\"generator\": \"{}\", \"classical\": {}, \"random_liu\": {}, \"neither\": {}, \"budget_exhausted\": {}}}",
                r.generator, r.classical, r.random_liu, r.neither, r.budget_exhausted
            )
        })
        .collect();
    std::fs::create_dir_all("results")?;
    std::fs::write(
        "results/phase7_applicability.json",
        format!("[\n{}\n]\n", json.join(",\n")),
    )?;
    println!();
    println!("  written results/phase7_applicability.json");
    Ok(())
}
