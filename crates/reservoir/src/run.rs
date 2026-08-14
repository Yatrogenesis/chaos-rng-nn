// SPDX-License-Identifier: MIT
//! The three sub-phases: the calibration that must pass before anything is
//! believed, the echo state screening, and the comparison itself.

use crate::esn::{check_echo_state_property, spectral_radius, EchoStateCheck};
use crate::fill::{build, Geometry, WeightSource, CONDITIONS};
use crate::reference::ReferenceRng;
use crate::tasks::{memory_capacity, narma10, narma10_nrmse};
use serde::Serialize;

/// Reservoir size, held fixed everywhere.
pub const N: usize = 100;
/// Spectral radius every condition is rescaled to, the one degree of freedom
/// equalised across conditions.
pub const SPECTRAL_RADIUS: f64 = 0.9;
/// Input scaling. Small enough to keep the units away from saturation, which is
/// the regime the memory capacity bound is stated in.
pub const INPUT_SCALING: f64 = 0.1;
/// Steps discarded before any state is recorded, so the measurement does not
/// depend on the zero initial state.
pub const WASHOUT: usize = 1_000;
/// Deepest delay evaluated for memory capacity, twice the reservoir size.
pub const MC_MAX_DELAY: usize = 2 * N;
/// Samples used to fit the memory capacity readouts.
pub const MC_TRAIN: usize = 4_000;
/// Samples held out to evaluate them.
pub const MC_TEST: usize = 3_000;
/// Samples used to fit the NARMA-10 readout.
pub const NARMA_TRAIN: usize = 4_000;
/// Samples held out to evaluate it.
pub const NARMA_TEST: usize = 2_000;
/// Ridge penalty for the memory capacity readouts.
pub const LAMBDA_MC: f64 = 1e-8;
/// Ridge penalty for the NARMA-10 readout.
pub const LAMBDA_NARMA: f64 = 1e-6;
/// Reservoir instances per condition.
pub const INSTANCES: usize = 20;
/// Length of the input used to drive the echo state property check.
pub const ECHO_INPUT_LEN: usize = 2_000;
/// Significance level, the same as every earlier phase.
pub const ALPHA: f64 = 0.05;

const MC_INPUT_LEN: usize = WASHOUT + MC_MAX_DELAY + MC_TRAIN + MC_TEST;
const NARMA_INPUT_LEN: usize = WASHOUT + NARMA_TRAIN + NARMA_TEST;

/// Uniform i.i.d. input on `[-0.5, 0.5]` for the memory capacity task, as
/// published.
fn mc_input(seed: u64) -> Vec<f64> {
    let mut r = ReferenceRng::from_seed(seed);
    (0..MC_INPUT_LEN).map(|_| r.next_range(-0.5, 0.5)).collect()
}

/// Uniform i.i.d. input on `[0, 0.5]` for NARMA-10, as published, together with
/// its target series.
///
/// The recursion diverges for some sequences. When that happens the sequence is
/// redrawn from a derived seed rather than damped, and the number of redraws is
/// returned so it can be reported instead of vanishing.
fn narma_input(seed: u64) -> (Vec<f64>, Vec<f64>, usize) {
    for attempt in 0..64 {
        let mut r = ReferenceRng::from_seed(seed.wrapping_add(attempt * 0x1000_0001));
        let u: Vec<f64> = (0..NARMA_INPUT_LEN)
            .map(|_| r.next_range(0.0, 0.5))
            .collect();
        if let Some(y) = narma10(&u) {
            return (u, y, attempt as usize);
        }
    }
    panic!("NARMA-10 diverged on 64 consecutive input sequences, which indicates a defect");
}

/// One point of the spectral radius sweep that calibrates the implementation.
#[derive(Debug, Clone, Serialize)]
pub struct CalibrationPoint {
    /// Spectral radius the reservoir was rescaled to.
    pub spectral_radius: f64,
    /// Mean memory capacity over the calibration instances.
    pub mean_memory_capacity: f64,
    /// The same divided by the reservoir size.
    pub mean_normalised: f64,
}

/// Result of Phase 8a.
#[derive(Debug, Clone, Serialize)]
pub struct Calibration {
    /// Reservoir size used throughout.
    pub n: usize,
    /// Memory capacity across the spectral radius sweep.
    pub sweep: Vec<CalibrationPoint>,
    /// NARMA-10 NRMSE of the standard reservoir, mean over instances.
    pub standard_narma_nrmse: f64,
    /// Whether memory capacity stayed at or below the theoretical ceiling.
    pub ceiling_respected: bool,
    /// Whether memory capacity increased with the spectral radius.
    pub memory_increases_with_radius: bool,
    /// Whether the NARMA-10 error fell inside the published band.
    pub narma_within_published_band: bool,
    /// All three gates together.
    pub passes: bool,
}

/// Upper edge of the NARMA-10 error a reservoir of this size is expected to
/// reach.
///
/// Echo state networks with a hundred units are reported in the reservoir
/// literature at NRMSE roughly between 0.2 and 0.5 on NARMA-10, with the exact
/// figure depending on the input scaling, the penalty and the readout length.
/// The gate is deliberately set at the loose end: it is meant to catch an
/// implementation that does not work, not to certify a particular tuning.
///
/// REF: [Jaeger and Haas, 2004] "Harnessing nonlinearity: predicting chaotic
///      systems and saving energy in wireless communication", Science
///      304(5667), pp. 78-80, DOI: 10.1126/science.1091277
pub const NARMA_PUBLISHED_CEILING: f64 = 0.7;

/// Phase 8a. Builds the canonical reservoir only, with none of the qualified
/// generators, and checks it against what the theory predicts.
///
/// Three predictions, each of which an implementation defect would break:
///
/// 1. Memory capacity cannot exceed the reservoir size. Exceeding it is the
///    signature of measuring in-sample rather than out of sample.
/// 2. Memory capacity grows as the spectral radius approaches one, because the
///    reservoir forgets more slowly. A measurement flat in the radius would mean
///    the recurrent matrix is not actually driving the state.
/// 3. The NARMA-10 error lands in the band the literature reports for a
///    reservoir of this size.
pub fn calibrate(instances: usize) -> Calibration {
    let radii = [0.5, 0.9, 0.99];
    let input = mc_input(0xCA11);
    let mut sweep = Vec::new();
    for radius in radii {
        let geometry = Geometry {
            n: N,
            spectral_radius: radius,
            input_scaling: INPUT_SCALING,
        };
        let mut totals = Vec::new();
        for instance in 0..instances {
            let (esn, _) = build(WeightSource::Standard, geometry, 1_000 + instance as u64);
            let mc = memory_capacity(&esn, &input, WASHOUT, MC_TRAIN, MC_MAX_DELAY, LAMBDA_MC)
                .expect("the calibration design must be solvable");
            totals.push(mc.total);
        }
        let mean = totals.iter().sum::<f64>() / totals.len() as f64;
        sweep.push(CalibrationPoint {
            spectral_radius: radius,
            mean_memory_capacity: mean,
            mean_normalised: mean / N as f64,
        });
    }

    let geometry = Geometry {
        n: N,
        spectral_radius: SPECTRAL_RADIUS,
        input_scaling: INPUT_SCALING,
    };
    let mut errors = Vec::new();
    for instance in 0..instances {
        let (u, y, _) = narma_input(2_000 + instance as u64);
        let (esn, _) = build(WeightSource::Standard, geometry, 1_000 + instance as u64);
        errors.push(
            narma10_nrmse(&esn, &u, &y, WASHOUT, NARMA_TRAIN, LAMBDA_NARMA)
                .expect("the calibration design must be solvable"),
        );
    }
    let standard_narma_nrmse = errors.iter().sum::<f64>() / errors.len() as f64;

    let ceiling_respected = sweep.iter().all(|p| p.mean_memory_capacity <= N as f64);
    let memory_increases_with_radius = sweep
        .windows(2)
        .all(|w| w[1].mean_memory_capacity > w[0].mean_memory_capacity);
    let narma_within_published_band = standard_narma_nrmse < NARMA_PUBLISHED_CEILING;

    Calibration {
        n: N,
        sweep,
        standard_narma_nrmse,
        ceiling_respected,
        memory_increases_with_radius,
        narma_within_published_band,
        passes: ceiling_respected && memory_increases_with_radius && narma_within_published_band,
    }
}

/// Everything measured for one condition.
#[derive(Debug, Clone, Serialize)]
pub struct ConditionResult {
    /// Stable identifier of the weight source.
    pub condition: String,
    /// Spectral radius of the raw fill, before rescaling, per instance.
    pub raw_spectral_radius: Vec<f64>,
    /// Echo state property check, per instance.
    pub echo_state: Vec<EchoStateCheck>,
    /// Whether every instance of this condition has the echo state property.
    pub echo_state_holds_everywhere: bool,
    /// Memory capacity, per instance.
    pub memory_capacity: Vec<f64>,
    /// NARMA-10 test NRMSE, per instance.
    pub narma_nrmse: Vec<f64>,
    /// How many times the NARMA-10 input had to be redrawn, summed.
    pub narma_redraws: usize,
}

/// Phase 8b and 8c: builds every condition, screens it, and measures it.
///
/// The task inputs are drawn once per instance and shared by all five
/// conditions, so instance `i` of every condition sees the identical driving
/// sequence and the only thing that differs is the recurrent matrix.
pub fn run_conditions(instances: usize) -> Vec<ConditionResult> {
    let geometry = Geometry {
        n: N,
        spectral_radius: SPECTRAL_RADIUS,
        input_scaling: INPUT_SCALING,
    };
    let mc_in = mc_input(0xC0FFEE);
    let echo_in: Vec<f64> = {
        let mut r = ReferenceRng::from_seed(0xECC0);
        (0..ECHO_INPUT_LEN)
            .map(|_| r.next_range(-0.5, 0.5))
            .collect()
    };
    let narma: Vec<(Vec<f64>, Vec<f64>, usize)> = (0..instances)
        .map(|i| narma_input(3_000 + i as u64))
        .collect();

    CONDITIONS
        .iter()
        .map(|source| {
            let mut raw_spectral_radius = Vec::new();
            let mut echo_state = Vec::new();
            let mut memory = Vec::new();
            let mut nrmse = Vec::new();
            let mut redraws = 0;

            for (instance, narma_case) in narma.iter().enumerate().take(instances) {
                let seed = 500 + instance as u64;
                let (esn, raw) = build(*source, geometry, seed);
                raw_spectral_radius.push(raw);
                debug_assert!((spectral_radius(&esn.w_res) - SPECTRAL_RADIUS).abs() < 1e-9);

                let check = check_echo_state_property(&esn, &echo_in, seed);
                echo_state.push(check);

                let mc = memory_capacity(&esn, &mc_in, WASHOUT, MC_TRAIN, MC_MAX_DELAY, LAMBDA_MC)
                    .expect("design must be solvable");
                memory.push(mc.total);

                let (u, y, r) = narma_case;
                redraws += r;
                nrmse.push(
                    narma10_nrmse(&esn, u, y, WASHOUT, NARMA_TRAIN, LAMBDA_NARMA)
                        .expect("design must be solvable"),
                );
            }

            ConditionResult {
                condition: source.as_str().to_string(),
                echo_state_holds_everywhere: echo_state.iter().all(|c| c.holds),
                raw_spectral_radius,
                echo_state,
                memory_capacity: memory,
                narma_nrmse: nrmse,
                narma_redraws: redraws,
            }
        })
        .collect()
}

/// One generator compared against the standard reservoir on one metric.
#[derive(Debug, Clone, Serialize)]
pub struct Comparison {
    /// Which condition was compared against the baseline.
    pub condition: String,
    /// Mean of the baseline sample.
    pub standard_mean: f64,
    /// Mean of this condition's sample.
    pub condition_mean: f64,
    /// Which test was applied, decided by the normality screen.
    pub test: String,
    /// Raw two-sided p-value.
    pub p_value: f64,
    /// The same after the Holm correction across the four comparisons.
    pub p_holm: f64,
    /// Cohen's d, reported whichever test was used.
    pub effect_size: f64,
}

/// The whole comparison for one metric.
#[derive(Debug, Clone, Serialize)]
pub struct MetricAnalysis {
    /// Metric name.
    pub metric: String,
    /// Whether every one of the five samples passed the normality screen.
    pub all_normal: bool,
    /// Omnibus test name.
    pub omnibus_test: String,
    /// Omnibus statistic.
    pub omnibus_statistic: f64,
    /// Omnibus p-value across all five conditions.
    pub omnibus_p: f64,
    /// The four generator-against-baseline comparisons.
    pub comparisons: Vec<Comparison>,
    /// Whether any Holm-adjusted comparison fell below alpha.
    pub any_significant: bool,
    /// Smallest standardised effect this design could detect with 80 percent
    /// power at the uncorrected level.
    pub minimum_detectable_effect: f64,
    /// The same at the most stringent level the Holm correction applies.
    pub minimum_detectable_effect_holm: f64,
    /// Effect size of the chacha8 comparison. That condition differs from the
    /// baseline only in the ChaCha round count, eight against twelve, so its
    /// value is one realisation of what this design registers when nothing of
    /// substance differs.
    pub noise_floor_effect: f64,
    /// Rank of the negative control's effect magnitude among the four
    /// comparisons, one being the largest.
    ///
    /// A rank of one is the informative case: the comparison that differs from
    /// the baseline in nothing but the ChaCha round count moved the metric more
    /// than any genuine change of generator did. That is a stronger statement
    /// than a p-value above the threshold, because it is a within-experiment
    /// demonstration that effects of that size arise here from nothing at all.
    /// A rank of four says only that this particular draw of the control landed
    /// close to the baseline, which is one realisation and calibrates nothing.
    pub negative_control_rank: usize,
}

/// Smallest standardised mean difference a two-sample comparison of `n` per
/// group can detect with the conventional eighty percent power.
///
/// `d = (z_{1-alpha/2} + z_{1-beta}) * sqrt(2/n)`, the normal approximation to
/// the two-sample t-test. It is reported because a result of "no significant
/// difference" is uninterpretable without it: a design that could only ever
/// have detected an enormous effect has not established that the effect is
/// small, and saying so is the difference between a null result and an absence
/// of evidence.
///
/// The approximation errs in the direction that flatters the design. Replacing
/// the t quantiles with normal ones ignores the heavier tails of the reference
/// distribution, so the value returned is a few percent **smaller** than the
/// exact noncentral-t figure: it claims slightly more resolution than the
/// design has. Against Cohen's tabulated `n = 26` per group for `d = 0.8` it
/// returns 0.777, about three percent low, and a test below pins both the size
/// and the direction of that gap. The exact calculation is not implemented
/// because the quantity is used to state a limitation, and a limitation stated
/// three percent too favourably is corrected in the report rather than by
/// carrying a noncentral-t implementation this project has no other use for.
pub fn minimum_detectable_effect(n: usize, alpha: f64) -> f64 {
    const POWER: f64 = 0.80;
    let z_alpha = xstats::normal_quantile(1.0 - alpha / 2.0);
    let z_beta = xstats::normal_quantile(POWER);
    (z_alpha + z_beta) * (2.0 / n as f64).sqrt()
}

/// Runs the pre-registered test sequence on one metric.
///
/// Shapiro-Wilk screens every sample first; Welch is used when both samples of
/// a pair pass and Mann-Whitney otherwise, the same rule as every earlier
/// phase. An omnibus test across all five conditions is reported first, so the
/// pairwise results are read in the context of whether anything varied at all.
pub fn analyse(results: &[ConditionResult], metric: &str) -> MetricAnalysis {
    let sample = |r: &ConditionResult| -> Vec<f64> {
        match metric {
            "memory_capacity" => r.memory_capacity.clone(),
            "narma_nrmse" => r.narma_nrmse.clone(),
            other => panic!("unknown metric {other}"),
        }
    };
    let samples: Vec<Vec<f64>> = results.iter().map(sample).collect();
    let normal: Vec<bool> = samples
        .iter()
        .map(|s| xstats::shapiro_wilk(s).p_value > ALPHA)
        .collect();
    let all_normal = normal.iter().all(|v| *v);

    let (omnibus_test, omnibus_statistic, omnibus_p) = if all_normal {
        let a = xstats::one_way_anova(&samples);
        ("one-way ANOVA".to_string(), a.f, a.p_value)
    } else {
        let k = xstats::kruskal_wallis(&samples);
        ("Kruskal-Wallis".to_string(), k.h, k.p_value)
    };

    let baseline = &samples[0];
    let mut raw_p = Vec::new();
    let mut partial = Vec::new();
    for (i, s) in samples.iter().enumerate().skip(1) {
        let (test, p) = if normal[0] && normal[i] {
            (
                "Welch".to_string(),
                xstats::welch_t_test(baseline, s).p_value,
            )
        } else {
            (
                "Mann-Whitney U".to_string(),
                xstats::mann_whitney_u(baseline, s).p_value,
            )
        };
        raw_p.push(p);
        partial.push((results[i].condition.clone(), xstats::mean(s), test, p));
    }
    let adjusted = xstats::holm_adjust(&raw_p);
    let standard_mean = xstats::mean(baseline);

    let comparisons: Vec<Comparison> = partial
        .into_iter()
        .zip(adjusted.iter())
        .enumerate()
        .map(
            |(i, ((condition, condition_mean, test, p_value), p_holm))| Comparison {
                condition,
                standard_mean,
                condition_mean,
                test,
                p_value,
                p_holm: *p_holm,
                effect_size: xstats::cohens_d(baseline, &samples[i + 1]),
            },
        )
        .collect();

    let n = baseline.len();
    // The Holm procedure tests the smallest p-value against alpha / m, which is
    // the most stringent level any of the four comparisons faces.
    let holm_alpha = ALPHA / comparisons.len() as f64;
    let noise_floor_effect = comparisons
        .iter()
        .find(|c| c.condition == "chacha8")
        .map(|c| c.effect_size.abs())
        .expect("the chacha8 negative control must be present");

    MetricAnalysis {
        metric: metric.to_string(),
        all_normal,
        omnibus_test,
        omnibus_statistic,
        omnibus_p,
        any_significant: comparisons.iter().any(|c| c.p_holm < ALPHA),
        minimum_detectable_effect: minimum_detectable_effect(n, ALPHA),
        minimum_detectable_effect_holm: minimum_detectable_effect(n, holm_alpha),
        negative_control_rank: 1 + comparisons
            .iter()
            .filter(|c| c.effect_size.abs() > noise_floor_effect + 1e-12)
            .count(),
        noise_floor_effect,
        comparisons,
    }
}

/// Everything Phase 8 produces, in the shape written to disk.
#[derive(Debug, Clone, Serialize)]
pub struct Phase8Report {
    /// Reservoir size.
    pub n: usize,
    /// Spectral radius equalised across conditions.
    pub spectral_radius: f64,
    /// Instances per condition.
    pub instances: usize,
    /// Phase 8a.
    pub calibration: Calibration,
    /// Phase 8b and the raw measurements of 8c.
    pub conditions: Vec<ConditionResult>,
    /// Phase 8c.
    pub analyses: Vec<MetricAnalysis>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_calibration_gates_pass() {
        // Blocking. Four instances is enough to establish the trend; the
        // reported figures use more.
        let c = calibrate(4);
        assert!(
            c.ceiling_respected,
            "memory capacity exceeded the reservoir size, which means it is \
             being measured in sample: {:?}",
            c.sweep
        );
        assert!(
            c.memory_increases_with_radius,
            "memory capacity did not grow with the spectral radius: {:?}",
            c.sweep
        );
        assert!(
            c.narma_within_published_band,
            "the standard reservoir reached NRMSE {} on NARMA-10, outside the \
             band reported for this size",
            c.standard_narma_nrmse
        );
    }

    #[test]
    fn the_detectable_effect_matches_the_published_table() {
        // Cohen tabulates n = 26 per group for d = 0.8 at alpha = 0.05 with
        // eighty percent power. The normal approximation used here should land
        // close to that.
        //   REF: [Cohen, 1988] "Statistical Power Analysis for the Behavioral
        //        Sciences", 2nd edition, Lawrence Erlbaum, table 2.4.1
        let d = minimum_detectable_effect(26, 0.05);
        assert!((d - 0.8).abs() < 0.03, "got {d}");
        // And it is optimistic rather than conservative, which is the direction
        // that matters when the number is used to bound what the design can
        // rule out.
        assert!(
            d < 0.8,
            "the approximation is expected to understate; got {d}"
        );
    }

    #[test]
    fn a_smaller_alpha_needs_a_larger_effect() {
        assert!(minimum_detectable_effect(20, 0.0125) > minimum_detectable_effect(20, 0.05));
    }

    #[test]
    fn the_narma_input_is_shared_across_conditions() {
        // The paired design depends on this.
        let (a, ya, _) = narma_input(77);
        let (b, yb, _) = narma_input(77);
        assert_eq!(a, b);
        assert_eq!(ya, yb);
    }

    #[test]
    fn running_is_reproducible() {
        let a = run_conditions(2);
        let b = run_conditions(2);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.memory_capacity, y.memory_capacity, "{}", x.condition);
            assert_eq!(x.narma_nrmse, y.narma_nrmse, "{}", x.condition);
        }
    }
}
