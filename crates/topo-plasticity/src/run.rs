// SPDX-License-Identifier: MIT
//! Phase 10c: the two hypotheses applied to the Phase 9 network, separately and
//! together, against the Phase 9 baseline.

use crate::plasticity::{self, Gate};
use crate::resilience::{self, Weights};
use chaos_rng::ChaChaRng;
use experiment::dataset::{make_moons, train_test_split, Dataset};
use nalgebra::DMatrix;
use predictive_coding::network::{Config, Network};
use predictive_coding::precision::Constant;
use serde::Serialize;

/// Runs per condition.
pub const SEEDS: usize = 20;
/// Significance level, unchanged across the project.
pub const ALPHA: f64 = 0.05;
/// Weight updates between recomputations of the topological signal.
///
/// Recomputing at every update would dominate the run time and would also track
/// batch-to-batch noise rather than the structure of the representation. Every
/// thirty-two updates is a little under once per epoch at this batch size.
pub const RECOMPUTE_EVERY: usize = 32;
/// Inputs used to probe a layer's activity when the signal is recomputed.
pub const PROBE_SAMPLES: usize = 128;
/// Steepness of the map from a standardised topological signal to a precision.
pub const SIGNAL_STEEPNESS: f64 = 2.0;
/// Weight of a new observation in the running mean and deviation.
pub const EMA_ALPHA: f64 = 0.1;

/// Turns a stream of topological readings into precisions.
///
/// The raw value of `T(S)` has no natural scale: it depends on the layer width,
/// on the weighting and on how far training has progressed. Feeding it into a
/// precision directly would make the modulation an arbitrary function of those.
/// Instead each reading is standardised against a running mean and a running
/// mean absolute deviation of the readings themselves, and the standardised
/// value goes through the same logistic map Phase 9 used.
///
/// The consequence, which is the point, is that the modulation has median one
/// by construction whatever the signal's units, so this condition differs from
/// the baseline in how precision varies and not in its level. Without that the
/// comparison would be a comparison of effective learning rates.
#[derive(Debug, Clone)]
pub struct SignalToPrecision {
    mean: f64,
    deviation: f64,
    initialised: bool,
}

impl Default for SignalToPrecision {
    fn default() -> Self {
        Self {
            mean: 0.0,
            deviation: 0.0,
            initialised: false,
        }
    }
}

impl SignalToPrecision {
    /// Absorbs a reading and returns the precision it implies.
    pub fn observe(&mut self, value: f64) -> f64 {
        if !self.initialised {
            self.mean = value;
            self.deviation = 0.0;
            self.initialised = true;
            return 1.0;
        }
        let z = if self.deviation > 1e-12 {
            (value - self.mean) / self.deviation
        } else {
            0.0
        };
        let deviation = (value - self.mean).abs();
        self.mean += EMA_ALPHA * (value - self.mean);
        self.deviation += EMA_ALPHA * (deviation - self.deviation);
        2.0 / (1.0 + (-SIGNAL_STEEPNESS * z).exp())
    }
}

/// What a condition switches on.
#[derive(Debug, Clone, Copy)]
pub struct Condition {
    /// Name used in the report.
    pub label: &'static str,
    /// Topological weighting, if precision is modulated by it.
    pub weights: Option<Weights>,
    /// Plasticity gate, if the learning rate is graded.
    pub gate: Option<Gate>,
    /// Whether the activations are shuffled before the signal is computed.
    ///
    /// This is the negative control. Shuffling each node's activity
    /// independently destroys the correlation structure between nodes while
    /// leaving every node's own distribution of activity exactly as it was, so
    /// the signal keeps its scale and loses its meaning. A difference between
    /// this and the genuine topological condition is attributable to the
    /// structure; agreement between them says the modulation is acting as
    /// noise of that magnitude and nothing more.
    pub shuffled: bool,
}

/// The five conditions of Phase 10c at a given pair of hyperparameter choices.
pub fn conditions(weights: Weights, gate: Gate) -> [Condition; 5] {
    [
        Condition {
            label: "baseline",
            weights: None,
            gate: None,
            shuffled: false,
        },
        Condition {
            label: "topological",
            weights: Some(weights),
            gate: None,
            shuffled: false,
        },
        Condition {
            label: "graded-plasticity",
            weights: None,
            gate: Some(gate),
            shuffled: false,
        },
        Condition {
            label: "both",
            weights: Some(weights),
            gate: Some(gate),
            shuffled: false,
        },
        Condition {
            label: "shuffled-control",
            weights: Some(weights),
            gate: None,
            shuffled: true,
        },
    ]
}

/// Shuffles each row independently, destroying correlation between nodes while
/// preserving each node's own activity distribution exactly.
fn shuffle_rows(activations: &mut [Vec<f64>], rng: &mut ChaChaRng) {
    for row in activations.iter_mut() {
        rng.shuffle(row);
    }
}

fn to_matrix(rows: &[Vec<f64>]) -> DMatrix<f64> {
    let n = rows.len();
    let m = rows.first().map(|r| r.len()).unwrap_or(0);
    let mut a = DMatrix::zeros(n, m);
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            a[(i, j)] = *v;
        }
    }
    a
}

/// One trained network.
#[derive(Debug, Clone, Serialize)]
pub struct RunRecord {
    /// Condition.
    pub condition: String,
    /// Run seed.
    pub seed: u64,
    /// Cross-entropy on the validation set.
    pub final_val_loss: f64,
    /// Accuracy on the validation set.
    pub final_val_accuracy: f64,
    /// Validation loss minus training loss.
    pub generalisation_gap: f64,
    /// Mean of the topological signal over the run, where one was computed.
    pub mean_signal: Option<f64>,
}

/// Trains one network under one condition.
pub fn train(
    condition: &Condition,
    seed: u64,
    cfg: &Config,
    train_set: &Dataset,
    val: &Dataset,
) -> RunRecord {
    let mut init = ChaChaRng::from_seed(seed);
    let mut net = Network::new(cfg.clone(), &mut init);
    let depth = net.depth();

    let mut order: Vec<usize> = (0..train_set.len()).collect();
    let mut shuffler = ChaChaRng::from_seed(seed ^ 0x5DEE_CE66_D125_ABCD);
    let mut permuter = ChaChaRng::from_seed(seed ^ 0x1234_5678_9ABC_DEF0);

    // A fixed probe set, so the signal reflects the network changing rather
    // than the sample changing under it.
    let probe: Vec<[f64; 2]> = train_set.x.iter().take(PROBE_SAMPLES).copied().collect();

    let gate_multipliers = match condition.gate {
        Some(g) => g.multipliers(depth),
        None => plasticity::uniform(depth),
    };

    // One converter per modulated error level. The output level is left at
    // constant precision: it has two nodes, and the topology of two points
    // carries no information.
    let modulated_levels = depth.saturating_sub(1);
    let mut converters = vec![SignalToPrecision::default(); modulated_levels];
    let mut precisions = vec![1.0; depth];
    let mut signals: Vec<f64> = Vec::new();
    let mut updates = 0usize;

    for _ in 0..cfg.epochs {
        shuffler.shuffle(&mut order);
        for chunk in order.chunks(cfg.batch_size) {
            if let Some(weights) = condition.weights {
                if updates % RECOMPUTE_EVERY == 0 {
                    for level in 0..modulated_levels {
                        let layer = level + 1;
                        let mut rows = net.layer_activations(&probe, layer);
                        if condition.shuffled {
                            shuffle_rows(&mut rows, &mut permuter);
                        }
                        let value = resilience::resilience(
                            &resilience::correlation_distances(&to_matrix(&rows)),
                            &weights,
                        );
                        signals.push(value);
                        precisions[level] = converters[level].observe(value);
                    }
                }
            }
            updates += 1;

            let (mut dw, mut db) = net.zero_accumulators();
            let mut inf = net.new_inference();
            for &i in chunk {
                let mut target = vec![0.0; Dataset::N_CLASSES];
                target[train_set.y[i]] = 1.0;
                net.feedforward(&train_set.x[i], &mut inf);
                // The precision schedule is fixed between recomputations, so a
                // constant supplier drives the relaxation and the accumulated
                // update is scaled by the current precisions afterwards.
                let _ = net.infer(&target, &mut inf, &mut Constant);
                net.accumulate(&inf, &precisions, &mut dw, &mut db);
            }
            net.apply_scaled(&dw, &db, chunk.len(), &gate_multipliers);
        }
    }

    let (final_val_loss, final_val_accuracy) = net.evaluate(&val.x, &val.y);
    let (final_train_loss, _) = net.evaluate(&train_set.x, &train_set.y);
    RunRecord {
        condition: condition.label.to_string(),
        seed,
        final_val_loss,
        final_val_accuracy,
        generalisation_gap: final_val_loss - final_train_loss,
        mean_signal: if signals.is_empty() {
            None
        } else {
            Some(signals.iter().sum::<f64>() / signals.len() as f64)
        },
    }
}

/// All runs of one condition.
#[derive(Debug, Clone, Serialize)]
pub struct ConditionResult {
    /// Condition.
    pub condition: String,
    /// Individual runs.
    pub runs: Vec<RunRecord>,
}

impl ConditionResult {
    /// Validation losses in run order.
    pub fn losses(&self) -> Vec<f64> {
        self.runs.iter().map(|r| r.final_val_loss).collect()
    }
    /// Generalisation gaps in run order.
    pub fn gaps(&self) -> Vec<f64> {
        self.runs.iter().map(|r| r.generalisation_gap).collect()
    }
}

/// Runs every condition at one pair of hyperparameter choices.
pub fn run_conditions(weights: Weights, gate: Gate, seeds: usize) -> Vec<ConditionResult> {
    let cfg = Config::default();
    let data = make_moons(2_000, 0.20, 20_260_813);
    let (train_set, val) = train_test_split(&data, 0.75, 20_260_814);

    conditions(weights, gate)
        .iter()
        .map(|condition| ConditionResult {
            condition: condition.label.to_string(),
            runs: (0..seeds)
                .map(|s| train(condition, 700 + s as u64, &cfg, &train_set, &val))
                .collect(),
        })
        .collect()
}

/// One condition against the baseline.
#[derive(Debug, Clone, Serialize)]
pub struct Comparison {
    /// Condition.
    pub condition: String,
    /// Mean of the baseline.
    pub baseline_mean: f64,
    /// Mean of this condition.
    pub condition_mean: f64,
    /// Test chosen by the normality screen.
    pub test: String,
    /// Raw p-value.
    pub p_value: f64,
    /// Holm-adjusted p-value.
    pub p_holm: f64,
    /// Cohen's d.
    pub effect_size: f64,
}

/// The analysis of one metric at one hyperparameter setting.
#[derive(Debug, Clone, Serialize)]
pub struct MetricAnalysis {
    /// Topological weighting in force.
    pub weights: String,
    /// Plasticity gate in force.
    pub gate: String,
    /// Metric name.
    pub metric: String,
    /// Whether every sample passed Shapiro-Wilk.
    pub all_normal: bool,
    /// Omnibus test name.
    pub omnibus_test: String,
    /// Omnibus statistic.
    pub omnibus_statistic: f64,
    /// Omnibus p-value.
    pub omnibus_p: f64,
    /// Comparisons against the baseline.
    pub comparisons: Vec<Comparison>,
    /// Whether any survived the correction.
    pub any_significant: bool,
    /// Smallest effect detectable at eighty percent power.
    pub minimum_detectable_effect: f64,
    /// Effect of the shuffled control against the baseline, the design's own
    /// noise scale.
    pub control_effect: f64,
}

/// Smallest standardised effect detectable with eighty percent power.
pub fn minimum_detectable_effect(n: usize, alpha: f64) -> f64 {
    (xstats::normal_quantile(1.0 - alpha / 2.0) + xstats::normal_quantile(0.80))
        * (2.0 / n as f64).sqrt()
}

/// Runs the pre-registered test sequence on one metric.
pub fn analyse(
    results: &[ConditionResult],
    metric: &str,
    weights: &Weights,
    gate: &Gate,
) -> MetricAnalysis {
    let samples: Vec<Vec<f64>> = results
        .iter()
        .map(|r| match metric {
            "final_val_loss" => r.losses(),
            "generalisation_gap" => r.gaps(),
            other => panic!("unknown metric {other}"),
        })
        .collect();
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
    let baseline_mean = xstats::mean(baseline);
    let comparisons: Vec<Comparison> = partial
        .into_iter()
        .zip(adjusted.iter())
        .enumerate()
        .map(
            |(i, ((condition, condition_mean, test, p_value), p_holm))| Comparison {
                condition,
                baseline_mean,
                condition_mean,
                test,
                p_value,
                p_holm: *p_holm,
                effect_size: xstats::cohens_d(baseline, &samples[i + 1]),
            },
        )
        .collect();
    let control_effect = comparisons
        .iter()
        .find(|c| c.condition == "shuffled-control")
        .map(|c| c.effect_size.abs())
        .unwrap_or(f64::NAN);

    MetricAnalysis {
        weights: weights.label.to_string(),
        gate: gate.label.to_string(),
        metric: metric.to_string(),
        all_normal,
        omnibus_test,
        omnibus_statistic,
        omnibus_p,
        any_significant: comparisons.iter().any(|c| c.p_holm < ALPHA),
        minimum_detectable_effect: minimum_detectable_effect(baseline.len(), ALPHA),
        control_effect,
        comparisons,
    }
}

/// Everything Phase 10 writes to disk.
#[derive(Debug, Clone, Serialize)]
pub struct Phase10Report {
    /// Runs per condition.
    pub seeds: usize,
    /// Weight updates between signal recomputations.
    pub recompute_every: usize,
    /// Phase 10a.
    pub calibration: resilience::Calibration,
    /// Raw runs, one block per hyperparameter setting.
    pub conditions: Vec<Vec<ConditionResult>>,
    /// Analyses across the sensitivity sweep.
    pub analyses: Vec<MetricAnalysis>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_signal_converter_is_centred_on_one() {
        // Whatever the scale of the incoming signal, the precision it produces
        // must sit around one, or the condition changes the effective step size
        // rather than its modulation.
        for scale in [1e-3, 1.0, 1e3] {
            let mut c = SignalToPrecision::default();
            let mut rng = ChaChaRng::from_seed(5);
            let mut seen = Vec::new();
            for _ in 0..4000 {
                seen.push(c.observe(scale * rng.next_f64()));
            }
            let tail = &seen[500..];
            let mean = tail.iter().sum::<f64>() / tail.len() as f64;
            assert!((mean - 1.0).abs() < 0.1, "scale {scale} gave mean {mean}");
            assert!(tail.iter().all(|v| *v > 0.0 && *v < 2.0));
        }
    }

    #[test]
    fn shuffling_preserves_each_row_and_destroys_the_structure() {
        let mut rows = vec![
            (0..64).map(|i| i as f64).collect::<Vec<f64>>(),
            (0..64).map(|i| i as f64).collect::<Vec<f64>>(),
        ];
        let before = to_matrix(&rows);
        let mut rng = ChaChaRng::from_seed(3);
        shuffle_rows(&mut rows, &mut rng);
        for row in rows.iter() {
            let mut sorted = row.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(sorted, (0..64).map(|i| i as f64).collect::<Vec<f64>>());
        }
        // Identical rows are perfectly correlated before, and are not after.
        let after = to_matrix(&rows);
        let d_before = resilience::correlation_distances(&before);
        let d_after = resilience::correlation_distances(&after);
        assert!(d_before[(0, 1)] < 1e-12);
        assert!(d_after[(0, 1)] > 0.5, "got {}", d_after[(0, 1)]);
    }

    #[test]
    fn training_is_reproducible_from_the_seed() {
        let cfg = Config {
            layers: vec![2, 8, 8, 2],
            epochs: 2,
            inference_steps: 4,
            ..Config::default()
        };
        let data = make_moons(200, 0.20, 20_260_813);
        let (tr, va) = train_test_split(&data, 0.75, 20_260_814);
        let c = conditions(resilience::SWEEP[1], plasticity::SWEEP[0]);
        for condition in c.iter() {
            let a = train(condition, 3, &cfg, &tr, &va);
            let b = train(condition, 3, &cfg, &tr, &va);
            assert_eq!(
                a.final_val_loss.to_bits(),
                b.final_val_loss.to_bits(),
                "{}",
                condition.label
            );
        }
    }

    #[test]
    fn the_baseline_computes_no_signal_and_the_others_do() {
        let cfg = Config {
            layers: vec![2, 8, 8, 2],
            epochs: 1,
            inference_steps: 4,
            ..Config::default()
        };
        let data = make_moons(200, 0.20, 20_260_813);
        let (tr, va) = train_test_split(&data, 0.75, 20_260_814);
        let c = conditions(resilience::SWEEP[1], plasticity::SWEEP[0]);
        assert!(train(&c[0], 3, &cfg, &tr, &va).mean_signal.is_none());
        assert!(train(&c[1], 3, &cfg, &tr, &va).mean_signal.is_some());
        assert!(train(&c[2], 3, &cfg, &tr, &va).mean_signal.is_none());
    }
}
