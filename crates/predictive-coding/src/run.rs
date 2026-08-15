// SPDX-License-Identifier: MIT
//! Phase 9a, the blocking check against backpropagation and the constant
//! precision baseline, and Phase 9b, the comparison across precision sources.

use crate::network::{Config, Network};
use crate::precision::{conditions, Constant, Precision};
use chaos_rng::ChaChaRng;
use experiment::dataset::{make_moons, train_test_split, Dataset};
use serde::Serialize;

/// Dataset parameters, identical to Phase 1 so that both phases see the same
/// points. The values are repeated here rather than imported because they are
/// private constants of that binary; the data they produce is shared through
/// the one implementation of the generator.
pub const DATASET_SEED: u64 = 20_260_813;
/// Seed of the train and validation split.
pub const SPLIT_SEED: u64 = 20_260_814;
/// Samples drawn.
pub const N_SAMPLES: usize = 2_000;
/// Standard deviation of the additive noise.
pub const NOISE: f64 = 0.20;
/// Fraction of the sample used for training.
pub const TRAIN_FRACTION: f64 = 0.75;
/// Runs per condition.
pub const SEEDS: usize = 20;
/// Significance level, unchanged from every earlier phase.
pub const ALPHA: f64 = 0.05;

/// One row of the Phase 9a table.
#[derive(Debug, Clone, Serialize)]
pub struct GateRow {
    /// Relaxation steps allowed before the update was read off.
    pub inference_steps: usize,
    /// Pearson correlation between the predictive coding update and the
    /// negated backpropagation gradient.
    pub correlation: f64,
    /// Cosine of the angle between the same two vectors.
    pub cosine: f64,
}

/// Result of the Phase 9a check.
#[derive(Debug, Clone, Serialize)]
pub struct Gate {
    /// The sweep over relaxation steps.
    pub rows: Vec<GateRow>,
    /// Correlation at the deepest relaxation.
    pub best_correlation: f64,
    /// Whether the correlation rose as the relaxation was allowed to run.
    pub improves_with_relaxation: bool,
    /// Whether the deepest relaxation cleared [`GATE_THRESHOLD`].
    pub clears_threshold: bool,
    /// Both conditions together.
    pub passes: bool,
}

/// Correlation the settled update must reach against the backpropagation
/// gradient.
///
/// Whittington and Bogacz show the predictive coding update converging on the
/// backpropagation gradient rather than equalling it, so an exact match is not
/// the expectation and would in fact suggest the inference loop had been
/// short-circuited. A high correlation is. The threshold is set where a working
/// implementation is comfortably above it and a wrong one has no chance of
/// reaching it.
///
/// REF: [Whittington and Bogacz, 2017] "An Approximation of the Error
///      Backpropagation Algorithm in a Predictive Coding Network with Local
///      Hebbian Synaptic Plasticity", Neural Computation 29(5), pp. 1229-1262,
///      DOI: 10.1162/NECO_a_00949. Title, journal, volume, issue, pages and
///      authors were checked against CrossRef rather than copied.
pub const GATE_THRESHOLD: f64 = 0.90;

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b.iter()) {
        let (da, db) = (x - ma, y - mb);
        saa += da * da;
        sbb += db * db;
        sab += da * db;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return 0.0;
    }
    sab / (saa * sbb).sqrt()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let na = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nb = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>() / (na * nb)
}

/// Phase 9a. Compares the update the network actually produces against the
/// gradient it is supposed to approximate.
///
/// The comparison is made on the same weights, the same input and the same
/// target, so the two vectors are directly commensurable. The predictive coding
/// side is the accumulated local update after the value nodes have relaxed; the
/// reference is the exact backpropagation gradient of the squared output error,
/// negated, since the local rule ascends the negative gradient.
///
/// It is run at a range of relaxation depths rather than at one, because the
/// claim in the literature is about convergence: an implementation that is
/// merely close at one arbitrary depth proves less than one whose agreement
/// improves as the inference is allowed to settle.
pub fn gate() -> Gate {
    let depths = [1usize, 2, 4, 8, 16, 32, 64, 128];
    let data = make_moons(64, NOISE, DATASET_SEED);
    let mut rows = Vec::new();

    for steps in depths {
        let mut correlations = Vec::new();
        let mut cosines = Vec::new();
        for trial in 0..8u64 {
            let cfg = Config {
                layers: vec![2, 8, 6, 2],
                inference_steps: steps,
                inference_rate: 0.1,
                ..Config::default()
            };
            let mut rng = ChaChaRng::from_seed(900 + trial);
            let net = Network::new(cfg, &mut rng);

            let idx = trial as usize % data.len();
            let input = data.x[idx];
            let mut target = vec![0.0; 2];
            target[data.y[idx]] = 1.0;

            let (gw, gb) = net.backprop_gradient(&input, &target);
            let mut inf = net.new_inference();
            net.feedforward(&input, &mut inf);
            let pi = net.infer(&target, &mut inf, &mut Constant);
            let (mut dw, mut db) = net.zero_accumulators();
            net.accumulate(&inf, &pi, &mut dw, &mut db);

            let mut pc = Vec::new();
            let mut bp = Vec::new();
            for l in 0..net.depth() {
                pc.extend_from_slice(&dw[l].data);
                bp.extend(gw[l].data.iter().map(|v| -v));
                pc.extend_from_slice(&db[l]);
                bp.extend(gb[l].iter().map(|v| -v));
            }
            correlations.push(pearson(&pc, &bp));
            cosines.push(cosine(&pc, &bp));
        }
        let n = correlations.len() as f64;
        rows.push(GateRow {
            inference_steps: steps,
            correlation: correlations.iter().sum::<f64>() / n,
            cosine: cosines.iter().sum::<f64>() / n,
        });
    }

    let best_correlation = rows.last().expect("the sweep is not empty").correlation;
    let improves_with_relaxation =
        rows.last().expect("non-empty").correlation > rows.first().expect("non-empty").correlation;
    let clears_threshold = best_correlation >= GATE_THRESHOLD;
    Gate {
        rows,
        best_correlation,
        improves_with_relaxation,
        clears_threshold,
        passes: improves_with_relaxation && clears_threshold,
    }
}

/// One trained network.
#[derive(Debug, Clone, Serialize)]
pub struct RunRecord {
    /// Precision condition.
    pub condition: String,
    /// Run seed.
    pub seed: u64,
    /// Cross-entropy on the validation set, the Phase 1 metric.
    pub final_val_loss: f64,
    /// Accuracy on the validation set.
    pub final_val_accuracy: f64,
    /// Validation loss minus training loss.
    pub generalisation_gap: f64,
}

/// Trains one network under one precision condition.
///
/// Everything except the precision is fixed by the seed and identical across
/// conditions: the initial weights, the order in which samples are presented,
/// and the data itself. The precision schedule is the only degree of freedom
/// the condition controls.
pub fn train(
    condition: &mut dyn Precision,
    seed: u64,
    cfg: &Config,
    train_set: &Dataset,
    val: &Dataset,
) -> RunRecord {
    let mut init = ChaChaRng::from_seed(seed);
    let mut net = Network::new(cfg.clone(), &mut init);
    let mut order: Vec<usize> = (0..train_set.len()).collect();
    let mut shuffler = ChaChaRng::from_seed(seed ^ 0x5DEE_CE66_D125_ABCD);

    for _ in 0..cfg.epochs {
        shuffler.shuffle(&mut order);
        for chunk in order.chunks(cfg.batch_size) {
            let (mut dw, mut db) = net.zero_accumulators();
            let mut inf = net.new_inference();
            for &i in chunk {
                let mut target = vec![0.0; Dataset::N_CLASSES];
                target[train_set.y[i]] = 1.0;
                net.feedforward(&train_set.x[i], &mut inf);
                let pi = net.infer(&target, &mut inf, condition);
                net.accumulate(&inf, &pi, &mut dw, &mut db);
            }
            net.apply(&dw, &db, chunk.len());
        }
    }

    let (final_val_loss, final_val_accuracy) = net.evaluate(&val.x, &val.y);
    let (final_train_loss, _) = net.evaluate(&train_set.x, &train_set.y);
    RunRecord {
        condition: condition.label().to_string(),
        seed,
        final_val_loss,
        final_val_accuracy,
        generalisation_gap: final_val_loss - final_train_loss,
    }
}

/// All runs of one condition.
#[derive(Debug, Clone, Serialize)]
pub struct ConditionResult {
    /// Precision condition.
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
    /// Validation accuracies in run order.
    pub fn accuracies(&self) -> Vec<f64> {
        self.runs.iter().map(|r| r.final_val_accuracy).collect()
    }
}

/// Phase 9b. Trains every condition over the same seeds.
pub fn run_conditions(seeds: usize, cfg: &Config) -> Vec<ConditionResult> {
    let data = make_moons(N_SAMPLES, NOISE, DATASET_SEED);
    let (train_set, val) = train_test_split(&data, TRAIN_FRACTION, SPLIT_SEED);

    let labels: Vec<&'static str> = conditions(0).iter().map(|c| c.label()).collect();
    labels
        .iter()
        .enumerate()
        .map(|(index, label)| {
            let runs = (0..seeds)
                .map(|s| {
                    let seed = 700 + s as u64;
                    // Each condition is rebuilt per run so its stream restarts
                    // from the run seed, matching how the earlier phases seed
                    // their generators.
                    let mut built = conditions(seed);
                    let condition = built
                        .get_mut(index)
                        .expect("the condition list is stable across calls");
                    train(condition.as_mut(), seed, cfg, &train_set, &val)
                })
                .collect();
            ConditionResult {
                condition: (*label).to_string(),
                runs,
            }
        })
        .collect()
}

/// One condition set against the constant precision baseline.
#[derive(Debug, Clone, Serialize)]
pub struct Comparison {
    /// Condition compared.
    pub condition: String,
    /// Mean of the baseline.
    pub baseline_mean: f64,
    /// Mean of this condition.
    pub condition_mean: f64,
    /// Test selected by the normality screen.
    pub test: String,
    /// Raw p-value.
    pub p_value: f64,
    /// Holm-adjusted p-value.
    pub p_holm: f64,
    /// Cohen's d.
    pub effect_size: f64,
}

/// The full analysis of one metric.
#[derive(Debug, Clone, Serialize)]
pub struct MetricAnalysis {
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
    /// Smallest effect detectable at eighty percent power, uncorrected.
    pub minimum_detectable_effect: f64,
    /// Effect between chacha8 and the ChaCha12 control, which differ only in
    /// round count and therefore measure this design's noise scale.
    pub negative_control_effect: f64,
}

/// Smallest standardised effect a two-sample comparison of `n` per group
/// detects with eighty percent power.
///
/// The normal approximation, which understates the requirement by a few percent
/// against the exact noncentral-t figure and therefore flatters the design.
pub fn minimum_detectable_effect(n: usize, alpha: f64) -> f64 {
    const POWER: f64 = 0.80;
    (xstats::normal_quantile(1.0 - alpha / 2.0) + xstats::normal_quantile(POWER))
        * (2.0 / n as f64).sqrt()
}

/// Runs the pre-registered test sequence on one metric.
pub fn analyse(results: &[ConditionResult], metric: &str) -> MetricAnalysis {
    let samples: Vec<Vec<f64>> = results
        .iter()
        .map(|r| match metric {
            "final_val_loss" => r.losses(),
            "generalisation_gap" => r.gaps(),
            "final_val_accuracy" => r.accuracies(),
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

    // The two cryptographic conditions differ only in the ChaCha round count,
    // so the effect between them is what this design registers from nothing.
    let index_of = |name: &str| results.iter().position(|r| r.condition == name);
    let negative_control_effect = match (index_of("chacha8"), index_of("chacha12-control")) {
        (Some(a), Some(b)) => xstats::cohens_d(&samples[a], &samples[b]).abs(),
        _ => f64::NAN,
    };

    MetricAnalysis {
        metric: metric.to_string(),
        all_normal,
        omnibus_test,
        omnibus_statistic,
        omnibus_p,
        any_significant: comparisons.iter().any(|c| c.p_holm < ALPHA),
        minimum_detectable_effect: minimum_detectable_effect(baseline.len(), ALPHA),
        negative_control_effect,
        comparisons,
    }
}

/// Everything Phase 9 writes to disk.
#[derive(Debug, Clone, Serialize)]
pub struct Phase9Report {
    /// Layer widths.
    pub layers: Vec<usize>,
    /// Relaxation steps per sample.
    pub inference_steps: usize,
    /// Inference step size.
    pub inference_rate: f64,
    /// Weight step size.
    pub learning_rate: f64,
    /// Training passes.
    pub epochs: usize,
    /// Runs per condition.
    pub seeds: usize,
    /// Phase 9a.
    pub gate: Gate,
    /// Phase 9b raw runs.
    pub conditions: Vec<ConditionResult>,
    /// Phase 9b analysis.
    pub analyses: Vec<MetricAnalysis>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_backprop_gate_passes() {
        // Blocking. Without this the comparison would run on a network that
        // does not approximate what it is supposed to approximate.
        let g = gate();
        assert!(
            g.improves_with_relaxation,
            "agreement did not improve as the inference settled: {:?}",
            g.rows
        );
        assert!(
            g.clears_threshold,
            "settled correlation was {} against a threshold of {GATE_THRESHOLD}: {:?}",
            g.best_correlation, g.rows
        );
    }

    #[test]
    fn training_is_reproducible_from_the_seed() {
        let cfg = Config {
            layers: vec![2, 8, 8, 2],
            epochs: 2,
            inference_steps: 4,
            ..Config::default()
        };
        let data = make_moons(200, NOISE, DATASET_SEED);
        let (tr, va) = train_test_split(&data, TRAIN_FRACTION, SPLIT_SEED);
        let run = || {
            let mut c = conditions(3);
            train(c[1].as_mut(), 3, &cfg, &tr, &va)
        };
        let a = run();
        let b = run();
        assert_eq!(a.final_val_loss.to_bits(), b.final_val_loss.to_bits());
        assert_eq!(
            a.generalisation_gap.to_bits(),
            b.generalisation_gap.to_bits()
        );
    }

    #[test]
    fn the_conditions_are_not_secretly_the_same_network() {
        // Only the precision differs, so the runs must still differ from each
        // other; if they did not, the comparison would be measuring nothing.
        let cfg = Config {
            layers: vec![2, 8, 8, 2],
            epochs: 2,
            inference_steps: 4,
            ..Config::default()
        };
        let data = make_moons(200, NOISE, DATASET_SEED);
        let (tr, va) = train_test_split(&data, TRAIN_FRACTION, SPLIT_SEED);
        let mut c = conditions(5);
        let constant = train(c[0].as_mut(), 5, &cfg, &tr, &va);
        let lorenz = train(c[1].as_mut(), 5, &cfg, &tr, &va);
        assert_ne!(constant.final_val_loss, lorenz.final_val_loss);
    }

    #[test]
    fn the_network_actually_learns_the_task() {
        // A network stuck at chance would make every condition equal for a
        // reason that has nothing to do with the question.
        let cfg = Config {
            layers: vec![2, 16, 16, 2],
            epochs: 10,
            ..Config::default()
        };
        let data = make_moons(400, NOISE, DATASET_SEED);
        let (tr, va) = train_test_split(&data, TRAIN_FRACTION, SPLIT_SEED);
        let mut c = conditions(9);
        let r = train(c[0].as_mut(), 9, &cfg, &tr, &va);
        assert!(
            r.final_val_accuracy > 0.8,
            "accuracy was only {}",
            r.final_val_accuracy
        );
    }
}
