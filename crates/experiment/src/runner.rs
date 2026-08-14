// SPDX-License-Identifier: MIT
//! Executes single training runs and whole conditions, and records their
//! outcomes.

use crate::mlp::{Config, Mlp};
use chaos_rng::{Rng, RngKind};
use experiment::dataset::Dataset;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Everything recorded about one training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    /// Generator used, either `lorenz` or `chacha8`.
    pub rng: String,
    /// Seed given to that generator.
    pub seed: u64,
    /// Mean training loss at the end of each epoch.
    pub train_loss_per_epoch: Vec<f64>,
    /// Cross-entropy on the validation split after the last epoch.
    pub final_val_loss: f64,
    /// Accuracy on the validation split after the last epoch.
    pub final_val_accuracy: f64,
    /// Training loss after the last epoch, kept to compute the generalisation
    /// gap.
    pub final_train_loss: f64,
    /// Difference between validation and training loss at the end.
    pub generalisation_gap: f64,
    /// Number of epochs needed for the validation loss to first fall below the
    /// threshold fixed in [`CONVERGENCE_THRESHOLD`], or `None` if it never did.
    pub epochs_to_threshold: Option<usize>,
    /// Wall-clock duration of the run in seconds.
    pub wall_clock_seconds: f64,
    /// SHA-256 of the final parameters, used for the reproducibility check.
    pub weight_hash: String,
}

/// Validation loss defining convergence for the speed metric. Fixed before any
/// run was executed, so it cannot be tuned to favour a condition.
pub const CONVERGENCE_THRESHOLD: f64 = 0.35;

/// Trains one network end to end, additionally returning the parameter vector
/// at the end of every epoch.
///
/// The randomness consumed is identical to [`run_once`]: snapshots are taken
/// from the network, never drawn from the generator, so a run recorded this way
/// is bit for bit the same run. Phase 3 verifies that rather than assuming it.
pub fn run_once_with_snapshots(
    kind: RngKind,
    seed: u64,
    train: &Dataset,
    val: &Dataset,
    cfg: Config,
) -> (RunRecord, Vec<Vec<f64>>) {
    let started = Instant::now();
    let mut rng = Rng::new(kind, seed);
    let mut net = Mlp::new(Dataset::N_FEATURES, Dataset::N_CLASSES, cfg, &mut rng);

    let mut train_loss_per_epoch = Vec::with_capacity(cfg.epochs);
    let mut epochs_to_threshold = None;
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut snapshots = Vec::with_capacity(cfg.epochs);

    for epoch in 0..cfg.epochs {
        rng.shuffle(&mut order);
        let mut epoch_loss = 0.0;
        let mut batches = 0usize;
        for chunk in order.chunks(cfg.batch_size) {
            let xs: Vec<[f64; 2]> = chunk.iter().map(|&i| train.x[i]).collect();
            let ys: Vec<usize> = chunk.iter().map(|&i| train.y[i]).collect();
            epoch_loss += net.train_batch(&xs, &ys, &mut rng);
            batches += 1;
        }
        train_loss_per_epoch.push(epoch_loss / batches as f64);
        snapshots.push(net.weight_vector());

        if epochs_to_threshold.is_none() {
            let (vl, _) = net.evaluate(&val.x, &val.y);
            if vl < CONVERGENCE_THRESHOLD {
                epochs_to_threshold = Some(epoch + 1);
            }
        }
    }

    let (final_val_loss, final_val_accuracy) = net.evaluate(&val.x, &val.y);
    let (final_train_loss, _) = net.evaluate(&train.x, &train.y);

    (
        RunRecord {
            rng: kind.as_str().to_string(),
            seed,
            train_loss_per_epoch,
            final_val_loss,
            final_val_accuracy,
            final_train_loss,
            generalisation_gap: final_val_loss - final_train_loss,
            epochs_to_threshold,
            wall_clock_seconds: started.elapsed().as_secs_f64(),
            weight_hash: net.weight_hash(),
        },
        snapshots,
    )
}

/// Trains one network end to end.
///
/// The single `rng` threaded through this function is the only source of
/// randomness in the run. It is consumed, in order, by weight initialisation,
/// then by minibatch shuffling and dropout masks within each epoch.
pub fn run_once(
    kind: RngKind,
    seed: u64,
    train: &Dataset,
    val: &Dataset,
    cfg: Config,
) -> RunRecord {
    let started = Instant::now();
    let mut rng = Rng::new(kind, seed);

    // Injection point one: initialisation.
    let mut net = Mlp::new(Dataset::N_FEATURES, Dataset::N_CLASSES, cfg, &mut rng);

    let mut train_loss_per_epoch = Vec::with_capacity(cfg.epochs);
    let mut epochs_to_threshold = None;
    let mut order: Vec<usize> = (0..train.len()).collect();

    for epoch in 0..cfg.epochs {
        // Injection point three: minibatch order.
        rng.shuffle(&mut order);

        let mut epoch_loss = 0.0;
        let mut batches = 0usize;
        for chunk in order.chunks(cfg.batch_size) {
            let xs: Vec<[f64; 2]> = chunk.iter().map(|&i| train.x[i]).collect();
            let ys: Vec<usize> = chunk.iter().map(|&i| train.y[i]).collect();
            // Injection point two happens inside, when dropout masks are drawn.
            epoch_loss += net.train_batch(&xs, &ys, &mut rng);
            batches += 1;
        }
        train_loss_per_epoch.push(epoch_loss / batches as f64);

        if epochs_to_threshold.is_none() {
            let (vl, _) = net.evaluate(&val.x, &val.y);
            if vl < CONVERGENCE_THRESHOLD {
                epochs_to_threshold = Some(epoch + 1);
            }
        }
    }

    let (final_val_loss, final_val_accuracy) = net.evaluate(&val.x, &val.y);
    let (final_train_loss, _) = net.evaluate(&train.x, &train.y);

    RunRecord {
        rng: kind.as_str().to_string(),
        seed,
        train_loss_per_epoch,
        final_val_loss,
        final_val_accuracy,
        final_train_loss,
        generalisation_gap: final_val_loss - final_train_loss,
        epochs_to_threshold,
        wall_clock_seconds: started.elapsed().as_secs_f64(),
        weight_hash: net.weight_hash(),
    }
}

/// Runs one condition `n_runs` times, with a distinct seed per run.
///
/// Both conditions receive the same list of seeds, so any difference between
/// them cannot come from the seeds themselves.
pub fn run_condition(
    kind: RngKind,
    seeds: &[u64],
    train: &Dataset,
    val: &Dataset,
    cfg: Config,
) -> Vec<RunRecord> {
    seeds
        .iter()
        .map(|&s| run_once(kind, s, train, val, cfg))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use experiment::dataset::{make_moons, train_test_split};

    #[test]
    fn a_run_is_reproducible_bit_for_bit() {
        let data = make_moons(200, 0.2, 1);
        let (tr, va) = train_test_split(&data, 0.8, 1);
        let cfg = Config {
            epochs: 3,
            ..Config::default()
        };
        let a = run_once(RngKind::Lorenz, 99, &tr, &va, cfg);
        let b = run_once(RngKind::Lorenz, 99, &tr, &va, cfg);
        assert_eq!(a.weight_hash, b.weight_hash);
        assert_eq!(a.final_val_loss.to_bits(), b.final_val_loss.to_bits());
    }

    #[test]
    fn snapshot_runs_match_the_published_records() {
        let raw = std::fs::read_to_string("../../results/phase1_runs.json")
            .or_else(|_| std::fs::read_to_string("results/phase1_runs.json"))
            .expect("published records are readable");
        let published: Vec<RunRecord> = serde_json::from_str(&raw).unwrap();
        let data = make_moons(2_000, 0.20, 20_260_813);
        let (tr, va) = train_test_split(&data, 0.75, 20_260_814);
        let cfg = Config::default();
        for kind in [RngKind::Lorenz, RngKind::ChaCha] {
            for seed in [1000u64, 1001, 1002] {
                let (snap, _) = run_once_with_snapshots(kind, seed, &tr, &va, cfg);
                let prior = published
                    .iter()
                    .find(|r| r.rng == kind.as_str() && r.seed == seed)
                    .unwrap();
                assert_eq!(
                    snap.weight_hash, prior.weight_hash,
                    "{:?} seed {seed} does not match the published record",
                    kind
                );
            }
        }
    }

    #[test]
    fn snapshots_do_not_perturb_the_run() {
        // The snapshot variant must consume randomness identically to the plain
        // one, otherwise Phase 3 would be measuring a different experiment from
        // the one Phase 1 reported.
        let data = make_moons(2_000, 0.20, 20_260_813);
        let (tr, va) = train_test_split(&data, 0.75, 20_260_814);
        let cfg = Config::default();
        for seed in [1000u64, 1001, 1002, 1006, 1009] {
            for kind in [RngKind::Lorenz, RngKind::ChaCha] {
                let plain = run_once(kind, seed, &tr, &va, cfg);
                let (snap, _) = run_once_with_snapshots(kind, seed, &tr, &va, cfg);
                assert_eq!(
                    plain.weight_hash, snap.weight_hash,
                    "{:?} seed {seed}: snapshotting changed the trajectory",
                    kind
                );
            }
        }
    }

    #[test]
    fn different_generators_take_different_paths() {
        let data = make_moons(200, 0.2, 1);
        let (tr, va) = train_test_split(&data, 0.8, 1);
        let cfg = Config {
            epochs: 3,
            ..Config::default()
        };
        let a = run_once(RngKind::Lorenz, 99, &tr, &va, cfg);
        let b = run_once(RngKind::ChaCha, 99, &tr, &va, cfg);
        assert_ne!(a.weight_hash, b.weight_hash);
    }
}
