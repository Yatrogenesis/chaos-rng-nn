// SPDX-License-Identifier: MIT
//! Phase 11: precision drawn from one continuous trajectory, read at a fixed
//! offset per level.
//!
//! **What changes from Phase 9, and what does not.** The network, the task, the
//! six generators, the logistic map from a variate to a precision, the number of
//! relaxation steps and every hyperparameter of training are unchanged. The only
//! difference is where the numbers come from.
//!
//! Phase 9 drew each level's precision independently, at each of the sixteen
//! relaxation steps of each sample. Nothing tied one draw to the next within a
//! level, and nothing tied one level to another. That gave the structure of a
//! generator, which is a property of its orbit over time, almost no way to
//! survive as far as the place it was supposed to act, and Phase 9's report said
//! so as an open limitation rather than a settled question.
//!
//! Here all three levels read from a single continuous orbit. Level `l` reads
//! the value at trajectory step `t + l * delta`, and `t` advances by one at
//! every relaxation step, without restarting between samples or between epochs.
//! The sixteen precisions a level sees while one sample relaxes are sixteen
//! consecutive states of that generator's own orbit, and the three levels are
//! the same orbit seen at three fixed offsets.

use crate::precision::{logistic_precision, ChaCha12Stream, Precision};
use chaos_rng::{Rng, RngKind};
use std::collections::VecDeque;

/// Where the trajectory comes from.
enum Source {
    Qualified(Box<Rng>),
    Control(Box<ChaCha12Stream>),
}

impl Source {
    fn draw(&mut self) -> f64 {
        match self {
            Source::Qualified(r) => r.next_f64(),
            Source::Control(r) => r.next_f64(),
        }
    }
}

/// One orbit, read at a fixed offset per level.
///
/// The buffer holds the window of the orbit that is currently addressable:
/// from the step the shallowest level is reading up to the step the deepest
/// level needs. It grows by drawing further along the orbit and shrinks from
/// the front as the shallowest level advances, so the memory is `(levels - 1)
/// * delta + 1` values however long training runs.
pub struct SharedTrajectory {
    source: Source,
    buffer: VecDeque<f64>,
    /// Trajectory index of `buffer[0]`.
    base: usize,
    /// Trajectory index level zero is reading now.
    step: usize,
    /// Offset in trajectory steps between consecutive levels.
    delta: usize,
    scratch: Vec<f64>,
    label: &'static str,
}

impl SharedTrajectory {
    /// A trajectory from one of the four qualified generators.
    pub fn qualified(kind: RngKind, seed: u64, delta: usize) -> Self {
        Self::new(
            Source::Qualified(Box::new(Rng::new(kind, seed))),
            delta,
            kind.as_str(),
        )
    }

    /// The negative control, ChaCha12, unchanged from Phase 9 in every respect
    /// but the sampling scheme.
    pub fn control(seed: u64, delta: usize) -> Self {
        Self::new(
            Source::Control(Box::new(ChaCha12Stream::from_seed(seed))),
            delta,
            "chacha12-control",
        )
    }

    fn new(source: Source, delta: usize, label: &'static str) -> Self {
        Self {
            source,
            buffer: VecDeque::new(),
            base: 0,
            step: 0,
            delta,
            scratch: Vec::new(),
            label,
        }
    }

    /// Offset in trajectory steps between consecutive levels.
    pub fn delta(&self) -> usize {
        self.delta
    }

    /// Reads the precisions for the current step without advancing.
    ///
    /// With `delta = 0` every level reads the same trajectory index and the
    /// three precisions are identical. That is the degenerate end of the sweep
    /// and it is deliberately reachable: it isolates the effect of reading one
    /// continuous orbit from the effect of separating the levels along it.
    fn fill(&mut self, levels: usize) {
        let deepest = self.step + levels.saturating_sub(1) * self.delta;
        while self.base + self.buffer.len() <= deepest {
            let v = self.source.draw();
            self.buffer.push_back(v);
        }
        while self.base < self.step {
            self.buffer.pop_front();
            self.base += 1;
        }
        self.scratch.clear();
        for level in 0..levels {
            let index = self.step + level * self.delta - self.base;
            self.scratch.push(logistic_precision(self.buffer[index]));
        }
    }

    /// The raw trajectory values each level is reading, before the logistic
    /// map. Used by the cross-correlation check.
    pub fn raw(&mut self, levels: usize) -> Vec<f64> {
        let deepest = self.step + levels.saturating_sub(1) * self.delta;
        while self.base + self.buffer.len() <= deepest {
            let v = self.source.draw();
            self.buffer.push_back(v);
        }
        (0..levels)
            .map(|l| self.buffer[self.step + l * self.delta - self.base])
            .collect()
    }

    /// Advances one trajectory step.
    pub fn advance(&mut self) {
        self.step += 1;
    }
}

impl Precision for SharedTrajectory {
    fn next(&mut self, depth: usize) -> &[f64] {
        // Every mutation happens before the borrow that is returned: the buffer
        // is extended and trimmed, the values are written into `scratch`, and
        // only then is the step advanced and the slice handed back. Advancing
        // here is what makes the orbit continuous across samples and epochs,
        // since nothing resets it between one sample and the next.
        self.fill(depth);
        self.step += 1;
        &self.scratch
    }

    fn label(&self) -> &'static str {
        self.label
    }
}

/// Offsets swept in this phase, in trajectory steps.
///
/// Zero is the degenerate reference where the levels are perfectly
/// synchronised. One separates them minimally. Four and sixteen separate them
/// by a quarter and by the whole of a sample's relaxation, so at sixteen the
/// value level one reads while a sample relaxes is the value level zero will
/// read while the next sample relaxes.
pub const DELTAS: [usize; 4] = [0, 1, 4, 16];

/// The six conditions at one offset, in report order.
pub fn conditions(seed: u64, delta: usize) -> Vec<Box<dyn Precision>> {
    vec![
        Box::new(crate::precision::Constant),
        Box::new(SharedTrajectory::qualified(RngKind::Lorenz, seed, delta)),
        Box::new(SharedTrajectory::qualified(RngKind::ChaCha, seed, delta)),
        Box::new(SharedTrajectory::qualified(RngKind::IfsLorenz, seed, delta)),
        Box::new(SharedTrajectory::qualified(RngKind::IfsChaCha, seed, delta)),
        Box::new(SharedTrajectory::control(seed, delta)),
    ]
}

/// One row of the cross-correlation check.
#[derive(Debug, Clone, serde::Serialize)]
pub struct OffsetCheck {
    /// Generator.
    pub condition: String,
    /// Offset the mechanism was configured with.
    pub delta: usize,
    /// Lag at which the cross-correlation between level zero and level one
    /// peaks.
    pub peak_lag: usize,
    /// Value of the peak.
    pub peak_value: f64,
    /// Whether the peak fell where the configuration says it should.
    pub peak_at_delta: bool,
    /// Median precision of level zero under this scheme.
    pub median_level_0: f64,
    /// Median precision of the deepest level.
    pub median_deepest: f64,
    /// Whether both medians are still one, the Phase 9 fairness property.
    pub medians_hold: f64,
}

/// Cross-correlation between two series at a lag, `corr(a[t + lag], b[t])`.
fn cross_correlation(a: &[f64], b: &[f64], lag: usize) -> f64 {
    let n = a.len().saturating_sub(lag);
    if n < 2 {
        return 0.0;
    }
    let x: Vec<f64> = (0..n).map(|t| a[t + lag]).collect();
    let y: Vec<f64> = (0..n).map(|t| b[t]).collect();
    let mx = x.iter().sum::<f64>() / n as f64;
    let my = y.iter().sum::<f64>() / n as f64;
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for (u, v) in x.iter().zip(y.iter()) {
        let (du, dv) = (u - mx, v - my);
        sxx += du * du;
        syy += dv * dv;
        sxy += du * dv;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return 0.0;
    }
    sxy / (sxx * syy).sqrt()
}

/// Samples used by the mechanism check.
pub const CHECK_STEPS: usize = 20_000;
/// Largest lag examined.
pub const MAX_LAG: usize = 24;
/// How far a median may sit from one before the fairness property is judged
/// broken.
pub const MEDIAN_TOLERANCE: f64 = 0.05;

/// Verifies that the mechanism does what it claims, before it is used.
///
/// Two checks, both blocking.
///
/// The offset must be real: the cross-correlation between level zero's
/// precision series and level one's must peak at the configured `delta` and not
/// somewhere else. An offset applied to the wrong index, or applied to the
/// relaxation counter instead of the trajectory, would still produce a
/// plausible-looking modulation and would make every result that followed a
/// measurement of something other than what was described.
///
/// The Phase 9 fairness property must survive: each level's median precision
/// must still be one. The logistic map is unchanged, so the marginal
/// distribution should be unchanged too, but correlation between levels is new
/// and the property is cheap to check and expensive to assume. Phase 9 and
/// Phase 10 both enforced their own version of this, and it is not inherited.
pub fn check_mechanism(delta: usize, levels: usize) -> Vec<OffsetCheck> {
    let mut out = Vec::new();
    for mut condition in conditions(4_242, delta) {
        let label = condition.label().to_string();
        if label == "constant" {
            continue;
        }
        let mut level_0 = Vec::with_capacity(CHECK_STEPS);
        let mut level_1 = Vec::with_capacity(CHECK_STEPS);
        let mut deepest = Vec::with_capacity(CHECK_STEPS);
        for _ in 0..CHECK_STEPS {
            let pi = condition.next(levels);
            level_0.push(pi[0]);
            level_1.push(pi[1.min(levels - 1)]);
            deepest.push(pi[levels - 1]);
        }

        let mut peak_lag = 0;
        let mut peak_value = f64::NEG_INFINITY;
        for lag in 0..=MAX_LAG {
            let c = cross_correlation(&level_0, &level_1, lag);
            if c > peak_value {
                peak_value = c;
                peak_lag = lag;
            }
        }

        let median = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).expect("precisions are finite"));
            v[v.len() / 2]
        };
        let m0 = median(level_0);
        let md = median(deepest);

        out.push(OffsetCheck {
            condition: label,
            delta,
            peak_lag,
            peak_value,
            peak_at_delta: peak_lag == delta,
            median_level_0: m0,
            median_deepest: md,
            medians_hold: if (m0 - 1.0).abs() < MEDIAN_TOLERANCE
                && (md - 1.0).abs() < MEDIAN_TOLERANCE
            {
                1.0
            } else {
                0.0
            },
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_offset_lands_where_it_is_configured() {
        // Blocking. An offset applied to the wrong index would still look like
        // a modulation and every later result would be measuring something
        // other than what is described.
        for delta in DELTAS {
            for row in check_mechanism(delta, 3) {
                assert!(
                    row.peak_at_delta,
                    "{} at delta {delta}: cross-correlation peaked at lag {} with {:.4}",
                    row.condition, row.peak_lag, row.peak_value
                );
            }
        }
    }

    #[test]
    fn the_fairness_property_of_phase_nine_survives() {
        // Blocking, and not inherited: the logistic map is unchanged but the
        // correlation between levels is new.
        for delta in DELTAS {
            for row in check_mechanism(delta, 3) {
                assert!(
                    row.medians_hold > 0.5,
                    "{} at delta {delta}: medians {:.4} and {:.4}",
                    row.condition,
                    row.median_level_0,
                    row.median_deepest
                );
            }
        }
    }

    #[test]
    fn zero_offset_makes_the_levels_identical() {
        let mut t = SharedTrajectory::qualified(RngKind::ChaCha, 7, 0);
        for _ in 0..200 {
            let pi = t.next(3);
            assert_eq!(pi[0], pi[1]);
            assert_eq!(pi[1], pi[2]);
        }
    }

    #[test]
    fn a_positive_offset_makes_them_differ() {
        let mut t = SharedTrajectory::qualified(RngKind::ChaCha, 7, 4);
        let mut identical = 0;
        for _ in 0..200 {
            let pi = t.next(3);
            if pi[0] == pi[1] {
                identical += 1;
            }
        }
        assert!(identical < 5, "levels coincided {identical} times in 200");
    }

    #[test]
    fn the_trajectory_is_continuous_across_calls() {
        // What level one reads now is what level zero reads delta steps later.
        // This is the property that makes the orbit continuous rather than
        // resampled, and it is the whole point of the phase.
        let delta = 4;
        let mut t = SharedTrajectory::qualified(RngKind::Lorenz, 11, delta);
        let mut level_1 = Vec::new();
        let mut level_0 = Vec::new();
        for _ in 0..100 {
            let pi = t.next(3);
            level_0.push(pi[0]);
            level_1.push(pi[1]);
        }
        for t0 in 0..(100 - delta) {
            assert!(
                (level_0[t0 + delta] - level_1[t0]).abs() < 1e-15,
                "at t = {t0}, level zero later read {} where level one read {}",
                level_0[t0 + delta],
                level_1[t0]
            );
        }
    }

    #[test]
    fn the_buffer_does_not_grow_without_bound() {
        let mut t = SharedTrajectory::qualified(RngKind::ChaCha, 3, 16);
        for _ in 0..50_000 {
            let _ = t.next(3);
        }
        assert!(
            t.buffer.len() <= 2 * 16 + 2,
            "buffer grew to {}",
            t.buffer.len()
        );
    }
}
