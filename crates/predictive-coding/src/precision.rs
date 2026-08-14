// SPDX-License-Identifier: MIT
//! Precision weighting: the one quantity this phase varies.
//!
//! In the free energy formulation the term each prediction error contributes is
//! weighted by a precision, the inverse variance the model assigns to that
//! level. It is the natural place to inject a modulating signal because it is
//! the quantity the theory already treats as varying over time, rather than an
//! arbitrary hook.
//!
//! REF: [Friston, 2013] "Life as we know it", Journal of The Royal Society
//!      Interface 10(86), 20130475, DOI: 10.1098/rsif.2013.0475. Cited as the
//!      motivation for why precision is the principled entry point. This crate
//!      implements no part of that framework.

use chaos_rng::{Rng, RngKind};

/// Supplies the precision of every error level, once per inference step.
pub trait Precision {
    /// Precisions for the `depth` error levels, index `k` weighting the error
    /// of layer `k + 1`.
    fn next(&mut self, depth: usize) -> &[f64];
    /// Stable identifier for output files.
    fn label(&self) -> &'static str;
}

/// Precision fixed at one everywhere, the standard case in the literature and
/// the baseline of Phase 9a.
#[derive(Debug, Default, Clone)]
pub struct Constant;

impl Precision for Constant {
    fn next(&mut self, depth: usize) -> &[f64] {
        const ONES: [f64; 16] = [1.0; 16];
        assert!(
            depth <= ONES.len(),
            "network deeper than the constant table"
        );
        &ONES[..depth]
    }
    fn label(&self) -> &'static str {
        "constant"
    }
}

/// Steepness of the logistic map from a uniform variate to a precision.
///
/// At two the interquartile range of the resulting precision is roughly 0.75 to
/// 1.33, which is a visible modulation without ever approaching either bound.
pub const STEEPNESS: f64 = 2.0;

/// Maps a variate uniform on `[0, 1)` to a precision.
///
/// `g(u) = 2 / (1 + exp(-k(2u - 1)))`. Three properties are wanted and this is
/// about the simplest function with all three: it is strictly positive, so a
/// precision never vanishes or turns negative and the inference stays a descent
/// on a real energy; it is bounded above by two, so no single step can be
/// scaled arbitrarily; and it sends the median of the input to exactly one, so
/// a modulated condition has the same central precision as the constant
/// baseline and differs from it in dispersion rather than in level.
///
/// That last property is what keeps the comparison fair. A modulation whose
/// mean precision drifted away from one would change the effective learning
/// rate, and any difference found would be a difference of step size wearing a
/// theoretical costume.
pub fn logistic_precision(u: f64) -> f64 {
    2.0 / (1.0 + (-STEEPNESS * (2.0 * u - 1.0)).exp())
}

/// ChaCha12, used only as the negative control.
///
/// This duplicates a few lines that also exist in the Phase 8 crate rather than
/// depending on it. The two phases are separate lines of investigation and
/// coupling them so one could borrow a thirty-line generator would be the wrong
/// trade. The bit-to-float conversion is the same 53-bit construction the
/// qualified generators use, so the control differs from the ChaCha8 condition
/// in the round count and in nothing else.
#[derive(Debug, Clone)]
pub struct ChaCha12Stream {
    inner: rand_chacha::ChaCha12Rng,
}

impl ChaCha12Stream {
    /// Creates the stream from a 64-bit seed.
    pub fn from_seed(seed: u64) -> Self {
        use rand_chacha::rand_core::SeedableRng as _;
        Self {
            inner: rand_chacha::ChaCha12Rng::seed_from_u64(seed),
        }
    }

    fn next_f64(&mut self) -> f64 {
        use rand_chacha::rand_core::Rng as _;
        (self.inner.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Where a modulated condition draws its stream from.
///
/// Both variants are boxed. The two cipher states differ in size by enough that
/// an unboxed enum would carry the larger everywhere, and the stream is drawn
/// from often enough that keeping the value small matters more than saving one
/// indirection.
#[derive(Debug, Clone)]
enum Source {
    Qualified(Box<Rng>),
    Control(Box<ChaCha12Stream>),
}

/// Precision redrawn from a generator at every inference step.
#[derive(Debug, Clone)]
pub struct Modulated {
    source: Source,
    buffer: Vec<f64>,
    label: &'static str,
}

impl Modulated {
    /// Modulation driven by one of the four qualified generators.
    pub fn qualified(kind: RngKind, seed: u64) -> Self {
        Self {
            source: Source::Qualified(Box::new(Rng::new(kind, seed))),
            buffer: Vec::new(),
            label: kind.as_str(),
        }
    }

    /// The negative control: modulation driven by ChaCha12.
    ///
    /// It differs from the chacha8 condition only in the number of ChaCha
    /// rounds, eight against twelve, and both are cryptographically strong. Any
    /// difference measured between those two conditions is therefore noise, and
    /// its size is what this design registers when nothing of substance
    /// differs. That number is what makes a null result here interpretable
    /// rather than merely unrejected.
    pub fn control(seed: u64) -> Self {
        Self {
            source: Source::Control(Box::new(ChaCha12Stream::from_seed(seed))),
            buffer: Vec::new(),
            label: "chacha12-control",
        }
    }

    fn draw(&mut self) -> f64 {
        match &mut self.source {
            Source::Qualified(r) => r.next_f64(),
            Source::Control(r) => r.next_f64(),
        }
    }
}

impl Precision for Modulated {
    fn next(&mut self, depth: usize) -> &[f64] {
        self.buffer.resize(depth, 1.0);
        for i in 0..depth {
            let u = self.draw();
            self.buffer[i] = logistic_precision(u);
        }
        &self.buffer
    }
    fn label(&self) -> &'static str {
        self.label
    }
}

/// The six conditions, in report order.
pub fn conditions(seed: u64) -> Vec<Box<dyn Precision>> {
    vec![
        Box::new(Constant),
        Box::new(Modulated::qualified(RngKind::Lorenz, seed)),
        Box::new(Modulated::qualified(RngKind::ChaCha, seed)),
        Box::new(Modulated::qualified(RngKind::IfsLorenz, seed)),
        Box::new(Modulated::qualified(RngKind::IfsChaCha, seed)),
        Box::new(Modulated::control(seed)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_map_is_centred_on_one() {
        assert!((logistic_precision(0.5) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn the_map_stays_strictly_inside_its_bounds() {
        for i in 0..=1000 {
            let p = logistic_precision(i as f64 / 1000.0);
            assert!(p > 0.0 && p < 2.0, "u = {} gave {p}", i as f64 / 1000.0);
        }
    }

    #[test]
    fn every_condition_has_a_median_precision_of_one() {
        // The fairness of the comparison rests on this: the conditions differ
        // in the dispersion of the precision, not in its level, so none of them
        // is quietly running at a different effective step size.
        for mut condition in conditions(31) {
            let label = condition.label();
            if label == "constant" {
                continue;
            }
            let mut sample: Vec<f64> = Vec::new();
            for _ in 0..4000 {
                sample.extend_from_slice(condition.next(3));
            }
            sample.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
            let median = sample[sample.len() / 2];
            let mean = sample.iter().sum::<f64>() / sample.len() as f64;
            assert!((median - 1.0).abs() < 0.05, "{label} median {median}");
            assert!((mean - 1.0).abs() < 0.05, "{label} mean {mean}");
        }
    }

    #[test]
    fn a_modulated_condition_actually_varies() {
        // A stream that returned a constant would look exactly like a null
        // result and would be invisible in the statistics.
        let mut m = Modulated::qualified(RngKind::Lorenz, 5);
        let mut seen: Vec<f64> = Vec::new();
        for _ in 0..500 {
            seen.extend_from_slice(m.next(3));
        }
        let lo = seen.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = seen.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(hi - lo > 0.5, "precision spanned only {lo} to {hi}");
    }

    #[test]
    fn constant_precision_is_exactly_one() {
        let mut c = Constant;
        assert_eq!(c.next(3), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn the_control_differs_from_chacha8_but_matches_its_statistics() {
        let mut eight = Modulated::qualified(RngKind::ChaCha, 77);
        let mut twelve = Modulated::control(77);
        let a: Vec<f64> = (0..2000).flat_map(|_| eight.next(2).to_vec()).collect();
        let b: Vec<f64> = (0..2000).flat_map(|_| twelve.next(2).to_vec()).collect();
        assert_ne!(a, b, "the control reproduced the chacha8 stream");
        let ma = a.iter().sum::<f64>() / a.len() as f64;
        let mb = b.iter().sum::<f64>() / b.len() as f64;
        assert!((ma - mb).abs() < 0.02, "means {ma} and {mb} diverge");
    }
}
