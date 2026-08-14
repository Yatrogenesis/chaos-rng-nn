// SPDX-License-Identifier: MIT
//! Where the reservoir weights come from. This is the only thing the phase
//! varies.

use crate::esn::{rescale_spectral_radius, Esn};
use chaos_rng::{Rng, RngKind};
use nalgebra::DMatrix;

/// The source filling the recurrent matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightSource {
    /// The canonical echo state network: entries drawn i.i.d. from the
    /// reference generator, see [`crate::reference::ReferenceRng`]. This is the
    /// literature baseline that 8a must reproduce before anything else is
    /// believed.
    Standard,
    /// Entries drawn from one of the generators qualified in Phase 0.
    Generator(RngKind),
}

impl WeightSource {
    /// Stable identifier used in the results files and the report.
    pub fn as_str(self) -> &'static str {
        match self {
            WeightSource::Standard => "standard-iid",
            WeightSource::Generator(k) => k.as_str(),
        }
    }
}

/// The five conditions, in the order used throughout the report.
pub const CONDITIONS: [WeightSource; 5] = [
    WeightSource::Standard,
    WeightSource::Generator(RngKind::Lorenz),
    WeightSource::Generator(RngKind::ChaCha),
    WeightSource::Generator(RngKind::IfsLorenz),
    WeightSource::Generator(RngKind::IfsChaCha),
];

/// A stream of weights uniform on `[-1, 1]`, from whichever source.
///
/// Both variants are boxed. The stream is drawn from once per matrix entry, so
/// the indirection is paid ten thousand times against a build that takes far
/// longer in the eigenvalue solve, and it keeps the enum from carrying the
/// larger of two very different cipher states in every value.
enum Stream {
    Standard(Box<crate::reference::ReferenceRng>),
    Generator(Box<Rng>),
}

impl Stream {
    fn new(source: WeightSource, seed: u64) -> Self {
        match source {
            WeightSource::Standard => {
                Stream::Standard(Box::new(crate::reference::ReferenceRng::from_seed(seed)))
            }
            WeightSource::Generator(kind) => Stream::Generator(Box::new(Rng::new(kind, seed))),
        }
    }

    /// Next weight, uniform on `[-1, 1]`.
    ///
    /// Both branches map a variate on `[0, 1)` through the same affine
    /// transform, so the marginal distribution is identical across conditions
    /// by construction and any difference that appears has to come from the
    /// dependence structure of the stream rather than from its histogram.
    fn next(&mut self) -> f64 {
        let u = match self {
            Stream::Standard(r) => r.next_f64(),
            Stream::Generator(r) => r.next_f64(),
        };
        2.0 * u - 1.0
    }
}

/// The reservoir geometry held fixed across every condition.
#[derive(Debug, Clone, Copy)]
pub struct Geometry {
    /// Reservoir size.
    pub n: usize,
    /// Spectral radius every condition is rescaled to.
    pub spectral_radius: f64,
    /// Multiplier on the input weights.
    pub input_scaling: f64,
}

/// Builds a reservoir of the given geometry from the given source.
///
/// Two choices are worth stating rather than leaving implicit.
///
/// **The matrix is dense.** Sparse reservoirs are common and are what Jaeger
/// originally proposed, mostly for speed. Sparsity here would be a second
/// variable, and it would also decide how much of the generator's stream ever
/// reaches the matrix. Dense means every one of the `n^2` entries comes from
/// the stream in order, which gives whatever correlation structure the
/// generator carries the largest surface on which to show itself. If the
/// source is going to matter anywhere, it should matter here.
///
/// **The input weights always come from the reference generator**, whatever
/// fills the recurrent matrix, and they are drawn from a separate stream seeded
/// independently. The phase asks about the fixed recurrent dynamics, so that is
/// the only thing allowed to differ; letting the input matrix vary too would
/// confound the two.
///
/// The returned reservoir is rescaled to the target spectral radius, and the
/// radius it had before rescaling is returned alongside, because that value is
/// itself a property of the source worth reporting.
pub fn build(source: WeightSource, geometry: Geometry, seed: u64) -> (Esn, f64) {
    let Geometry {
        n,
        spectral_radius,
        input_scaling,
    } = geometry;

    let mut stream = Stream::new(source, seed);
    // Filled in row-major order so the stream's consecutive values land in
    // consecutive entries of a row.
    let mut w_res = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            w_res[(i, j)] = stream.next();
        }
    }
    let raw_radius = rescale_spectral_radius(&mut w_res, spectral_radius);

    let mut input_stream = Stream::new(WeightSource::Standard, seed ^ INPUT_SEED_MASK);
    let w_in = DMatrix::from_fn(n, 1, |_, _| input_scaling * input_stream.next());

    (Esn { w_res, w_in, n }, raw_radius)
}

/// Separates the input stream from the recurrent stream for the standard
/// condition, which would otherwise draw both from the same sequence and make
/// the two matrices share structure that no other condition shares.
const INPUT_SEED_MASK: u64 = 0x9E37_79B9_7F4A_7C15;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esn::spectral_radius;

    const GEOMETRY: Geometry = Geometry {
        n: 40,
        spectral_radius: 0.9,
        input_scaling: 0.1,
    };

    #[test]
    fn every_condition_lands_on_the_target_radius() {
        for source in CONDITIONS {
            let (esn, raw) = build(source, GEOMETRY, 42);
            assert!(
                (spectral_radius(&esn.w_res) - GEOMETRY.spectral_radius).abs() < 1e-9,
                "{} missed the target radius",
                source.as_str()
            );
            assert!(
                raw > 0.0,
                "{} produced a degenerate matrix",
                source.as_str()
            );
        }
    }

    #[test]
    fn the_conditions_differ_from_one_another() {
        // If two sources produced the same matrix the comparison would be
        // vacuous, and a wiring mistake that fed every condition from the same
        // stream would look exactly like a null result.
        let built: Vec<_> = CONDITIONS
            .iter()
            .map(|s| build(*s, GEOMETRY, 42).0)
            .collect();
        for i in 0..built.len() {
            for j in i + 1..built.len() {
                let diff = (&built[i].w_res - &built[j].w_res).norm();
                assert!(
                    diff > 1e-6,
                    "{} and {} produced the same recurrent matrix",
                    CONDITIONS[i].as_str(),
                    CONDITIONS[j].as_str()
                );
            }
        }
    }

    #[test]
    fn the_input_matrix_is_identical_across_conditions() {
        // The isolation of the variable depends on this.
        let reference = build(CONDITIONS[0], GEOMETRY, 42).0.w_in;
        for source in CONDITIONS.iter().skip(1) {
            let other = build(*source, GEOMETRY, 42).0.w_in;
            assert!(
                (&reference - &other).norm() < 1e-15,
                "{} received a different input matrix",
                source.as_str()
            );
        }
    }

    #[test]
    fn building_is_deterministic_from_the_seed() {
        for source in CONDITIONS {
            let a = build(source, GEOMETRY, 7).0;
            let b = build(source, GEOMETRY, 7).0;
            assert_eq!(a.w_res, b.w_res, "{} is not reproducible", source.as_str());
            assert_eq!(a.w_in, b.w_in);
        }
    }

    #[test]
    fn different_seeds_give_different_reservoirs() {
        for source in CONDITIONS {
            let a = build(source, GEOMETRY, 7).0;
            let b = build(source, GEOMETRY, 8).0;
            assert!(
                (&a.w_res - &b.w_res).norm() > 1e-6,
                "{} ignored the seed",
                source.as_str()
            );
        }
    }

    #[test]
    fn the_weights_span_the_intended_range() {
        for source in CONDITIONS {
            // Before rescaling the entries are uniform on [-1, 1]; check the
            // mean and the extremes of the pre-rescaling stream directly.
            let mut stream = Stream::new(source, 3);
            let sample: Vec<f64> = (0..20_000).map(|_| stream.next()).collect();
            let mean = sample.iter().sum::<f64>() / sample.len() as f64;
            let lo = sample.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = sample.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert!(
                lo >= -1.0 && hi <= 1.0,
                "{} left the range",
                source.as_str()
            );
            assert!(
                lo < -0.99 && hi > 0.99,
                "{} did not fill it",
                source.as_str()
            );
            assert!(mean.abs() < 0.03, "{} mean was {mean}", source.as_str());
        }
    }
}
