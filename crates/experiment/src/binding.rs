// SPDX-License-Identifier: MIT
//! Phase 5: holographic against non-holographic binding of training
//! trajectories.
//!
//! Two vector-symbolic schemes are compared on the same task: store the sixty
//! per-epoch weight vectors of a training run in a single fixed-width trace,
//! then recover individual epochs from it, with and without corruption of the
//! trace. The schemes differ in one respect only, the binding operation.
//!
//! REF: [Plate, 1995] "Holographic Reduced Representations", IEEE Transactions
//!      on Neural Networks 6(3), pp. 623-641
//!      DOI: 10.1109/72.377968
//!      Binding by circular convolution, B = x conv y, computed as
//!      F^-1(F(x) . F(y)), with unbinding by circular correlation against the
//!      approximate inverse.
//!
//! REF: [Gayler, 1998] "Multiplicative Binding, Representation Operators and
//!      Analogy", in Advances in Analogy Research, New Bulgarian University.
//!      https://arxiv.org/abs/cs/0412059
//!      The MAP architecture: binding by element-wise multiplication, bundling
//!      by addition, the same approximate retrieval, no transform involved.
//!
//! Why the comparison is not decided in advance. Both schemes are analysed
//! under the assumption that the stored items are close to orthogonal, which
//! random high-dimensional vectors are. Real training trajectories are not:
//! Phase 3 measured their effective dimension at about 2.3 against 2178
//! nominal, and consecutive epochs are strongly correlated because the
//! optimiser moves smoothly. The regime here therefore violates the premise
//! under which either scheme is guaranteed to behave, which is what makes the
//! question worth asking rather than answering from the literature.
//!
//! The FFT comes from `yatrosci-fft`, the author's own crate, which delegates
//! to a mixed-radix implementation for lengths that are not powers of two. That
//! matters here: the vectors are 2178 wide, which factors as 2 * 3^2 * 11^2, so
//! a radix-2 transform would have to pad, and padding turns circular
//! convolution into linear convolution, which would break unbinding outright.

use ndarray::Array1;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Which binding operation a trace uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Scheme {
    /// Circular convolution in the frequency domain, after Plate.
    Hrr,
    /// Element-wise multiplication, after Gayler.
    Map,
}

impl Scheme {
    /// Stable identifier for output files.
    pub fn as_str(self) -> &'static str {
        match self {
            Scheme::Hrr => "hrr",
            Scheme::Map => "map",
        }
    }
}

/// Discrete Fourier transform of a real vector.
fn forward(v: &[f64]) -> Array1<Complex64> {
    let a = Array1::from_vec(v.to_vec());
    yatrosci_fft::fft(&a.view(), None)
}

/// Inverse transform, discarding the imaginary residue left by rounding.
fn inverse(spectrum: &Array1<Complex64>) -> Vec<f64> {
    yatrosci_fft::ifft(&spectrum.view(), None)
        .iter()
        .map(|c| c.re)
        .collect()
}

/// Circular convolution, the HRR binding operation.
pub fn circular_convolution(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "binding needs equal widths");
    let (fa, fb) = (forward(a), forward(b));
    let prod: Array1<Complex64> = fa
        .iter()
        .zip(fb.iter())
        .map(|(x, y)| x * y)
        .collect::<Vec<_>>()
        .into();
    inverse(&prod)
}

/// Circular correlation, the HRR unbinding operation.
///
/// Correlation with `a` is convolution with its involution, which in the
/// frequency domain is the complex conjugate. Plate calls this the approximate
/// inverse: it is exact only when the spectrum has unit magnitude, and for
/// Gaussian vectors it is close enough that retrieval works, which is precisely
/// what the calibration measures.
pub fn circular_correlation(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "unbinding needs equal widths");
    let (fa, fb) = (forward(a), forward(b));
    let prod: Array1<Complex64> = fa
        .iter()
        .zip(fb.iter())
        .map(|(x, y)| x.conj() * y)
        .collect::<Vec<_>>()
        .into();
    inverse(&prod)
}

/// Element-wise product, the MAP binding operation.
pub fn elementwise_bind(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "binding needs equal widths");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Element-wise unbinding, dividing by the key.
///
/// The exact inverse of multiplication is division, which is unstable wherever
/// a key component approaches zero. A key drawn from N(0, 1/d) has components
/// near zero with non-negligible probability, so the divisor is floored away
/// from zero. Without that floor a single small component turns into an
/// enormous one and destroys the retrieved vector, which would misattribute an
/// implementation artefact to the scheme.
#[allow(
    dead_code,
    reason = "kept as the documented counter-example: this is what fails"
)]
pub fn elementwise_unbind(key: &[f64], trace: &[f64]) -> Vec<f64> {
    assert_eq!(key.len(), trace.len(), "unbinding needs equal widths");
    let floor = 1e-6;
    key.iter()
        .zip(trace.iter())
        .map(|(k, t)| {
            let d = if k.abs() < floor {
                floor.copysign(if *k == 0.0 { 1.0 } else { *k })
            } else {
                *k
            };
            t / d
        })
        .collect()
}

/// Cosine similarity, the fidelity measure used throughout this phase.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Generates a key vector with components from N(0, 1/d), the distribution
/// under which both schemes are analysed.
pub fn make_key(d: usize, rng: &mut chaos_rng::Rng) -> Vec<f64> {
    let scale = 1.0 / (d as f64).sqrt();
    (0..d).map(|_| rng.next_normal() * scale).collect()
}

/// Generates a bipolar key with components drawn uniformly from {-1, +1}.
///
/// This is the key distribution MAP is defined over, and using it is not a
/// detail. With Gaussian keys the natural inverse of element-wise
/// multiplication is division, and the crosstalk terms then carry ratios of
/// Gaussians, which are Cauchy distributed and have no finite variance. The
/// calibration showed exactly that failure: retrieval fidelity of 0.02, 0.008,
/// -0.003 and 0.002 for bundles of 5, 15, 30 and 60 items, which is to say
/// nothing at all. With bipolar components the square of a key is the all-ones
/// vector, so multiplication is its own exact inverse and the crosstalk stays
/// bounded.
///
/// REF: [Gayler, 1998] "Multiplicative Binding, Representation Operators and
///      Analogy", https://arxiv.org/abs/cs/0412059
pub fn make_bipolar_key(d: usize, rng: &mut chaos_rng::Rng) -> Vec<f64> {
    (0..d)
        .map(|_| if rng.next_f64() < 0.5 { -1.0 } else { 1.0 })
        .collect()
}

/// Generates a unitary key: a vector whose Fourier magnitudes are all one, with
/// random phases.
///
/// This exists for fairness, not convenience. Unbinding in HRR is usually done
/// with the involution, whose spectrum is the complex conjugate of the key's.
/// That is an approximate inverse: the round trip multiplies each frequency by
/// |F(k)|^2, which for a Gaussian key is an exponential variable of mean one
/// rather than the constant one. The resulting retrieval similarity converges to
/// E[W]/sqrt(E[W^2]) with W exponential, that is 1/sqrt(2) = 0.7071, and it does
/// not improve with width. Measured here at 0.7272, 0.7196, 0.7051, 0.7037 and
/// 0.7062 for widths 64, 256, 1024, 2178 and 8192: flat, and converging on the
/// predicted constant.
///
/// MAP, by contrast, unbinds by division, which is its exact inverse. Comparing
/// an approximate inverse against an exact one would measure that asymmetry
/// rather than the property under study. With a unitary key the conjugate is the
/// exact inverse, so both schemes retrieve exactly in the noiseless case and the
/// comparison isolates the binding operation itself.
///
/// REF: [Plate, 1995] section on unitary vectors, DOI 10.1109/72.377968
pub fn make_unitary_key(d: usize, rng: &mut chaos_rng::Rng) -> Vec<f64> {
    // Random phases, conjugate-symmetric so the inverse transform is real.
    let mut spectrum = vec![Complex64::new(0.0, 0.0); d];
    spectrum[0] = Complex64::new(1.0, 0.0);
    if d % 2 == 0 {
        spectrum[d / 2] = Complex64::new(if rng.next_f64() < 0.5 { -1.0 } else { 1.0 }, 0.0);
    }
    let half = (d - 1) / 2;
    for i in 1..=half {
        let theta = rng.next_f64() * std::f64::consts::TAU;
        let c = Complex64::new(theta.cos(), theta.sin());
        spectrum[i] = c;
        spectrum[d - i] = c.conj();
    }
    let arr = Array1::from_vec(spectrum);
    yatrosci_fft::ifft(&arr.view(), None)
        .iter()
        .map(|c| c.re)
        .collect()
}

/// Bundles a set of items into one trace by binding each to its key and
/// summing.
pub fn bundle(scheme: Scheme, keys: &[Vec<f64>], items: &[Vec<f64>]) -> Vec<f64> {
    assert_eq!(keys.len(), items.len(), "one key per item");
    let d = items[0].len();
    let mut trace = vec![0.0f64; d];
    for (k, item) in keys.iter().zip(items.iter()) {
        let bound = match scheme {
            Scheme::Hrr => circular_convolution(k, item),
            Scheme::Map => elementwise_bind(k, item),
        };
        for (t, b) in trace.iter_mut().zip(bound.iter()) {
            *t += b;
        }
    }
    trace
}

/// Recovers the item bound to `key` from a trace.
pub fn retrieve(scheme: Scheme, key: &[f64], trace: &[f64]) -> Vec<f64> {
    match scheme {
        Scheme::Hrr => circular_correlation(key, trace),
        // Multiplication is its own inverse for a bipolar key, so this is
        // exact, matching the exactness of unbinding a unitary key in HRR.
        Scheme::Map => elementwise_bind(key, trace),
    }
}

/// How a trace is damaged before retrieval is attempted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Corruption {
    /// A fraction of components set to zero.
    ///
    /// Chosen as the primary model because it represents loss of stored
    /// content, which is the failure mode a distributed representation is
    /// supposed to tolerate, and because it has no free scale parameter: the
    /// only quantity is the fraction lost, so the two schemes cannot be
    /// separated by an arbitrary choice of noise magnitude.
    Erase,
    /// Additive Gaussian noise whose standard deviation is a multiple of the
    /// trace's own, reported alongside erasure so the conclusion does not rest
    /// on a single damage model.
    Noise,
}

/// Applies corruption to a copy of the trace.
pub fn corrupt(
    trace: &[f64],
    kind: Corruption,
    fraction: f64,
    rng: &mut chaos_rng::Rng,
) -> Vec<f64> {
    let mut out = trace.to_vec();
    match kind {
        Corruption::Erase => {
            let mut idx: Vec<usize> = (0..out.len()).collect();
            rng.shuffle(&mut idx);
            let n = (out.len() as f64 * fraction).round() as usize;
            for &i in idx.iter().take(n) {
                out[i] = 0.0;
            }
        }
        Corruption::Noise => {
            let sd = {
                let m = trace.iter().sum::<f64>() / trace.len() as f64;
                (trace.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / trace.len() as f64).sqrt()
            };
            for v in out.iter_mut() {
                *v += rng.next_normal() * sd * fraction;
            }
        }
    }
    out
}

/// One point of a calibration curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationPoint {
    /// Scheme measured.
    pub scheme: String,
    /// Number of items bundled into the trace.
    pub items: usize,
    /// Mean cosine similarity of retrieval across probes and repeats.
    pub mean_fidelity: f64,
    /// Standard deviation of that similarity.
    pub std_dev: f64,
}

/// Calibration in the regime where both schemes are analysed: independent
/// items drawn from N(0, 1/d).
///
/// This is the blocking gate of Phase 5a. If a scheme cannot retrieve here, the
/// implementation is wrong and nothing measured on real trajectories would mean
/// anything.
#[allow(
    dead_code,
    reason = "the Gaussian-key variant, retained for the measurement recorded in its documentation"
)]
pub fn calibrate(
    scheme: Scheme,
    d: usize,
    item_counts: &[usize],
    repeats: usize,
    rng: &mut chaos_rng::Rng,
) -> Vec<CalibrationPoint> {
    let mut out = Vec::new();
    for &m in item_counts {
        let mut fidelities = Vec::with_capacity(repeats * m.min(5));
        for _ in 0..repeats {
            let keys: Vec<Vec<f64>> = (0..m).map(|_| make_key(d, rng)).collect();
            let items: Vec<Vec<f64>> = (0..m).map(|_| make_key(d, rng)).collect();
            let trace = bundle(scheme, &keys, &items);
            // Probe a spread of positions rather than only the first.
            for p in probe_positions(m) {
                let got = retrieve(scheme, &keys[p], &trace);
                fidelities.push(cosine_similarity(&got, &items[p]));
            }
        }
        out.push(CalibrationPoint {
            scheme: scheme.as_str().to_string(),
            items: m,
            mean_fidelity: xstats::mean(&fidelities),
            std_dev: xstats::std_dev(&fidelities),
        });
    }
    out
}

/// Calibration using unitary keys, the variant used by the real comparison.
pub fn calibrate_unitary(
    scheme: Scheme,
    d: usize,
    item_counts: &[usize],
    repeats: usize,
    rng: &mut chaos_rng::Rng,
) -> Vec<CalibrationPoint> {
    let mut out = Vec::new();
    for &m in item_counts {
        let mut fidelities = Vec::new();
        for _ in 0..repeats {
            // Each scheme gets the key distribution it is defined over, so
            // both unbind exactly in the noiseless case.
            let keys: Vec<Vec<f64>> = (0..m)
                .map(|_| match scheme {
                    Scheme::Hrr => make_unitary_key(d, rng),
                    Scheme::Map => make_bipolar_key(d, rng),
                })
                .collect();
            let items: Vec<Vec<f64>> = (0..m).map(|_| make_key(d, rng)).collect();
            let trace = bundle(scheme, &keys, &items);
            for p in probe_positions(m) {
                let got = retrieve(scheme, &keys[p], &trace);
                fidelities.push(cosine_similarity(&got, &items[p]));
            }
        }
        out.push(CalibrationPoint {
            scheme: scheme.as_str().to_string(),
            items: m,
            mean_fidelity: xstats::mean(&fidelities),
            std_dev: xstats::std_dev(&fidelities),
        });
    }
    out
}

/// Positions probed within a bundle of `m` items: first, quarter, middle,
/// three quarters and last, deduplicated for small `m`.
pub fn probe_positions(m: usize) -> Vec<usize> {
    let mut v = vec![0, m / 4, m / 2, (3 * m) / 4, m - 1];
    v.sort_unstable();
    v.dedup();
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use chaos_rng::{Rng, RngKind};

    const D: usize = 256;

    #[test]
    fn circular_convolution_has_an_identity() {
        // Convolving with the delta at position zero must return the input.
        let mut rng = Rng::new(RngKind::ChaCha, 1);
        let x = make_key(D, &mut rng);
        let mut delta = vec![0.0; D];
        delta[0] = 1.0;
        let y = circular_convolution(&x, &delta);
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-9, "identity failed: {a} against {b}");
        }
    }

    #[test]
    fn circular_convolution_is_commutative() {
        let mut rng = Rng::new(RngKind::ChaCha, 2);
        let a = make_key(D, &mut rng);
        let b = make_key(D, &mut rng);
        let ab = circular_convolution(&a, &b);
        let ba = circular_convolution(&b, &a);
        for (x, y) in ab.iter().zip(ba.iter()) {
            assert!((x - y).abs() < 1e-9);
        }
    }

    #[test]
    fn convolution_length_is_preserved_for_a_non_power_of_two() {
        // 2178 is the real width and is not a power of two. A radix-2 transform
        // would have to pad, and padding would silently change the operation
        // from circular to linear convolution.
        let mut rng = Rng::new(RngKind::ChaCha, 3);
        let a = make_key(2178, &mut rng);
        let b = make_key(2178, &mut rng);
        assert_eq!(circular_convolution(&a, &b).len(), 2178);
    }

    #[test]
    fn hrr_with_a_gaussian_key_hits_the_theoretical_constant() {
        // With a Gaussian key the involution is only an approximate inverse:
        // the round trip scales each frequency by |F(k)|^2, an exponential
        // variable of mean one, and the resulting similarity converges to
        // E[W]/sqrt(E[W^2]) = 1/sqrt(2) = 0.7071 regardless of width. The
        // assertion is against that constant rather than against a hopeful
        // threshold, because matching it is what shows the implementation is
        // right. See make_unitary_key for the fair-comparison variant.
        let mut rng = Rng::new(RngKind::ChaCha, 4);
        let mut acc = 0.0;
        let reps = 40;
        for _ in 0..reps {
            let key = make_key(D, &mut rng);
            let item = make_key(D, &mut rng);
            let trace = circular_convolution(&key, &item);
            let got = circular_correlation(&key, &trace);
            acc += cosine_similarity(&got, &item);
        }
        let mean = acc / reps as f64;
        let expected = 1.0 / 2f64.sqrt();
        assert!(
            (mean - expected).abs() < 0.05,
            "mean similarity {mean} against the predicted {expected}"
        );
    }

    #[test]
    fn hrr_fidelity_scales_with_dimension() {
        // Distinguishes an implementation error from the known behaviour of the
        // approximate inverse, whose noise falls as the width grows.
        let mut rng = Rng::new(RngKind::ChaCha, 42);
        for d in [64usize, 256, 1024, 2178, 8192] {
            let mut s = 0.0;
            let reps = 20;
            for _ in 0..reps {
                let key = make_key(d, &mut rng);
                let item = make_key(d, &mut rng);
                let trace = circular_convolution(&key, &item);
                let got = circular_correlation(&key, &trace);
                s += cosine_similarity(&got, &item);
            }
            println!("  d = {d:>5}: similitud media HRR = {:.4}", s / reps as f64);
        }
    }

    #[test]
    fn map_retrieves_a_single_bound_pair() {
        let mut rng = Rng::new(RngKind::ChaCha, 5);
        let key = make_key(D, &mut rng);
        let item = make_key(D, &mut rng);
        let trace = elementwise_bind(&key, &item);
        let got = elementwise_unbind(&key, &trace);
        let s = cosine_similarity(&got, &item);
        assert!(s > 0.9, "similarity was only {s}");
    }

    #[test]
    fn both_schemes_degrade_as_the_bundle_grows() {
        // Crosstalk must increase with load. A scheme whose fidelity did not
        // fall would indicate the items are not really sharing the trace.
        let mut rng = Rng::new(RngKind::ChaCha, 6);
        for scheme in [Scheme::Hrr, Scheme::Map] {
            let c = calibrate(scheme, D, &[2, 32], 3, &mut rng);
            assert!(
                c[0].mean_fidelity > c[1].mean_fidelity,
                "{:?}: fidelity did not fall with load, {} then {}",
                scheme,
                c[0].mean_fidelity,
                c[1].mean_fidelity
            );
        }
    }

    #[test]
    fn erasure_removes_the_requested_fraction() {
        let mut rng = Rng::new(RngKind::ChaCha, 7);
        let t = make_key(1000, &mut rng);
        let c = corrupt(&t, Corruption::Erase, 0.3, &mut rng);
        let zeros = c.iter().filter(|v| **v == 0.0).count();
        assert!((zeros as i64 - 300).abs() <= 1, "erased {zeros} of 1000");
    }

    #[test]
    fn cosine_similarity_matches_known_cases() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-12);
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-12);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-12);
    }
}

#[cfg(test)]
mod unitary_tests {
    use super::*;
    use chaos_rng::{Rng, RngKind};

    #[test]
    fn unitary_keys_have_flat_spectra() {
        let mut rng = Rng::new(RngKind::ChaCha, 21);
        let k = make_unitary_key(512, &mut rng);
        let f = forward(&k);
        for c in f.iter() {
            assert!(
                (c.norm() - 1.0).abs() < 1e-9,
                "magnitude was {} rather than 1",
                c.norm()
            );
        }
    }

    #[test]
    fn unitary_keys_make_hrr_retrieval_exact() {
        // With a unitary key the conjugate is the exact inverse, so a single
        // bound pair must come back essentially perfectly. This is what makes
        // the comparison against MAP fair.
        let mut rng = Rng::new(RngKind::ChaCha, 22);
        for d in [256usize, 2178] {
            let key = make_unitary_key(d, &mut rng);
            let item = make_key(d, &mut rng);
            let trace = circular_convolution(&key, &item);
            let got = circular_correlation(&key, &trace);
            let s = cosine_similarity(&got, &item);
            assert!(s > 0.9999, "d = {d}: similarity was {s}");
        }
    }
}
