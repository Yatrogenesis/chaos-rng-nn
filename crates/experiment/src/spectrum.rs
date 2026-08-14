// SPDX-License-Identifier: MIT
//! Phase 6: spectrum of a superposed operator built from one training run.
//!
//! Three 60 by 60 matrices are derived from the same sixty per-epoch weight
//! vectors of a run, normalised, averaged, and the spectrum of the result is
//! compared against a null.
//!
//! REF: [Wigner, 1958] "On the Distribution of the Roots of Certain Symmetric
//!      Matrices", Annals of Mathematics 67(2), pp. 325-327
//!      DOI: 10.2307/1970008
//!
//! REF: [Mehta, 2004] "Random Matrices", 3rd edition, Pure and Applied
//!      Mathematics vol. 142, Elsevier. ISBN 978-0-12-088409-4
//!
//! **The null is not a plain Gaussian ensemble, and the reason matters.** A
//! synthetic prototype run before this phase superposed a circulant matrix, a
//! block matrix of contractive affine maps, and a Euclidean distance matrix at
//! size 200, and found a spectral gap of 0.246 against 0.003 for a superposition
//! of three independent Gaussian ensembles, with a Kolmogorov-Smirnov test
//! against the semicircle at p = 0.0003. Most of that was an artefact. A
//! distance matrix is non-negative by construction, and any non-negative matrix
//! carries a dominant Perron-Frobenius eigenvalue that has nothing to do with
//! geometry or fractals. Replacing the real distance matrix with the absolute
//! value of a generic Gaussian matrix, which shares only the non-negativity,
//! reproduced most of the effect: a gap of 0.193 against the original 0.246.
//!
//! REF: [Perron, 1907] "Zur Theorie der Matrices", Mathematische Annalen 64(2),
//!      pp. 248-263, DOI: 10.1007/BF01449896, and
//!      [Frobenius, 1912] "Über Matrizen aus nicht negativen Elementen",
//!      Sitzungsberichte der Königlich Preussischen Akademie der Wissenschaften.
//!
//! The null used here is therefore the corrected one: the same circulant and
//! affine terms, with the distance matrix replaced by the absolute value of a
//! Gaussian matrix with a zero diagonal. That isolates what a real Euclidean
//! distance matrix contributes beyond merely being non-negative, which is the
//! only question the design can answer.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Number of epochs, and therefore the size of every matrix here.
pub const N: usize = 60;

/// Summary of one spectrum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumSummary {
    /// Largest eigenvalue.
    pub max_eigenvalue: f64,
    /// Difference between the largest and the second largest.
    pub spectral_gap: f64,
    /// Kolmogorov-Smirnov statistic of the standardised spectrum against the
    /// semicircle law.
    pub ks_statistic: f64,
    /// Its p-value.
    pub ks_p_value: f64,
}

/// Eigenvalues of a symmetric matrix, in descending order.
fn eigenvalues_descending(m: &DMatrix<f64>) -> Vec<f64> {
    let sym = nalgebra::SymmetricEigen::new(m.clone());
    let mut v: Vec<f64> = sym.eigenvalues.iter().cloned().collect();
    v.sort_by(|a, b| b.partial_cmp(a).expect("eigenvalues are finite"));
    v
}

/// Summarises a symmetric matrix's spectrum.
///
/// The comparison against the semicircle is made on the bulk, with the largest
/// eigenvalue excluded. A single outlier of Perron-Frobenius type would
/// otherwise dominate the statistic and the test would report a rejection that
/// says nothing about the shape of the bulk, which is what the semicircle law
/// describes.
pub fn summarise(m: &DMatrix<f64>) -> SpectrumSummary {
    let ev = eigenvalues_descending(m);
    let max = ev[0];
    let gap = ev[0] - ev[1];

    let bulk = &ev[1..];
    let mean = xstats::mean(bulk);
    let sd = xstats::std_dev(bulk);
    // The semicircle law is supported on [-2, 2], so the bulk is standardised
    // to unit variance and then scaled by two.
    let standardised: Vec<f64> = if sd > 0.0 {
        bulk.iter().map(|v| 2.0 * (v - mean) / (sd * 2.0)).collect()
    } else {
        bulk.to_vec()
    };
    let (d, p) = xstats::ks_test(&standardised, xstats::semicircle_cdf);

    SpectrumSummary {
        max_eigenvalue: max,
        spectral_gap: gap,
        ks_statistic: d,
        ks_p_value: p,
    }
}

/// Divides a matrix by its Frobenius norm, so that the three terms enter the
/// superposition on a common footing rather than by whichever happens to have
/// the largest entries.
pub fn normalise(m: &DMatrix<f64>) -> DMatrix<f64> {
    let norm = m.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm == 0.0 {
        m.clone()
    } else {
        m / norm
    }
}

/// Pairwise Euclidean distances between the epoch vectors: the TDA term.
pub fn distance_matrix(trajectory: &[Vec<f64>]) -> DMatrix<f64> {
    let n = trajectory.len();
    let mut m = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = trajectory[i]
                .iter()
                .zip(trajectory[j].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            m[(i, j)] = d;
            m[(j, i)] = d;
        }
    }
    m
}

/// Pairwise circular correlation between epoch vectors, reduced to a scalar:
/// the HRR term.
///
/// The circular correlation of two vectors is itself a vector. It is reduced
/// here to the largest absolute component, which is the standard read-out in
/// holographic representations: retrieval asks where the correlation peaks, and
/// how high. The alternative, taking the value at lag zero, would collapse to
/// an ordinary inner product and discard exactly the shift structure that
/// distinguishes circular correlation from a dot product.
///
/// The matrix is symmetrised, since the correlation of i with j peaks at the
/// mirrored lag of the correlation of j with i and the two maxima agree in
/// magnitude up to rounding.
pub fn circular_correlation_matrix(trajectory: &[Vec<f64>]) -> DMatrix<f64> {
    let n = trajectory.len();
    let mut m = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let c = crate::binding::circular_correlation(&trajectory[i], &trajectory[j]);
            let peak = c.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
            m[(i, j)] = peak;
            m[(j, i)] = peak;
        }
    }
    m
}

/// Block matrix built from the affine maps of the Sierpinski chaos game: the
/// IFS term.
///
/// Each map is written in homogeneous form, a three by three matrix carrying
/// both the contraction and the translation, and twenty such blocks are tiled
/// along the diagonal, cycling through the three maps, which fills sixty rows
/// exactly.
///
/// **This term is identical for every run, and that is a finding rather than a
/// shortcut.** The chaos game's three maps are fixed by the triangle; what the
/// randomness selects is which map to apply at each step, not what the maps
/// are. So the affine operator that drove an `ifs-lorenz` run is the same one
/// that drove an `ifs-chacha8` run, and the same one used as a reference for
/// the two conditions that involve no chaos game at all. The asymmetry the
/// design anticipated between IFS and non-IFS conditions therefore does not
/// arise, but the price is that this term carries no run-level information: it
/// contributes a constant to every superposition.
pub fn affine_matrix() -> DMatrix<f64> {
    let vertices = chaos_rng::ifs::VERTICES;
    let mut m = DMatrix::zeros(N, N);
    for block in 0..(N / 3) {
        let v = vertices[block % 3];
        let o = block * 3;
        // Homogeneous form of w(x) = x/2 + v/2.
        m[(o, o)] = 0.5;
        m[(o + 1, o + 1)] = 0.5;
        m[(o, o + 2)] = v.x * 0.5;
        m[(o + 1, o + 2)] = v.y * 0.5;
        m[(o + 2, o + 2)] = 1.0;
    }
    // Symmetrised, because the spectral tools here are for symmetric matrices
    // and an affine block is not symmetric. (M + M^T)/2 preserves the diagonal
    // and halves the off-diagonal translation entries.
    let t = m.transpose();
    (m + t) / 2.0
}

/// A Gaussian symmetric matrix with entries scaled to the standard ensemble.
pub fn goe(n: usize, rng: &mut chaos_rng::Rng) -> DMatrix<f64> {
    let mut m = DMatrix::zeros(n, n);
    let scale = 1.0 / (n as f64).sqrt();
    for i in 0..n {
        for j in i..n {
            let v = rng.next_normal() * scale;
            m[(i, j)] = v;
            m[(j, i)] = v;
        }
    }
    m
}

/// The absolute value of a Gaussian matrix, with a zero diagonal: the corrected
/// stand-in for a distance matrix.
///
/// It shares non-negativity and a vanishing diagonal with a real distance
/// matrix, and shares nothing else. It does not obey the triangle inequality
/// and is not the distance matrix of any point configuration.
pub fn abs_goe(n: usize, rng: &mut chaos_rng::Rng) -> DMatrix<f64> {
    let mut m = goe(n, rng);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = m[(i, j)].abs();
        }
        m[(i, i)] = 0.0;
    }
    m
}

/// The superposition of three normalised terms.
pub fn superpose(a: &DMatrix<f64>, b: &DMatrix<f64>, c: &DMatrix<f64>) -> DMatrix<f64> {
    (normalise(a) + normalise(b) + normalise(c)) / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use chaos_rng::{Rng, RngKind};

    #[test]
    fn distance_matrix_is_a_metric_on_its_face() {
        let t: Vec<Vec<f64>> = (0..5)
            .map(|i| vec![i as f64, (i * i) as f64, -(i as f64)])
            .collect();
        let m = distance_matrix(&t);
        for i in 0..5 {
            assert!(m[(i, i)].abs() < 1e-12, "diagonal must vanish");
            for j in 0..5 {
                assert!(m[(i, j)] >= 0.0, "distances are non-negative");
                assert!((m[(i, j)] - m[(j, i)]).abs() < 1e-12, "must be symmetric");
                for k in 0..5 {
                    assert!(
                        m[(i, j)] <= m[(i, k)] + m[(k, j)] + 1e-9,
                        "triangle inequality violated"
                    );
                }
            }
        }
    }

    #[test]
    fn affine_matrix_is_symmetric_and_constant() {
        let a = affine_matrix();
        let b = affine_matrix();
        assert_eq!(a, b, "the affine term must not vary between calls");
        for i in 0..N {
            for j in 0..N {
                assert!((a[(i, j)] - a[(j, i)]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn goe_bulk_follows_the_semicircle() {
        // The calibration of the spectral tools: a Gaussian ensemble must not be
        // rejected against the law it obeys. If this fails, no rejection
        // reported elsewhere in this phase means anything.
        let mut rng = Rng::new(RngKind::ChaCha, 900);
        let m = goe(400, &mut rng);
        let s = summarise(&m);
        assert!(
            s.ks_p_value > 0.01,
            "a Gaussian ensemble was rejected against the semicircle, p = {}, D = {}",
            s.ks_p_value,
            s.ks_statistic
        );
    }

    #[test]
    fn a_non_negative_matrix_shows_a_dominant_eigenvalue() {
        // The artefact this phase is designed around: non-negativity alone
        // produces a large leading eigenvalue.
        let mut rng = Rng::new(RngKind::ChaCha, 901);
        let plain = goe(200, &mut rng);
        let mut rng2 = Rng::new(RngKind::ChaCha, 901);
        let nonneg = abs_goe(200, &mut rng2);
        let gp = summarise(&plain).spectral_gap;
        let gn = summarise(&nonneg).spectral_gap;
        assert!(
            gn > gp * 3.0,
            "non-negativity should dominate: gaps {gn} against {gp}"
        );
    }

    #[test]
    fn normalisation_puts_terms_on_a_common_scale() {
        let mut rng = Rng::new(RngKind::ChaCha, 902);
        let a = goe(20, &mut rng);
        let b = &goe(20, &mut rng) * 1000.0;
        for m in [normalise(&a), normalise(&b)] {
            let f: f64 = m.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((f - 1.0).abs() < 1e-12, "Frobenius norm was {f}");
        }
    }
}
