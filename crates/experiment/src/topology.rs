// SPDX-License-Identifier: MIT
//! Phase 0.5: topological fingerprint of the generators.
//!
//! The Phase 0 battery only sees first and second order structure: a histogram
//! and a correlation. A chaotic source could in principle leave the geometry of
//! its attractor in the extracted stream while passing both. This module looks
//! for that geometry with persistent homology over a Takens embedding, and
//! judges what it finds against an empirical null built from uniform noise.
//!
//! REF: [Takens, 1981] "Detecting strange attractors in turbulence", in
//!      Dynamical Systems and Turbulence, Lecture Notes in Mathematics 898,
//!      pp. 366-381. DOI: 10.1007/BFb0091924
//!
//! REF: [Fraser and Swinney, 1986] "Independent coordinates for strange
//!      attractors from mutual information", Physical Review A 33(2),
//!      pp. 1134-1140. DOI: 10.1103/PhysRevA.33.1134
//!
//! REF: [Kennel, Brown and Abarbanel, 1992] "Determining embedding dimension
//!      for phase-space reconstruction using a geometrical construction",
//!      Physical Review A 45(6), pp. 3403-3411
//!      DOI: 10.1103/PhysRevA.45.3403
//!
//! REF: [Edelsbrunner, Letscher and Zomorodian, 2002] "Topological Persistence
//!      and Simplification", Discrete and Computational Geometry 28,
//!      pp. 511-533. DOI: 10.1007/s00454-002-2885-2
//!
//! The question this phase asks was suggested by two pieces of prior work that
//! apply the same family of tools to learned representations and to training
//! trajectories rather than to generators:
//!
//! REF: [Birdal, Lou, Guibas and Simsekli, 2021] "Intrinsic Dimension,
//!      Persistent Homology and Generalization in Neural Networks", Advances
//!      in Neural Information Processing Systems 34
//!      DOI: 10.48550/arXiv.2111.13171
//!
//! REF: Embedding-Manifold-Compression (PP25),
//!      https://github.com/Yatrogenesis/Embedding-Manifold-Compression, which
//!      uses the Grassberger-Procaccia correlation dimension and Lyapunov
//!      exponents on BERT embeddings.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use tda::{
    distances::euclidean_distance_matrix, persistent_homology::compute_persistence,
    simplicial_complex::vietoris_rips_complex,
};

/// Points used per cloud in the Rips computation.
///
/// A Vietoris-Rips complex over n points carries on the order of n^3/6
/// triangles, so this figure is a compromise between resolution and the cost of
/// the null distribution, which repeats the computation thirty times.
pub const CLOUD_POINTS: usize = 120;

/// Size of the null distribution.
pub const NULL_RESAMPLES: usize = 30;

/// Number of bins used by the mutual information estimate.
const AMI_BINS: usize = 16;

/// Threshold below which the false-nearest-neighbour fraction is considered to
/// have vanished, as used by Kennel et al.
const FNN_THRESHOLD: f64 = 0.01;

/// Ratio test constant of the false-nearest-neighbour criterion.
const FNN_RTOL: f64 = 15.0;

/// Points used by the false-neighbour search.
///
/// The search is quadratic in the number of points, so it runs on a prefix of
/// the series rather than all of it. Two thousand points is well above the
/// few hundred typically used in the literature for this diagnostic, and keeps
/// the cost at a few million distance evaluations per candidate dimension.
const FNN_POINTS: usize = 2_000;

/// Result of the embedding parameter search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingParams {
    /// Delay chosen as the first local minimum of average mutual information.
    pub delay: usize,
    /// Dimension chosen as the first at which the false neighbour fraction
    /// falls below [`FNN_THRESHOLD`].
    pub dimension: usize,
    /// Average mutual information at each candidate delay, for the record.
    pub ami_curve: Vec<f64>,
    /// False neighbour fraction at each candidate dimension, for the record.
    pub fnn_curve: Vec<f64>,
}

/// Average mutual information between a series and itself at a given delay,
/// estimated with a two-dimensional histogram.
///
/// I(tau) = sum_ij p_ij log( p_ij / (p_i p_j) )
pub fn average_mutual_information(x: &[f64], delay: usize, bins: usize) -> f64 {
    if delay >= x.len() {
        return 0.0;
    }
    let n = x.len() - delay;
    let lo = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = (hi - lo).max(f64::MIN_POSITIVE);
    let bin_of = |v: f64| (((v - lo) / span) * bins as f64).min(bins as f64 - 1.0) as usize;

    let mut joint = vec![0f64; bins * bins];
    let mut px = vec![0f64; bins];
    let mut py = vec![0f64; bins];
    for i in 0..n {
        let a = bin_of(x[i]);
        let b = bin_of(x[i + delay]);
        joint[a * bins + b] += 1.0;
        px[a] += 1.0;
        py[b] += 1.0;
    }
    let n_f = n as f64;
    let mut mi = 0.0;
    for a in 0..bins {
        for b in 0..bins {
            let p_ab = joint[a * bins + b] / n_f;
            if p_ab > 0.0 {
                let p_a = px[a] / n_f;
                let p_b = py[b] / n_f;
                mi += p_ab * (p_ab / (p_a * p_b)).ln();
            }
        }
    }
    mi
}

/// Fraction of false nearest neighbours when going from dimension `m` to
/// `m + 1`, by the criterion of Kennel et al.
///
/// A neighbour is false when the extra coordinate separates the pair by more
/// than [`FNN_RTOL`] times their distance in the lower dimension.
pub fn false_nearest_neighbour_fraction(x: &[f64], m: usize, delay: usize) -> f64 {
    let span = (m - 1) * delay;
    if x.len() <= span + delay + 1 {
        return 0.0;
    }
    let n = (x.len() - span - delay).min(FNN_POINTS);
    let point = |i: usize, dim: usize| -> Vec<f64> { (0..dim).map(|k| x[i + k * delay]).collect() };

    let mut false_count = 0usize;
    let mut total = 0usize;
    for i in 0..n {
        let pi = point(i, m);
        // Nearest neighbour in dimension m, excluding the point itself.
        let mut best = f64::INFINITY;
        let mut best_j = None;
        for j in 0..n {
            if i == j {
                continue;
            }
            let pj = point(j, m);
            let d2: f64 = pi
                .iter()
                .zip(pj.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if d2 < best {
                best = d2;
                best_j = Some(j);
            }
        }
        let Some(j) = best_j else { continue };
        let d_m = best.sqrt();
        if d_m <= f64::MIN_POSITIVE {
            continue;
        }
        let extra = (x[i + span + delay] - x[j + span + delay]).abs();
        total += 1;
        if extra / d_m > FNN_RTOL {
            false_count += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        false_count as f64 / total as f64
    }
}

/// Chooses the delay and the embedding dimension by the standard methods.
///
/// The delay is the first local minimum of average mutual information, which is
/// the criterion of Fraser and Swinney: the first delay at which the coordinate
/// carries the least redundant information about the previous one. The
/// dimension is the first at which the false neighbour fraction falls below one
/// percent, the criterion of Kennel et al.
pub fn choose_embedding(x: &[f64], max_delay: usize, max_dim: usize) -> EmbeddingParams {
    let ami_curve: Vec<f64> = (1..=max_delay)
        .map(|d| average_mutual_information(x, d, AMI_BINS))
        .collect();
    let mut delay = 1;
    for i in 1..ami_curve.len() - 1 {
        if ami_curve[i] < ami_curve[i - 1] && ami_curve[i] < ami_curve[i + 1] {
            delay = i + 1;
            break;
        }
        // No local minimum found so far; fall back to the global minimum below.
        delay = ami_curve
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
            .map(|(i, _)| i + 1)
            .unwrap_or(1);
    }

    let fnn_curve: Vec<f64> = (1..=max_dim)
        .map(|m| false_nearest_neighbour_fraction(x, m, delay))
        .collect();
    let dimension = fnn_curve
        .iter()
        .position(|&f| f < FNN_THRESHOLD)
        .map(|i| i + 1)
        .unwrap_or(max_dim);

    EmbeddingParams {
        delay,
        dimension,
        ami_curve,
        fnn_curve,
    }
}

/// Builds a Takens delay embedding and returns at most `max_points` rows.
pub fn takens_embedding(
    x: &[f64],
    dimension: usize,
    delay: usize,
    max_points: usize,
) -> DMatrix<f64> {
    let span = (dimension - 1) * delay;
    let available = x.len().saturating_sub(span);
    let n = available.min(max_points);
    // Stride through the available window so the sample covers the whole series
    // rather than only its beginning.
    let stride = (available / n).max(1);
    let mut data = Vec::with_capacity(n * dimension);
    for i in 0..n {
        let base = i * stride;
        for k in 0..dimension {
            data.push(x[base + k * delay]);
        }
    }
    DMatrix::from_row_slice(n, dimension, &data)
}

/// Total finite persistence in dimension one for a point cloud.
///
/// Note on scale. Persistence has the units of the data: multiplying every
/// coordinate by a constant multiplies every bar length by the same constant.
/// Totals are therefore comparable only between clouds on a common scale. In
/// this phase the streams that live on the unit interval (the fractional part,
/// the mixed output, and the ChaCha8 control) are mutually comparable and
/// comparable to the uniform null; the raw attractor states and the coordinate
/// scaled by 2^28 are not, and their totals are reported as evidence that the
/// pipeline detects geometry at all, not as values to be ranked against the
/// others.
///
/// Infinite bars are excluded. The implementation used here reports a number of
/// unpaired one-simplices as infinite even for a contractible cloud, which
/// would distort a raw count; total finite persistence is unaffected by that
/// and is the statistic the protocol asks for. The same statistic is applied to
/// every condition and to the null, so any residual bias is common to all.
pub fn total_h1_persistence(points: &DMatrix<f64>, max_radius: f64) -> f64 {
    let d = euclidean_distance_matrix(points).expect("point cloud is non-empty");
    let complex = vietoris_rips_complex(&d, max_radius, 2).expect("rips complex builds");
    let pairs = compute_persistence(&complex, 2).expect("persistence computes");
    pairs
        .iter()
        .filter(|p| p.dimension == 1 && !p.is_infinite())
        .map(|p| p.persistence())
        .sum()
}

/// A scale for the Rips filtration derived from the cloud itself, so that
/// clouds of different spread are compared on comparable footing.
///
/// The median pairwise distance is used, doubled, which comfortably exceeds the
/// scale at which a one-dimensional cycle in such a cloud would die.
pub fn adaptive_radius(points: &DMatrix<f64>) -> f64 {
    let d = euclidean_distance_matrix(points).expect("point cloud is non-empty");
    let n = d.nrows();
    let mut v: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            v.push(d[(i, j)]);
        }
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("distances are finite"));
    2.0 * v[v.len() / 2]
}

/// One row of the Phase 0.5 table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyRow {
    /// What was measured.
    pub label: String,
    /// Points in the cloud.
    pub points: usize,
    /// Embedding dimension used, or the native dimension for raw states.
    pub dimension: usize,
    /// Delay used, or zero when not applicable.
    pub delay: usize,
    /// Filtration radius used.
    pub radius: f64,
    /// Total finite persistence in dimension one.
    pub total_h1: f64,
    /// Empirical p-value against the uniform null, when computed.
    pub p_value: Option<f64>,
}

/// Empirical p-value of an observed statistic against a null sample.
///
/// Two sided: the proportion of null values at least as extreme as the
/// observation, in either direction, with the usual add-one correction so that
/// a p-value of exactly zero cannot be reported from a finite null.
///
/// REF: [Phipson and Smyth, 2010] "Permutation P-values Should Never Be Zero",
///      Statistical Applications in Genetics and Molecular Biology 9(1)
///      DOI: 10.2202/1544-6115.1585
pub fn empirical_p_value(observed: f64, null: &[f64]) -> f64 {
    let median = {
        let mut v = null.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        v[v.len() / 2]
    };
    let deviation = (observed - median).abs();
    let at_least_as_extreme = null
        .iter()
        .filter(|v| (*v - median).abs() >= deviation)
        .count();
    (at_least_as_extreme as f64 + 1.0) / (null.len() as f64 + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutual_information_is_maximal_at_zero_delay() {
        let x: Vec<f64> = (0..500).map(|i| (i as f64 * 0.3).sin()).collect();
        let i0 = average_mutual_information(&x, 0, AMI_BINS);
        let i5 = average_mutual_information(&x, 5, AMI_BINS);
        assert!(i0 > i5, "a series shares most information with itself");
    }

    #[test]
    fn circle_shows_the_hole_that_theory_predicts() {
        // A circle has exactly one one-dimensional hole. This is the control on
        // the persistence backend itself: if it fails, nothing downstream can
        // be trusted.
        let n = 30;
        let mut data = Vec::new();
        for i in 0..n {
            let a = std::f64::consts::TAU * i as f64 / n as f64;
            data.push(a.cos());
            data.push(a.sin());
        }
        let pts = DMatrix::from_row_slice(n, 2, &data);
        let d = euclidean_distance_matrix(&pts).unwrap();
        let cx = vietoris_rips_complex(&d, 2.5, 2).unwrap();
        let pairs = compute_persistence(&cx, 2).unwrap();
        let top = pairs
            .iter()
            .filter(|p| p.dimension == 1 && !p.is_infinite())
            .map(|p| p.persistence())
            .fold(0.0f64, f64::max);
        // Birth at the point spacing, death near sqrt(3) for the unit circle.
        assert!(
            top > 1.4 && top < 1.7,
            "dominant H1 persistence of a unit circle was {top}, expected about 1.52"
        );
    }

    #[test]
    fn filled_region_has_far_less_h1_than_a_circle() {
        let n = 30;
        let mut data = Vec::new();
        for i in 0..n {
            data.push(((i * 37) % 13) as f64 / 13.0);
            data.push(((i * 17) % 11) as f64 / 11.0);
        }
        let pts = DMatrix::from_row_slice(n, 2, &data);
        let total = total_h1_persistence(&pts, 2.5);
        assert!(
            total < 0.5,
            "a filled region should carry little H1, got {total}"
        );
    }

    #[test]
    fn empirical_p_value_is_bounded_and_never_zero() {
        let null: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let p_extreme = empirical_p_value(1000.0, &null);
        let p_central = empirical_p_value(15.0, &null);
        assert!(p_extreme > 0.0, "a finite null cannot justify p = 0");
        assert!(p_extreme < 0.05);
        assert!(p_central > 0.5);
    }
}
