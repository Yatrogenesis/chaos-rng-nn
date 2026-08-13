// SPDX-License-Identifier: MIT
//! Phase 3: fractal dimension of the optimisation trajectory.
//!
//! The training trajectory is the sequence of parameter vectors, one per epoch,
//! treated as a point cloud in parameter space. Its persistent-homology
//! dimension is estimated and compared between the two generators, and against
//! the generalisation gap already measured in Phase 1.
//!
//! REF: [Birdal, Lou, Guibas and Simsekli, 2021] "Intrinsic Dimension,
//!      Persistent Homology and Generalization in Neural Networks", Advances
//!      in Neural Information Processing Systems 34
//!      DOI: 10.48550/arXiv.2111.13171
//!
//! REF: [Adams et al., 2020] "A Fractal Dimension for Measures via Persistent
//!      Homology", in Topological Data Analysis, Abel Symposia 15, pp. 1-31
//!      DOI: 10.1007/978-3-030-43408-3_1
//!
//! The estimator follows the alpha-weighted construction of Birdal et al. In
//! their notation, for a finite set X of n points,
//!
//!   E^alpha(X) = sum over edges e of the minimum spanning tree of |e|^alpha
//!
//! and the persistent-homology dimension is recovered from the growth rate of
//! that quantity with n:
//!
//!   E^alpha(X_n) ~ n^((d - alpha) / d),   so   d = alpha / (1 - m)
//!
//! where m is the slope of log E^alpha against log n. Following the paper, and
//! its own reference to Adams et al., alpha is fixed at one, so E is simply the
//! total edge length of the minimum spanning tree and d = 1 / (1 - m).
//!
//! The zero-dimensional persistence of a Vietoris-Rips filtration is exactly
//! the minimum spanning tree of the point cloud, so computing the tree directly
//! is the same object arrived at by a cheaper route.

use serde::{Deserialize, Serialize};

/// Exponent of the edge weights. Fixed at one, as in Birdal et al.
pub const ALPHA: f64 = 1.0;

/// Euclidean distance between two parameter vectors.
fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Total edge weight of the minimum spanning tree, with each edge raised to
/// [`ALPHA`], computed by Prim's algorithm on the complete graph.
///
/// REF: [Prim, 1957] "Shortest connection networks and some generalizations",
///      The Bell System Technical Journal 36(6), pp. 1389-1401
///      DOI: 10.1002/j.1538-7305.1957.tb01515.x
pub fn mst_weight(points: &[Vec<f64>]) -> f64 {
    let n = points.len();
    if n < 2 {
        return 0.0;
    }
    let mut in_tree = vec![false; n];
    let mut best = vec![f64::INFINITY; n];
    best[0] = 0.0;
    let mut total = 0.0;

    for _ in 0..n {
        // Cheapest vertex not yet in the tree.
        let mut u = usize::MAX;
        let mut u_cost = f64::INFINITY;
        for v in 0..n {
            if !in_tree[v] && best[v] < u_cost {
                u_cost = best[v];
                u = v;
            }
        }
        if u == usize::MAX {
            break;
        }
        in_tree[u] = true;
        if u_cost.is_finite() && u_cost > 0.0 {
            total += u_cost.powf(ALPHA);
        }
        for v in 0..n {
            if !in_tree[v] {
                let d = distance(&points[u], &points[v]);
                if d < best[v] {
                    best[v] = d;
                }
            }
        }
    }
    total
}

/// Outcome of one dimension estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhDimEstimate {
    /// Subsample sizes used.
    pub sample_sizes: Vec<usize>,
    /// Mean minimum spanning tree weight at each size.
    pub mst_weights: Vec<f64>,
    /// Slope of log weight against log size.
    pub slope: f64,
    /// Coefficient of determination of that linear fit, as a check that the
    /// power law it assumes actually holds.
    pub r_squared: f64,
    /// The estimate itself, alpha / (1 - slope).
    pub dimension: f64,
}

/// Estimates the persistent-homology dimension of a point cloud.
///
/// For each subsample size the tree weight is averaged over `repeats` draws,
/// taken deterministically from `rng` so the estimate is reproducible.
pub fn estimate(
    points: &[Vec<f64>],
    sizes: &[usize],
    repeats: usize,
    rng: &mut chaos_rng::Rng,
) -> PhDimEstimate {
    let mut mst_weights = Vec::with_capacity(sizes.len());
    for &m in sizes {
        let mut acc = 0.0;
        for _ in 0..repeats {
            let mut idx: Vec<usize> = (0..points.len()).collect();
            rng.shuffle(&mut idx);
            let sample: Vec<Vec<f64>> = idx[..m].iter().map(|&i| points[i].clone()).collect();
            acc += mst_weight(&sample);
        }
        mst_weights.push(acc / repeats as f64);
    }

    // Ordinary least squares of log E against log n.
    let xs: Vec<f64> = sizes.iter().map(|&s| (s as f64).ln()).collect();
    let ys: Vec<f64> = mst_weights.iter().map(|w| w.max(1e-300).ln()).collect();
    let mx = xs.iter().sum::<f64>() / xs.len() as f64;
    let my = ys.iter().sum::<f64>() / ys.len() as f64;
    let sxy: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - mx) * (y - my))
        .sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };

    let syy: f64 = ys.iter().map(|y| (y - my) * (y - my)).sum();
    let r_squared = if sxx > 0.0 && syy > 0.0 {
        (sxy * sxy) / (sxx * syy)
    } else {
        0.0
    };

    // d = alpha / (1 - slope). A slope at or above one would place the estimate
    // at infinity or make it negative, which signals that the power law does
    // not hold on this cloud rather than an enormous dimension.
    let dimension = if (1.0 - slope).abs() < 1e-9 {
        f64::INFINITY
    } else {
        ALPHA / (1.0 - slope)
    };

    PhDimEstimate {
        sample_sizes: sizes.to_vec(),
        mst_weights,
        slope,
        r_squared,
        dimension,
    }
}

/// Pearson product-moment correlation.
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxy: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (a - mx) * (b - my))
        .sum();
    let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    let syy: f64 = y.iter().map(|b| (b - my) * (b - my)).sum();
    if sxx <= 0.0 || syy <= 0.0 {
        return 0.0;
    }
    sxy / (sxx * syy).sqrt()
}

/// Spearman rank correlation, with average ranks for ties.
pub fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let rank = |v: &[f64]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).expect("finite"));
        let mut r = vec![0.0; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let avg = ((i + 1) + (j + 1)) as f64 / 2.0;
            for k in i..=j {
                r[idx[k]] = avg;
            }
            i = j + 1;
        }
        r
    };
    pearson(&rank(x), &rank(y))
}

/// Two-sided p-value for a correlation coefficient, from the Student t
/// transformation t = r sqrt((n - 2) / (1 - r^2)) with n - 2 degrees of freedom.
pub fn correlation_p_value(r: f64, n: usize) -> f64 {
    if n < 3 {
        return 1.0;
    }
    let df = (n - 2) as f64;
    if (1.0 - r * r).abs() < 1e-15 {
        return 0.0;
    }
    let t = r * (df / (1.0 - r * r)).sqrt();
    xstats::betai(df / 2.0, 0.5, df / (df + t * t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chaos_rng::{Rng, RngKind};

    #[test]
    fn mst_of_a_line_is_its_length() {
        // Points evenly spaced on a line: the tree is the line itself.
        let pts: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let w = mst_weight(&pts);
        assert!((w - 9.0).abs() < 1e-9, "expected 9, got {w}");
    }

    #[test]
    fn mst_of_a_square_is_three_sides() {
        let pts = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ];
        let w = mst_weight(&pts);
        assert!((w - 3.0).abs() < 1e-9, "expected 3, got {w}");
    }

    #[test]
    fn estimator_recovers_the_dimension_of_a_uniform_square() {
        // Points spread uniformly over a two-dimensional square should give an
        // estimate near two. This is the calibration of the estimator itself;
        // without it, any number it produces on a trajectory is meaningless.
        let mut rng = Rng::new(RngKind::ChaCha, 4242);
        let pts: Vec<Vec<f64>> = (0..800)
            .map(|_| vec![rng.next_f64(), rng.next_f64()])
            .collect();
        let mut est_rng = Rng::new(RngKind::ChaCha, 7);
        let est = estimate(&pts, &[100, 200, 400, 800], 3, &mut est_rng);
        assert!(
            est.dimension > 1.6 && est.dimension < 2.5,
            "estimate for a uniform square was {}, slope {}, r2 {}",
            est.dimension,
            est.slope,
            est.r_squared
        );
        assert!(
            est.r_squared > 0.98,
            "the power law should fit well on a square"
        );
    }

    #[test]
    fn estimator_recovers_the_dimension_of_a_uniform_line() {
        let mut rng = Rng::new(RngKind::ChaCha, 99);
        let pts: Vec<Vec<f64>> = (0..800).map(|_| vec![rng.next_f64()]).collect();
        let mut est_rng = Rng::new(RngKind::ChaCha, 7);
        let est = estimate(&pts, &[100, 200, 400, 800], 3, &mut est_rng);
        assert!(
            est.dimension > 0.7 && est.dimension < 1.4,
            "estimate for a uniform line was {}",
            est.dimension
        );
    }

    #[test]
    fn correlations_agree_on_a_monotone_relation() {
        let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v * 3.0 + 1.0).collect();
        assert!((pearson(&x, &y) - 1.0).abs() < 1e-9);
        assert!((spearman(&x, &y) - 1.0).abs() < 1e-9);
        assert!(correlation_p_value(pearson(&x, &y), 10) < 1e-6);
    }
}
