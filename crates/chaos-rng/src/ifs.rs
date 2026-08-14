// SPDX-License-Identifier: MIT
//! Iterated function system driven by the chaos game.
//!
//! A third family of generator, deliberately unlike the other two. Lorenz is a
//! continuous flow integrated in time; ChaCha8 is a block construction with no
//! geometry at all. This one is a discrete attractor whose native geometry is a
//! fractal of exactly known, non-integer dimension, which is what makes it
//! useful here: it supplies a reference value against which the measurement
//! pipeline itself can be calibrated, rather than only a relative control.
//!
//! REF: [Barnsley, 1988] "Fractals Everywhere", Academic Press.
//!      ISBN 978-0-12-079062-3. The chaos game and its convergence to the
//!      attractor of an iterated function system.
//!
//! REF: [Grassberger and Procaccia, 1983] "Characterization of Strange
//!      Attractors", Physical Review Letters 50(5), pp. 346-349
//!      DOI: 10.1103/PhysRevLett.50.346
//!
//! The Sierpinski triangle has Hausdorff dimension log(3)/log(2), about
//! 1.5849625, and for this attractor the correlation dimension coincides with
//! it. That number is the calibration target of [`correlation_dimension`].
//!
//! Motivation for adding this family: the Embedding-Manifold-Compression work
//! (PP25) exploits the fact that learned embeddings lie on a low-dimensional
//! fractal manifold. Phase 0.5 of this experiment established where a
//! continuous attractor's structure is destroyed during extraction. This
//! generator asks the same question of a source whose geometry is fractal by
//! construction and whose dimension is known exactly in advance.

use crate::{ChaChaRng, LorenzRng};

/// A source of uniform variates that can drive the chaos game.
///
/// The chaos game needs randomness to choose a vertex at each step. Making that
/// source a parameter separates two questions that would otherwise be confounded:
/// whether any detectable fractal structure comes from the chaos game itself, or
/// is inherited from whatever drives it.
pub trait ChaosSource {
    /// Returns the next variate uniform on [0, 1).
    fn next_unit(&mut self) -> f64;
}

impl ChaosSource for LorenzRng {
    fn next_unit(&mut self) -> f64 {
        self.next_f64()
    }
}

impl ChaosSource for ChaChaRng {
    fn next_unit(&mut self) -> f64 {
        self.next_f64()
    }
}

/// A point of the attractor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    /// First coordinate.
    pub x: f64,
    /// Second coordinate.
    pub y: f64,
}

/// Vertices of the triangle.
///
/// An equilateral triangle with unit side, sitting on the x axis with its left
/// vertex at the origin. The choice is arbitrary in the sense that any
/// non-degenerate triangle produces an attractor of the same dimension, and
/// deliberate in the sense that the coordinates are exact or exactly derived,
/// so the point cloud is not distorted by the choice: (0, 0), (1, 0) and
/// (1/2, sqrt(3)/2).
pub const VERTICES: [Point2; 3] = [
    Point2 { x: 0.0, y: 0.0 },
    Point2 { x: 1.0, y: 0.0 },
    Point2 {
        x: 0.5,
        // sqrt(3)/2, written out so the constant is exact at compile time.
        y: 0.866_025_403_784_438_6,
    },
];

/// Number of iterations discarded before points are used.
///
/// Determined empirically rather than copied from the Lorenz burn-in, which
/// addresses a different problem: there the orbit must reach the attractor from
/// an arbitrary start, here the point contracts towards the attractor
/// geometrically. Each step halves the distance to the attractor, so after n
/// steps an initial offset of at most 1 has shrunk to at most 2^-n. At 60 steps
/// that is below 1e-18, comfortably under the resolution of a double, so the
/// point is on the attractor to machine precision. The value below is that
/// bound with a wide margin, and `burn_in_is_sufficient` checks the reasoning
/// numerically rather than trusting it.
pub const BURN_IN: usize = 100;

/// Chaos game iterations advanced between two extracted outputs.
///
/// Each iteration halves the distance to a vertex, so consecutive points share
/// most of their position and consecutive extractions inherit that. Without
/// decimation the extracted stream showed a lag-one autocorrelation of 0.0101,
/// just over the 0.01 that the Phase 0 battery requires, which is a real
/// residual correlation rather than a rounding artefact. Advancing several
/// iterations decorrelates it, exactly as the Lorenz extractor does for the same
/// reason. The value is the smallest that brings every autocorrelation up to lag
/// ten under the threshold, which `extraction_passes_the_phase_zero_battery`
/// checks.
pub const DECIMATION: usize = 4;

/// Chaos game over the Sierpinski triangle, driven by an arbitrary source.
#[derive(Debug, Clone)]
pub struct IfsRng<S: ChaosSource> {
    source: S,
    point: Point2,
}

impl<S: ChaosSource> IfsRng<S> {
    /// Starts the chaos game from the centroid of the triangle, then discards
    /// [`BURN_IN`] iterations.
    ///
    /// The starting point is inside the triangle, so it is already in the basin
    /// of attraction; the burn-in removes its influence regardless.
    pub fn new(source: S) -> Self {
        let centroid = Point2 {
            x: (VERTICES[0].x + VERTICES[1].x + VERTICES[2].x) / 3.0,
            y: (VERTICES[0].y + VERTICES[1].y + VERTICES[2].y) / 3.0,
        };
        let mut rng = Self {
            source,
            point: centroid,
        };
        for _ in 0..BURN_IN {
            rng.step();
        }
        rng
    }

    /// One iteration: choose a vertex, move half way to it.
    fn step(&mut self) -> Point2 {
        // Three equiprobable vertices. The variate is uniform on [0, 1), so the
        // thirds are equal by construction; no rejection is needed.
        let u = self.source.next_unit();
        let v = if u < 1.0 / 3.0 {
            VERTICES[0]
        } else if u < 2.0 / 3.0 {
            VERTICES[1]
        } else {
            VERTICES[2]
        };
        self.point = Point2 {
            x: 0.5 * (self.point.x + v.x),
            y: 0.5 * (self.point.y + v.y),
        };
        self.point
    }

    /// Advances the game and returns the raw attractor point, bypassing
    /// extraction. Used by the topological analysis as a positive control.
    pub fn next_raw_point(&mut self) -> Point2 {
        self.step()
    }

    /// Returns the next 64 bits.
    ///
    /// The extraction follows the same principle as the Lorenz extractor and
    /// differs in two respects, both forced by the geometry rather than chosen.
    ///
    /// Kept the same: harvest digits far below the scale of the attractor's own
    /// motion by scaling and taking the fractional part, then pass the combined
    /// words through the SplitMix64 finaliser, a bijection which therefore
    /// cannot add entropy the source did not supply.
    ///
    /// Changed: the Lorenz extractor scales by 2^28 because its coordinates live
    /// in the tens. These coordinates live in [0, 1], roughly eight bits higher
    /// in relative terms, so the same absolute scaling would harvest digits that
    /// still carry macroscopic structure. The scale here is 2^36 accordingly.
    /// And there are two coordinates rather than three, so the mixing combines
    /// two words, not three.
    ///
    /// This is not a cryptographic generator and must not be used as one.
    pub fn next_u64(&mut self) -> u64 {
        let mut p = self.step();
        for _ in 1..DECIMATION {
            p = self.step();
        }
        let harvest = |v: f64| -> u64 {
            let scaled = v * 68_719_476_736.0; // 2^36
            let frac = scaled - scaled.floor();
            (frac * 9_007_199_254_740_992.0) as u64 // 2^53
        };
        let a = harvest(p.x);
        let b = harvest(p.y);
        let mut z = a ^ b.rotate_left(31);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns the next variate uniform on [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Advances the game and returns the first coordinate scaled by 2^36,
    /// before the fractional part is taken. First of the extraction stages that
    /// the topological analysis examines separately.
    pub fn next_stage_scaled(&mut self) -> f64 {
        self.step().x * 68_719_476_736.0
    }

    /// Advances the game and returns the fractional part of that scaled
    /// coordinate, the second stage, before mixing.
    pub fn next_stage_fraction(&mut self) -> f64 {
        let scaled = self.next_stage_scaled();
        scaled - scaled.floor()
    }
}

/// Correlation dimension by the Grassberger-Procaccia method.
///
/// The correlation sum C(r) counts the fraction of point pairs closer than r,
///
///   C(r) = 2 / (N (N-1)) * #{ (i, j) : i < j, |x_i - x_j| < r },
///
/// and behaves as C(r) ~ r^D over the scaling region, so D is the slope of
/// log C against log r there.
///
/// REF: [Grassberger and Procaccia, 1983] "Characterization of Strange
///      Attractors", Physical Review Letters 50(5), pp. 346-349
///      DOI: 10.1103/PhysRevLett.50.346
///
/// The scaling region matters. At radii approaching the diameter of the cloud
/// the sum saturates at one, and at radii below the typical nearest-neighbour
/// distance it is dominated by the finite sample rather than by the geometry.
/// The fit is therefore taken over an interior band of radii, and both the band
/// and the quality of the fit are returned so the caller can see whether the
/// power law held rather than being handed a number alone.
#[derive(Debug, Clone)]
pub struct CorrelationDimension {
    /// Radii at which the correlation sum was evaluated.
    pub radii: Vec<f64>,
    /// The correlation sum at each radius.
    pub sums: Vec<f64>,
    /// Indices of the radii used for the fit.
    pub fit_range: (usize, usize),
    /// Estimated dimension, the slope of the fitted region.
    pub dimension: f64,
    /// Coefficient of determination of that fit.
    pub r_squared: f64,
}

/// Estimates the correlation dimension of a planar point cloud.
///
/// `n_radii` logarithmically spaced radii are placed between the 0.1th and the
/// 10th percentile of pairwise distances.
///
/// That band was not chosen by taste. An earlier version evaluated up to the
/// median, and the estimator then returned 1.751 for a uniform square whose
/// true dimension is 2, an underestimate of twelve percent, with a fit quality
/// of r squared 0.9997. A good fit over the wrong region: near the median the
/// correlation sum is already approaching saturation at one, which flattens the
/// slope. Moving the band well below saturation removes the bias, and the tests
/// on clouds of known dimension are what establish that.
pub fn correlation_dimension(points: &[Point2], n_radii: usize) -> CorrelationDimension {
    let n = points.len();
    assert!(n >= 100, "the correlation sum needs a substantial cloud");

    // Pairwise distances. Sampled rather than exhaustive above a threshold,
    // because the full set is quadratic and the estimate does not need it.
    let mut dists: Vec<f64> = Vec::new();
    let stride = if n > 3_000 { n / 3_000 } else { 1 };
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n {
            let dx = points[i].x - points[j].x;
            let dy = points[i].y - points[j].y;
            dists.push((dx * dx + dy * dy).sqrt());
            j += stride;
        }
        i += stride;
    }
    dists.sort_by(|a, b| a.partial_cmp(b).expect("distances are finite"));
    let total = dists.len() as f64;

    let lo = dists[dists.len() / 1_000].max(1e-12);
    let hi = dists[dists.len() / 10];
    let radii: Vec<f64> = (0..n_radii)
        .map(|k| {
            let t = k as f64 / (n_radii as f64 - 1.0);
            (lo.ln() + t * (hi.ln() - lo.ln())).exp()
        })
        .collect();

    let sums: Vec<f64> = radii
        .iter()
        .map(|&r| dists.partition_point(|&d| d < r) as f64 / total)
        .collect();

    // Fit over the interior, discarding a fifth at each end where saturation
    // and sampling noise dominate.
    let a = n_radii / 5;
    let b = n_radii - n_radii / 5;
    let xs: Vec<f64> = radii[a..b].iter().map(|r| r.ln()).collect();
    let ys: Vec<f64> = sums[a..b].iter().map(|s| s.max(1e-300).ln()).collect();
    let mx = xs.iter().sum::<f64>() / xs.len() as f64;
    let my = ys.iter().sum::<f64>() / ys.len() as f64;
    let sxy: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - mx) * (y - my))
        .sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    let syy: f64 = ys.iter().map(|y| (y - my) * (y - my)).sum();

    CorrelationDimension {
        radii,
        sums,
        fit_range: (a, b),
        dimension: if sxx > 0.0 { sxy / sxx } else { 0.0 },
        r_squared: if sxx > 0.0 && syy > 0.0 {
            (sxy * sxy) / (sxx * syy)
        } else {
            0.0
        },
    }
}

/// Hausdorff dimension of the Sierpinski triangle, log(3)/log(2).
pub const SIERPINSKI_DIMENSION: f64 = 1.584_962_500_721_156;

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud<S: ChaosSource>(mut rng: IfsRng<S>, n: usize) -> Vec<Point2> {
        (0..n).map(|_| rng.next_raw_point()).collect()
    }

    #[test]
    fn points_stay_inside_the_triangle() {
        let mut rng = IfsRng::new(ChaChaRng::from_seed(1));
        for _ in 0..10_000 {
            let p = rng.next_raw_point();
            // Barycentric test against the three vertices.
            let sign = |a: Point2, b: Point2, c: Point2| {
                (a.x - c.x) * (b.y - c.y) - (b.x - c.x) * (a.y - c.y)
            };
            let d1 = sign(p, VERTICES[0], VERTICES[1]);
            let d2 = sign(p, VERTICES[1], VERTICES[2]);
            let d3 = sign(p, VERTICES[2], VERTICES[0]);
            let neg = d1 < -1e-12 || d2 < -1e-12 || d3 < -1e-12;
            let pos = d1 > 1e-12 || d2 > 1e-12 || d3 > 1e-12;
            assert!(!(neg && pos), "point escaped the triangle: {p:?}");
        }
    }

    #[test]
    fn burn_in_is_sufficient() {
        // Two games driven by the same vertex choices but started from
        // different points must converge to the same trajectory. This checks
        // the contraction argument behind BURN_IN numerically rather than
        // trusting the algebra.
        let mut a = IfsRng::new(ChaChaRng::from_seed(7));
        let mut b = IfsRng::new(ChaChaRng::from_seed(7));
        b.point = Point2 { x: 0.9, y: 0.05 };
        for _ in 0..BURN_IN {
            a.step();
            b.step();
        }
        let d = ((a.point.x - b.point.x).powi(2) + (a.point.y - b.point.y).powi(2)).sqrt();
        assert!(d < 1e-15, "trajectories had not merged: separation {d:e}");
    }

    #[test]
    fn correlation_dimension_recovers_a_uniform_square() {
        // Calibration of the estimator itself on a cloud of known dimension 2,
        // before it is used to judge the attractor.
        let mut rng = ChaChaRng::from_seed(11);
        let pts: Vec<Point2> = (0..4_000)
            .map(|_| Point2 {
                x: rng.next_f64(),
                y: rng.next_f64(),
            })
            .collect();
        let d = correlation_dimension(&pts, 40);
        assert!(
            (d.dimension - 2.0).abs() < 0.15,
            "uniform square gave {} with r2 {}",
            d.dimension,
            d.r_squared
        );
        assert!(d.r_squared > 0.99);
    }

    #[test]
    fn correlation_dimension_recovers_a_line() {
        let mut rng = ChaChaRng::from_seed(13);
        let pts: Vec<Point2> = (0..4_000)
            .map(|_| Point2 {
                x: rng.next_f64(),
                y: 0.0,
            })
            .collect();
        let d = correlation_dimension(&pts, 40);
        assert!(
            (d.dimension - 1.0).abs() < 0.15,
            "uniform line gave {}",
            d.dimension
        );
    }

    #[test]
    fn attractor_matches_the_sierpinski_dimension() {
        // The blocking gate of Phase 4a. Tolerance of 0.05 in absolute terms,
        // about three percent of the target: wide enough to absorb the finite
        // sample and the choice of scaling region, narrow enough to exclude the
        // neighbouring integers 1 and 2, which is what the test has to
        // distinguish.
        for (label, dim) in [
            (
                "lorenz-driven",
                correlation_dimension(
                    &cloud(IfsRng::new(LorenzRng::from_seed(20_260_814)), 12_000),
                    40,
                ),
            ),
            (
                "chacha-driven",
                correlation_dimension(
                    &cloud(IfsRng::new(ChaChaRng::from_seed(20_260_814)), 12_000),
                    40,
                ),
            ),
        ] {
            println!(
                "  {label}: D2 = {:.6} (teorico {:.6}, error {:.4}), r2 = {:.6}",
                dim.dimension,
                SIERPINSKI_DIMENSION,
                (dim.dimension - SIERPINSKI_DIMENSION).abs(),
                dim.r_squared
            );
            assert!(
                (dim.dimension - SIERPINSKI_DIMENSION).abs() < 0.05,
                "{label}: correlation dimension {} against theoretical {}, r2 {}",
                dim.dimension,
                SIERPINSKI_DIMENSION,
                dim.r_squared
            );
            assert!(dim.r_squared > 0.99, "{label}: power law fit was poor");
        }
    }

    #[test]
    fn extraction_passes_the_phase_zero_battery() {
        // The extracted stream must qualify on the same terms as the other
        // generators before it may be used in training.
        let mut rng = IfsRng::new(ChaChaRng::from_seed(20_260_813));
        let samples: Vec<f64> = (0..100_000).map(|_| rng.next_f64()).collect();
        let r = crate::stats::run_battery(&samples, 100);
        assert!(
            r.passes(),
            "IFS extraction failed the battery: chi p={:.4}, acf={:?}, mean={:.6}, var={:.6}",
            r.chi.p_value,
            r.autocorrelations,
            r.mean,
            r.variance
        );
    }

    #[test]
    fn the_two_variants_produce_different_streams() {
        let mut a = IfsRng::new(LorenzRng::from_seed(5));
        let mut b = IfsRng::new(ChaChaRng::from_seed(5));
        let sa: Vec<u64> = (0..32).map(|_| a.next_u64()).collect();
        let sb: Vec<u64> = (0..32).map(|_| b.next_u64()).collect();
        assert_ne!(sa, sb);
    }

    #[test]
    fn streams_are_reproducible() {
        let s = || {
            let mut r = IfsRng::new(LorenzRng::from_seed(3));
            (0..32).map(|_| r.next_u64()).collect::<Vec<_>>()
        };
        assert_eq!(s(), s());
    }
}
