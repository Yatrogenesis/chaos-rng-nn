// SPDX-License-Identifier: MIT
//! Weighted topological resilience of a layer's correlation structure.
//!
//! `T(S) = sum_d w_d sum_i max(0, pers(f_i) - sigma_d)`: for each homology
//! dimension, the total persistence of the features that outlive a threshold,
//! combined across dimensions by a weight vector.
//!
//! **Provenance.** The shape of this expression is taken from a draft of the
//! author's, and it is reused here as a design hypothesis of this project and
//! nothing more. None of the weights, thresholds or reported outcomes of that
//! draft are carried over: the quantities below are hyperparameters of this
//! experiment, swept rather than inherited, because there is no way to check
//! where the draft's values came from. Nothing in this crate should be read as
//! confirming any result stated there.

use nalgebra::DMatrix;
use serde::Serialize;
use tda::{persistent_homology::compute_persistence, simplicial_complex::vietoris_rips_complex};

/// Highest homology dimension the weighting covers.
pub const MAX_DIMENSION: usize = 2;

/// What has to be passed to `tda` to obtain homology up to [`MAX_DIMENSION`].
///
/// The crate returns homology only up to `max_dimension - 1`: asking for two
/// yields nothing in dimension two, silently and without error. Verified on a
/// circle of twelve points, where asking for one, two and three returns
/// dimensions {0}, {0,1} and {0,1,2} respectively. Without this offset the
/// `w_2` term of the weighting would have been multiplying an empty set for the
/// whole phase, and nothing would have signalled it.
pub const TDA_DIMENSION_REQUEST: usize = MAX_DIMENSION + 1;

/// Weights and thresholds of one configuration of `T(S)`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Weights {
    /// Name used in the report.
    pub label: &'static str,
    /// Weight of each homology dimension, indices 0 to [`MAX_DIMENSION`].
    pub w: [f64; MAX_DIMENSION + 1],
    /// Persistence a feature must exceed before it counts, per dimension.
    pub sigma: [f64; MAX_DIMENSION + 1],
}

impl Weights {
    /// Whether the weights form a convex combination, as the hypothesis states.
    pub fn is_normalised(&self) -> bool {
        (self.w.iter().sum::<f64>() - 1.0).abs() < 1e-12
    }
}

/// The configurations swept in this phase.
///
/// Four weightings, kept deliberately including one that fails, and two
/// threshold levels. None reproduces anyone's reported optimum; they are
/// hyperparameters of this experiment.
///
/// The weight on dimension zero is zero in all but the first, and that is a
/// correction the calibration forced rather than a tuning choice. Total
/// dimension-zero persistence is the weight of the minimum spanning tree, so it
/// is largest when every point is far from every other, which is exactly what
/// independent noise produces and the opposite of what the hypothesis needs. No
/// positive `w_0` can rank structure above noise. The `with-components`
/// configuration is retained so the sweep shows that failure instead of hiding
/// it; it is excluded from the later comparison, with the reason recorded.
///
/// The threshold is what makes the dimension-one term work. Noise generates
/// hundreds of very short one-dimensional bars whose sum exceeds a genuine
/// loop's single long bar, so an unthresholded total would also rank noise
/// first. At 0.10 every noise bar falls below the cut and contributes nothing,
/// while a real cycle survives.
pub const SWEEP: [Weights; 5] = [
    Weights {
        label: "with-components",
        w: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        sigma: [0.10, 0.10, 0.10],
    },
    Weights {
        label: "loops",
        w: [0.0, 1.0, 0.0],
        sigma: [0.10, 0.10, 0.10],
    },
    Weights {
        label: "loops-and-voids",
        w: [0.0, 0.7, 0.3],
        sigma: [0.10, 0.10, 0.10],
    },
    Weights {
        label: "voids-heavy",
        w: [0.0, 0.5, 0.5],
        sigma: [0.10, 0.10, 0.10],
    },
    Weights {
        label: "loops-strict",
        w: [0.0, 1.0, 0.0],
        sigma: [0.30, 0.30, 0.30],
    },
];

/// Converts an activation matrix to the distance matrix of its nodes.
///
/// Rows are nodes, columns are the samples over which activity was recorded.
/// Each row is centred and scaled to unit norm, after which the Euclidean
/// distance between two rows is exactly `sqrt(2(1 - rho))` for their
/// correlation `rho`. Building the distance directly from that identity keeps
/// the geometry a genuine metric, which the Rips construction requires, instead
/// of an ad hoc function of correlation that need not satisfy the triangle
/// inequality.
///
/// A node whose activity does not vary has no correlation with anything. Its
/// row is left at zero, which places it at distance one from every other node:
/// far, but not infinitely so, and without introducing a non-finite entry that
/// would propagate through the filtration.
pub fn correlation_distances(activations: &DMatrix<f64>) -> DMatrix<f64> {
    let (n, m) = (activations.nrows(), activations.ncols());
    let mut z = DMatrix::zeros(n, m);
    for i in 0..n {
        let row: Vec<f64> = (0..m).map(|j| activations[(i, j)]).collect();
        let mean = row.iter().sum::<f64>() / m as f64;
        let norm = row
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            .sqrt();
        if norm > 1e-12 {
            for j in 0..m {
                z[(i, j)] = (row[j] - mean) / norm;
            }
        }
    }
    let mut d = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut acc = 0.0;
            for k in 0..m {
                let diff = z[(i, k)] - z[(j, k)];
                acc += diff * diff;
            }
            let value = acc.sqrt();
            d[(i, j)] = value;
            d[(j, i)] = value;
        }
    }
    d
}

/// Largest distance the filtration runs to.
///
/// Standardised rows sit on a unit sphere, so no two are further apart than
/// two; running the filtration to that bound means every feature that can be
/// born has been, and none is truncated by the choice of scale.
pub const MAX_RADIUS: f64 = 2.0;

/// Finite bar lengths per homology dimension, with the crate's defects undone.
///
/// Two corrections are applied, both established against inputs whose answer is
/// known by hand rather than assumed, and both written up in full at
/// `docs/tda-0.1.0-persistence-defects.md`, root cause and suggested patch
/// included:
///
/// - Every finite dimension-zero pair is returned twice, because the crate
///   computes dimension zero once by union-find and again by reducing the edge
///   boundary matrix, keeping both. On four points at 0, 1, 3 and 6, whose
///   spanning tree has edges 1, 2 and 3, it returns `[1,1,2,2,3,3]`. The
///   duplicates are exact, so sorting and taking every second entry recovers
///   the true multiset. Halving the sum would fix a total but would apply the
///   per-feature threshold to phantom features, so the deduplication is done on
///   the bars themselves.
/// - Homology is returned only up to one less than the dimension requested, so
///   [`TDA_DIMENSION_REQUEST`] asks for one more than is wanted.
///
/// Infinite bars are dropped throughout. The same exclusion applies to every
/// condition and to the calibration.
pub fn bar_lengths(distances: &DMatrix<f64>) -> [Vec<f64>; MAX_DIMENSION + 1] {
    bar_lengths_to(distances, MAX_DIMENSION)
}

/// The same, computing only up to `highest`.
///
/// A weighting that puts no weight on dimension two does not need it, and
/// stopping a dimension earlier is worth roughly a factor of thirty in time:
/// on thirty-two nodes the complex through dimension two costs about nine
/// milliseconds and through dimension three about three hundred and thirty.
/// That difference decides whether a signal can be recomputed during training
/// at all.
pub fn bar_lengths_to(distances: &DMatrix<f64>, highest: usize) -> [Vec<f64>; MAX_DIMENSION + 1] {
    let complex = vietoris_rips_complex(distances, MAX_RADIUS, highest + 1)
        .expect("the Rips complex builds from a finite distance matrix");
    let pairs = compute_persistence(&complex, highest + 1).expect("persistence computes");
    let mut out: [Vec<f64>; MAX_DIMENSION + 1] = Default::default();
    for pair in pairs.iter() {
        if pair.is_infinite() || pair.dimension > MAX_DIMENSION {
            continue;
        }
        out[pair.dimension].push(pair.persistence());
    }
    out[0].sort_by(|a, b| a.partial_cmp(b).expect("bar lengths are finite"));
    out[0] = out[0].iter().step_by(2).copied().collect();
    out
}

/// Total finite persistence per dimension, after the corrections.
pub fn persistence_totals(distances: &DMatrix<f64>) -> [f64; MAX_DIMENSION + 1] {
    let bars = bar_lengths(distances);
    [
        bars[0].iter().sum(),
        bars[1].iter().sum(),
        bars[2].iter().sum(),
    ]
}

/// Computes `T(S)` from a distance matrix.
pub fn resilience(distances: &DMatrix<f64>, weights: &Weights) -> f64 {
    let highest = (0..=MAX_DIMENSION)
        .rev()
        .find(|d| weights.w[*d] > 0.0)
        .unwrap_or(0);
    let bars = bar_lengths_to(distances, highest);
    let mut total = 0.0;
    for (d, lengths) in bars.iter().enumerate() {
        if weights.w[d] == 0.0 {
            continue;
        }
        for length in lengths {
            let excess = length - weights.sigma[d];
            if excess > 0.0 {
                total += weights.w[d] * excess;
            }
        }
    }
    total
}

/// `T(S)` straight from an activation matrix.
pub fn resilience_of(activations: &DMatrix<f64>, weights: &Weights) -> f64 {
    resilience(&correlation_distances(activations), weights)
}

/// One row of the calibration table.
#[derive(Debug, Clone, Serialize)]
pub struct CalibrationRow {
    /// Weight configuration.
    pub weights: String,
    /// `T(S)` for a modular structure, which has clusters but no cycle.
    pub modular: f64,
    /// `T(S)` for a ring structure, which carries a genuine loop.
    pub ring: f64,
    /// `T(S)` for modules arranged on a ring, which carries both.
    pub modular_ring: f64,
    /// `T(S)` for independent noise.
    pub noise: f64,
    /// Whether both loop-carrying cases exceed noise, which is what the
    /// hypothesis asserts. The plain modular case is not part of the condition:
    /// it has no cycle, so a measure of loops scoring it at noise level is
    /// correct behaviour rather than a failure.
    pub loops_exceed_noise: bool,
}

/// Result of the mandatory calibration.
#[derive(Debug, Clone, Serialize)]
pub struct Calibration {
    /// One row per weight configuration.
    pub rows: Vec<CalibrationRow>,
    /// Whether every configuration ranked structure above noise.
    pub passes: bool,
}

/// Builds synthetic activations whose correlation structure is known.
///
/// Three cases, following the logic of a synthetic validation on graphs of
/// controlled topology, which is a reasonable construction independently of any
/// result claimed from it:
///
/// - `modular`: nodes fall into blocks driven by a shared latent signal, so the
///   correlation matrix is block structured and the point cloud separates into
///   well-spaced clusters that persist as components.
/// - `ring`: node `i` is driven by a phase on a circle, so correlation decays
///   with angular separation and the cloud lies on a closed curve, which is the
///   one construction here that carries a genuine one-dimensional cycle.
/// - `noise`: independent activity, which should carry neither.
fn synthetic(kind: &str, nodes: usize, samples: usize, seed: u64) -> DMatrix<f64> {
    let mut rng = chaos_rng::ChaChaRng::from_seed(seed);
    let mut a = DMatrix::zeros(nodes, samples);
    match kind {
        "modular" => {
            let blocks = 4;
            let per = nodes / blocks;
            for s in 0..samples {
                let latent: Vec<f64> = (0..blocks).map(|_| rng.next_normal()).collect();
                for i in 0..nodes {
                    let block = (i / per).min(blocks - 1);
                    a[(i, s)] = latent[block] + 0.25 * rng.next_normal();
                }
            }
        }
        "ring" => {
            for s in 0..samples {
                let phase = 2.0 * std::f64::consts::PI * rng.next_f64();
                for i in 0..nodes {
                    let angle = 2.0 * std::f64::consts::PI * i as f64 / nodes as f64;
                    a[(i, s)] = (phase - angle).cos() + 0.15 * rng.next_normal();
                }
            }
        }
        "modular-ring" => {
            // Blocks arranged around a circle: clusters and a genuine cycle at
            // once. This is the case the hypothesis actually speaks about,
            // "modular structure with loops", which the plain modular case does
            // not exhibit.
            let blocks = 6;
            let per = nodes / blocks;
            for s in 0..samples {
                let phase = 2.0 * std::f64::consts::PI * rng.next_f64();
                for i in 0..nodes {
                    let block = (i / per).min(blocks - 1);
                    let angle = 2.0 * std::f64::consts::PI * block as f64 / blocks as f64;
                    a[(i, s)] = (phase - angle).cos() + 0.15 * rng.next_normal();
                }
            }
        }
        "noise" => {
            for s in 0..samples {
                for i in 0..nodes {
                    a[(i, s)] = rng.next_normal();
                }
            }
        }
        other => panic!("unknown synthetic structure {other}"),
    }
    a
}

/// Nodes used by the calibration, matching a hidden layer of the Phase 9
/// network.
pub const CALIBRATION_NODES: usize = 32;
/// Samples used by the calibration.
pub const CALIBRATION_SAMPLES: usize = 128;

/// Checks that `T(S)` responds in the direction the hypothesis requires.
///
/// If a measure of topological structure does not separate structure from noise
/// on cases built to have or lack it, then whatever it is measuring on a real
/// network is not what its name says, and no comparison built on it would mean
/// anything. This runs before any of it is used.
pub fn calibrate() -> Calibration {
    let modular = synthetic("modular", CALIBRATION_NODES, CALIBRATION_SAMPLES, 4_001);
    let ring = synthetic("ring", CALIBRATION_NODES, CALIBRATION_SAMPLES, 4_002);
    let modular_ring = synthetic(
        "modular-ring",
        CALIBRATION_NODES,
        CALIBRATION_SAMPLES,
        4_004,
    );
    let noise = synthetic("noise", CALIBRATION_NODES, CALIBRATION_SAMPLES, 4_003);

    let rows: Vec<CalibrationRow> = SWEEP
        .iter()
        .map(|w| {
            let m = resilience_of(&modular, w);
            let r = resilience_of(&ring, w);
            let mr = resilience_of(&modular_ring, w);
            let n = resilience_of(&noise, w);
            CalibrationRow {
                weights: w.label.to_string(),
                modular: m,
                ring: r,
                modular_ring: mr,
                noise: n,
                loops_exceed_noise: r > n && mr > n,
            }
        })
        .collect();
    // The gate is on the configurations actually carried forward.
    let passes = rows
        .iter()
        .filter(|r| r.weights != "with-components")
        .all(|r| r.loops_exceed_noise);
    Calibration { rows, passes }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_swept_weighting_is_a_convex_combination() {
        for w in SWEEP {
            assert!(w.is_normalised(), "{} does not sum to one", w.label);
            assert!(
                w.w.iter().all(|v| *v >= 0.0),
                "{} has a negative weight",
                w.label
            );
        }
    }

    #[test]
    fn the_distance_matches_the_correlation_identity() {
        // Two rows with a known correlation must land at sqrt(2(1 - rho)).
        let a = DMatrix::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0]);
        let d = correlation_distances(&a);
        // Perfectly correlated, so distance zero.
        assert!(d[(0, 1)].abs() < 1e-12, "got {}", d[(0, 1)]);

        let b = DMatrix::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0]);
        let d = correlation_distances(&b);
        // Perfectly anticorrelated, so sqrt(2 * 2) = 2.
        assert!((d[(0, 1)] - 2.0).abs() < 1e-12, "got {}", d[(0, 1)]);
    }

    #[test]
    fn a_constant_node_is_placed_at_a_finite_distance() {
        let a = DMatrix::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]);
        let d = correlation_distances(&a);
        assert!(d[(0, 1)].is_finite());
        assert!((d[(0, 1)] - 1.0).abs() < 1e-12, "got {}", d[(0, 1)]);
    }

    #[test]
    fn the_distance_matrix_is_a_metric_on_these_inputs() {
        let a = synthetic("modular", 12, 40, 7);
        let d = correlation_distances(&a);
        for i in 0..d.nrows() {
            assert_eq!(d[(i, i)], 0.0);
            for j in 0..d.ncols() {
                assert!((d[(i, j)] - d[(j, i)]).abs() < 1e-15);
                for k in 0..d.ncols() {
                    assert!(
                        d[(i, k)] <= d[(i, j)] + d[(j, k)] + 1e-9,
                        "triangle inequality failed at {i} {j} {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_higher_threshold_never_raises_the_score() {
        let a = synthetic("ring", 20, 64, 9);
        let d = correlation_distances(&a);
        let light = Weights {
            label: "t",
            w: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            sigma: [0.05, 0.05, 0.05],
        };
        let firm = Weights {
            sigma: [0.5, 0.5, 0.5],
            ..light
        };
        assert!(resilience(&d, &firm) <= resilience(&d, &light));
    }

    #[test]
    fn deduplication_recovers_the_true_dimension_zero_multiset() {
        // Four points on a line, spaced so the whole spanning tree fits inside
        // MAX_RADIUS: edges of 0.3, 0.6 and 0.9. The crate returns each of
        // those twice; deduplication must recover exactly one of each.
        let pts = [0.0f64, 0.3, 0.9, 1.8];
        let n = pts.len();
        let mut d = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] = (pts[i] - pts[j]).abs();
            }
        }
        let bars = bar_lengths(&d);
        assert_eq!(bars[0].len(), 3, "got {:?}", bars[0]);
        for (got, want) in bars[0].iter().zip([0.3, 0.6, 0.9].iter()) {
            assert!((got - want).abs() < 1e-9, "got {:?}", bars[0]);
        }
    }

    #[test]
    fn the_requested_dimension_is_actually_computed() {
        // Guards the offset. If the crate is fixed upstream this fails, and the
        // offset must come out rather than silently double-counting dimensions.
        let a = synthetic("ring", 14, 64, 3);
        let d = correlation_distances(&a);
        let complex = vietoris_rips_complex(&d, MAX_RADIUS, TDA_DIMENSION_REQUEST).unwrap();
        let pairs = compute_persistence(&complex, TDA_DIMENSION_REQUEST).unwrap();
        assert!(
            pairs.iter().any(|p| p.dimension == MAX_DIMENSION),
            "nothing came back in dimension {MAX_DIMENSION}"
        );
    }

    #[test]
    fn per_dimension_evidence() {
        // Not an assertion: the measurement that decides the weight sweep.
        for kind in ["modular", "ring", "modular-ring", "noise"] {
            let a = synthetic(kind, CALIBRATION_NODES, CALIBRATION_SAMPLES, 4_001);
            let bars = bar_lengths(&correlation_distances(&a));
            let longest = |d: usize| bars[d].iter().cloned().fold(0.0, f64::max);
            let over = |d: usize, s: f64| bars[d].iter().filter(|v| **v > s).count();
            println!(
                "{kind:>13}  n(H1)={:<5} longest H1 {:>7.4}  over 0.10: {:<3}  longest H2 {:>7.4}  over 0.10: {}",
                bars[1].len(),
                longest(1),
                over(1, 0.10),
                longest(2),
                over(2, 0.10)
            );
        }
    }

    #[test]
    fn total_dimension_zero_persistence_is_the_spanning_tree_weight() {
        // The identity behind the calibration result. Every dimension-zero bar
        // is born at zero and dies when its component merges, and those merge
        // radii are exactly the edges of the minimum spanning tree. Total
        // dimension-zero persistence is therefore a measure of how spread out
        // the cloud is, maximised by points all mutually far apart, which is
        // what independent noise gives. That is why no positive weight on
        // dimension zero can rank structure above noise, and why the sweep
        // keeps one such weighting only to show it failing.
        for kind in ["modular", "ring", "noise"] {
            let a = synthetic(kind, 16, 64, 21);
            let d = correlation_distances(&a);
            let n = d.nrows();

            // Prim's algorithm on the complete distance graph.
            let mut in_tree = vec![false; n];
            let mut best = vec![f64::INFINITY; n];
            best[0] = 0.0;
            let mut mst = 0.0;
            for _ in 0..n {
                let mut u = usize::MAX;
                for v in 0..n {
                    if !in_tree[v] && (u == usize::MAX || best[v] < best[u]) {
                        u = v;
                    }
                }
                in_tree[u] = true;
                mst += best[u];
                for v in 0..n {
                    if !in_tree[v] && d[(u, v)] < best[v] {
                        best[v] = d[(u, v)];
                    }
                }
            }

            let h0: f64 = bar_lengths(&d)[0].iter().sum();
            assert!(
                (h0 - mst).abs() < 1e-6,
                "{kind}: deduplicated dimension-zero total {h0} against spanning tree weight {mst}"
            );
        }
    }

    #[test]
    fn the_calibration_separates_loops_from_noise() {
        // Blocking. Without this the quantity would be a name, not a measure.
        let c = calibrate();
        for row in c.rows.iter().filter(|r| r.weights != "with-components") {
            assert!(
                row.loops_exceed_noise,
                "{}: ring {:.4} and modular-ring {:.4} against noise {:.4}",
                row.weights, row.ring, row.modular_ring, row.noise
            );
        }
        assert!(c.passes);
    }

    #[test]
    fn a_weighting_that_includes_dimension_zero_fails_as_expected() {
        // The failure is part of the result, so it is pinned. If this ever
        // passes, the claim in the report that no positive w_0 can work needs
        // revisiting rather than quietly benefiting from the change.
        let c = calibrate();
        let row = c
            .rows
            .iter()
            .find(|r| r.weights == "with-components")
            .expect("the failing configuration is retained deliberately");
        assert!(
            !row.loops_exceed_noise,
            "dimension zero no longer ranks noise first: ring {:.4}, modular-ring {:.4}, noise {:.4}",
            row.ring, row.modular_ring, row.noise
        );
    }
}
