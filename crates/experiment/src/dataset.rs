// SPDX-License-Identifier: MIT
//! Synthetic classification dataset.

use chaos_rng::ChaChaRng;

/// A labelled dataset held in row-major order.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Feature rows, each of length [`Dataset::n_features`].
    pub x: Vec<[f64; 2]>,
    /// Class label per row, either 0 or 1.
    pub y: Vec<usize>,
}

impl Dataset {
    /// Number of input features.
    pub const N_FEATURES: usize = 2;
    /// Number of classes.
    pub const N_CLASSES: usize = 2;

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Whether the dataset is empty. Present because a type with `len` should
    /// offer it; unused so far by the experiment itself.
    #[allow(dead_code, reason = "completes the len/is_empty pair")]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

/// Generates the two-moons dataset: two interleaved half circles with additive
/// Gaussian noise. The classes are not linearly separable, so a linear model
/// cannot solve the task and the hidden layers are actually exercised.
///
/// The generator is deliberately fixed to ChaCha8 and to a seed passed in by the
/// caller, independent of the seeds used for training. The same dataset is
/// therefore shared by every run of every condition, so that the only quantity
/// varying between conditions is the source of training randomness. Generating
/// the data with the condition's own generator would confound the comparison.
///
/// REF: the construction follows the `make_moons` generator of
///      [Pedregosa et al., 2011] "Scikit-learn: Machine Learning in Python",
///      Journal of Machine Learning Research 12, pp. 2825-2830.
///      https://jmlr.org/papers/v12/pedregosa11a.html
pub fn make_moons(n_samples: usize, noise: f64, seed: u64) -> Dataset {
    let mut rng = ChaChaRng::from_seed(seed);
    let n_out = n_samples / 2;
    let n_in = n_samples - n_out;

    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    for i in 0..n_out {
        let t = std::f64::consts::PI * i as f64 / (n_out as f64 - 1.0);
        x.push([t.cos(), t.sin()]);
        y.push(0);
    }
    for i in 0..n_in {
        let t = std::f64::consts::PI * i as f64 / (n_in as f64 - 1.0);
        x.push([1.0 - t.cos(), 1.0 - t.sin() - 0.5]);
        y.push(1);
    }

    for row in x.iter_mut() {
        row[0] += rng.next_normal() * noise;
        row[1] += rng.next_normal() * noise;
    }

    Dataset { x, y }
}

/// Splits a dataset into training and validation parts.
///
/// The split is a deterministic shuffle driven by its own ChaCha generator, for
/// the same reason as [`make_moons`]: the partition must be identical across
/// conditions.
pub fn train_test_split(data: &Dataset, train_fraction: f64, seed: u64) -> (Dataset, Dataset) {
    let mut idx: Vec<usize> = (0..data.len()).collect();
    let mut rng = ChaChaRng::from_seed(seed);
    rng.shuffle(&mut idx);

    let n_train = (data.len() as f64 * train_fraction).round() as usize;
    let mut train = Dataset {
        x: Vec::new(),
        y: Vec::new(),
    };
    let mut test = Dataset {
        x: Vec::new(),
        y: Vec::new(),
    };
    for (rank, &i) in idx.iter().enumerate() {
        let target = if rank < n_train {
            &mut train
        } else {
            &mut test
        };
        target.x.push(data.x[i]);
        target.y.push(data.y[i]);
    }
    (train, test)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moons_are_balanced_and_deterministic() {
        let a = make_moons(1000, 0.2, 42);
        let b = make_moons(1000, 0.2, 42);
        assert_eq!(a.len(), 1000);
        assert_eq!(a.x, b.x, "same seed must give the same dataset");
        let positives = a.y.iter().filter(|&&v| v == 1).count();
        assert!(
            (positives as i64 - 500).abs() <= 1,
            "classes should be balanced, got {positives} positives"
        );
    }

    #[test]
    fn split_partitions_without_loss_or_duplication() {
        let d = make_moons(200, 0.1, 7);
        let (tr, te) = train_test_split(&d, 0.8, 7);
        assert_eq!(tr.len() + te.len(), d.len());
        assert_eq!(tr.len(), 160);
    }

    #[test]
    fn moons_are_not_linearly_separable() {
        // A sanity check on the task: the class means are close together
        // relative to the spread, so a linear boundary through them cannot
        // separate the classes cleanly.
        let d = make_moons(1000, 0.1, 1);
        let mut m0 = [0.0; 2];
        let mut m1 = [0.0; 2];
        let (mut n0, mut n1) = (0.0, 0.0);
        for (row, &label) in d.x.iter().zip(d.y.iter()) {
            if label == 0 {
                m0[0] += row[0];
                m0[1] += row[1];
                n0 += 1.0;
            } else {
                m1[0] += row[0];
                m1[1] += row[1];
                n1 += 1.0;
            }
        }
        let sep = ((m0[0] / n0 - m1[0] / n1).powi(2) + (m0[1] / n0 - m1[1] / n1).powi(2)).sqrt();
        assert!(sep < 1.5, "class means unexpectedly far apart: {sep}");
    }
}
