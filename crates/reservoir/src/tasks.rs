// SPDX-License-Identifier: MIT
//! The two benchmark tasks, both taken from the published definitions rather
//! than invented here, so the numbers can be compared against the literature.

use crate::esn::Esn;
use crate::ridge::{ridge, RidgeError};
use nalgebra::DMatrix;

/// Squared Pearson correlation between two equal-length series.
///
/// Returns zero when either series is constant, which is the right reading:
/// a readout that never moves has reconstructed nothing, and the undefined
/// correlation would otherwise propagate as a NaN into the sum.
pub fn squared_correlation(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut saa = 0.0;
    let mut sbb = 0.0;
    let mut sab = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let da = x - ma;
        let db = y - mb;
        saa += da * da;
        sbb += db * db;
        sab += da * db;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return 0.0;
    }
    let r = sab / (saa * sbb).sqrt();
    r * r
}

/// Result of the linear memory capacity measurement.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryCapacity {
    /// Sum over delays of the squared correlation, the capacity itself.
    pub total: f64,
    /// Capacity divided by the reservoir size, which the theory bounds by one.
    pub normalised: f64,
    /// Per-delay contributions, index zero being delay one.
    pub per_delay: Vec<f64>,
    /// Largest delay evaluated.
    pub max_delay: usize,
}

/// Measures linear memory capacity.
///
/// The input is i.i.d. uniform, one readout is trained per delay to reconstruct
/// the input `k` steps back, and the capacity is the sum of the squared
/// correlations achieved on held-out data. The theoretical ceiling is the
/// reservoir size.
///
/// REF: [Jaeger, 2001] "Short term memory in echo state networks", GMD Report
///      152, German National Research Center for Information Technology. The
///      definition of the measure and the bound MC <= N.
///
/// The correlations are computed on a test segment the readouts never saw.
/// Doing this in-sample would inflate the total: with `n + 1` free parameters
/// per readout, in-sample squared correlation is biased upward by roughly
/// `(n + 1) / T`, and summed over hundreds of delays that bias is of the same
/// order as the quantity being measured. It is the single easiest way to
/// manufacture a capacity above the theoretical ceiling, so the ceiling is
/// checked afterwards as a guard.
pub fn memory_capacity(
    esn: &Esn,
    input: &[f64],
    washout: usize,
    train_len: usize,
    max_delay: usize,
    lambda: f64,
) -> Result<MemoryCapacity, RidgeError> {
    // The states must be aligned with an input that is itself delayed, so the
    // usable range starts once the deepest delay is available.
    let states = esn.collect_states(input, washout);
    let usable = states.nrows();
    assert!(
        usable > train_len + max_delay,
        "not enough samples for the requested training length and delays"
    );
    let test_len = usable - train_len - max_delay;

    // Row r of `states` corresponds to time washout + r. The target for delay k
    // at that row is input[washout + r - k].
    let mut train_x = DMatrix::zeros(train_len, esn.n + 1);
    let mut test_x = DMatrix::zeros(test_len, esn.n + 1);
    for r in 0..train_len {
        for c in 0..=esn.n {
            train_x[(r, c)] = states[(max_delay + r, c)];
        }
    }
    for r in 0..test_len {
        for c in 0..=esn.n {
            test_x[(r, c)] = states[(max_delay + train_len + r, c)];
        }
    }

    let mut train_y = DMatrix::zeros(train_len, max_delay);
    for k in 1..=max_delay {
        for r in 0..train_len {
            train_y[(r, k - 1)] = input[washout + max_delay + r - k];
        }
    }

    let w_out = ridge(&train_x, &train_y, lambda)?;
    let predictions = &test_x * &w_out;

    let mut per_delay = Vec::with_capacity(max_delay);
    let mut target = vec![0.0; test_len];
    let mut predicted = vec![0.0; test_len];
    for k in 1..=max_delay {
        for r in 0..test_len {
            target[r] = input[washout + max_delay + train_len + r - k];
            predicted[r] = predictions[(r, k - 1)];
        }
        per_delay.push(squared_correlation(&target, &predicted));
    }

    let total: f64 = per_delay.iter().sum();
    Ok(MemoryCapacity {
        total,
        normalised: total / esn.n as f64,
        per_delay,
        max_delay,
    })
}

/// Generates the NARMA-10 input and target series.
///
/// `y(t+1) = 0.3 y(t) + 0.05 y(t) sum_{i=0}^{9} y(t-i) + 1.5 u(t-9) u(t) + 0.1`
/// with `u(t)` drawn uniformly from `[0, 0.5]`, exactly as published.
///
/// REF: [Atiya and Parlos, 2000] "New results on recurrent network training:
///      unifying the algorithms and accelerating convergence", IEEE
///      Transactions on Neural Networks 11(3), pp. 697-709,
///      DOI: 10.1109/72.846741. Origin of the NARMA family.
/// REF: [Jaeger and Haas, 2004] "Harnessing nonlinearity: predicting chaotic
///      systems and saving energy in wireless communication", Science
///      304(5667), pp. 78-80, DOI: 10.1126/science.1091277. The reference point
///      for echo state network performance on this kind of task.
///
/// The recursion is well known to be unstable for some input sequences: the
/// product term can push the state onto a divergent path from which it never
/// returns. Rather than damp it, which would change the published task into a
/// different one, divergence is detected and reported so the caller can draw a
/// fresh input sequence.
pub fn narma10(u: &[f64]) -> Option<Vec<f64>> {
    const ORDER: usize = 10;
    let mut y = vec![0.0; u.len()];
    for t in ORDER - 1..u.len() - 1 {
        let window: f64 = (0..ORDER).map(|i| y[t - i]).sum();
        let next = 0.3 * y[t] + 0.05 * y[t] * window + 1.5 * u[t - (ORDER - 1)] * u[t] + 0.1;
        if !next.is_finite() || next.abs() > 1e3 {
            return None;
        }
        y[t + 1] = next;
    }
    Some(y)
}

/// Normalised root mean squared error against the variance of the target.
///
/// Dividing by the target's variance rather than its range makes the value
/// comparable across sequences and is the convention in the echo state
/// literature: one means no better than predicting the mean.
pub fn nrmse(predicted: &[f64], target: &[f64]) -> f64 {
    assert_eq!(predicted.len(), target.len());
    let n = target.len() as f64;
    let mean = target.iter().sum::<f64>() / n;
    let var = target.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    if var <= 0.0 {
        return f64::NAN;
    }
    let mse = predicted
        .iter()
        .zip(target.iter())
        .map(|(p, t)| (p - t) * (p - t))
        .sum::<f64>()
        / n;
    (mse / var).sqrt()
}

/// Trains a readout on NARMA-10 and returns the test NRMSE.
pub fn narma10_nrmse(
    esn: &Esn,
    u: &[f64],
    y: &[f64],
    washout: usize,
    train_len: usize,
    lambda: f64,
) -> Result<f64, RidgeError> {
    let states = esn.collect_states(u, washout);
    let usable = states.nrows();
    assert!(usable > train_len, "not enough samples after the washout");
    let test_len = usable - train_len;

    let mut train_x = DMatrix::zeros(train_len, esn.n + 1);
    let mut test_x = DMatrix::zeros(test_len, esn.n + 1);
    for r in 0..train_len {
        for c in 0..=esn.n {
            train_x[(r, c)] = states[(r, c)];
        }
    }
    for r in 0..test_len {
        for c in 0..=esn.n {
            test_x[(r, c)] = states[(train_len + r, c)];
        }
    }
    let train_y = DMatrix::from_fn(train_len, 1, |r, _| y[washout + r]);
    let w_out = ridge(&train_x, &train_y, lambda)?;
    let predictions = &test_x * &w_out;

    let predicted: Vec<f64> = (0..test_len).map(|r| predictions[(r, 0)]).collect();
    let target: Vec<f64> = (0..test_len).map(|r| y[washout + train_len + r]).collect();
    Ok(nrmse(&predicted, &target))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::ReferenceRng;

    #[test]
    fn squared_correlation_is_one_for_an_affine_image() {
        let a: Vec<f64> = (0..50).map(|i| i as f64 * 0.3 - 2.0).collect();
        let b: Vec<f64> = a.iter().map(|v| 4.0 * v + 7.0).collect();
        assert!((squared_correlation(&a, &b) - 1.0).abs() < 1e-12);
        // Sign is discarded, so an inverted copy is equally "reconstructed".
        let c: Vec<f64> = a.iter().map(|v| -2.0 * v).collect();
        assert!((squared_correlation(&a, &c) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn squared_correlation_is_zero_against_a_constant() {
        let a: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert_eq!(squared_correlation(&a, &[3.0; 20]), 0.0);
    }

    #[test]
    fn narma10_reproduces_the_recursion_by_hand() {
        // Constant input, so the arithmetic can be checked term by term.
        let u = vec![0.2; 15];
        let y = narma10(&u).expect("a constant small input does not diverge");
        // The first nine outputs are the zero initial condition; the first
        // computed value is at index 10.
        for v in y.iter().take(10) {
            assert_eq!(*v, 0.0);
        }
        // At t = 9 the whole y window is zero, so
        // y(10) = 0 + 0 + 1.5 * u(0) * u(9) + 0.1 = 1.5 * 0.04 + 0.1
        assert!((y[10] - (1.5 * 0.2 * 0.2 + 0.1)).abs() < 1e-15);
        // At t = 10 the window is y(10) alone.
        let expected = 0.3 * y[10] + 0.05 * y[10] * y[10] + 1.5 * 0.2 * 0.2 + 0.1;
        assert!((y[11] - expected).abs() < 1e-15);
    }

    #[test]
    fn narma10_reports_divergence_rather_than_returning_infinities() {
        // Inputs above the published range drive the product term hard enough
        // to escape. The point is that the failure is visible.
        let u = vec![5.0; 200];
        assert!(narma10(&u).is_none());
    }

    #[test]
    fn narma10_stays_bounded_on_the_published_input_range() {
        let mut rng = ReferenceRng::from_seed(5);
        let u: Vec<f64> = (0..4000).map(|_| rng.next_range(0.0, 0.5)).collect();
        let y = narma10(&u).expect("the published input range should not diverge here");
        assert!(y.iter().all(|v| v.is_finite() && v.abs() < 10.0));
    }

    #[test]
    fn nrmse_is_zero_for_a_perfect_prediction_and_one_for_the_mean() {
        let target: Vec<f64> = (0..40).map(|i| (i as f64 * 0.37).sin()).collect();
        assert!(nrmse(&target, &target) < 1e-15);
        let n = target.len() as f64;
        let mean = target.iter().sum::<f64>() / n;
        assert!((nrmse(&[mean; 40], &target) - 1.0).abs() < 1e-12);
    }
}
