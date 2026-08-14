// SPDX-License-Identifier: MIT
//! Ridge regression in closed form: the readout training, and the point where
//! this phase departs from every earlier one.
//!
//! Phases 1 through 7 trained by stochastic gradient descent, an iterative
//! procedure whose path depends on the order of the data, on the dropout masks
//! and on the initial weights, which is exactly why the randomness source had
//! three separate places to enter. Here there is no iteration and no schedule.
//! The readout is the solution of a linear system, determined by the data and
//! the penalty alone, and identical to the last bit whatever generator filled
//! the reservoir. All the randomness lives in the fixed recurrent weights.

use nalgebra::DMatrix;

/// Solves `min ||X W - Y||^2 + lambda ||W||^2` for `W`.
///
/// The normal equations are `(X'X + lambda I) W = X'Y`. The penalised Gram
/// matrix is symmetric positive definite for any positive `lambda`, so a
/// Cholesky factorisation both solves it and asserts that conditioning has not
/// been lost: if the factorisation fails, the caller learns that rather than
/// receiving a silently wrong solution from a more forgiving solver.
///
/// The bias column is not penalised. Shrinking an intercept toward zero would
/// make the fit depend on where the target happens to sit, which is a
/// well-known defect of the naive formulation rather than a modelling choice.
///
/// REF: [Hoerl and Kennard, 1970] "Ridge Regression: Biased Estimation for
///      Nonorthogonal Problems", Technometrics 12(1), pp. 55-67,
///      DOI: 10.1080/00401706.1970.10488634
pub fn ridge(x: &DMatrix<f64>, y: &DMatrix<f64>, lambda: f64) -> Result<DMatrix<f64>, RidgeError> {
    assert_eq!(
        x.nrows(),
        y.nrows(),
        "design and targets disagree on samples"
    );
    assert!(lambda > 0.0, "the penalty must be positive");
    let p = x.ncols();
    let mut gram = x.transpose() * x;
    // Column 0 is the bias, left unpenalised for the reason above.
    for i in 1..p {
        gram[(i, i)] += lambda;
    }
    let rhs = x.transpose() * y;
    let chol = gram.cholesky().ok_or(RidgeError::NotPositiveDefinite)?;
    Ok(chol.solve(&rhs))
}

/// Why a ridge solve failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RidgeError {
    /// The penalised Gram matrix did not factorise, which means the design is
    /// far worse conditioned than the penalty can repair.
    NotPositiveDefinite,
}

impl std::fmt::Display for RidgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RidgeError::NotPositiveDefinite => {
                write!(f, "the penalised Gram matrix is not positive definite")
            }
        }
    }
}

impl std::error::Error for RidgeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_an_exact_linear_relation() {
        // y = 3 + 2 a - 1 b, with a tiny penalty, must come back very close.
        let x = DMatrix::from_row_slice(
            5,
            3,
            &[
                1.0, 0.0, 0.0, //
                1.0, 1.0, 0.0, //
                1.0, 0.0, 1.0, //
                1.0, 2.0, 1.0, //
                1.0, 1.0, 3.0,
            ],
        );
        let truth = [3.0, 2.0, -1.0];
        let y = DMatrix::from_fn(5, 1, |i, _| {
            truth[0] + truth[1] * x[(i, 1)] + truth[2] * x[(i, 2)]
        });
        let w = ridge(&x, &y, 1e-12).unwrap();
        for (i, expected) in truth.iter().enumerate() {
            assert!((w[(i, 0)] - expected).abs() < 1e-6, "coefficient {i}");
        }
    }

    #[test]
    fn a_larger_penalty_shrinks_the_slopes_but_not_the_intercept() {
        let x = DMatrix::from_row_slice(4, 2, &[1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
        let y = DMatrix::from_row_slice(4, 1, &[8.0, 10.0, 12.0, 14.0]);
        let light = ridge(&x, &y, 1e-10).unwrap();
        let heavy = ridge(&x, &y, 50.0).unwrap();
        assert!(
            heavy[(1, 0)].abs() < light[(1, 0)].abs(),
            "slope must shrink"
        );

        // With the intercept unpenalised, the fit still passes through the
        // centroid of the data: b0 = mean(y) - b1 * mean(x). It is worth
        // stating in that form rather than as "b0 stays at mean(y)", which
        // holds only for a centred predictor and is false here, where the
        // predictor averages 0.5.
        let mean_x = 0.5;
        let mean_y = 11.0;
        for w in [&light, &heavy] {
            assert!(
                (w[(0, 0)] - (mean_y - w[(1, 0)] * mean_x)).abs() < 1e-9,
                "the fit left the centroid"
            );
        }
        // The point of leaving it unpenalised: it does not collapse toward zero
        // as the slope is suppressed.
        assert!(heavy[(0, 0)] > 10.0);
    }

    #[test]
    fn solves_several_targets_at_once() {
        // The memory capacity task fits hundreds of readouts against one
        // design matrix, so multi-column targets must give the same answer as
        // separate solves.
        let x = DMatrix::from_row_slice(4, 2, &[1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
        let y = DMatrix::from_row_slice(4, 2, &[8.0, 1.0, 10.0, 0.0, 12.0, 1.0, 14.0, 4.0]);
        let both = ridge(&x, &y, 1e-6).unwrap();
        let col = |j: usize| DMatrix::from_fn(y.nrows(), 1, |r, _| y[(r, j)]);
        let first = ridge(&x, &col(0), 1e-6).unwrap();
        let second = ridge(&x, &col(1), 1e-6).unwrap();
        for i in 0..2 {
            assert!((both[(i, 0)] - first[(i, 0)]).abs() < 1e-9);
            assert!((both[(i, 1)] - second[(i, 0)]).abs() < 1e-9);
        }
    }
}
