// SPDX-License-Identifier: MIT
//! The echo state network itself: a fixed recurrent reservoir and the linear
//! readout trained on top of it.

use nalgebra::DMatrix;

/// A reservoir with its input matrix, before any readout is trained.
///
/// The reservoir is fixed for the lifetime of the network. Nothing in this
/// struct is ever updated by training, which is the property that makes the
/// phase meaningful: the weights drawn from a generator are the weights that
/// do the computing, not an initial condition that gradient descent walks away
/// from.
#[derive(Debug, Clone)]
pub struct Esn {
    /// Recurrent weights, `n` by `n`, already rescaled to the target radius.
    pub w_res: DMatrix<f64>,
    /// Input weights, `n` by 1, one scalar input channel.
    pub w_in: DMatrix<f64>,
    /// Reservoir size.
    pub n: usize,
}

impl Esn {
    /// Advances the state one step under input `u`.
    ///
    /// `x(t+1) = tanh(W_res x(t) + W_in u(t))`, the canonical formulation.
    pub fn step(&self, x: &mut DMatrix<f64>, u: f64) {
        let mut next = &self.w_res * &*x;
        next += &self.w_in * u;
        next.apply(|v| *v = v.tanh());
        *x = next;
    }

    /// Runs the reservoir over `input` from a zero state and returns the
    /// collected states after discarding `washout` of them.
    ///
    /// Each returned row is `[1, x_1, ..., x_n]`: the leading constant is the
    /// bias term of the readout, included here rather than bolted on later so
    /// the design matrix is complete where it is built.
    pub fn collect_states(&self, input: &[f64], washout: usize) -> DMatrix<f64> {
        assert!(
            input.len() > washout,
            "the washout consumes the whole input, leaving nothing to fit"
        );
        let kept = input.len() - washout;
        let mut design = DMatrix::zeros(kept, self.n + 1);
        let mut x = DMatrix::zeros(self.n, 1);
        for (t, &u) in input.iter().enumerate() {
            self.step(&mut x, u);
            if t >= washout {
                let row = t - washout;
                design[(row, 0)] = 1.0;
                for i in 0..self.n {
                    design[(row, i + 1)] = x[(i, 0)];
                }
            }
        }
        design
    }
}

/// How far apart two reservoir states remain after being driven by the same
/// input from different starting points.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct EchoStateCheck {
    /// Euclidean distance between the two states at the start.
    pub initial_separation: f64,
    /// Euclidean distance after the driving input has been applied.
    pub final_separation: f64,
    /// Whether the final separation is below [`ECHO_TOLERANCE`].
    pub holds: bool,
}

/// Separation below which two trajectories are taken to have merged.
///
/// The states live in `[-1, 1]^n`, so this is roughly four orders of magnitude
/// below the double-precision noise floor of quantities of that size, and many
/// orders below any separation that would matter for the readout.
pub const ECHO_TOLERANCE: f64 = 1e-11;

/// Verifies the echo state property numerically.
///
/// A spectral radius below one is the usual rule of thumb, but it is neither
/// necessary nor sufficient for the echo state property in a `tanh` network,
/// so a rescaled matrix is not evidence on its own. This drives two different
/// initial states with the same input and measures whether they converge,
/// which is the property's definition rather than a proxy for it.
///
/// REF: [Jaeger, 2001] "The 'echo state' approach to analysing and training
///      recurrent neural networks", GMD Report 148, German National Research
///      Center for Information Technology. Section 3 states the property and
///      notes that the spectral radius condition is a rule of thumb.
pub fn check_echo_state_property(esn: &Esn, input: &[f64], seed: u64) -> EchoStateCheck {
    let mut rng = crate::reference::ReferenceRng::from_seed(seed);
    let mut a: DMatrix<f64> = DMatrix::from_fn(esn.n, 1, |_, _| rng.next_range(-1.0, 1.0));
    let mut b: DMatrix<f64> = DMatrix::from_fn(esn.n, 1, |_, _| rng.next_range(-1.0, 1.0));
    let initial_separation = (&a - &b).norm();
    for &u in input {
        esn.step(&mut a, u);
        esn.step(&mut b, u);
    }
    let final_separation = (&a - &b).norm();
    EchoStateCheck {
        initial_separation,
        final_separation,
        holds: final_separation < ECHO_TOLERANCE,
    }
}

/// Spectral radius, computed from the full complex spectrum.
///
/// This is the value used for rescaling. See
/// [`spectral_radius_power_iteration`] for why the cheaper method is not the
/// one relied on.
pub fn spectral_radius(m: &DMatrix<f64>) -> f64 {
    m.complex_eigenvalues()
        .iter()
        .map(|z| z.norm())
        .fold(0.0, f64::max)
}

/// Spectral radius by power iteration, kept for the comparison that justifies
/// not using it.
///
/// Power iteration converges to the dominant eigenvalue only when that
/// eigenvalue is real and strictly larger in magnitude than the rest. The
/// recurrent matrix of a reservoir is not symmetric and its dominant
/// eigenvalues are routinely a complex conjugate pair of equal magnitude, in
/// which case the iterate rotates instead of converging and the returned value
/// is whatever the loop happened to stop on. A test in this module exhibits
/// exactly that failure on a rotation, and it is the reason the rescaling uses
/// the full spectrum instead.
pub fn spectral_radius_power_iteration(m: &DMatrix<f64>, iterations: usize) -> f64 {
    let n = m.nrows();
    let mut v = DMatrix::from_element(n, 1, 1.0 / (n as f64).sqrt());
    let mut estimate = 0.0;
    for _ in 0..iterations {
        let next = m * &v;
        let norm = next.norm();
        if norm < f64::EPSILON {
            return 0.0;
        }
        estimate = norm;
        v = next / norm;
    }
    estimate
}

/// Rescales `m` in place so that its spectral radius equals `target`.
///
/// The spectral radius is homogeneous of degree one, so a single division
/// achieves the target exactly rather than approximately. A matrix whose
/// radius is already zero, which would be a degenerate fill, is left alone and
/// reported by the caller.
pub fn rescale_spectral_radius(m: &mut DMatrix<f64>, target: f64) -> f64 {
    let rho = spectral_radius(m);
    if rho > 0.0 {
        *m *= target / rho;
    }
    rho
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rotation(theta: f64) -> DMatrix<f64> {
        DMatrix::from_row_slice(2, 2, &[theta.cos(), -theta.sin(), theta.sin(), theta.cos()])
    }

    #[test]
    fn spectral_radius_matches_a_known_diagonal() {
        let m = DMatrix::from_diagonal(&nalgebra::DVector::from_row_slice(&[0.3, -0.7, 0.5]));
        assert!((spectral_radius(&m) - 0.7).abs() < 1e-12);
    }

    #[test]
    fn power_iteration_agrees_on_a_symmetric_matrix() {
        // Symmetric with a dominant real eigenvalue, the case power iteration
        // is valid for.
        let m = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let exact = spectral_radius(&m);
        let iterated = spectral_radius_power_iteration(&m, 500);
        assert!((exact - 3.0).abs() < 1e-12);
        assert!((iterated - exact).abs() < 1e-9);
    }

    #[test]
    fn power_iteration_is_correct_here_but_cannot_be_relied_on_generally() {
        // A rotation has a complex conjugate pair of equal magnitude, so power
        // iteration never converges to an eigenvector: the iterate keeps
        // rotating. It happens to return the right magnitude for a pure
        // rotation because the matrix is orthogonal and preserves the norm,
        // which is precisely the accident that makes this test worth writing:
        // the value is right for the wrong reason and does not generalise.
        let m = rotation(0.7);
        assert!((spectral_radius(&m) - 1.0).abs() < 1e-12);
        assert!((spectral_radius_power_iteration(&m, 500) - 1.0).abs() < 1e-9);

        // Scale one axis so the matrix is no longer orthogonal. The spectrum
        // is still a conjugate pair, and now the iterate's norm oscillates
        // instead of settling, so the estimate is wrong.
        let mut skewed = m;
        skewed[(0, 0)] *= 4.0;
        skewed[(1, 0)] *= 4.0;
        let exact = spectral_radius(&skewed);
        let iterated = spectral_radius_power_iteration(&skewed, 500);
        assert!(
            (iterated - exact).abs() > 1e-3,
            "power iteration returned {iterated} against an exact {exact}; if this \
             ever agrees, the justification for using the full spectrum needs revisiting"
        );
    }

    #[test]
    fn rescaling_hits_the_target_exactly() {
        let mut rng = crate::reference::ReferenceRng::from_seed(7);
        let mut m: DMatrix<f64> = DMatrix::from_fn(40, 40, |_, _| rng.next_range(-1.0, 1.0));
        rescale_spectral_radius(&mut m, 0.9);
        assert!((spectral_radius(&m) - 0.9).abs() < 1e-9);
    }

    #[test]
    fn a_contracting_reservoir_has_the_echo_state_property() {
        let mut rng = crate::reference::ReferenceRng::from_seed(11);
        let mut w_res: DMatrix<f64> = DMatrix::from_fn(50, 50, |_, _| rng.next_range(-1.0, 1.0));
        rescale_spectral_radius(&mut w_res, 0.8);
        let w_in: DMatrix<f64> = DMatrix::from_fn(50, 1, |_, _| rng.next_range(-1.0, 1.0));
        let esn = Esn { w_res, w_in, n: 50 };
        let input: Vec<f64> = (0..2000).map(|_| rng.next_range(-0.5, 0.5)).collect();
        assert!(check_echo_state_property(&esn, &input, 3).holds);
    }

    #[test]
    fn a_strongly_expanding_reservoir_does_not() {
        // The check must be able to fail, or it certifies nothing. A radius far
        // above one drives the units into saturation, where distinct states
        // persist instead of merging.
        let mut rng = crate::reference::ReferenceRng::from_seed(13);
        let mut w_res: DMatrix<f64> = DMatrix::from_fn(50, 50, |_, _| rng.next_range(-1.0, 1.0));
        rescale_spectral_radius(&mut w_res, 8.0);
        let w_in: DMatrix<f64> = DMatrix::from_fn(50, 1, |_, _| rng.next_range(-1.0, 1.0));
        let esn = Esn { w_res, w_in, n: 50 };
        let input: Vec<f64> = (0..2000).map(|_| rng.next_range(-0.5, 0.5)).collect();
        assert!(!check_echo_state_property(&esn, &input, 3).holds);
    }

    #[test]
    fn the_state_stays_inside_the_tanh_range() {
        let mut rng = crate::reference::ReferenceRng::from_seed(17);
        let mut w_res: DMatrix<f64> = DMatrix::from_fn(30, 30, |_, _| rng.next_range(-1.0, 1.0));
        rescale_spectral_radius(&mut w_res, 0.9);
        let w_in: DMatrix<f64> = DMatrix::from_fn(30, 1, |_, _| rng.next_range(-1.0, 1.0));
        let esn = Esn { w_res, w_in, n: 30 };
        let input: Vec<f64> = (0..500).map(|_| rng.next_range(-0.5, 0.5)).collect();
        let states = esn.collect_states(&input, 100);
        assert_eq!(states.nrows(), 400);
        assert_eq!(states.ncols(), 31);
        for j in 1..31 {
            for i in 0..400 {
                assert!(states[(i, j)].abs() <= 1.0);
            }
        }
        // The bias column is exactly one everywhere.
        for i in 0..400 {
            assert_eq!(states[(i, 0)], 1.0);
        }
    }
}
