// SPDX-License-Identifier: MIT
//! The predictive coding network: value nodes, error nodes, an inference loop
//! that relaxes the value nodes, and a purely local weight update.

use crate::precision::Precision;

/// Hyperbolic tangent, the nonlinearity used throughout.
///
/// Chosen over a rectifier because the inference loop needs `f'` at the current
/// value of every node, and a rectifier's derivative is zero over half its
/// domain: a value node that drifts negative would stop receiving the top-down
/// term entirely and the relaxation would stall there. `tanh` is smooth,
/// bounded and has a derivative that never vanishes at finite argument.
fn f(v: f64) -> f64 {
    v.tanh()
}

/// Derivative of [`f`], written in terms of the activation to avoid recomputing
/// the tangent.
fn f_prime(v: f64) -> f64 {
    let t = v.tanh();
    1.0 - t * t
}

/// A dense weight matrix, `rows` by `cols`, stored row-major.
#[derive(Debug, Clone)]
pub struct Matrix {
    /// Row count.
    pub rows: usize,
    /// Column count.
    pub cols: usize,
    /// Entries, row-major.
    pub data: Vec<f64>,
}

impl Matrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// `y = M x`, with `y` overwritten.
    fn multiply_into(&self, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(y.len(), self.rows);
        for (r, out) in y.iter_mut().enumerate() {
            let row = &self.data[r * self.cols..(r + 1) * self.cols];
            *out = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        }
    }

    /// `y = M^T x`, with `y` overwritten.
    fn transpose_multiply_into(&self, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), self.rows);
        debug_assert_eq!(y.len(), self.cols);
        y.fill(0.0);
        for (r, &scale) in x.iter().enumerate() {
            if scale == 0.0 {
                continue;
            }
            let row = &self.data[r * self.cols..(r + 1) * self.cols];
            for (c, w) in row.iter().enumerate() {
                y[c] += scale * w;
            }
        }
    }
}

/// Everything held fixed across conditions.
#[derive(Debug, Clone)]
pub struct Config {
    /// Node counts, from the input layer to the output layer.
    pub layers: Vec<usize>,
    /// Relaxation steps of the inference loop, per presented sample.
    pub inference_steps: usize,
    /// Step size of the inference loop, separate from the weight step size.
    pub inference_rate: f64,
    /// Step size of the weight update.
    pub learning_rate: f64,
    /// Passes over the training set.
    pub epochs: usize,
    /// Samples per weight update.
    pub batch_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // The same widths as the Phase 1 network, so the architecture is
            // not a hidden difference when the two are set side by side.
            layers: vec![2, 32, 32, 2],
            inference_steps: 16,
            inference_rate: 0.2,
            learning_rate: 0.05,
            epochs: 30,
            batch_size: 32,
        }
    }
}

/// A predictive coding network.
///
/// The state is the weights and biases. Value and error nodes are transient,
/// recreated for each presented sample, because they are the network's inference
/// about that sample rather than anything it retains.
#[derive(Debug, Clone)]
pub struct Network {
    /// `w[l]` maps layer `l` to layer `l + 1`.
    pub w: Vec<Matrix>,
    /// `b[l]` is the bias of the prediction of layer `l + 1`.
    pub b: Vec<Vec<f64>>,
    cfg: Config,
}

/// The transient state of one inference: value nodes, error nodes and the
/// activations feeding each prediction.
#[derive(Debug, Clone)]
pub struct Inference {
    /// Value nodes, `x[0]` being the clamped input.
    pub x: Vec<Vec<f64>>,
    /// Error nodes. `e[0]` is unused and stays empty; `e[l]` for `l >= 1` is
    /// `x[l] - mu[l]`.
    pub e: Vec<Vec<f64>>,
    /// Predictions.
    pub mu: Vec<Vec<f64>>,
}

impl Network {
    /// Builds a network with weights drawn from `rng`.
    ///
    /// Initialisation is deliberately **not** a place where the conditions
    /// differ. Every condition receives the same weights for the same run seed,
    /// drawn from the same reference stream, so that the precision schedule is
    /// the only thing under comparison. Phase 1 already established that the
    /// source of the initial weights makes no difference, and repeating that
    /// here would confound the question this phase asks.
    ///
    /// The scale is `1/sqrt(fan_in)`, which keeps the pre-activations of a
    /// `tanh` layer away from saturation at the start.
    pub fn new(cfg: Config, rng: &mut chaos_rng::ChaChaRng) -> Self {
        let mut w = Vec::new();
        let mut b = Vec::new();
        for l in 0..cfg.layers.len() - 1 {
            let (fan_in, fan_out) = (cfg.layers[l], cfg.layers[l + 1]);
            let scale = 1.0 / (fan_in as f64).sqrt();
            let mut m = Matrix::zeros(fan_out, fan_in);
            for v in m.data.iter_mut() {
                *v = rng.next_normal() * scale;
            }
            w.push(m);
            b.push(vec![0.0; fan_out]);
        }
        Self { w, b, cfg }
    }

    /// Number of weight matrices.
    pub fn depth(&self) -> usize {
        self.w.len()
    }

    /// Allocates the transient state for one sample.
    pub fn new_inference(&self) -> Inference {
        let x: Vec<Vec<f64>> = self.cfg.layers.iter().map(|n| vec![0.0; *n]).collect();
        let e = x.clone();
        let mu = x.clone();
        Inference { x, e, mu }
    }

    /// Runs the network forward, leaving every value node at its prediction.
    ///
    /// This is the state of zero prediction error, and it is where the
    /// inference loop starts: before the target is clamped there is nothing to
    /// explain, and the errors are exactly zero.
    pub fn feedforward(&self, input: &[f64], inf: &mut Inference) {
        inf.x[0].copy_from_slice(input);
        for l in 0..self.depth() {
            let activated: Vec<f64> = inf.x[l].iter().map(|v| f(*v)).collect();
            self.w[l].multiply_into(&activated, &mut inf.mu[l + 1]);
            for (m, bias) in inf.mu[l + 1].iter_mut().zip(self.b[l].iter()) {
                *m += bias;
            }
            inf.x[l + 1].copy_from_slice(&inf.mu[l + 1]);
            inf.e[l + 1].fill(0.0);
        }
    }

    /// Recomputes predictions and errors from the current value nodes.
    fn refresh_errors(&self, inf: &mut Inference) {
        for l in 0..self.depth() {
            let activated: Vec<f64> = inf.x[l].iter().map(|v| f(*v)).collect();
            self.w[l].multiply_into(&activated, &mut inf.mu[l + 1]);
            for (m, bias) in inf.mu[l + 1].iter_mut().zip(self.b[l].iter()) {
                *m += bias;
            }
            for i in 0..inf.x[l + 1].len() {
                inf.e[l + 1][i] = inf.x[l + 1][i] - inf.mu[l + 1][i];
            }
        }
    }

    /// Relaxes the value nodes with the input and the target both clamped.
    ///
    /// The update of an interior value node is
    /// `dx[l] = -pi[l] e[l] + f'(x[l]) * (W[l]^T (pi[l+1] e[l+1]))`,
    /// gradient descent on the precision-weighted sum of squared prediction
    /// errors. Every term is available at the node itself or at the synapses
    /// touching it: nothing is transported from the far end of the network,
    /// which is the property that distinguishes this from backpropagation.
    ///
    /// The precision is redrawn at **every** step of this loop rather than once
    /// per sample or once per epoch, because the weighting is what the theory
    /// says modulates the inference dynamics, and holding it fixed while the
    /// dynamics run would test something else.
    /// Returns the precision in force at the end of the relaxation, which is
    /// the one the weight update is entitled to use: the update is a step at
    /// the point the inference settled on, not at some earlier point of the
    /// trajectory.
    pub fn infer(
        &self,
        target: &[f64],
        inf: &mut Inference,
        precision: &mut dyn Precision,
    ) -> Vec<f64> {
        let depth = self.depth();
        inf.x[depth].copy_from_slice(target);
        let mut top_down = vec![0.0; self.cfg.layers.iter().copied().max().unwrap_or(0)];
        let mut last = vec![1.0; depth];

        for _ in 0..self.cfg.inference_steps {
            self.refresh_errors(inf);
            let pi = precision.next(depth);
            last.copy_from_slice(pi);

            for l in 1..depth {
                let weighted_above: Vec<f64> = inf.e[l + 1].iter().map(|v| pi[l] * v).collect();
                let slice = &mut top_down[..self.cfg.layers[l]];
                self.w[l].transpose_multiply_into(&weighted_above, slice);
                for (i, &top) in slice.iter().enumerate() {
                    let dx = -pi[l - 1] * inf.e[l][i] + f_prime(inf.x[l][i]) * top;
                    inf.x[l][i] += self.cfg.inference_rate * dx;
                }
            }
        }
        self.refresh_errors(inf);
        last
    }

    /// Accumulates the local weight update for the current inference.
    ///
    /// `dW[l] = pi[l+1] e[l+1] f(x[l])^T`, the product of the activity of the
    /// error node above and the activity of the value node below. It is Hebbian
    /// in the strict sense: each synapse changes by the product of the two
    /// quantities it already connects.
    pub fn accumulate(&self, inf: &Inference, pi: &[f64], dw: &mut [Matrix], db: &mut [Vec<f64>]) {
        for l in 0..self.depth() {
            let activated: Vec<f64> = inf.x[l].iter().map(|v| f(*v)).collect();
            let err = &inf.e[l + 1];
            let scale = pi[l];
            for r in 0..dw[l].rows {
                let contribution = scale * err[r];
                if contribution == 0.0 {
                    continue;
                }
                let row = &mut dw[l].data[r * dw[l].cols..(r + 1) * dw[l].cols];
                for (c, a) in activated.iter().enumerate() {
                    row[c] += contribution * a;
                }
                db[l][r] += contribution;
            }
        }
    }

    /// Applies an accumulated update, averaged over `count` samples.
    pub fn apply(&mut self, dw: &[Matrix], db: &[Vec<f64>], count: usize) {
        let step = self.cfg.learning_rate / count as f64;
        for l in 0..self.depth() {
            for (v, d) in self.w[l].data.iter_mut().zip(dw[l].data.iter()) {
                *v += step * d;
            }
            for (v, d) in self.b[l].iter_mut().zip(db[l].iter()) {
                *v += step * d;
            }
        }
    }

    /// Applies an accumulated update with a per-layer multiplier on the step.
    ///
    /// [`Self::apply`] is left exactly as it was rather than delegating here,
    /// so that Phase 9's published numbers cannot move by so much as a bit
    /// through a refactor of a function they depend on.
    pub fn apply_scaled(&mut self, dw: &[Matrix], db: &[Vec<f64>], count: usize, gate: &[f64]) {
        assert_eq!(gate.len(), self.depth(), "one multiplier per weight matrix");
        let base = self.cfg.learning_rate / count as f64;
        for l in 0..self.depth() {
            let step = base * gate[l];
            for (v, d) in self.w[l].data.iter_mut().zip(dw[l].data.iter()) {
                *v += step * d;
            }
            for (v, d) in self.b[l].iter_mut().zip(db[l].iter()) {
                *v += step * d;
            }
        }
    }

    /// The configuration this network was built with.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Value-node activations of one layer over a set of inputs.
    ///
    /// Rows are nodes, columns are inputs. This is the quantity Phase 10 takes
    /// the correlation structure of, and it is produced by a plain feedforward
    /// pass so that probing never disturbs the training state.
    pub fn layer_activations(&self, inputs: &[[f64; 2]], layer: usize) -> Vec<Vec<f64>> {
        let mut inf = self.new_inference();
        let mut out = vec![Vec::with_capacity(inputs.len()); self.cfg.layers[layer]];
        for input in inputs {
            self.feedforward(input, &mut inf);
            for (node, row) in out.iter_mut().enumerate() {
                row.push(inf.x[layer][node]);
            }
        }
        out
    }

    /// Zeroed accumulators shaped like the parameters.
    pub fn zero_accumulators(&self) -> (Vec<Matrix>, Vec<Vec<f64>>) {
        let dw = self
            .w
            .iter()
            .map(|m| Matrix::zeros(m.rows, m.cols))
            .collect();
        let db = self.b.iter().map(|v| vec![0.0; v.len()]).collect();
        (dw, db)
    }

    /// The network's output for one input, by a pure feedforward pass.
    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        let mut inf = self.new_inference();
        self.feedforward(input, &mut inf);
        inf.x[self.depth()].clone()
    }

    /// Cross-entropy loss and accuracy over a labelled set.
    ///
    /// The network is trained on squared prediction error, which is what the
    /// predictive coding formulation minimises, but it is *evaluated* with the
    /// softmax cross-entropy used by Phase 1, so that the reported numbers are
    /// the same quantity measured on the same data. The training objectives
    /// differ, and the report says so; the metric does not.
    pub fn evaluate(&self, xs: &[[f64; 2]], ys: &[usize]) -> (f64, f64) {
        let mut loss = 0.0;
        let mut correct = 0usize;
        for (x, &y) in xs.iter().zip(ys.iter()) {
            let out = self.predict(x);
            let max = out.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = out.iter().map(|v| (v - max).exp()).collect();
            let total: f64 = exps.iter().sum();
            let probs: Vec<f64> = exps.iter().map(|v| v / total).collect();
            loss += -(probs[y].max(1e-15)).ln();
            let predicted = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("outputs must not be NaN"))
                .map(|(i, _)| i)
                .expect("at least one class");
            if predicted == y {
                correct += 1;
            }
        }
        let n = xs.len() as f64;
        (loss / n, correct as f64 / n)
    }

    /// The exact backpropagation gradient of the squared output error.
    ///
    /// Present only so that Phase 9a can check the predictive coding update
    /// against it. It is never used to train anything: no gradient computed
    /// here is ever applied to a weight.
    pub fn backprop_gradient(&self, input: &[f64], target: &[f64]) -> (Vec<Matrix>, Vec<Vec<f64>>) {
        let depth = self.depth();
        let mut inf = self.new_inference();
        self.feedforward(input, &mut inf);

        let (mut gw, mut gb) = self.zero_accumulators();
        // dL/dmu at the output, for L = 1/2 ||target - mu||^2.
        let mut delta: Vec<f64> = (0..target.len())
            .map(|i| inf.mu[depth][i] - target[i])
            .collect();

        for l in (0..depth).rev() {
            let activated: Vec<f64> = inf.x[l].iter().map(|v| f(*v)).collect();
            let cols = gw[l].cols;
            for r in 0..gw[l].rows {
                let d = delta[r];
                let row = &mut gw[l].data[r * cols..(r + 1) * cols];
                for (c, a) in activated.iter().enumerate() {
                    row[c] = d * a;
                }
                gb[l][r] = d;
            }
            if l > 0 {
                let mut below = vec![0.0; self.w[l].cols];
                self.w[l].transpose_multiply_into(&delta, &mut below);
                for (i, v) in below.iter_mut().enumerate() {
                    *v *= f_prime(inf.x[l][i]);
                }
                delta = below;
            }
        }
        (gw, gb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::Constant;

    fn small() -> (Network, Config) {
        let cfg = Config {
            layers: vec![2, 5, 4, 2],
            inference_steps: 200,
            inference_rate: 0.1,
            learning_rate: 0.05,
            epochs: 1,
            batch_size: 1,
        };
        let mut rng = chaos_rng::ChaChaRng::from_seed(11);
        (Network::new(cfg.clone(), &mut rng), cfg)
    }

    #[test]
    fn the_feedforward_pass_leaves_no_prediction_error() {
        let (net, _) = small();
        let mut inf = net.new_inference();
        net.feedforward(&[0.3, -0.7], &mut inf);
        for l in 1..=net.depth() {
            for v in inf.e[l].iter() {
                assert_eq!(*v, 0.0);
            }
        }
    }

    #[test]
    fn clamping_the_target_creates_error_only_at_the_top() {
        let (net, _) = small();
        let mut inf = net.new_inference();
        net.feedforward(&[0.3, -0.7], &mut inf);
        let depth = net.depth();
        inf.x[depth].copy_from_slice(&[1.0, 0.0]);
        net.refresh_errors(&mut inf);
        for l in 1..depth {
            assert!(inf.e[l].iter().all(|v| v.abs() < 1e-15), "layer {l}");
        }
        assert!(inf.e[depth].iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn inference_reduces_the_total_prediction_error() {
        // Relaxation is gradient descent on that quantity, so it must not go up.
        let (net, _) = small();
        let mut inf = net.new_inference();
        net.feedforward(&[0.3, -0.7], &mut inf);
        let depth = net.depth();
        inf.x[depth].copy_from_slice(&[1.0, 0.0]);
        net.refresh_errors(&mut inf);
        let before: f64 = (1..=depth)
            .map(|l| inf.e[l].iter().map(|v| v * v).sum::<f64>())
            .sum();
        let _ = net.infer(&[1.0, 0.0], &mut inf, &mut Constant);
        let after: f64 = (1..=depth)
            .map(|l| inf.e[l].iter().map(|v| v * v).sum::<f64>())
            .sum();
        assert!(after < before, "error rose from {before} to {after}");
    }

    #[test]
    fn the_backprop_gradient_matches_finite_differences() {
        // The reference the Phase 9a gate compares against must itself be
        // right, or the gate certifies nothing.
        let (net, _) = small();
        let input = [0.4, -0.2];
        let target = [1.0, 0.0];
        let (gw, _) = net.backprop_gradient(&input, &target);

        let loss = |n: &Network| -> f64 {
            let out = n.predict(&input);
            0.5 * out
                .iter()
                .zip(target.iter())
                .map(|(o, t)| (o - t) * (o - t))
                .sum::<f64>()
        };

        let h = 1e-6;
        for (l, reference) in gw.iter().enumerate() {
            for idx in [0usize, 3, 7] {
                if idx >= net.w[l].data.len() {
                    continue;
                }
                let mut up = net.clone();
                up.w[l].data[idx] += h;
                let mut down = net.clone();
                down.w[l].data[idx] -= h;
                let numeric = (loss(&up) - loss(&down)) / (2.0 * h);
                let analytic = reference.data[idx];
                assert!(
                    (numeric - analytic).abs() < 1e-6,
                    "layer {l} index {idx}: finite difference {numeric}, analytic {analytic}"
                );
            }
        }
    }

    #[test]
    fn the_transpose_product_agrees_with_the_plain_one() {
        let m = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let x = [1.0, -1.0];
        let mut y = vec![0.0; 3];
        m.transpose_multiply_into(&x, &mut y);
        assert_eq!(y, vec![1.0 - 4.0, 2.0 - 5.0, 3.0 - 6.0]);
    }
}
