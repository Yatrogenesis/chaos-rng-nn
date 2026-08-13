// SPDX-License-Identifier: MIT
//! Multilayer perceptron with explicit forward and backward passes.
//!
//! Written by hand rather than with a tensor framework for one reason: the
//! experiment must control every random draw. The three injection points named
//! in the protocol are weight initialisation, dropout masks and minibatch
//! order, and here each of them consumes from the generator under test and from
//! nothing else. A framework that draws its own randomness internally would
//! silently mix a third source into both conditions.
//!
//! The loops below index by position rather than iterating, because the weight
//! matrix is a flat slice addressed as (out, in) and the backward pass needs
//! both indices at once. Rewriting them as iterator chains would obscure the
//! arithmetic without changing it.
#![allow(
    clippy::needless_range_loop,
    reason = "flat matrices addressed by two indices read more clearly this way"
)]

use chaos_rng::Rng;
use sha2::{Digest, Sha256};

/// A fully connected layer.
#[derive(Debug, Clone)]
struct Layer {
    /// Weights, row-major, shape (out, in).
    w: Vec<f64>,
    /// Biases, length out.
    b: Vec<f64>,
    /// Input width.
    n_in: usize,
    /// Output width.
    n_out: usize,
    // Adam moments, same shapes as the parameters.
    m_w: Vec<f64>,
    v_w: Vec<f64>,
    m_b: Vec<f64>,
    v_b: Vec<f64>,
}

impl Layer {
    /// Creates a layer with He normal initialisation, drawing from `rng`.
    ///
    /// REF: [He, Zhang, Ren and Sun, 2015] "Delving Deep into Rectifiers:
    ///      Surpassing Human-Level Performance on ImageNet Classification",
    ///      IEEE International Conference on Computer Vision (ICCV)
    ///      DOI: 10.1109/ICCV.2015.123
    fn new(n_in: usize, n_out: usize, rng: &mut Rng) -> Self {
        let scale = (2.0 / n_in as f64).sqrt();
        let w = (0..n_in * n_out)
            .map(|_| rng.next_normal() * scale)
            .collect();
        Self {
            w,
            b: vec![0.0; n_out],
            n_in,
            n_out,
            m_w: vec![0.0; n_in * n_out],
            v_w: vec![0.0; n_in * n_out],
            m_b: vec![0.0; n_out],
            v_b: vec![0.0; n_out],
        }
    }

    /// Computes `w x + b`.
    fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut out = self.b.clone();
        for o in 0..self.n_out {
            let row = &self.w[o * self.n_in..(o + 1) * self.n_in];
            let mut acc = 0.0;
            for i in 0..self.n_in {
                acc += row[i] * x[i];
            }
            out[o] += acc;
        }
        out
    }
}

/// Hyperparameters shared by every run, so that the two conditions differ only
/// in their source of randomness.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Widths of the hidden layers.
    pub hidden: [usize; 2],
    /// Probability that a hidden unit is dropped during training.
    pub dropout: f64,
    /// Adam step size.
    pub learning_rate: f64,
    /// Observations per gradient step.
    pub batch_size: usize,
    /// Passes over the training set.
    pub epochs: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hidden: [32, 32],
            dropout: 0.1,
            learning_rate: 0.01,
            batch_size: 32,
            epochs: 60,
        }
    }
}

/// A trainable network.
#[derive(Debug, Clone)]
pub struct Mlp {
    layers: Vec<Layer>,
    cfg: Config,
    step: usize,
}

impl Mlp {
    /// Builds a network whose weights are drawn from `rng`.
    ///
    /// This is randomness injection point one of three.
    pub fn new(n_in: usize, n_out: usize, cfg: Config, rng: &mut Rng) -> Self {
        let mut layers = Vec::new();
        let mut prev = n_in;
        for &h in cfg.hidden.iter() {
            layers.push(Layer::new(prev, h, rng));
            prev = h;
        }
        layers.push(Layer::new(prev, n_out, cfg_rng_guard(rng)));
        Self {
            layers,
            cfg,
            step: 0,
        }
    }

    /// Forward pass returning the logits, and, when training, the intermediate
    /// activations and dropout masks needed by the backward pass.
    fn forward_train(&self, x: &[f64], masks: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut acts: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len());
        let mut cur = x.to_vec();
        for (li, layer) in self.layers.iter().enumerate() {
            acts.push(cur.clone());
            let mut z = layer.forward(&cur);
            if li + 1 < self.layers.len() {
                for v in z.iter_mut() {
                    *v = v.max(0.0); // ReLU
                }
                // Inverted dropout: scale at training time so inference needs no
                // adjustment.
                for (v, m) in z.iter_mut().zip(masks[li].iter()) {
                    *v *= m;
                }
            }
            cur = z;
        }
        (acts, cur)
    }

    /// Forward pass for evaluation, with dropout disabled.
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        let mut cur = x.to_vec();
        for (li, layer) in self.layers.iter().enumerate() {
            let mut z = layer.forward(&cur);
            if li + 1 < self.layers.len() {
                for v in z.iter_mut() {
                    *v = v.max(0.0);
                }
            }
            cur = z;
        }
        cur
    }

    /// One Adam update over a minibatch, returning the mean loss on it.
    ///
    /// Dropout masks are drawn here, which is randomness injection point two of
    /// three.
    ///
    /// REF: [Kingma and Ba, 2015] "Adam: A Method for Stochastic Optimization",
    ///      International Conference on Learning Representations
    ///      DOI: 10.48550/arXiv.1412.6980
    ///
    /// REF: [Srivastava, Hinton, Krizhevsky, Sutskever and Salakhutdinov, 2014]
    ///      "Dropout: A Simple Way to Prevent Neural Networks from
    ///      Overfitting", Journal of Machine Learning Research 15, pp. 1929-1958
    ///      https://jmlr.org/papers/v15/srivastava14a.html
    pub fn train_batch(&mut self, xs: &[[f64; 2]], ys: &[usize], rng: &mut Rng) -> f64 {
        let n_layers = self.layers.len();
        let mut grad_w: Vec<Vec<f64>> = self.layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
        let mut grad_b: Vec<Vec<f64>> = self.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();
        let mut total_loss = 0.0;

        for (x, &y) in xs.iter().zip(ys.iter()) {
            // Dropout masks for this observation.
            let masks: Vec<Vec<f64>> = (0..n_layers - 1)
                .map(|li| {
                    let keep = 1.0 - self.cfg.dropout;
                    (0..self.layers[li].n_out)
                        .map(|_| {
                            if self.cfg.dropout <= 0.0 {
                                1.0
                            } else if rng.next_f64() < keep {
                                1.0 / keep
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect();

            let (acts, logits) = self.forward_train(x, &masks);

            // Softmax cross-entropy.
            let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = logits.iter().map(|v| (v - max_logit).exp()).collect();
            let sum_exp: f64 = exps.iter().sum();
            let probs: Vec<f64> = exps.iter().map(|v| v / sum_exp).collect();
            total_loss += -(probs[y].max(1e-15)).ln();

            // Backward pass. delta starts as dL/dlogits.
            let mut delta: Vec<f64> = probs.clone();
            delta[y] -= 1.0;

            for li in (0..n_layers).rev() {
                let layer = &self.layers[li];
                let a_in = &acts[li];
                for o in 0..layer.n_out {
                    let d = delta[o];
                    grad_b[li][o] += d;
                    let base = o * layer.n_in;
                    for i in 0..layer.n_in {
                        grad_w[li][base + i] += d * a_in[i];
                    }
                }
                if li > 0 {
                    let mut new_delta = vec![0.0; layer.n_in];
                    for o in 0..layer.n_out {
                        let d = delta[o];
                        let base = o * layer.n_in;
                        for i in 0..layer.n_in {
                            new_delta[i] += d * layer.w[base + i];
                        }
                    }
                    // Through the dropout mask and the ReLU of layer li-1.
                    for (i, nd) in new_delta.iter_mut().enumerate() {
                        *nd *= masks[li - 1][i];
                        if acts[li][i] <= 0.0 {
                            *nd = 0.0;
                        }
                    }
                    delta = new_delta;
                }
            }
        }

        let n = xs.len() as f64;
        self.step += 1;
        let t = self.step as f64;
        const BETA1: f64 = 0.9;
        const BETA2: f64 = 0.999;
        const EPS: f64 = 1e-8;
        for li in 0..n_layers {
            let layer = &mut self.layers[li];
            for k in 0..layer.w.len() {
                let g = grad_w[li][k] / n;
                layer.m_w[k] = BETA1 * layer.m_w[k] + (1.0 - BETA1) * g;
                layer.v_w[k] = BETA2 * layer.v_w[k] + (1.0 - BETA2) * g * g;
                let m_hat = layer.m_w[k] / (1.0 - BETA1.powf(t));
                let v_hat = layer.v_w[k] / (1.0 - BETA2.powf(t));
                layer.w[k] -= self.cfg.learning_rate * m_hat / (v_hat.sqrt() + EPS);
            }
            for k in 0..layer.b.len() {
                let g = grad_b[li][k] / n;
                layer.m_b[k] = BETA1 * layer.m_b[k] + (1.0 - BETA1) * g;
                layer.v_b[k] = BETA2 * layer.v_b[k] + (1.0 - BETA2) * g * g;
                let m_hat = layer.m_b[k] / (1.0 - BETA1.powf(t));
                let v_hat = layer.v_b[k] / (1.0 - BETA2.powf(t));
                layer.b[k] -= self.cfg.learning_rate * m_hat / (v_hat.sqrt() + EPS);
            }
        }

        total_loss / n
    }

    /// Mean cross-entropy loss and accuracy over a dataset, without dropout.
    pub fn evaluate(&self, xs: &[[f64; 2]], ys: &[usize]) -> (f64, f64) {
        let mut loss = 0.0;
        let mut correct = 0usize;
        for (x, &y) in xs.iter().zip(ys.iter()) {
            let logits = self.predict(x);
            let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = logits.iter().map(|v| (v - max_logit).exp()).collect();
            let sum_exp: f64 = exps.iter().sum();
            let probs: Vec<f64> = exps.iter().map(|v| v / sum_exp).collect();
            loss += -(probs[y].max(1e-15)).ln();
            let pred = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits must not be NaN"))
                .map(|(i, _)| i)
                .expect("at least one class");
            if pred == y {
                correct += 1;
            }
        }
        let n = xs.len() as f64;
        (loss / n, correct as f64 / n)
    }

    /// The full parameter vector, flattened in a fixed order.
    ///
    /// Used by Phase 3, which treats the sequence of these vectors across
    /// epochs as a point cloud in parameter space.
    pub fn weight_vector(&self) -> Vec<f64> {
        let mut v = Vec::new();
        for layer in self.layers.iter() {
            v.extend_from_slice(&layer.w);
            v.extend_from_slice(&layer.b);
        }
        v
    }

    /// SHA-256 over every parameter, in a fixed order, as the fingerprint used
    /// to verify bit-for-bit reproducibility between two runs of one
    /// configuration.
    pub fn weight_hash(&self) -> String {
        let mut hasher = Sha256::new();
        for layer in self.layers.iter() {
            for v in layer.w.iter().chain(layer.b.iter()) {
                hasher.update(v.to_le_bytes());
            }
        }
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

/// Identity helper that makes the borrow of the generator explicit at the call
/// site of the output layer, keeping the draw order unambiguous.
fn cfg_rng_guard(rng: &mut Rng) -> &mut Rng {
    rng
}

#[cfg(test)]
mod tests {
    use super::*;
    use chaos_rng::RngKind;

    #[test]
    fn identical_seeds_give_identical_networks() {
        let cfg = Config::default();
        let mut r1 = Rng::new(RngKind::Lorenz, 5);
        let mut r2 = Rng::new(RngKind::Lorenz, 5);
        let a = Mlp::new(2, 2, cfg, &mut r1);
        let b = Mlp::new(2, 2, cfg, &mut r2);
        assert_eq!(a.weight_hash(), b.weight_hash());
    }

    #[test]
    fn different_seeds_give_different_networks() {
        let cfg = Config::default();
        let mut r1 = Rng::new(RngKind::Lorenz, 5);
        let mut r2 = Rng::new(RngKind::Lorenz, 6);
        let a = Mlp::new(2, 2, cfg, &mut r1);
        let b = Mlp::new(2, 2, cfg, &mut r2);
        assert_ne!(a.weight_hash(), b.weight_hash());
    }

    #[test]
    fn training_reduces_loss_on_a_separable_problem() {
        // A trivially separable problem: the network must be able to drive the
        // loss down, otherwise the backward pass is wrong.
        let cfg = Config {
            dropout: 0.0,
            epochs: 1,
            ..Config::default()
        };
        let mut rng = Rng::new(RngKind::ChaCha, 1);
        let mut net = Mlp::new(2, 2, cfg, &mut rng);
        let xs: Vec<[f64; 2]> = (0..64)
            .map(|i| if i % 2 == 0 { [-2.0, -2.0] } else { [2.0, 2.0] })
            .collect();
        let ys: Vec<usize> = (0..64).map(|i| i % 2).collect();
        let (before, _) = net.evaluate(&xs, &ys);
        for _ in 0..50 {
            net.train_batch(&xs, &ys, &mut rng);
        }
        let (after, acc) = net.evaluate(&xs, &ys);
        assert!(after < before, "loss did not fall: {before} then {after}");
        assert!(acc > 0.99, "accuracy only reached {acc}");
    }

    #[test]
    fn dropout_consumes_randomness_and_changes_the_trajectory() {
        // With dropout active the training path must depend on the generator;
        // if it did not, injection point two would be inert.
        let cfg = Config {
            epochs: 1,
            ..Config::default()
        };
        let xs: Vec<[f64; 2]> = (0..32)
            .map(|i| [i as f64 * 0.1, -(i as f64) * 0.1])
            .collect();
        let ys: Vec<usize> = (0..32).map(|i| i % 2).collect();

        let mut init = Rng::new(RngKind::ChaCha, 11);
        let base = Mlp::new(2, 2, cfg, &mut init);

        let mut a = base.clone();
        let mut ra = Rng::new(RngKind::ChaCha, 100);
        a.train_batch(&xs, &ys, &mut ra);

        let mut b = base.clone();
        let mut rb = Rng::new(RngKind::ChaCha, 200);
        b.train_batch(&xs, &ys, &mut rb);

        assert_ne!(
            a.weight_hash(),
            b.weight_hash(),
            "different dropout streams must give different weights"
        );
    }
}
