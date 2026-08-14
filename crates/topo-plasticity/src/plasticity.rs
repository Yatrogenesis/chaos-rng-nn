// SPDX-License-Identifier: MIT
//! Graded plasticity: a learning rate that varies with depth in the hierarchy.
//!
//! `alpha_i = alpha_max / (1 + exp(-beta (l_i - l_threshold)))`, with `l_i` the
//! normalised position of a layer.
//!
//! **Provenance.** The shape of this expression comes from a draft of the
//! author's, and is reused here as a design hypothesis of this project. That
//! draft's reported outcomes cannot be verified and none is cited or relied on;
//! `beta` and `l_threshold` are hyperparameters of this experiment, swept and
//! documented, not values inherited as anyone's optimum.
//!
//! The idea itself is older and has a verifiable origin, which is what is cited.
//!
//! REF: [Grossberg, 2013] "Adaptive Resonance Theory: How a brain learns to
//!      consciously attend, learn, and recognize a changing world", Neural
//!      Networks 37, pp. 1-47, DOI: 10.1016/j.neunet.2012.09.017. Title,
//!      journal, volume, pages and year checked against CrossRef. Adaptive
//!      resonance makes plasticity conditional rather than uniform, which is
//!      the substance the expression above is a particular parameterisation of.

use serde::Serialize;

/// One setting of the graded plasticity gate.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Gate {
    /// Name used in the report.
    pub label: &'static str,
    /// Steepness of the transition across the hierarchy.
    pub beta: f64,
    /// Normalised depth at which the gate is half open.
    pub threshold: f64,
}

/// The settings swept in this phase.
///
/// Three, spanning a gentle and a sharp transition and a shift of where the
/// transition sits. As with the topological weights, these are this
/// experiment's hyperparameters; nothing is carried over as an optimum.
pub const SWEEP: [Gate; 3] = [
    Gate {
        label: "gentle-midpoint",
        beta: 6.0,
        threshold: 0.5,
    },
    Gate {
        label: "sharp-midpoint",
        beta: 12.0,
        threshold: 0.5,
    },
    Gate {
        label: "gentle-early",
        beta: 6.0,
        threshold: 0.25,
    },
];

impl Gate {
    /// Multipliers on the learning rate, one per weight matrix.
    ///
    /// The raw logistic is computed at each layer's normalised depth and then
    /// **rescaled so the multipliers average exactly one**. That normalisation
    /// is not cosmetic and it is enforced by a test. Without it the gate would
    /// change the mean learning rate as well as its distribution across layers,
    /// and any difference measured against the baseline would be a difference
    /// of step size rather than of graded plasticity. It is the same trap the
    /// precision map in Phase 9 was built to avoid, in a new place.
    ///
    /// What survives the normalisation is the shape: deeper layers learn faster
    /// than shallow ones, or the reverse, at the same total rate.
    pub fn multipliers(&self, depth: usize) -> Vec<f64> {
        assert!(depth > 0, "a network with no weight matrices has no gate");
        if depth == 1 {
            return vec![1.0];
        }
        let raw: Vec<f64> = (0..depth)
            .map(|i| {
                let l = i as f64 / (depth - 1) as f64;
                1.0 / (1.0 + (-self.beta * (l - self.threshold)).exp())
            })
            .collect();
        let mean = raw.iter().sum::<f64>() / depth as f64;
        assert!(mean > 0.0, "the gate closed everywhere");
        raw.iter().map(|v| v / mean).collect()
    }
}

/// The uniform gate, which is what the baseline uses.
pub fn uniform(depth: usize) -> Vec<f64> {
    vec![1.0; depth]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_multipliers_average_exactly_one() {
        // The fairness of every comparison in Phase 10c rests on this.
        for gate in SWEEP {
            for depth in 2..=6 {
                let m = gate.multipliers(depth);
                let mean = m.iter().sum::<f64>() / depth as f64;
                assert!(
                    (mean - 1.0).abs() < 1e-12,
                    "{} at depth {depth} averaged {mean}",
                    gate.label
                );
            }
        }
    }

    #[test]
    fn the_gate_is_monotone_in_depth() {
        for gate in SWEEP {
            let m = gate.multipliers(4);
            for pair in m.windows(2) {
                assert!(
                    pair[1] > pair[0],
                    "{} was not increasing: {m:?}",
                    gate.label
                );
            }
        }
    }

    #[test]
    fn every_multiplier_stays_positive() {
        // A non-positive multiplier would reverse or freeze a layer's learning,
        // which is not what a graded rate means.
        for gate in SWEEP {
            for depth in 2..=8 {
                assert!(gate.multipliers(depth).iter().all(|v| *v > 0.0));
            }
        }
    }

    #[test]
    fn a_sharper_gate_spreads_the_multipliers_further() {
        let gentle = SWEEP[0].multipliers(4);
        let sharp = SWEEP[1].multipliers(4);
        let spread = |v: &[f64]| {
            v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - v.iter().cloned().fold(f64::INFINITY, f64::min)
        };
        assert!(spread(&sharp) > spread(&gentle));
    }

    #[test]
    fn shifting_the_threshold_moves_the_transition() {
        // The early threshold opens the gate sooner, so the shallowest layer
        // gets a larger share than it does at the midpoint.
        let midpoint = SWEEP[0].multipliers(4);
        let early = SWEEP[2].multipliers(4);
        assert!(early[0] > midpoint[0]);
    }

    #[test]
    fn the_uniform_gate_is_the_identity() {
        assert_eq!(uniform(3), vec![1.0, 1.0, 1.0]);
    }
}
