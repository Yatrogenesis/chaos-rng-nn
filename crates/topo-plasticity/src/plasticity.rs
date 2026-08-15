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
    /// Floor below which the raw logistic is not allowed to fall, as a
    /// fraction of its maximum.
    ///
    /// Zero reproduces the Phase 10 formula exactly, which is what makes the
    /// two phases comparable rather than merely similar.
    pub floor: f64,
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
        floor: 0.0,
    },
    Gate {
        label: "sharp-midpoint",
        beta: 12.0,
        threshold: 0.5,
        floor: 0.0,
    },
    Gate {
        label: "gentle-early",
        beta: 6.0,
        threshold: 0.25,
        floor: 0.0,
    },
];

/// Floors swept in Phase 12, as a fraction of the gate's maximum.
///
/// Zero is Phase 10 exactly, and is included so the comparison has a reference
/// rather than being a fresh experiment beside an old one.
///
/// The second value is derived rather than picked. The hypothesis asks for a
/// setting where no layer falls below a tenth of the base rate, and the
/// sharpest gate is the binding case: its raw logistic at the shallowest layer
/// is 0.002473, so the floor `f` that puts the normalised multiplier exactly at
/// 0.10 solves `(f + (1-f)*0.002473) / (f + (1-f)*0.500000) = 0.10`, giving
/// `f = 0.050159`. A floor of 0.05 lands at 0.0997, just under, so 0.06 is used
/// and reaches 0.1176.
///
/// The two larger values flatten the gate progressively. They are there because
/// a floor high enough to erase the grading would defeat the purpose of grading,
/// and that failure mode should be visible in the sweep rather than argued
/// about.
pub const FLOORS: [f64; 4] = [0.0, 0.06, 0.15, 0.35];

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
    ///
    /// The floor is applied before the normalisation, so raising it does not
    /// raise the mean rate, it compresses the spread around it. That is the
    /// only way the comparison stays a comparison of grading rather than of
    /// step size, and it is why the averages-to-one test below covers the whole
    /// floor sweep and not just the floorless case.
    pub fn multipliers(&self, depth: usize) -> Vec<f64> {
        assert!(depth > 0, "a network with no weight matrices has no gate");
        if depth == 1 {
            return vec![1.0];
        }
        // alpha_i = alpha_min + (alpha_max - alpha_min) * logistic, with
        // alpha_max fixed at one because the scale is removed by the
        // normalisation below. Only the ratio between the floor and the
        // maximum survives it, which is the quantity the hypothesis is about.
        let raw: Vec<f64> = (0..depth)
            .map(|i| {
                let l = i as f64 / (depth - 1) as f64;
                let logistic = 1.0 / (1.0 + (-self.beta * (l - self.threshold)).exp());
                self.floor + (1.0 - self.floor) * logistic
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
        // The fairness of every comparison in Phase 10c and Phase 12 rests on
        // this, and the floor changes the family, so the constraint is
        // rechecked across the whole sweep rather than inherited from the
        // floorless case.
        for gate in SWEEP {
            for floor in FLOORS {
                let g = Gate { floor, ..gate };
                for depth in 2..=6 {
                    let m = g.multipliers(depth);
                    let mean = m.iter().sum::<f64>() / depth as f64;
                    assert!(
                        (mean - 1.0).abs() < 1e-12,
                        "{} with floor {floor} at depth {depth} averaged {mean}",
                        g.label
                    );
                }
            }
        }
    }

    #[test]
    fn a_zero_floor_reproduces_the_phase_ten_gate_exactly() {
        // The comparison against Phase 10 is only a comparison if the
        // floorless case is bit-for-bit what Phase 10 ran.
        for gate in SWEEP {
            assert_eq!(gate.floor, 0.0);
            let m = gate.multipliers(3);
            let raw: Vec<f64> = (0..3)
                .map(|i| {
                    let l = i as f64 / 2.0;
                    1.0 / (1.0 + (-gate.beta * (l - gate.threshold)).exp())
                })
                .collect();
            let mean = raw.iter().sum::<f64>() / 3.0;
            for (got, want) in m.iter().zip(raw.iter().map(|v| v / mean)) {
                assert_eq!(*got, want, "{}", gate.label);
            }
        }
    }

    #[test]
    fn a_higher_floor_compresses_the_spread_without_moving_the_mean() {
        // This is the mechanism the phase is testing, stated as a property.
        for gate in SWEEP {
            let spread = |f: f64| {
                let m = Gate { floor: f, ..gate }.multipliers(3);
                m.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    - m.iter().cloned().fold(f64::INFINITY, f64::min)
            };
            for pair in FLOORS.windows(2) {
                assert!(
                    spread(pair[1]) < spread(pair[0]),
                    "{}: floor {} did not compress against {}",
                    gate.label,
                    pair[1],
                    pair[0]
                );
            }
        }
    }

    #[test]
    fn the_intended_floor_keeps_every_layer_above_a_tenth_of_the_base_rate() {
        // The hypothesis is about not starving a layer, so the sweep has to
        // contain a setting where no layer is starved, and the sharpest gate is
        // the hardest case.
        let sharp = SWEEP.iter().find(|g| g.label == "sharp-midpoint").unwrap();
        let starved = sharp
            .multipliers(3)
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        assert!(
            starved < 0.01,
            "the floorless case should starve: {starved}"
        );
        let floored = Gate {
            floor: 0.06,
            ..*sharp
        }
        .multipliers(3)
        .into_iter()
        .fold(f64::INFINITY, f64::min);
        // The derivation in FLOORS puts the exact threshold at f = 0.050159,
        // so 0.06 clears it and 0.05 would not. That boundary is recorded here
        // rather than assumed.
        assert!(
            floored >= 0.10,
            "floor 0.06 left the shallowest layer at {floored}"
        );
        let under = Gate {
            floor: 0.05,
            ..*sharp
        }
        .multipliers(3)
        .into_iter()
        .fold(f64::INFINITY, f64::min);
        assert!(
            under < 0.10,
            "0.05 was expected to fall just short, got {under}"
        );
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
