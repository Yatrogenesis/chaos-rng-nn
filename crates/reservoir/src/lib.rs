// SPDX-License-Identifier: MIT
//! Phase 8: reservoir computing with the validated generators as fixed
//! reservoir weights.
//!
//! **This is a change of paradigm, not a repetition.** Phases 1 through 7 asked
//! whether the source of randomness matters inside stochastic gradient descent,
//! and seven phases of converging evidence said no: no empirical effect in
//! phases 1, 3, 4c and 6, and, in phase 7, no formal channel through which one
//! of the four generators could even be described by the machinery being
//! measured. This phase does not ask that question an eighth time.
//!
//! It changes the learning paradigm. In reservoir computing the recurrent
//! dynamics are **never trained**. They are drawn once and frozen, and only a
//! linear readout is fitted, in closed form by ridge regression rather than by
//! gradient descent. Under backpropagation the initial weights are a starting
//! point that training moves away from, which is precisely why the source that
//! drew them had so little room to matter. Here the drawn weights are the
//! computation: whatever structure the generator's stream carries stays in the
//! matrix that transforms the state at every timestep, for the whole life of
//! the network. That gives the question a place to have an answer that it did
//! not have before.
//!
//! REF: [Jaeger, 2001] "The 'echo state' approach to analysing and training
//!      recurrent neural networks", GMD Report 148, German National Research
//!      Center for Information Technology. The echo state property and the
//!      architecture.
//! REF: [Maass, Natschlaeger and Markram, 2002] "Real-time computing without
//!      stable states: a new framework for neural computation based on
//!      perturbations", Neural Computation 14(11), pp. 2531-2560,
//!      DOI: 10.1162/089976602760407955. The independent formulation of the
//!      same idea as liquid state machines.
//! REF: [Jaeger and Haas, 2004] "Harnessing nonlinearity: predicting chaotic
//!      systems and saving energy in wireless communication", Science
//!      304(5667), pp. 78-80, DOI: 10.1126/science.1091277.
//!
//! This phase shares no code and no concept with Phase 7. It depends on nothing
//! outside this workspace.

pub mod esn;
pub mod fill;
pub mod reference;
pub mod ridge;
pub mod run;
pub mod tasks;
