// SPDX-License-Identifier: MIT
//! Phase 9: a predictive coding network whose precision weighting is modulated
//! by the validated generators.
//!
//! **Scope, stated before anything else.** This is the perception and learning
//! side only. It builds a hierarchy of value nodes and error nodes in which each
//! level predicts the one below, relaxes the value nodes to settle on an
//! explanation of the input, and then changes each synapse by the product of the
//! two activities it already connects. There is no action on the world, no
//! policy selection and no expected free energy: the full active inference
//! framework is not implemented here and nothing in this crate should be read as
//! implementing it. What is implemented is the tractable subset, and the name
//! should not be allowed to suggest more than that.
//!
//! REF: [Whittington and Bogacz, 2017] "An Approximation of the Error
//!      Backpropagation Algorithm in a Predictive Coding Network with Local
//!      Hebbian Synaptic Plasticity", Neural Computation 29(5), pp. 1229-1262,
//!      DOI: 10.1162/NECO_a_00949. The architecture and the claim that the
//!      local update approximates the backpropagation gradient, which Phase 9a
//!      checks rather than assumes.
//! REF: [Friston, 2013] "Life as we know it", Journal of The Royal Society
//!      Interface 10(86), 20130475, DOI: 10.1098/rsif.2013.0475. Cited as the
//!      motivation for treating precision as the principled place to inject a
//!      modulating signal, not as something implemented here.
//!
//! This is a third, separate line of investigation. It shares no code and no
//! concept with the Phase 7 pilot or the Phase 8 reservoir, and reuses only the
//! qualified generators and the Phase 1 dataset, both unmodified.

pub mod network;
pub mod precision;
pub mod run;
pub mod trajectory;
