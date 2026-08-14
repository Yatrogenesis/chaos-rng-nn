// SPDX-License-Identifier: MIT
//! Phase 10: topological resilience and graded plasticity as design hypotheses.
//!
//! **Where these two ideas come from, stated before anything else.** Both are
//! taken from drafts of the author's. What is reused is the mathematical shape
//! of two expressions and nothing else. Those drafts also contain experimental
//! results that cannot be verified and that carry the marks of fabrication, so
//! no figure from them appears anywhere in this crate or in the report, and no
//! claim here rests on anything reported there. The weights and thresholds are
//! hyperparameters of this experiment, swept and documented, not values
//! inherited as anyone's optimum.
//!
//! The two expressions are therefore treated as hypotheses this project is
//! testing for the first time, with the burden of evidence that implies.
//!
//! REF: [Grossberg, 2013] "Adaptive Resonance Theory: How a brain learns to
//!      consciously attend, learn, and recognize a changing world", Neural
//!      Networks 37, pp. 1-47, DOI: 10.1016/j.neunet.2012.09.017. The real and
//!      verifiable origin of graded plasticity as an idea: a learning rate that
//!      depends on how well the current input matches what the system already
//!      represents. Title, journal, volume, pages and year were checked against
//!      CrossRef.

pub mod plasticity;
pub mod resilience;
pub mod run;
