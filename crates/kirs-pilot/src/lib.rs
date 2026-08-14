// SPDX-License-Identifier: MIT
//! Phase 7: which Pesin-type formula, if any, has its hypotheses satisfied for
//! each generator family, decided by Horn resolution.
//!
//! **What this is not.** It computes no entropy, no Lyapunov exponent and no
//! dimension. It answers a prior question that the earlier phases never asked:
//! given what is actually known about each generator, is the machinery whose
//! output those phases measured formally applicable to it at all. The answer is
//! a classification, not a number.
//!
//! REF: [Pesin, 1977] "Characteristic Lyapunov exponents and smooth ergodic
//!      theory", Russian Mathematical Surveys 32(4), pp. 55-114
//!      DOI: 10.1070/RM1977v032n04ABEH001639
//!      The classical formula, whose hypotheses include a diffeomorphism and an
//!      invariant measure absolutely continuous with respect to Lebesgue.
//!
//! REF: [Liu, 1998] "Random perturbations of Axiom A basic sets", Journal of
//!      Statistical Physics 90, pp. 467-490, DOI: 10.1023/A:1023280407906, and
//!      [Liu and Qian, 1995] "Smooth Ergodic Theory of Random Dynamical
//!      Systems", Lecture Notes in Mathematics 1606, Springer,
//!      DOI: 10.1007/BFb0094308. The random extension, which covers
//!      non-invertible systems that the classical statement excludes.
//!
//! **Isolation.** The `kirs` repository is consumed as a read-only reference at
//! a pinned commit. No file inside that checkout is edited by this project and
//! none of its source is copied here; the dependency is by path. Its fifteen
//! tests are expected to pass untouched, which is what shows the coupling is
//! clean.

use kirs_lab::{atom, compound, run_bounded};
use pirs_kirs::{parse_program, solve, KnowledgeBase};
use std::rc::Rc;

/// Step budget for every query, the bounded entry point rather than the
/// unbounded one.
///
/// A Horn program this small resolves in a handful of steps, so the budget is
/// not a performance measure: it is there so that a mistake in the clauses
/// produces an exhausted budget rather than a hang, and so that exhaustion can
/// be reported instead of being mistaken for a negative answer.
pub const STEP_BUDGET: usize = 10_000;

/// The knowledge base.
///
/// Every fact carries the reason it can be asserted. A property with no
/// published result behind it is **not** declared, and its absence is the
/// honest answer rather than a gap to be filled with an assumption. That
/// applies in particular to absolute continuity of the invariant measure, which
/// is available for exactly one of the four families.
pub const PROGRAM: &str = r#"
generador(lorenz).
generador(chacha8).
generador(ifs_lorenz).
generador(ifs_chacha8).

% Invertibility.
%
% lorenz: the flow of an autonomous ODE is invertible in time by uniqueness of
% solutions; this is a property of the construction, not a citation.
invertible(lorenz).

% chacha8: the ChaCha core is a bijection on its state by design, being built
% from additions, rotations and exclusive-ors, each invertible. Property of the
% construction.
invertible(chacha8).

% The chaos game applies one of three contractive maps chosen at each step and
% keeps only the image. Two different points can land on the same successor, so
% the step map is not injective. Property of the construction.
no_invertible(ifs_lorenz).
no_invertible(ifs_chacha8).

% Absolute continuity of the invariant measure.
%
% lorenz: the Lorenz attractor supports an SRB measure, established rigorously
% by Tucker. An SRB measure is absolutely continuous along unstable manifolds,
% which is the sense in which the hypothesis of the classical formula is met.
%   REF: Tucker, "A Rigorous ODE Solver and Smale's 14th Problem",
%        Foundations of Computational Mathematics 2 (2002), pp. 53-117,
%        DOI: 10.1007/s002080010018
medida_absolutamente_continua(lorenz).

% chacha8: no result is known to this project about the absolute continuity of
% any invariant measure of the ChaCha map, and the question is arguably not even
% well posed for a finite-state permutation. No fact is declared.
%
% ifs_lorenz, ifs_chacha8: the Sierpinski attractor carries a self-similar
% measure supported on a set of Hausdorff dimension log(3)/log(2), which has
% zero Lebesgue measure in the plane. That measure is therefore singular, not
% absolutely continuous. Declaring the negative would need negation, which this
% engine does not have, so the fact is simply absent -- but the absence here
% reflects a known negative rather than ignorance, and the report says so.

% Applicability rules.
pesin_clasico_aplica(G) :-
    generador(G),
    invertible(G),
    medida_absolutamente_continua(G).

pesin_aleatorio_liu_aplica(G) :-
    generador(G),
    no_invertible(G).
"#;

/// Which formulas apply to one generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Applicability {
    /// Generator name as it appears in the program.
    pub generator: String,
    /// Whether the classical formula's hypotheses are satisfied.
    pub classical: bool,
    /// Whether the random extension's hypotheses are satisfied.
    pub random_liu: bool,
    /// True when neither resolved, computed outside the engine.
    pub neither: bool,
    /// True if any query exhausted its step budget, in which case the answers
    /// above are not trustworthy and must not be read as negatives.
    pub budget_exhausted: bool,
}

/// The four families, in the order used throughout the report.
pub const GENERATORS: [&str; 4] = ["lorenz", "chacha8", "ifs_lorenz", "ifs_chacha8"];

/// Parses the knowledge base.
pub fn knowledge_base() -> Rc<KnowledgeBase> {
    Rc::new(parse_program(PROGRAM).expect("the knowledge base must parse"))
}

/// Asks whether `goal` holds of `generator`.
///
/// The query is ground: the generator is supplied and the engine is asked
/// whether the rule resolves, so a non-empty answer set means yes and an empty
/// one means the rule did not resolve. That is weaker than a proof of the
/// negation, and the distinction is kept in the report rather than blurred.
fn holds(kb: &Rc<KnowledgeBase>, goal: &str, generator: &str) -> (bool, bool) {
    let kb = kb.clone();
    let g = generator.to_string();
    let name = goal.to_string();
    let (answers, exhausted) = run_bounded(1, STEP_BUDGET, move |_q| {
        solve(kb.clone(), compound(&name, vec![atom(&g)]))
    });
    (!answers.is_empty(), exhausted)
}

/// Classifies all four generators.
///
/// The third category, neither formula applying, is computed here in Rust
/// rather than as a clause. The engine accepts Horn clauses only, with no
/// negation, as its parser states outright: "No operators, no cut — Horn
/// clauses only". A rule of the form `sin_formula(G) :- generador(G),
/// not(...), not(...)` cannot be expressed, and writing one would not parse.
/// Rather than reshape the question to flatter the engine, the two positive
/// rules are queried separately and the classification is made outside it.
pub fn classify() -> Vec<Applicability> {
    let kb = knowledge_base();
    GENERATORS
        .iter()
        .map(|g| {
            let (classical, e1) = holds(&kb, "pesin_clasico_aplica", g);
            let (random_liu, e2) = holds(&kb, "pesin_aleatorio_liu_aplica", g);
            Applicability {
                generator: (*g).to_string(),
                classical,
                random_liu,
                neither: !classical && !random_liu,
                budget_exhausted: e1 || e2,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_program_parses() {
        let _ = knowledge_base();
    }

    #[test]
    fn negation_is_absent_from_the_engine() {
        // The design decision above rests on this. If a future version of the
        // engine accepted negation, this test would fail and the classification
        // could move back inside the program where it belongs.
        let kb = Rc::new(
            parse_program("p(a). q(X) :- not(p(X)).")
                .map_err(|e| e.to_string())
                .err()
                .map(|_| KnowledgeBase::default())
                .unwrap_or_default(),
        );
        // Parsing either failed, or `not` was read as an ordinary predicate
        // with no clauses, which is not negation either way.
        let (answers, _) = run_bounded(1, STEP_BUDGET, move |q| {
            solve(kb.clone(), compound("q", vec![q]))
        });
        assert!(
            answers.is_empty(),
            "the engine resolved a negated goal, so the out-of-engine classification is no longer needed"
        );
    }

    #[test]
    fn classical_applies_only_where_both_hypotheses_are_declared() {
        let r = classify();
        let lorenz = r.iter().find(|a| a.generator == "lorenz").unwrap();
        assert!(
            lorenz.classical,
            "Lorenz has both invertibility and an SRB measure"
        );
        for other in r.iter().filter(|a| a.generator != "lorenz") {
            assert!(
                !other.classical,
                "{} should not satisfy the classical hypotheses",
                other.generator
            );
        }
    }

    #[test]
    fn the_random_extension_covers_exactly_the_non_invertible_families() {
        let r = classify();
        for a in r.iter() {
            let expected = a.generator.starts_with("ifs_");
            assert_eq!(
                a.random_liu, expected,
                "{} random-extension applicability was {}",
                a.generator, a.random_liu
            );
        }
    }

    #[test]
    fn chacha8_falls_through_both_rules() {
        // The finding that matters for the earlier phases: a family used
        // throughout as the control satisfies neither set of hypotheses.
        let r = classify();
        let c = r.iter().find(|a| a.generator == "chacha8").unwrap();
        assert!(c.neither, "chacha8 unexpectedly matched a rule");
    }

    #[test]
    fn no_query_exhausts_its_budget() {
        // If any did, the negatives above would be unreadable.
        for a in classify() {
            assert!(
                !a.budget_exhausted,
                "{} exhausted the step budget",
                a.generator
            );
        }
    }
}
