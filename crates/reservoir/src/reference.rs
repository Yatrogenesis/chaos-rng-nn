// SPDX-License-Identifier: MIT
//! The reference i.i.d. source for the canonical echo state network.

use rand_chacha::rand_core::{Rng as _, SeedableRng as _};

/// The baseline generator: ChaCha12, the reference stream cipher generator of
/// the `rand` crate.
///
/// Two points about this choice matter for reading the results.
///
/// It is deliberately **not** the `rand` crate's own float conversion that is
/// used, but the same 53-bit construction the qualified generators use, so that
/// the mapping from bits to weights is byte-identical across all five
/// conditions. Only the bit source differs. Had the baseline used a different
/// conversion, any difference found would have had two possible causes instead
/// of one.
///
/// It is also **not** ChaCha8, even though the `rand` crate's own default is a
/// ChaCha variant, because ChaCha8 is itself one of the four conditions under
/// test. Using the same primitive for the baseline and for a treatment would
/// have made that comparison a test of nothing. As it stands the chacha8
/// condition and this baseline are near-identical by construction anyway, which
/// is a fact about the design worth stating rather than hiding: chacha8 serves
/// as a negative control on the measurement harness. If it were to differ
/// significantly from this baseline, that would indicate a defect in the
/// harness, not a discovery about generators.
///
/// REF: [Bernstein, 2008] "ChaCha, a variant of Salsa20", State of the Art of
///      Stream Ciphers workshop (SASC 2008)
///      <https://cr.yp.to/chacha/chacha-20080128.pdf>
#[derive(Debug, Clone)]
pub struct ReferenceRng {
    inner: rand_chacha::ChaCha12Rng,
}

impl ReferenceRng {
    /// Creates the generator from a 64-bit seed.
    pub fn from_seed(seed: u64) -> Self {
        Self {
            inner: rand_chacha::ChaCha12Rng::seed_from_u64(seed),
        }
    }

    /// Returns the next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// Returns the next variate uniform on `[0, 1)`.
    ///
    /// The identical construction to the qualified generators: the top 53 bits
    /// scaled, so every output is exactly representable and the spacing is
    /// uniform.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Returns the next variate uniform on `[low, high)`.
    pub fn next_range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic_from_the_seed() {
        let a: Vec<f64> = (0..64)
            .map({
                let mut r = ReferenceRng::from_seed(9);
                move |_| r.next_f64()
            })
            .collect();
        let b: Vec<f64> = (0..64)
            .map({
                let mut r = ReferenceRng::from_seed(9);
                move |_| r.next_f64()
            })
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn stays_in_the_unit_interval_and_covers_it() {
        let mut r = ReferenceRng::from_seed(4);
        let sample: Vec<f64> = (0..50_000).map(|_| r.next_f64()).collect();
        assert!(sample.iter().all(|v| (0.0..1.0).contains(v)));
        let mean = sample.iter().sum::<f64>() / sample.len() as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean was {mean}");
    }

    #[test]
    fn the_range_helper_respects_its_bounds() {
        let mut r = ReferenceRng::from_seed(6);
        for _ in 0..10_000 {
            let v = r.next_range(-0.5, 0.5);
            assert!((-0.5..0.5).contains(&v));
        }
    }
}
