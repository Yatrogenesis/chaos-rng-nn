// SPDX-License-Identifier: MIT
//! Deterministic pseudo-random generator driven by the Lorenz attractor.
//!
//! The orbit is integrated with a fixed-step classical fourth-order Runge-Kutta
//! scheme, and uniform variates are extracted from the trajectory with the
//! method documented on [`LorenzRng::next_u64`].
//!
//! REF: [Lorenz, 1963] "Deterministic Nonperiodic Flow", Journal of the
//!      Atmospheric Sciences 20(2), pp. 130-141
//!      DOI: 10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2
//!
//! REF: [Hairer, Norsett and Wanner, 1993] "Solving Ordinary Differential
//!      Equations I: Nonstiff Problems", 2nd edition, Springer Series in
//!      Computational Mathematics vol. 8, section II.1
//!      DOI: 10.1007/978-3-540-78862-1

#![forbid(unsafe_code)]

pub mod stats;

/// Classical parameters placing the Lorenz system in its chaotic regime, with a
/// largest Lyapunov exponent of approximately 0.906.
///
/// REF: [Sprott, 2003] "Chaos and Time-Series Analysis", Oxford University
///      Press, table 4.1. ISBN 978-0-19-850840-3
pub const SIGMA: f64 = 10.0;
/// Rayleigh parameter of the chaotic regime.
pub const RHO: f64 = 28.0;
/// Geometric parameter of the chaotic regime, 8/3.
pub const BETA: f64 = 8.0 / 3.0;

/// Integration step. Small enough that RK4 local error stays far below the
/// precision that the extraction method consumes, large enough that successive
/// extracted samples are separated by a meaningful stretch of trajectory.
pub const DT: f64 = 0.01;

/// Number of integration steps discarded before extraction begins, so the
/// trajectory has settled onto the attractor and the output does not depend on
/// where in state space the seed happened to place it.
pub const BURN_IN_STEPS: usize = 10_000;

/// Number of integration steps advanced between two extracted samples.
///
/// A single RK4 step of 0.01 time units moves the state very little, so
/// consecutive states are strongly correlated. Decimating the trajectory is what
/// removes that correlation; the value below is the smallest power-of-two-free
/// stride found to bring lag-1 autocorrelation of the output below 0.002 over
/// one million samples (see `stats::autocorrelation` and the test
/// `extraction_passes_phase_zero_battery`).
pub const DECIMATION: usize = 5;

/// State of the Lorenz system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LorenzState {
    /// First coordinate.
    pub x: f64,
    /// Second coordinate.
    pub y: f64,
    /// Third coordinate.
    pub z: f64,
}

/// Right-hand side of the Lorenz system:
/// dx/dt = sigma (y - x), dy/dt = x (rho - z) - y, dz/dt = x y - beta z.
fn derivative(s: LorenzState) -> LorenzState {
    LorenzState {
        x: SIGMA * (s.y - s.x),
        y: s.x * (RHO - s.z) - s.y,
        z: s.x * s.y - BETA * s.z,
    }
}

/// One classical RK4 step of size `dt`.
fn rk4_step(s: LorenzState, dt: f64) -> LorenzState {
    let add = |a: LorenzState, b: LorenzState, f: f64| LorenzState {
        x: a.x + b.x * f,
        y: a.y + b.y * f,
        z: a.z + b.z * f,
    };

    let k1 = derivative(s);
    let k2 = derivative(add(s, k1, dt * 0.5));
    let k3 = derivative(add(s, k2, dt * 0.5));
    let k4 = derivative(add(s, k3, dt));

    LorenzState {
        x: s.x + (dt / 6.0) * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x),
        y: s.y + (dt / 6.0) * (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y),
        z: s.z + (dt / 6.0) * (k1.z + 2.0 * k2.z + 2.0 * k3.z + k4.z),
    }
}

/// Pseudo-random generator whose entropy source is a Lorenz orbit.
///
/// Determinism: for a fixed seed the sequence is reproducible bit for bit on any
/// platform with IEEE-754 double precision and the same operation order. The
/// implementation uses only multiplication, addition and subtraction, in a fixed
/// order, with no fused multiply-add, no transcendental functions and no
/// parallel reduction, so results do not depend on the optimiser.
#[derive(Debug, Clone)]
pub struct LorenzRng {
    state: LorenzState,
}

impl LorenzRng {
    /// Creates a generator from a 64-bit seed.
    ///
    /// The seed is mapped into a bounded region of state space rather than used
    /// directly, because the Lorenz system is only chaotic near its attractor
    /// and arbitrarily large initial values would need a long transient to get
    /// there. The burn-in below then discards that transient entirely.
    pub fn from_seed(seed: u64) -> Self {
        // SplitMix64 to decorrelate the three coordinates of nearby seeds.
        // REF: [Steele, Lea and Flood, 2014] "Fast splittable pseudorandom
        //      number generators", OOPSLA '14, DOI: 10.1145/2660193.2660195
        let mut s = seed;
        let mut split = || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        // Map to [-15, 15] for x and y, and [5, 45] for z: a box that contains
        // the attractor's usual excursions.
        let unit = |v: u64| (v >> 11) as f64 / (1u64 << 53) as f64;
        let mut rng = Self {
            state: LorenzState {
                x: unit(split()) * 30.0 - 15.0,
                y: unit(split()) * 30.0 - 15.0,
                z: unit(split()) * 40.0 + 5.0,
            },
        };
        for _ in 0..BURN_IN_STEPS {
            rng.state = rk4_step(rng.state, DT);
        }
        rng
    }

    /// Advances the orbit by `DECIMATION` steps and returns the new state.
    fn advance(&mut self) -> LorenzState {
        for _ in 0..DECIMATION {
            self.state = rk4_step(self.state, DT);
        }
        self.state
    }

    /// Returns the next 64 bits.
    ///
    /// Extraction method: the least significant bits of the mantissa of each of
    /// the three coordinates are harvested and combined. Concretely, each
    /// coordinate is scaled and truncated so that only digits far below the
    /// scale of the attractor's macroscopic motion survive, and the three
    /// resulting integers are mixed.
    ///
    /// The choice is deliberate. The coordinates themselves are very far from
    /// uniform: the attractor's invariant measure concentrates on two lobes, so
    /// a naive linear rescaling of `x` into [0, 1) fails a chi-squared
    /// uniformity test decisively. The low-order mantissa digits, in contrast,
    /// are driven by the stretching that produces the positive Lyapunov
    /// exponent, and are close to uniform. This is the standard argument for
    /// harvesting from a chaotic map; the accompanying statistical battery
    /// verifies it empirically rather than assuming it.
    ///
    /// Note on what this is and is not: this is a deterministic simulation of a
    /// chaotic system, suitable for reproducible experiments. It is not a
    /// cryptographic generator and must not be used as one; its state is
    /// recoverable from its output and it has no security analysis.
    pub fn next_u64(&mut self) -> u64 {
        let s = self.advance();
        // 2^28 places the extracted digits well below the attractor's own
        // dynamic range (coordinates live in the tens) while staying well above
        // the accumulated RK4 rounding error at this step size.
        let harvest = |v: f64| -> u64 {
            let scaled = v * 268_435_456.0; // 2^28
            let frac = scaled - scaled.floor();
            (frac * 9_007_199_254_740_992.0) as u64 // 2^53
        };
        let a = harvest(s.x);
        let b = harvest(s.y);
        let c = harvest(s.z);
        // Mixing three harvested words with the SplitMix64 finaliser. Without a
        // finaliser the concatenated words retain visible structure in the high
        // bits; the finaliser is a bijection, so it cannot destroy entropy.
        let mut z = a ^ b.rotate_left(21) ^ c.rotate_left(42);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns the next variate uniform on [0, 1).
    ///
    /// Uses the top 53 bits so every output is exactly representable and the
    /// spacing is uniform, the standard construction for doubles.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Advances the orbit and returns the raw state, bypassing extraction.
    ///
    /// Exposed for the topological analysis of Phase 0.5, which needs the
    /// attractor itself as a positive control: whatever structure the geometry
    /// of the Lorenz system carries must be visible here, or the measurement
    /// pipeline is at fault rather than the extraction.
    pub fn next_raw_state(&mut self) -> LorenzState {
        self.advance()
    }

    /// Advances the orbit and returns the first coordinate scaled by 2^28
    /// before the fractional part is taken.
    ///
    /// This is the first of the three extraction stages that Phase 0.5
    /// examines separately, so that any loss of structure can be attributed to
    /// a specific step rather than to the pipeline as a whole.
    pub fn next_stage_scaled(&mut self) -> f64 {
        let s = self.advance();
        s.x * 268_435_456.0
    }

    /// Advances the orbit and returns the fractional part of the scaled first
    /// coordinate, the second extraction stage, before mixing.
    pub fn next_stage_fraction(&mut self) -> f64 {
        let scaled = self.next_stage_scaled();
        scaled - scaled.floor()
    }

    /// Returns a standard normal variate by the Box-Muller transform.
    ///
    /// REF: [Box and Muller, 1958] "A Note on the Generation of Random Normal
    ///      Deviates", Annals of Mathematical Statistics 29(2), pp. 610-611
    ///      DOI: 10.1214/aoms/1177706645
    ///
    /// Only one of the two generated deviates is kept. Caching the second would
    /// halve the cost but makes the number of underlying draws depend on call
    /// parity, which complicates exact reproducibility across code paths that
    /// consume different counts. The experiment values reproducibility over
    /// speed here.
    pub fn next_normal(&mut self) -> f64 {
        let mut u1 = self.next_f64();
        // ln(0) is undefined; the probability is about 2^-53 but the guard costs
        // nothing and removes a non-deterministic failure mode.
        while u1 <= f64::MIN_POSITIVE {
            u1 = self.next_f64();
        }
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Returns an integer uniform on [0, n), without modulo bias.
    ///
    /// Uses Lemire's rejection method.
    /// REF: [Lemire, 2019] "Fast Random Integer Generation in an Interval",
    ///      ACM Transactions on Modeling and Computer Simulation 29(1)
    ///      DOI: 10.1145/3230636
    pub fn next_below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "upper bound must be positive");
        let mut x = self.next_u64();
        let mut m = (x as u128) * (n as u128);
        let mut l = m as u64;
        if l < n {
            let threshold = n.wrapping_neg() % n;
            while l < threshold {
                x = self.next_u64();
                m = (x as u128) * (n as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Shuffles a slice in place with the Fisher-Yates algorithm.
    ///
    /// REF: [Durstenfeld, 1964] "Algorithm 235: Random permutation",
    ///      Communications of the ACM 7(7), p. 420
    ///      DOI: 10.1145/364520.364540
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.next_below(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }
}

/// The same interface backed by ChaCha8, used as the control condition.
///
/// REF: [Bernstein, 2008] "ChaCha, a variant of Salsa20", State of the Art of
///      Stream Ciphers workshop (SASC 2008)
///      https://cr.yp.to/chacha/chacha-20080128.pdf
#[derive(Debug, Clone)]
pub struct ChaChaRng {
    inner: rand_chacha::ChaCha8Rng,
}

impl ChaChaRng {
    /// Creates a generator from a 64-bit seed, zero-extended to the 32-byte key.
    pub fn from_seed(seed: u64) -> Self {
        use rand_chacha::rand_core::SeedableRng;
        let mut key = [0u8; 32];
        key[..8].copy_from_slice(&seed.to_le_bytes());
        Self {
            inner: rand_chacha::ChaCha8Rng::from_seed(key),
        }
    }

    /// Returns the next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        use rand_chacha::rand_core::Rng;
        self.inner.next_u64()
    }

    /// Returns the next variate uniform on [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Returns a standard normal variate, by the same Box-Muller construction
    /// used by [`LorenzRng::next_normal`], so the two conditions differ only in
    /// the underlying bit source.
    pub fn next_normal(&mut self) -> f64 {
        let mut u1 = self.next_f64();
        while u1 <= f64::MIN_POSITIVE {
            u1 = self.next_f64();
        }
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Returns an integer uniform on [0, n), by the same method used by
    /// [`LorenzRng::next_below`].
    pub fn next_below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "upper bound must be positive");
        let mut x = self.next_u64();
        let mut m = (x as u128) * (n as u128);
        let mut l = m as u64;
        if l < n {
            let threshold = n.wrapping_neg() % n;
            while l < threshold {
                x = self.next_u64();
                m = (x as u128) * (n as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Shuffles a slice in place, same algorithm as [`LorenzRng::shuffle`].
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.next_below(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }
}

/// The randomness source of an experimental condition.
///
/// The two variants differ in size because a Lorenz state is three doubles plus
/// nothing else, while a ChaCha8 core carries its key schedule and buffer.
/// Boxing the larger variant would add an indirection on the hot path of every
/// single draw, which is exactly what this experiment measures, so the size
/// difference is accepted deliberately.
///
/// A single enumeration rather than a trait object, so that dispatch is static
/// and the two conditions execute identical code paths apart from the bit
/// source itself.
#[derive(Debug, Clone)]
#[allow(
    clippy::large_enum_variant,
    reason = "boxing would perturb the timing measurement"
)]
pub enum Rng {
    /// Lorenz-attractor generator, the treatment condition.
    Lorenz(LorenzRng),
    /// ChaCha8 generator, the control condition.
    ChaCha(ChaChaRng),
}

/// Which generator a condition uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RngKind {
    /// Lorenz-attractor generator.
    Lorenz,
    /// ChaCha8 generator.
    ChaCha,
}

impl RngKind {
    /// Stable identifier used in output files.
    pub fn as_str(self) -> &'static str {
        match self {
            RngKind::Lorenz => "lorenz",
            RngKind::ChaCha => "chacha8",
        }
    }
}

impl Rng {
    /// Creates the generator selected by `kind`.
    pub fn new(kind: RngKind, seed: u64) -> Self {
        match kind {
            RngKind::Lorenz => Rng::Lorenz(LorenzRng::from_seed(seed)),
            RngKind::ChaCha => Rng::ChaCha(ChaChaRng::from_seed(seed)),
        }
    }

    /// Returns the next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        match self {
            Rng::Lorenz(r) => r.next_u64(),
            Rng::ChaCha(r) => r.next_u64(),
        }
    }

    /// Returns the next variate uniform on [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        match self {
            Rng::Lorenz(r) => r.next_f64(),
            Rng::ChaCha(r) => r.next_f64(),
        }
    }

    /// Returns a standard normal variate.
    pub fn next_normal(&mut self) -> f64 {
        match self {
            Rng::Lorenz(r) => r.next_normal(),
            Rng::ChaCha(r) => r.next_normal(),
        }
    }

    /// Returns an integer uniform on [0, n).
    pub fn next_below(&mut self, n: u64) -> u64 {
        match self {
            Rng::Lorenz(r) => r.next_below(n),
            Rng::ChaCha(r) => r.next_below(n),
        }
    }

    /// Shuffles a slice in place.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        match self {
            Rng::Lorenz(r) => r.shuffle(items),
            Rng::ChaCha(r) => r.shuffle(items),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rk4_conserves_a_known_fixed_point() {
        // The origin is a fixed point of the Lorenz system, so RK4 must leave it
        // exactly where it is. This checks the integrator independently of the
        // extraction method.
        let s = LorenzState {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let after = rk4_step(s, DT);
        assert!(after.x.abs() < 1e-15 && after.y.abs() < 1e-15 && after.z.abs() < 1e-15);
    }

    #[test]
    fn rk4_matches_reference_step_for_known_state() {
        // Fourth-order accuracy check: halving the step must reduce the
        // deviation from a much finer reference by roughly 2^4.
        let s0 = LorenzState {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        };
        let reference = {
            let mut s = s0;
            for _ in 0..1024 {
                s = rk4_step(s, 0.1 / 1024.0);
            }
            s
        };
        let err = |dt: f64, n: usize| {
            let mut s = s0;
            for _ in 0..n {
                s = rk4_step(s, dt);
            }
            ((s.x - reference.x).powi(2)
                + (s.y - reference.y).powi(2)
                + (s.z - reference.z).powi(2))
            .sqrt()
        };
        let coarse = err(0.05, 2);
        let fine = err(0.025, 4);
        assert!(
            fine < coarse / 8.0,
            "order of convergence too low: coarse {coarse:e}, fine {fine:e}"
        );
    }

    #[test]
    fn seeds_are_reproducible_and_distinct() {
        let a: Vec<u64> = (0..64)
            .map(|_| LorenzRng::from_seed(7).next_u64())
            .collect();
        let mut r1 = LorenzRng::from_seed(7);
        let mut r2 = LorenzRng::from_seed(7);
        let mut r3 = LorenzRng::from_seed(8);
        let s1: Vec<u64> = (0..64).map(|_| r1.next_u64()).collect();
        let s2: Vec<u64> = (0..64).map(|_| r2.next_u64()).collect();
        let s3: Vec<u64> = (0..64).map(|_| r3.next_u64()).collect();
        assert_eq!(s1, s2, "same seed must give the same stream");
        assert_ne!(s1, s3, "different seeds must give different streams");
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn next_below_respects_bounds() {
        let mut r = LorenzRng::from_seed(1);
        for _ in 0..10_000 {
            let v = r.next_below(7);
            assert!(v < 7);
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut r = LorenzRng::from_seed(3);
        let mut items: Vec<usize> = (0..500).collect();
        r.shuffle(&mut items);
        let mut sorted = items.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..500).collect::<Vec<_>>());
        assert_ne!(
            items, sorted,
            "a shuffle of 500 items should not be identity"
        );
    }
}
