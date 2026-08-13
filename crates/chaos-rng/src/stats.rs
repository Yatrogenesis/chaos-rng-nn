// SPDX-License-Identifier: MIT
//! Statistical battery applied to a generator before it is allowed into the
//! experiment. These are basic sanity tests, not a substitute for a suite such
//! as TestU01 or the NIST battery; passing them is necessary, not sufficient.

/// Outcome of a chi-squared goodness-of-fit test for uniformity.
#[derive(Debug, Clone, Copy)]
pub struct ChiSquared {
    /// The test statistic.
    pub statistic: f64,
    /// Degrees of freedom, bins minus one.
    pub degrees_of_freedom: usize,
    /// Upper-tail probability of the statistic under the null hypothesis.
    pub p_value: f64,
}

/// Chi-squared goodness-of-fit against the uniform distribution on [0, 1).
///
/// Statistic: sum over bins of (observed - expected)^2 / expected, with
/// expected = n / bins. Under the null hypothesis it follows a chi-squared
/// distribution with bins - 1 degrees of freedom.
///
/// REF: [Knuth, 1997] "The Art of Computer Programming, Volume 2:
///      Seminumerical Algorithms", 3rd edition, section 3.3.1
///      ISBN 978-0-201-89684-8
pub fn chi_squared_uniformity(samples: &[f64], bins: usize) -> ChiSquared {
    assert!(bins >= 2, "need at least two bins");
    let mut counts = vec![0usize; bins];
    for &s in samples {
        // Values are expected in [0, 1); clamp defensively so an out-of-range
        // value lands in the last bin instead of panicking, which would hide a
        // generator defect behind a crash.
        let idx = ((s * bins as f64) as usize).min(bins - 1);
        counts[idx] += 1;
    }
    let expected = samples.len() as f64 / bins as f64;
    let statistic: f64 = counts
        .iter()
        .map(|&o| {
            let d = o as f64 - expected;
            d * d / expected
        })
        .sum();
    let dof = bins - 1;
    ChiSquared {
        statistic,
        degrees_of_freedom: dof,
        p_value: chi_squared_sf(statistic, dof as f64),
    }
}

/// Autocorrelation of a sequence at a given lag.
///
/// r_k = sum_{i} (x_i - mean)(x_{i+k} - mean) / sum_i (x_i - mean)^2
///
/// REF: [Box, Jenkins and Reinsel, 2008] "Time Series Analysis: Forecasting
///      and Control", 4th edition, section 2.1.4
///      DOI: 10.1002/9781118619193
pub fn autocorrelation(samples: &[f64], lag: usize) -> f64 {
    assert!(lag < samples.len(), "lag must be shorter than the series");
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let d = samples[i] - mean;
        den += d * d;
        if i + lag < n {
            num += d * (samples[i + lag] - mean);
        }
    }
    if den == 0.0 {
        return 0.0;
    }
    num / den
}

/// Sample mean.
pub fn mean(samples: &[f64]) -> f64 {
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// Unbiased sample variance, with Bessel's correction.
pub fn variance(samples: &[f64]) -> f64 {
    let n = samples.len();
    assert!(n > 1, "variance needs at least two samples");
    let m = mean(samples);
    samples.iter().map(|s| (s - m) * (s - m)).sum::<f64>() / (n as f64 - 1.0)
}

/// Regularised lower incomplete gamma function P(a, x), by series expansion for
/// x < a + 1 and by continued fraction otherwise.
///
/// REF: [Abramowitz and Stegun, 1964] "Handbook of Mathematical Functions",
///      equations 6.5.29 and 6.5.31. National Bureau of Standards, Applied
///      Mathematics Series 55.
fn gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // Series representation.
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        // Continued fraction for Q(a, x), then P = 1 - Q.
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        let q = (-x + a * x.ln() - ln_gamma(a)).exp() * h;
        1.0 - q
    }
}

/// Natural logarithm of the gamma function, by the Lanczos approximation.
///
/// REF: [Lanczos, 1964] "A Precision Approximation of the Gamma Function",
///      Journal of the SIAM: Series B, Numerical Analysis 1, pp. 86-96
///      DOI: 10.1137/0701008
pub fn ln_gamma(x: f64) -> f64 {
    const COF: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.120_865_097_386_617_7e-2,
        -0.539_523_938_495_3e-5,
    ];
    let mut y = x;
    let tmp = x + 5.5 - (x + 0.5) * (x + 5.5).ln();
    let mut ser = 1.000_000_000_190_015;
    for c in COF.iter() {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.506_628_274_631_000_5 * ser / x).ln()
}

/// Upper-tail probability of the chi-squared distribution.
pub fn chi_squared_sf(statistic: f64, dof: f64) -> f64 {
    if statistic <= 0.0 {
        return 1.0;
    }
    1.0 - gamma_p(dof / 2.0, statistic / 2.0)
}

/// Full Phase 0 report for one generator.
#[derive(Debug, Clone)]
pub struct BatteryReport {
    /// Number of samples examined.
    pub n: usize,
    /// Chi-squared uniformity result.
    pub chi: ChiSquared,
    /// Autocorrelation at lags 1 through 10.
    pub autocorrelations: Vec<f64>,
    /// Observed sample mean.
    pub mean: f64,
    /// Observed sample variance.
    pub variance: f64,
}

impl BatteryReport {
    /// Expected mean of the uniform distribution on [0, 1).
    pub const EXPECTED_MEAN: f64 = 0.5;
    /// Expected variance of the uniform distribution on [0, 1), 1/12.
    pub const EXPECTED_VARIANCE: f64 = 1.0 / 12.0;

    /// Acceptance criteria, fixed in advance of running the battery so the
    /// threshold cannot be adjusted to fit the observed numbers.
    ///
    /// The generator passes when the uniformity test is not rejected at
    /// alpha = 0.01, every autocorrelation up to lag 10 is below 0.01 in
    /// absolute value, and the sample mean and variance are within 1 percent
    /// of their theoretical values.
    pub fn passes(&self) -> bool {
        self.chi.p_value > 0.01
            && self.autocorrelations.iter().all(|r| r.abs() < 0.01)
            && (self.mean - Self::EXPECTED_MEAN).abs() < 0.01 * Self::EXPECTED_MEAN
            && (self.variance - Self::EXPECTED_VARIANCE).abs() < 0.01 * Self::EXPECTED_VARIANCE
    }
}

/// Runs the whole Phase 0 battery over a slice of variates on [0, 1).
pub fn run_battery(samples: &[f64], bins: usize) -> BatteryReport {
    BatteryReport {
        n: samples.len(),
        chi: chi_squared_uniformity(samples, bins),
        autocorrelations: (1..=10).map(|k| autocorrelation(samples, k)).collect(),
        mean: mean(samples),
        variance: variance(samples),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChaChaRng, LorenzRng};

    /// Number of samples used by the battery tests. Kept at 10^5 in the unit
    /// tests so the suite stays fast; the binary runs the full 10^6 required by
    /// the protocol and writes those numbers to the report.
    const N: usize = 100_000;

    #[test]
    fn ln_gamma_matches_known_values() {
        // ln(Gamma(n)) = ln((n-1)!) for positive integers.
        assert!((ln_gamma(1.0) - 0.0).abs() < 1e-10);
        assert!((ln_gamma(2.0) - 0.0).abs() < 1e-10);
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-10);
        // ln(Gamma(1/2)) = ln(sqrt(pi))
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-10);
    }

    #[test]
    fn chi_squared_sf_matches_known_values() {
        // Median of chi-squared with 1 dof is about 0.4549; survival there ~0.5.
        assert!((chi_squared_sf(0.454_936_4, 1.0) - 0.5).abs() < 1e-4);
        // The 95th percentile with 1 dof is 3.8415.
        assert!((chi_squared_sf(3.841_459, 1.0) - 0.05).abs() < 1e-4);
        // The 95th percentile with 10 dof is 18.307.
        assert!((chi_squared_sf(18.307_038, 10.0) - 0.05).abs() < 1e-4);
    }

    #[test]
    fn battery_rejects_a_deliberately_bad_generator() {
        // A ramp is perfectly uniform in its histogram but maximally correlated,
        // so the battery must fail it on autocorrelation. This guards against a
        // battery that passes everything.
        let ramp: Vec<f64> = (0..N).map(|i| i as f64 / N as f64).collect();
        let r = run_battery(&ramp, 100);
        assert!(!r.passes(), "battery must reject a monotone ramp");
        assert!(r.autocorrelations[0] > 0.9);
    }

    #[test]
    fn battery_rejects_a_non_uniform_generator() {
        // Squaring uniform variates biases them towards zero; the chi-squared
        // test must catch it.
        let mut rng = ChaChaRng::from_seed(1);
        let biased: Vec<f64> = (0..N).map(|_| rng.next_f64().powi(2)).collect();
        let r = run_battery(&biased, 100);
        assert!(!r.passes(), "battery must reject a squared-uniform stream");
        assert!(r.chi.p_value < 0.01);
    }

    #[test]
    fn chacha_passes_the_battery() {
        // The control generator must pass, otherwise the battery itself is
        // suspect rather than the generator under test.
        let mut rng = ChaChaRng::from_seed(20_260_813);
        let samples: Vec<f64> = (0..N).map(|_| rng.next_f64()).collect();
        let r = run_battery(&samples, 100);
        assert!(
            r.passes(),
            "ChaCha8 failed the battery: chi p={:.4}, acf={:?}, mean={:.6}, var={:.6}",
            r.chi.p_value,
            r.autocorrelations,
            r.mean,
            r.variance
        );
    }

    #[test]
    fn extraction_passes_phase_zero_battery() {
        // The blocking gate of Phase 0: the Lorenz extraction must pass the same
        // battery as the control before it may be used in training.
        let mut rng = LorenzRng::from_seed(20_260_813);
        let samples: Vec<f64> = (0..N).map(|_| rng.next_f64()).collect();
        let r = run_battery(&samples, 100);
        assert!(
            r.passes(),
            "Lorenz extraction failed the battery: chi p={:.4}, acf={:?}, mean={:.6}, var={:.6}",
            r.chi.p_value,
            r.autocorrelations,
            r.mean,
            r.variance
        );
    }
}
