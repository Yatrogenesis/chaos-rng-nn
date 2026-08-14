// SPDX-License-Identifier: MIT
//! Hypothesis tests implemented directly from their published closed forms, so
//! the experiment carries no heavyweight statistical dependency and every
//! formula in use is visible, cited and unit tested against known values.

#![forbid(unsafe_code)]

/// Sample mean.
pub fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Unbiased sample variance, with Bessel's correction.
pub fn variance(x: &[f64]) -> f64 {
    let n = x.len();
    assert!(n > 1, "variance needs at least two observations");
    let m = mean(x);
    x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n as f64 - 1.0)
}

/// Sample standard deviation.
pub fn std_dev(x: &[f64]) -> f64 {
    variance(x).sqrt()
}

// ---------------------------------------------------------------------------
// Special functions
// ---------------------------------------------------------------------------

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

/// Continued fraction expansion used by [`betai`], evaluated by Lentz's method.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 300;
    const EPS: f64 = 3.0e-16;
    const FPMIN: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAXIT {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;

        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularised incomplete beta function I_x(a, b).
///
/// REF: [Press, Teukolsky, Vetterling and Flannery, 2007] "Numerical Recipes:
///      The Art of Scientific Computing", 3rd edition, section 6.4
///      ISBN 978-0-521-88068-8
pub fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Complementary error function, by the rational Chebyshev fit of Numerical
/// Recipes, accurate to about 1.2e-7 in relative terms.
///
/// REF: [Press, Teukolsky, Vetterling and Flannery, 2007] "Numerical Recipes:
///      The Art of Scientific Computing", 3rd edition, section 6.2
///      ISBN 978-0-521-88068-8
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

/// Cumulative distribution function of the standard normal distribution.
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

/// Quantile function of the standard normal distribution, algorithm AS 241
/// (the PPND16 variant, accurate to about 1e-16).
///
/// REF: [Wichura, 1988] "Algorithm AS 241: The Percentage Points of the Normal
///      Distribution", Journal of the Royal Statistical Society Series C
///      (Applied Statistics) 37(3), pp. 477-484
///      DOI: 10.2307/2347330
pub fn normal_quantile(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "quantile argument must lie in (0, 1)");
    let q = p - 0.5;

    if q.abs() <= 0.425 {
        let r = 0.180_625 - q * q;
        let num = ((((((2_509.080_928_730_122_7 * r + 33_430.575_583_588_13) * r
            + 67_265.770_927_008_7)
            * r
            + 45_921.953_931_549_87)
            * r
            + 13_731.693_765_509_46)
            * r
            + 1_971.590_950_306_551_4)
            * r
            + 133.141_667_891_784_38)
            * r
            + 3.387_132_872_796_366_6;
        let den = ((((((5_226.495_278_852_854 * r + 28_729.085_735_721_942) * r
            + 39_307.895_800_092_71)
            * r
            + 21_213.794_301_586_596)
            * r
            + 5_394.196_021_424_751)
            * r
            + 687.187_007_492_057_9)
            * r
            + 42.313_330_701_600_91)
            * r
            + 1.0;
        return q * num / den;
    }

    let r_outer = if q < 0.0 { p } else { 1.0 - p };
    let mut r = (-r_outer.ln()).sqrt();

    let val = if r <= 5.0 {
        r -= 1.6;
        let num = ((((((7.745_450_142_783_414e-4 * r + 0.022_723_844_989_269_184) * r
            + 0.241_780_725_177_450_6)
            * r
            + 1.270_458_252_452_368_4)
            * r
            + 3.647_848_324_763_204_6)
            * r
            + 5.769_497_221_460_691)
            * r
            + 4.630_337_846_156_545)
            * r
            + 1.423_437_110_749_683_6;
        let den = ((((((1.050_750_071_644_416_8e-9 * r + 5.475_938_084_995_345e-4) * r
            + 0.015_198_666_563_616_457)
            * r
            + 0.148_103_976_427_480_07)
            * r
            + 0.689_767_334_985_1)
            * r
            + 1.676_384_830_183_803_9)
            * r
            + 2.053_191_626_637_759)
            * r
            + 1.0;
        num / den
    } else {
        r -= 5.0;
        let num = ((((((2.010_334_399_292_288e-7 * r + 2.711_555_568_743_487_6e-5) * r
            + 0.001_242_660_947_388_078_4)
            * r
            + 0.026_532_189_526_576_124)
            * r
            + 0.296_560_571_828_504_9)
            * r
            + 1.784_826_539_917_291_3)
            * r
            + 5.463_784_911_164_114)
            * r
            + 6.657_904_643_501_104;
        let den = ((((((2.044_263_103_389_939_8e-15 * r + 1.421_511_758_316_446e-7) * r
            + 1.846_318_317_510_055e-5)
            * r
            + 7.868_691_311_456_133e-4)
            * r
            + 0.014_875_361_290_850_615)
            * r
            + 0.136_929_880_922_735_8)
            * r
            + 0.599_832_206_555_888)
            * r
            + 1.0;
        num / den
    };

    if q < 0.0 {
        -val
    } else {
        val
    }
}

// ---------------------------------------------------------------------------
// Welch's t-test
// ---------------------------------------------------------------------------

/// Result of a two-sided Welch test.
#[derive(Debug, Clone, Copy)]
pub struct WelchResult {
    /// The t statistic.
    pub t: f64,
    /// Degrees of freedom from the Welch-Satterthwaite equation.
    pub df: f64,
    /// Two-sided probability of a statistic at least this extreme under H0.
    pub p_value: f64,
}

/// Welch's unequal-variances t-test, two sided.
///
/// t = (m1 - m2) / sqrt(s1^2/n1 + s2^2/n2), with the Welch-Satterthwaite
/// degrees of freedom
/// df = (s1^2/n1 + s2^2/n2)^2 / ( (s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1) ).
/// The two-sided p-value is I_{df/(df+t^2)}(df/2, 1/2).
///
/// REF: [Welch, 1947] "The Generalization of Student's Problem when Several
///      Different Population Variances are Involved", Biometrika 34(1/2),
///      pp. 28-35. DOI: 10.2307/2332510
pub fn welch_t_test(a: &[f64], b: &[f64]) -> WelchResult {
    let (n1, n2) = (a.len() as f64, b.len() as f64);
    let (v1, v2) = (variance(a), variance(b));
    let se1 = v1 / n1;
    let se2 = v2 / n2;
    let se = (se1 + se2).sqrt();

    if se == 0.0 {
        // Both samples constant. The difference is either exactly zero, in
        // which case there is no evidence of a difference, or non-zero with
        // zero variance, which no t-test can characterise.
        let t = if mean(a) == mean(b) {
            0.0
        } else {
            f64::INFINITY
        };
        return WelchResult {
            t,
            df: n1 + n2 - 2.0,
            p_value: if t == 0.0 { 1.0 } else { 0.0 },
        };
    }

    let t = (mean(a) - mean(b)) / se;
    let df = (se1 + se2).powi(2) / (se1 * se1 / (n1 - 1.0) + se2 * se2 / (n2 - 1.0));
    let p = betai(df / 2.0, 0.5, df / (df + t * t));
    WelchResult { t, df, p_value: p }
}

/// Cohen's d for two independent samples, using the pooled standard deviation.
///
/// d = (m1 - m2) / s_pooled, with
/// s_pooled = sqrt( ((n1-1)s1^2 + (n2-1)s2^2) / (n1+n2-2) ).
///
/// REF: [Cohen, 1988] "Statistical Power Analysis for the Behavioral
///      Sciences", 2nd edition, Lawrence Erlbaum Associates
///      DOI: 10.4324/9780203771587
pub fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    let (n1, n2) = (a.len() as f64, b.len() as f64);
    let (v1, v2) = (variance(a), variance(b));
    let pooled = (((n1 - 1.0) * v1 + (n2 - 1.0) * v2) / (n1 + n2 - 2.0)).sqrt();
    if pooled == 0.0 {
        return 0.0;
    }
    (mean(a) - mean(b)) / pooled
}

// ---------------------------------------------------------------------------
// Mann-Whitney U
// ---------------------------------------------------------------------------

/// Result of a two-sided Mann-Whitney U test.
#[derive(Debug, Clone, Copy)]
pub struct MannWhitneyResult {
    /// The U statistic of the first sample.
    pub u: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// True when the p-value came from the exact null distribution rather than
    /// the normal approximation.
    pub exact: bool,
}

/// Mann-Whitney U test, two sided.
///
/// Ranks are assigned to the pooled sample with ties receiving their average
/// rank. U1 = R1 - n1(n1+1)/2. When there are no ties and the samples are small
/// the exact null distribution is enumerated by the standard recurrence;
/// otherwise the normal approximation with a tie correction and a continuity
/// correction is used.
///
/// REF: [Mann and Whitney, 1947] "On a Test of Whether one of Two Random
///      Variables is Stochastically Larger than the Other", Annals of
///      Mathematical Statistics 18(1), pp. 50-60
///      DOI: 10.1214/aoms/1177730491
pub fn mann_whitney_u(a: &[f64], b: &[f64]) -> MannWhitneyResult {
    let n1 = a.len();
    let n2 = b.len();
    assert!(n1 > 0 && n2 > 0, "both samples must be non-empty");

    // Pooled ranking with average ranks for ties.
    let mut pooled: Vec<(f64, usize)> = a
        .iter()
        .map(|&v| (v, 0usize))
        .chain(b.iter().map(|&v| (v, 1usize)))
        .collect();
    pooled.sort_by(|p, q| p.0.partial_cmp(&q.0).expect("samples must not contain NaN"));

    let n = pooled.len();
    let mut ranks = vec![0.0f64; n];
    let mut tie_groups: Vec<usize> = Vec::new();
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && pooled[j + 1].0 == pooled[i].0 {
            j += 1;
        }
        let avg = ((i + 1) + (j + 1)) as f64 / 2.0;
        for r in ranks.iter_mut().take(j + 1).skip(i) {
            *r = avg;
        }
        if j > i {
            tie_groups.push(j - i + 1);
        }
        i = j + 1;
    }

    let r1: f64 = ranks
        .iter()
        .zip(pooled.iter())
        .filter(|(_, p)| p.1 == 0)
        .map(|(r, _)| *r)
        .sum();

    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u2 = (n1 * n2) as f64 - u1;
    let u_min = u1.min(u2);

    // Exact enumeration when feasible and untied.
    if tie_groups.is_empty() && n1 <= 20 && n2 <= 20 {
        let p = exact_two_sided_p(n1, n2, u_min);
        return MannWhitneyResult {
            u: u1,
            p_value: p,
            exact: true,
        };
    }

    // Normal approximation with tie and continuity corrections.
    let mu = (n1 * n2) as f64 / 2.0;
    let n_f = n as f64;
    let tie_term: f64 = tie_groups
        .iter()
        .map(|&t| {
            let t = t as f64;
            t * t * t - t
        })
        .sum();
    let sigma = (((n1 * n2) as f64 / 12.0) * ((n_f + 1.0) - tie_term / (n_f * (n_f - 1.0)))).sqrt();
    if sigma == 0.0 {
        return MannWhitneyResult {
            u: u1,
            p_value: 1.0,
            exact: false,
        };
    }
    let z = ((u_min - mu).abs() - 0.5) / sigma;
    let p = (2.0 * (1.0 - normal_cdf(z))).min(1.0);
    MannWhitneyResult {
        u: u1,
        p_value: p,
        exact: false,
    }
}

/// Exact two-sided p-value for the untied Mann-Whitney null distribution.
///
/// Enumerates the C(n1+n2, n1) equally likely interleavings by the standard
/// recurrence
///
/// f(m, n, u) = f(m-1, n, u-n) + f(m, n-1, u),
///
/// where f(m, n, u) counts the arrangements of m and n observations whose
/// statistic equals u, with f(m, 0, 0) = f(0, n, 0) = 1 and f = 0 for u < 0.
/// The lower tail at `u_min` is then doubled to obtain the two-sided value.
///
/// REF: [Mann and Whitney, 1947] "On a Test of Whether one of Two Random
///      Variables is Stochastically Larger than the Other", Annals of
///      Mathematical Statistics 18(1), pp. 50-60, section 2 (the recurrence)
///      DOI: 10.1214/aoms/1177730491
fn exact_two_sided_p(n1: usize, n2: usize, u_min: f64) -> f64 {
    let max_u = n1 * n2;
    // table[m][n][u], flattened over u for each (m, n).
    let mut table = vec![vec![vec![0f64; max_u + 1]; n2 + 1]; n1 + 1];
    for m in 0..=n1 {
        for n in 0..=n2 {
            if m == 0 || n == 0 {
                table[m][n][0] = 1.0;
                continue;
            }
            for u in 0..=max_u {
                let mut acc = table[m][n - 1][u];
                if u >= n {
                    acc += table[m - 1][n][u - n];
                }
                table[m][n][u] = acc;
            }
        }
    }
    let counts = &table[n1][n2];
    let total: f64 = counts.iter().sum();
    if total == 0.0 {
        return 1.0;
    }
    let cutoff = (u_min.floor() as usize).min(max_u);
    let tail: f64 = counts[..=cutoff].iter().sum();
    ((2.0 * tail) / total).min(1.0)
}

// ---------------------------------------------------------------------------
// Shapiro-Wilk
// ---------------------------------------------------------------------------

/// Result of a Shapiro-Wilk normality test.
#[derive(Debug, Clone, Copy)]
pub struct ShapiroWilkResult {
    /// The W statistic, in (0, 1]; values near one indicate normality.
    pub w: f64,
    /// Probability of a W at least this small under the normality hypothesis.
    pub p_value: f64,
}

/// Shapiro-Wilk test for normality, following algorithm AS R94.
///
/// REF: [Royston, 1995] "Remark AS R94: A Remark on Algorithm AS 181: The
///      W-test for Normality", Journal of the Royal Statistical Society
///      Series C (Applied Statistics) 44(4), pp. 547-551
///      DOI: 10.2307/2986146
///
/// REF: [Shapiro and Wilk, 1965] "An Analysis of Variance Test for Normality
///      (Complete Samples)", Biometrika 52(3/4), pp. 591-611
///      DOI: 10.2307/2333709
pub fn shapiro_wilk(x: &[f64]) -> ShapiroWilkResult {
    let n = x.len();
    assert!(n >= 3, "Shapiro-Wilk requires at least three observations");

    let mut s: Vec<f64> = x.to_vec();
    s.sort_by(|p, q| p.partial_cmp(q).expect("samples must not contain NaN"));

    let n_f = n as f64;

    // Expected values of the standard normal order statistics, Blom's
    // approximation, as prescribed by AS R94.
    let m: Vec<f64> = (1..=n)
        .map(|i| normal_quantile((i as f64 - 0.375) / (n_f + 0.25)))
        .collect();
    let ssumm2: f64 = m.iter().map(|v| v * v).sum();
    let rsn = 1.0 / n_f.sqrt();

    let mut a = vec![0.0f64; n];

    let an = -2.706_056 * rsn.powi(5) + 4.434_685 * rsn.powi(4)
        - 2.071_190 * rsn.powi(3)
        - 0.147_981 * rsn * rsn
        + 0.221_157 * rsn
        + m[n - 1] / ssumm2.sqrt();

    let phi;
    if n > 5 {
        let an1 = -3.582_633 * rsn.powi(5) + 5.682_633 * rsn.powi(4)
            - 1.752_461 * rsn.powi(3)
            - 0.293_762 * rsn * rsn
            + 0.042_981 * rsn
            + m[n - 2] / ssumm2.sqrt();
        phi = (ssumm2 - 2.0 * m[n - 1] * m[n - 1] - 2.0 * m[n - 2] * m[n - 2])
            / (1.0 - 2.0 * an * an - 2.0 * an1 * an1);
        a[n - 1] = an;
        a[0] = -an;
        a[n - 2] = an1;
        a[1] = -an1;
        for i in 2..n - 2 {
            a[i] = m[i] / phi.sqrt();
        }
    } else {
        phi = (ssumm2 - 2.0 * m[n - 1] * m[n - 1]) / (1.0 - 2.0 * an * an);
        a[n - 1] = an;
        a[0] = -an;
        for i in 1..n - 1 {
            a[i] = m[i] / phi.sqrt();
        }
    }

    let xbar = mean(&s);
    let numerator: f64 = a.iter().zip(s.iter()).map(|(ai, si)| ai * si).sum::<f64>();
    let denominator: f64 = s.iter().map(|si| (si - xbar) * (si - xbar)).sum();
    if denominator == 0.0 {
        // A constant sample has no dispersion; normality is undefined.
        return ShapiroWilkResult {
            w: 1.0,
            p_value: 1.0,
        };
    }
    let w = (numerator * numerator / denominator).min(1.0);

    // p-value by Royston's normalising transformations.
    let p_value = if n == 3 {
        // Exact for n = 3.
        let pi6 = 6.0 / std::f64::consts::PI;
        let stqr = (0.75f64).sqrt().asin();
        let p = pi6 * (w.sqrt().asin() - stqr);
        p.clamp(0.0, 1.0)
    } else if n <= 11 {
        let gamma = -2.273 + 0.459 * n_f;
        let mu = 0.544_0 - 0.399_78 * n_f + 0.025_054 * n_f * n_f - 0.000_671_4 * n_f.powi(3);
        let sigma =
            (1.382_2 - 0.778_57 * n_f + 0.062_767 * n_f * n_f - 0.002_032_2 * n_f.powi(3)).exp();
        let arg = gamma - (1.0 - w).ln();
        if arg <= 0.0 {
            return ShapiroWilkResult { w, p_value: 1.0 };
        }
        let z = (-arg.ln() - mu) / sigma;
        1.0 - normal_cdf(z)
    } else {
        let ln_n = n_f.ln();
        let mu = -1.586_1 - 0.310_82 * ln_n - 0.083_751 * ln_n * ln_n + 0.003_891_5 * ln_n.powi(3);
        let sigma = (-0.480_3 - 0.082_676 * ln_n + 0.003_030_2 * ln_n * ln_n).exp();
        let z = ((1.0 - w).ln() - mu) / sigma;
        1.0 - normal_cdf(z)
    };

    ShapiroWilkResult {
        w,
        p_value: p_value.clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_quantile_matches_published_values() {
        // Standard normal percentage points, widely tabulated.
        assert!((normal_quantile(0.5)).abs() < 1e-12);
        assert!((normal_quantile(0.975) - 1.959_963_984_540_054).abs() < 1e-9);
        assert!((normal_quantile(0.025) + 1.959_963_984_540_054).abs() < 1e-9);
        assert!((normal_quantile(0.99) - 2.326_347_874_040_841).abs() < 1e-9);
        assert!((normal_quantile(0.001) + 3.090_232_306_167_814).abs() < 1e-8);
    }

    #[test]
    fn normal_cdf_and_quantile_are_inverse() {
        for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let z = normal_quantile(p);
            assert!(
                (normal_cdf(z) - p).abs() < 1e-6,
                "p={p}, round trip gave {}",
                normal_cdf(z)
            );
        }
    }

    #[test]
    fn welch_on_identical_samples_finds_no_difference() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = welch_t_test(&a, &a);
        assert!(r.t.abs() < 1e-12);
        assert!((r.p_value - 1.0).abs() < 1e-9);
    }

    #[test]
    fn welch_matches_hand_computed_statistic() {
        // Two samples with means 2 and 5 and identical variance 1.
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let r = welch_t_test(&a, &b);
        // t = (2 - 5) / sqrt(1/3 + 1/3) = -3 / 0.816496... = -3.674234...
        assert!((r.t + 3.674_234_614_174_767).abs() < 1e-9, "t = {}", r.t);
        // Equal variances and equal n give df = n1 + n2 - 2 = 4.
        assert!((r.df - 4.0).abs() < 1e-9, "df = {}", r.df);
        assert!(r.p_value < 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn welch_separates_clearly_different_populations() {
        let a: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let b: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 0.1).collect();
        let r = welch_t_test(&a, &b);
        assert!(r.p_value < 1e-9, "p = {}", r.p_value);
    }

    #[test]
    fn cohens_d_is_scale_free_and_signed() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let d = cohens_d(&a, &b);
        // Means differ by 3, pooled sd is 1, so d = -3.
        assert!((d + 3.0).abs() < 1e-9, "d = {d}");
        assert!((cohens_d(&b, &a) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn mann_whitney_matches_hand_computed_u() {
        // Perfect separation: every value of b exceeds every value of a, so
        // U1 = 0 and U2 = n1 n2.
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let r = mann_whitney_u(&a, &b);
        assert!((r.u - 0.0).abs() < 1e-12, "u = {}", r.u);
        assert!(r.exact, "small untied samples should use the exact test");
        // With n1 = n2 = 3 there are C(6,3) = 20 arrangements, one of which is
        // this extreme, so the two-sided p is 2/20 = 0.1.
        assert!((r.p_value - 0.1).abs() < 1e-9, "p = {}", r.p_value);
    }

    #[test]
    fn mann_whitney_on_interleaved_samples_finds_no_difference() {
        let a = [1.0, 3.0, 5.0, 7.0, 9.0];
        let b = [2.0, 4.0, 6.0, 8.0, 10.0];
        let r = mann_whitney_u(&a, &b);
        assert!(r.p_value > 0.5, "p = {}", r.p_value);
    }

    #[test]
    fn mann_whitney_handles_ties_without_panicking() {
        let a = [1.0, 1.0, 2.0, 2.0, 3.0];
        let b = [1.0, 2.0, 2.0, 3.0, 3.0];
        let r = mann_whitney_u(&a, &b);
        assert!(!r.exact, "tied samples must fall back to the approximation");
        assert!(r.p_value > 0.0 && r.p_value <= 1.0);
    }

    #[test]
    fn shapiro_wilk_accepts_normal_looking_data() {
        // Symmetric, bell-shaped sample; W should be near one and p large.
        let x = [
            -1.62, -1.28, -0.92, -0.67, -0.43, -0.21, 0.0, 0.21, 0.43, 0.67, 0.92, 1.28, 1.62,
        ];
        let r = shapiro_wilk(&x);
        assert!(r.w > 0.95, "W = {}", r.w);
        assert!(r.p_value > 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn shapiro_wilk_rejects_strongly_skewed_data() {
        // Exponential-like sample, strongly right skewed.
        let x = [
            0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.22, 0.35, 0.6, 1.1, 2.4, 6.0, 15.0,
        ];
        let r = shapiro_wilk(&x);
        assert!(r.p_value < 0.05, "p = {} with W = {}", r.p_value, r.w);
    }

    #[test]
    fn anova_finds_no_difference_between_identical_groups() {
        let g = vec![vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0, 4.0]];
        let r = one_way_anova(&g);
        assert!(r.f.abs() < 1e-12, "F = {}", r.f);
        assert!((r.p_value - 1.0).abs() < 1e-9);
    }

    #[test]
    fn anova_matches_a_hand_computed_case() {
        // Three groups of three, means 2, 5 and 8, each with variance 1.
        // SS_between = 3*((2-5)^2 + 0 + (8-5)^2) = 54, df 2, MS 27.
        // SS_within  = 6*1 = 6, df 6, MS 1. So F = 27 exactly.
        let g = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let r = one_way_anova(&g);
        assert!((r.f - 27.0).abs() < 1e-9, "F = {}", r.f);
        assert!((r.df_between - 2.0).abs() < 1e-12);
        assert!((r.df_within - 6.0).abs() < 1e-12);
        assert!(r.p_value < 0.01, "p = {}", r.p_value);
    }

    #[test]
    fn anova_and_welch_agree_on_two_groups() {
        // With two groups of equal size and variance, F must equal t squared.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![3.0, 4.0, 5.0, 6.0, 7.0];
        let f = one_way_anova(&[a.clone(), b.clone()]).f;
        let t = welch_t_test(&a, &b).t;
        assert!((f - t * t).abs() < 1e-9, "F = {f}, t^2 = {}", t * t);
    }

    #[test]
    fn kruskal_wallis_finds_no_difference_between_interleaved_groups() {
        let g = vec![
            vec![1.0, 4.0, 7.0, 10.0],
            vec![2.0, 5.0, 8.0, 11.0],
            vec![3.0, 6.0, 9.0, 12.0],
        ];
        let r = kruskal_wallis(&g);
        assert!(r.p_value > 0.5, "p = {}", r.p_value);
    }

    #[test]
    fn kruskal_wallis_separates_disjoint_groups() {
        let g = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![11.0, 12.0, 13.0, 14.0, 15.0],
            vec![21.0, 22.0, 23.0, 24.0, 25.0],
        ];
        let r = kruskal_wallis(&g);
        assert!((r.degrees_of_freedom - 2.0).abs() < 1e-12);
        assert!(r.p_value < 0.01, "p = {}", r.p_value);
    }

    #[test]
    fn kruskal_wallis_handles_ties() {
        let g = vec![vec![1.0, 1.0, 2.0], vec![1.0, 2.0, 2.0]];
        let r = kruskal_wallis(&g);
        assert!(r.p_value > 0.0 && r.p_value <= 1.0);
        assert!(r.h.is_finite());
    }

    #[test]
    fn holm_matches_its_defining_properties() {
        let raw = [0.01, 0.04, 0.03, 0.005];
        let adj = holm_adjust(&raw);
        // Smallest raw value is multiplied by m, and no adjusted value falls
        // below the raw one or above one.
        assert!((adj[3] - 0.02).abs() < 1e-12, "adj = {adj:?}");
        for (r, a) in raw.iter().zip(adj.iter()) {
            assert!(*a >= r - 1e-12 && *a <= 1.0);
        }
        // Monotone in the order of the raw values.
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by(|&x, &y| raw[x].partial_cmp(&raw[y]).unwrap());
        for w in order.windows(2) {
            assert!(adj[w[0]] <= adj[w[1]] + 1e-12);
        }
    }

    #[test]
    fn holm_is_never_more_conservative_than_bonferroni() {
        let raw = [0.001, 0.02, 0.03, 0.5];
        let adj = holm_adjust(&raw);
        for (r, a) in raw.iter().zip(adj.iter()) {
            assert!(*a <= (raw.len() as f64 * r).min(1.0) + 1e-12);
        }
    }

    #[test]
    fn shapiro_wilk_statistic_stays_in_range() {
        let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0];
        let r = shapiro_wilk(&x);
        assert!(r.w > 0.0 && r.w <= 1.0, "W = {}", r.w);
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0, "p = {}", r.p_value);
    }
}

// ---------------------------------------------------------------------------
// Comparisons across more than two groups
// ---------------------------------------------------------------------------

/// Result of a one-way analysis of variance.
#[derive(Debug, Clone, Copy)]
pub struct AnovaResult {
    /// The F statistic.
    pub f: f64,
    /// Degrees of freedom between groups.
    pub df_between: f64,
    /// Degrees of freedom within groups.
    pub df_within: f64,
    /// Upper-tail probability of F under the null.
    pub p_value: f64,
}

/// One-way analysis of variance.
///
/// F = (SS_between / df_between) / (SS_within / df_within), and the p-value is
/// the upper tail of the F distribution, obtained from the regularised
/// incomplete beta function as
/// P(F > f) = I_{df2/(df2 + df1 f)}(df2/2, df1/2).
///
/// REF: [Fisher, 1925] "Statistical Methods for Research Workers", Oliver and
///      Boyd. The analysis of variance.
///      DOI: 10.1007/978-1-4612-4380-9_6 (reprint in Breakthroughs in
///      Statistics, Springer Series in Statistics)
pub fn one_way_anova(groups: &[Vec<f64>]) -> AnovaResult {
    let k = groups.len();
    assert!(k >= 2, "analysis of variance needs at least two groups");
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    let grand_mean = groups.iter().flatten().sum::<f64>() / n_total as f64;

    let ss_between: f64 = groups
        .iter()
        .map(|g| {
            let d = mean(g) - grand_mean;
            g.len() as f64 * d * d
        })
        .sum();
    let ss_within: f64 = groups
        .iter()
        .map(|g| {
            let m = mean(g);
            g.iter().map(|v| (v - m) * (v - m)).sum::<f64>()
        })
        .sum();

    let df_between = (k - 1) as f64;
    let df_within = (n_total - k) as f64;
    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;

    if ms_within <= 0.0 {
        return AnovaResult {
            f: f64::INFINITY,
            df_between,
            df_within,
            p_value: 0.0,
        };
    }
    let f = ms_between / ms_within;
    let p = betai(
        df_within / 2.0,
        df_between / 2.0,
        df_within / (df_within + df_between * f),
    );
    AnovaResult {
        f,
        df_between,
        df_within,
        p_value: p,
    }
}

/// Result of a Kruskal-Wallis test.
#[derive(Debug, Clone, Copy)]
pub struct KruskalWallisResult {
    /// The H statistic, corrected for ties.
    pub h: f64,
    /// Degrees of freedom, groups minus one.
    pub degrees_of_freedom: f64,
    /// Upper-tail probability under the chi-squared approximation.
    pub p_value: f64,
}

/// Kruskal-Wallis one-way analysis of variance on ranks.
///
/// H = 12 / (N(N+1)) * sum_i R_i^2 / n_i - 3(N+1), divided by the tie
/// correction 1 - sum(t^3 - t) / (N^3 - N). Under the null H follows a
/// chi-squared distribution with k - 1 degrees of freedom, an approximation
/// that is adequate when each group has at least five observations.
///
/// REF: [Kruskal and Wallis, 1952] "Use of Ranks in One-Criterion Variance
///      Analysis", Journal of the American Statistical Association 47(260),
///      pp. 583-621. DOI: 10.1080/01621459.1952.10483441
pub fn kruskal_wallis(groups: &[Vec<f64>]) -> KruskalWallisResult {
    assert!(
        groups.len() >= 2,
        "Kruskal-Wallis needs at least two groups"
    );

    let mut pooled: Vec<(f64, usize)> = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        for &v in g {
            pooled.push((v, gi));
        }
    }
    pooled.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("values must not be NaN"));
    let n = pooled.len();

    let mut ranks = vec![0.0f64; n];
    let mut tie_term = 0.0f64;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && pooled[j + 1].0 == pooled[i].0 {
            j += 1;
        }
        let avg = ((i + 1) + (j + 1)) as f64 / 2.0;
        for r in ranks.iter_mut().take(j + 1).skip(i) {
            *r = avg;
        }
        let t = (j - i + 1) as f64;
        if t > 1.0 {
            tie_term += t * t * t - t;
        }
        i = j + 1;
    }

    let n_f = n as f64;
    let mut sum = 0.0;
    for (gi, g) in groups.iter().enumerate() {
        let r_sum: f64 = ranks
            .iter()
            .zip(pooled.iter())
            .filter(|(_, p)| p.1 == gi)
            .map(|(r, _)| *r)
            .sum();
        sum += r_sum * r_sum / g.len() as f64;
    }
    let mut h = 12.0 / (n_f * (n_f + 1.0)) * sum - 3.0 * (n_f + 1.0);
    let correction = 1.0 - tie_term / (n_f * n_f * n_f - n_f);
    if correction > 0.0 {
        h /= correction;
    }
    let df = (groups.len() - 1) as f64;
    KruskalWallisResult {
        h,
        degrees_of_freedom: df,
        p_value: chi_squared_sf(h, df),
    }
}

/// Upper-tail probability of the chi-squared distribution.
pub fn chi_squared_sf(statistic: f64, dof: f64) -> f64 {
    if statistic <= 0.0 {
        return 1.0;
    }
    1.0 - gamma_p(dof / 2.0, statistic / 2.0)
}

/// Regularised lower incomplete gamma function P(a, x).
///
/// REF: [Abramowitz and Stegun, 1964] "Handbook of Mathematical Functions",
///      equations 6.5.29 and 6.5.31, National Bureau of Standards Applied
///      Mathematics Series 55.
fn gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
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
        1.0 - (-x + a * x.ln() - ln_gamma(a)).exp() * h
    }
}

/// Adjusts a family of p-values by the Holm step-down procedure.
///
/// Holm controls the family-wise error rate under any dependence structure and
/// is uniformly more powerful than Bonferroni, which it contains as its first
/// step. Returns the adjusted values in the order the inputs were given, so a
/// caller can compare each against the original alpha.
///
/// REF: [Holm, 1979] "A Simple Sequentially Rejective Multiple Test Procedure",
///      Scandinavian Journal of Statistics 6(2), pp. 65-70
///      https://www.jstor.org/stable/4615733
pub fn holm_adjust(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| p_values[a].partial_cmp(&p_values[b]).expect("finite"));

    let mut adjusted = vec![0.0f64; m];
    let mut running = 0.0f64;
    for (rank, &i) in idx.iter().enumerate() {
        let scaled = ((m - rank) as f64 * p_values[i]).min(1.0);
        // Enforce monotonicity: an adjusted value can never fall below one
        // already assigned to a smaller raw p-value.
        running = running.max(scaled);
        adjusted[i] = running;
    }
    adjusted
}
