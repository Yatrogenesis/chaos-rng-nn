// SPDX-License-Identifier: MIT
//! Hypothesis testing and figures for the recorded runs.

use crate::runner::RunRecord;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};

/// Significance level, fixed by the protocol before any run was executed.
pub const ALPHA: f64 = 0.05;

/// Font used by the figures, vendored under `assets/fonts` together with its
/// licence so that figure generation needs nothing from the host system.
const FONT_BYTES: &[u8] = include_bytes!("../../../assets/fonts/DejaVuSans.ttf");

/// Registers the vendored font with plotters. Safe to call repeatedly.
fn ensure_font() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if plotters::style::register_font("sans-serif", FontStyle::Normal, FONT_BYTES).is_err() {
            panic!("the vendored font under assets/fonts is not a valid OpenType file");
        }
    });
}

/// Summary of one condition for one metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    /// Condition name.
    pub rng: String,
    /// Metric name.
    pub metric: String,
    /// Number of runs.
    pub n: usize,
    /// Sample mean.
    pub mean: f64,
    /// Sample standard deviation.
    pub std_dev: f64,
    /// Smallest observation.
    pub min: f64,
    /// Largest observation.
    pub max: f64,
}

/// Outcome of comparing the two conditions on one metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comparison {
    /// Metric compared.
    pub metric: String,
    /// Shapiro-Wilk W for the Lorenz condition.
    pub shapiro_w_lorenz: f64,
    /// Shapiro-Wilk p-value for the Lorenz condition.
    pub shapiro_p_lorenz: f64,
    /// Shapiro-Wilk W for the ChaCha8 condition.
    pub shapiro_w_chacha: f64,
    /// Shapiro-Wilk p-value for the ChaCha8 condition.
    pub shapiro_p_chacha: f64,
    /// Which test was applied, and why.
    pub test_used: String,
    /// Test statistic, t for Welch and U for Mann-Whitney.
    pub statistic: f64,
    /// Two-sided p-value of the comparison.
    pub p_value: f64,
    /// Cohen's d, reported regardless of significance.
    pub cohens_d: f64,
    /// Whether the null hypothesis is rejected at [`ALPHA`].
    pub rejects_h0: bool,
}

/// Descriptive statistics for one metric of one condition.
pub fn summarise(records: &[RunRecord], rng: &str, metric: &str, values: &[f64]) -> Summary {
    Summary {
        rng: rng.to_string(),
        metric: metric.to_string(),
        n: records.len(),
        mean: xstats::mean(values),
        std_dev: xstats::std_dev(values),
        min: values.iter().cloned().fold(f64::INFINITY, f64::min),
        max: values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    }
}

/// Compares the two conditions on one metric.
///
/// Normality of each sample is checked first with Shapiro-Wilk. Welch's test is
/// used only when neither sample is rejected at [`ALPHA`]; otherwise the
/// comparison falls back to Mann-Whitney U, which assumes no distributional
/// form. The choice is therefore made by the data rather than by preference,
/// and the reason is recorded in the output.
pub fn compare(metric: &str, lorenz: &[f64], chacha: &[f64]) -> Comparison {
    let sw_l = xstats::shapiro_wilk(lorenz);
    let sw_c = xstats::shapiro_wilk(chacha);
    let both_normal = sw_l.p_value > ALPHA && sw_c.p_value > ALPHA;

    let (test_used, statistic, p_value) = if both_normal {
        let w = xstats::welch_t_test(lorenz, chacha);
        (
            format!(
                "Welch t-test (Shapiro-Wilk did not reject normality: p = {:.4} and {:.4})",
                sw_l.p_value, sw_c.p_value
            ),
            w.t,
            w.p_value,
        )
    } else {
        let m = xstats::mann_whitney_u(lorenz, chacha);
        (
            format!(
                "Mann-Whitney U ({}, Shapiro-Wilk rejected normality: p = {:.4} and {:.4})",
                if m.exact {
                    "exact"
                } else {
                    "normal approximation"
                },
                sw_l.p_value,
                sw_c.p_value
            ),
            m.u,
            m.p_value,
        )
    };

    Comparison {
        metric: metric.to_string(),
        shapiro_w_lorenz: sw_l.w,
        shapiro_p_lorenz: sw_l.p_value,
        shapiro_w_chacha: sw_c.w,
        shapiro_p_chacha: sw_c.p_value,
        test_used,
        statistic,
        p_value,
        cohens_d: xstats::cohens_d(lorenz, chacha),
        rejects_h0: p_value < ALPHA,
    }
}

/// Cartesian chart context over two floating point axes, factored out because
/// the full type is unwieldy at the call site.
type FloatChart<'a, 'b> = ChartContext<
    'a,
    BitMapBackend<'b>,
    Cartesian2d<plotters::coord::types::RangedCoordf64, plotters::coord::types::RangedCoordf64>,
>;

/// Draws the mean training loss per epoch for both conditions, with a shaded
/// band of one standard deviation.
pub fn plot_loss_curves(
    records: &[RunRecord],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let lorenz: Vec<&RunRecord> = records.iter().filter(|r| r.rng == "lorenz").collect();
    let chacha: Vec<&RunRecord> = records.iter().filter(|r| r.rng == "chacha8").collect();
    let epochs = lorenz[0].train_loss_per_epoch.len();

    let curve = |group: &[&RunRecord]| -> (Vec<f64>, Vec<f64>) {
        let mut means = Vec::with_capacity(epochs);
        let mut sds = Vec::with_capacity(epochs);
        for e in 0..epochs {
            let vals: Vec<f64> = group.iter().map(|r| r.train_loss_per_epoch[e]).collect();
            means.push(xstats::mean(&vals));
            sds.push(xstats::std_dev(&vals));
        }
        (means, sds)
    };
    let (lm, ls) = curve(&lorenz);
    let (cm, cs) = curve(&chacha);

    let y_max = lm
        .iter()
        .chain(cm.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        * 1.1;

    ensure_font();
    let root = BitMapBackend::new(path, (1000, 620)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Training loss per epoch, mean of 10 runs with one standard deviation",
            ("sans-serif", 20).into_font(),
        )
        .margin(18)
        .x_label_area_size(48)
        .y_label_area_size(64)
        .build_cartesian_2d(0f64..epochs as f64, 0f64..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Mean training loss")
        .draw()?;

    let band = |chart: &mut FloatChart<'_, '_>,
                means: &[f64],
                sds: &[f64],
                colour: RGBColor|
     -> Result<(), Box<dyn std::error::Error>> {
        let upper: Vec<(f64, f64)> = means
            .iter()
            .zip(sds.iter())
            .enumerate()
            .map(|(i, (m, s))| (i as f64, m + s))
            .collect();
        let lower: Vec<(f64, f64)> = means
            .iter()
            .zip(sds.iter())
            .enumerate()
            .map(|(i, (m, s))| (i as f64, (m - s).max(0.0)))
            .rev()
            .collect();
        let mut poly = upper.clone();
        poly.extend(lower);
        chart.draw_series(std::iter::once(Polygon::new(poly, colour.mix(0.15))))?;
        Ok(())
    };

    band(&mut chart, &lm, &ls, BLUE)?;
    band(&mut chart, &cm, &cs, RED)?;

    chart
        .draw_series(LineSeries::new(
            lm.iter().enumerate().map(|(i, v)| (i as f64, *v)),
            BLUE.stroke_width(2),
        ))?
        .label("Lorenz")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(2)));

    chart
        .draw_series(LineSeries::new(
            cm.iter().enumerate().map(|(i, v)| (i as f64, *v)),
            RED.stroke_width(2),
        ))?
        .label("ChaCha8")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

/// Draws the distribution of final validation loss as a strip of points per
/// condition, which shows overlap directly rather than through a summary
/// statistic.
pub fn plot_final_losses(
    records: &[RunRecord],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let lorenz: Vec<f64> = records
        .iter()
        .filter(|r| r.rng == "lorenz")
        .map(|r| r.final_val_loss)
        .collect();
    let chacha: Vec<f64> = records
        .iter()
        .filter(|r| r.rng == "chacha8")
        .map(|r| r.final_val_loss)
        .collect();

    let all: Vec<f64> = lorenz.iter().chain(chacha.iter()).cloned().collect();
    let lo = all.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = all.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let pad = (hi - lo).max(1e-6) * 0.25;

    ensure_font();
    let root = BitMapBackend::new(path, (860, 520)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Final validation loss, one point per run",
            ("sans-serif", 20).into_font(),
        )
        .margin(18)
        .x_label_area_size(48)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..3f64, (lo - pad)..(hi + pad))?;

    chart
        .configure_mesh()
        .x_desc("1 = Lorenz, 2 = ChaCha8")
        .y_desc("Final validation loss")
        .draw()?;

    chart.draw_series(
        lorenz
            .iter()
            .map(|v| Circle::new((1.0, *v), 5, BLUE.filled())),
    )?;
    chart.draw_series(
        chacha
            .iter()
            .map(|v| Circle::new((2.0, *v), 5, RED.filled())),
    )?;

    // Mean markers.
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.8, xstats::mean(&lorenz)), (1.2, xstats::mean(&lorenz))],
        BLACK.stroke_width(2),
    )))?;
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(1.8, xstats::mean(&chacha)), (2.2, xstats::mean(&chacha))],
        BLACK.stroke_width(2),
    )))?;

    root.present()?;
    Ok(())
}
