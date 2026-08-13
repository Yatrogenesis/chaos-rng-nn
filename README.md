# chaos-rng-nn

A controlled experiment: is a deterministic pseudo-random generator driven by
the Lorenz attractor statistically distinguishable from ChaCha8 when used as the
randomness source in neural network training?

The randomness enters at three points of the pipeline: weight initialisation,
dropout masks, and minibatch order. The experiment measures whether that choice
changes final validation loss, convergence speed or generalisation gap.

**Results and their limits are in [REPORT.md](REPORT.md).** In short: within a
small MLP on a synthetic task, with ten runs per condition, no significant
difference was found on the learning metrics, and the chaotic generator was
three times slower. Ten runs per condition cannot establish equivalence, only
fail to detect a difference.

## Layout

```
crates/chaos-rng    Lorenz generator, ChaCha8 control, and the qualification battery
crates/xstats       Welch, Mann-Whitney U, Shapiro-Wilk and the special functions they need
crates/experiment   Dataset, MLP, run harness, analysis and figures
assets/fonts        Font vendored for figure rendering, with its licence
results/            Machine-readable output, committed so the report can be checked
figures/            Generated plots
```

## Reproducing

Requires a Rust toolchain (1.85 or newer). No system libraries, no network
access at run time, no GPU.

```bash
# 1. Generator qualification. Blocking: exits non-zero if a generator fails,
#    and the later phases are not meaningful if it does.
cargo run --release -p experiment -- phase0

# 2. The comparison itself: 10 runs per condition, plus a bit-for-bit
#    reproducibility check that exits non-zero on mismatch.
cargo run --release -p experiment -- phase1

# 3. Hypothesis tests and figures.
cargo run --release -p experiment -- analyse

# Unit tests, including the Phase 0 battery and the statistical functions
# checked against published values.
cargo test --workspace --release
```

Expected runtime on a four-core CPU: a few seconds for phase 0, about fifteen
seconds for phase 1, under a second for the analysis.

## What is fixed in advance

These were chosen before any run and are stated here so that later changes are
visible in the history:

- Significance level alpha = 0.05; ten runs per condition.
- Test selection is data-driven: Shapiro-Wilk on both samples, then Welch when
  neither is rejected, Mann-Whitney U otherwise.
- Phase 0 acceptance: uniformity not rejected at alpha = 0.01, all
  autocorrelations to lag 10 below 0.01 in absolute value, mean and variance
  within one percent of theory.
- Dataset and split are generated once with fixed seeds by the control
  generator, so both conditions see identical data.
- Both conditions receive the same ten training seeds.

One pre-registered metric, epochs to reach a validation loss of 0.35, turned out
to be degenerate: every run reached it in the first epoch. It is reported
unchanged, with a clearly labelled post-hoc replacement alongside it. See the
report.

## Determinism

Every run is reproducible bit for bit from its seed. This is enforced rather
than asserted: `phase1` repeats the first configuration of each condition and
compares the SHA-256 of the final parameters and the bit pattern of the final
loss, exiting non-zero on any mismatch, and a unit test does the same.

The Lorenz orbit uses only addition, subtraction and multiplication in a fixed
order, with no fused multiply-add and no parallel reduction, so the same
sequence should be produced on any platform with IEEE-754 doubles. That
reasoning has not been verified on a second platform.

## What this is not

The Lorenz generator here is a deterministic simulation of a chaotic system,
intended for reproducible experiments. It is not a cryptographic generator: its
state is recoverable from its output and it has had no security analysis. The
Phase 0 battery is a sanity check, not TestU01 or the NIST suite, neither of
which was run.

## Licence

MIT. See [LICENSE](LICENSE). The vendored font under `assets/fonts` carries its
own licence, included alongside it.
