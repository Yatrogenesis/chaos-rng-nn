# Chaos-driven pseudo-randomness against ChaCha8 in neural network training

Status: Phases 0, 0.5, 1 and 3 complete, including the Phase 0.5 null
distribution. Phase 2 not executed; see
[Phase 2](#7-phase-2-not-executed). Sections 1 to 10 describe Phases 0 and 1 and
are unchanged since they were first published; Phases 0.5 and 3 are added as
sections 11 and 12.

All numbers below were produced by the code in this repository on the date and
hardware stated in [Execution environment](#execution-environment). Nothing is
quoted from literature except where a reference is given, and nothing was
selected after the fact except where explicitly labelled post hoc.

## 1. Question and hypothesis

A deterministic generator driven by the Lorenz attractor is compared against
ChaCha8 as the source of randomness at three points of a training pipeline:
weight initialisation, dropout masks, and minibatch order.

The question is not whether the chaotic generator is better. It is whether it is
**statistically distinguishable** from a standard generator, while offering
exact deterministic reproducibility from a seed.

**H0.** Under identical architecture, identical data and an identical compute
budget, there is no statistically significant difference (alpha = 0.05) between
the two generators in final validation loss, convergence speed, or
generalisation gap.

**Falsification criterion, fixed before any run.** With N = 10 runs per
condition, normality of each sample is assessed with Shapiro-Wilk; Welch's test
is applied when neither sample is rejected, and Mann-Whitney U otherwise. If the
resulting two-sided p-value for final validation loss is below 0.05, H0 is
rejected and Cohen's d is reported.

## 2. Method

**Generator under test.** The Lorenz system, integrated with fixed-step
classical RK4 at dt = 0.01, with the standard chaotic parameters sigma = 10,
rho = 28, beta = 8/3. Ten thousand steps are discarded as burn-in so the orbit
settles on the attractor. Between samples the orbit advances five steps.

**Extraction method.** The macroscopic coordinates are strongly non-uniform: the
invariant measure of the attractor concentrates on two lobes, so rescaling a
coordinate into the unit interval fails a uniformity test outright. Instead each
coordinate is scaled by 2^28 and its fractional part taken, which keeps only
digits far below the scale of the attractor's own motion and above the
accumulated integration error. The three resulting words are combined and passed
through the SplitMix64 finaliser, a bijection, so the mixing cannot add entropy
that the orbit did not supply. The claim that these digits are near-uniform is
not assumed: it is what the Phase 0 battery measures.

**Control.** ChaCha8 seeded from the same 64-bit seed, exposed through an
identical interface. Both conditions use the same Box-Muller construction for
normal variates, the same Lemire rejection method for bounded integers, and the
same Fisher-Yates shuffle, so the only difference between conditions is the
underlying bit source.

**Controls against confounding.** The dataset and the train/validation split are
generated once, by ChaCha8, with fixed seeds distinct from the training seeds, so
both conditions see identical data. Both conditions receive the same list of ten
training seeds. The network, hyperparameters and number of optimisation steps are
identical.

## 3. Phase 0: generator qualification

One million variates per generator, chi-squared over 1000 equiprobable bins.
Acceptance criteria were fixed before running: uniformity not rejected at
alpha = 0.01, every autocorrelation up to lag 10 below 0.01 in absolute value,
and mean and variance within one percent of theory.

| Quantity | Lorenz | ChaCha8 | Theoretical |
|---|---|---|---|
| Samples | 1 000 000 | 1 000 000 | |
| Chi-squared (999 dof) | 1059.556 | 1014.616 | |
| Chi-squared p-value | 0.0896 | 0.3585 | |
| Mean | 0.500009 | 0.500281 | 0.5 |
| Variance | 0.083316 | 0.083191 | 0.083333 |
| Largest absolute autocorrelation, lags 1 to 10 | 0.00236 | 0.00203 | 0 |
| Verdict | PASS | PASS | |

Autocorrelations, lags 1 to 10:

- Lorenz: 0.00092, 0.00070, 0.00068, -0.00071, 0.00115, -0.00139, -0.00048, -0.00091, -0.00236, -0.00091
- ChaCha8: 0.00203, 0.00118, -0.00150, 0.00088, -0.00008, 0.00037, 0.00077, -0.00136, -0.00064, -0.00201

The Lorenz p-value of 0.0896 is the weakest result in the table. It clears the
0.01 gate that was fixed in advance, but it is also close enough to conventional
thresholds to deserve mention rather than silence: a single p-value near 0.09 is
not evidence of a defect, and equally it is not the comfortable margin that
ChaCha8 shows.

The battery is verified to have discriminating power: it rejects a monotone ramp
on autocorrelation and a squared-uniform stream on uniformity. A battery that
accepted everything would prove nothing about the generator.

**Scope.** These are basic sanity tests. They are necessary, not sufficient.
No claim is made that this generator would survive TestU01 or the NIST suite,
neither of which was run. It is not a cryptographic generator and must not be
used as one.

## 4. Phase 1: MLP on synthetic classification

**Task.** Two interleaved moons, 2000 samples, Gaussian noise of standard
deviation 0.20, split 75/25 into 1500 training and 500 validation observations.
The classes are not linearly separable.

**Network.** Two hidden layers of 32 units, ReLU, inverted dropout at p = 0.1,
softmax cross-entropy, Adam at learning rate 0.01, batch size 32, 60 epochs.
He normal initialisation. N = 10 runs per condition with seeds 1000 to 1009.

### Results

| Metric | Lorenz, mean ± sd | ChaCha8, mean ± sd | Test | p | Cohen's d | H0 |
|---|---|---|---|---|---|---|
| Final validation loss | 0.122441 ± 0.009684 | 0.118844 ± 0.006167 | Mann-Whitney U, exact | 0.3150 | 0.443 | not rejected |
| Final validation accuracy | 0.969200 ± 0.002860 | 0.968600 ± 0.003658 | Mann-Whitney U, normal approx. | 1.0000 | 0.183 | not rejected |
| Generalisation gap | 0.062599 ± 0.007468 | 0.061116 ± 0.005239 | Mann-Whitney U, exact | 0.4359 | 0.230 | not rejected |
| Epochs to threshold, pre-registered | 1.000 ± 0.000 | 1.000 ± 0.000 | Welch | 1.0000 | 0.000 | not rejected |
| Epochs to training loss < 0.10, post hoc | 7.200 ± 1.476 | 7.400 ± 1.075 | Welch | 0.7334 | -0.155 | not rejected |
| Wall-clock seconds per run | 1.116828 ± 0.010399 | 0.371316 ± 0.001490 | Mann-Whitney U, exact | 0.000011 | 100.363 | **rejected** |

The test applied to each metric was chosen by the data, not by preference:
Shapiro-Wilk was run on both samples first, and Welch was used only where
neither sample was rejected at alpha = 0.05. For final validation loss the
Lorenz sample gave Shapiro-Wilk p = 0.0473, marginally below the threshold,
which is why the non-parametric test was used there.

**A defect in the pre-registered convergence metric.** The threshold of 0.35 on
validation loss was fixed before running, and turned out to be reached by every
one of the twenty runs in its first epoch. The metric therefore has zero
variance and carries no information about convergence speed. The row is reported
anyway, because deleting a pre-registered metric that produced an uninformative
result would misrepresent the protocol. The post-hoc row beneath it uses a
tighter threshold on the training curves already recorded, and is exploratory:
it was chosen after seeing the data and must not be read as a confirmatory test.

**Reproducibility.** Both conditions reproduce bit for bit. Re-running the first
configuration of each condition reproduced the SHA-256 of the final parameters
and the exact bit pattern of the final validation loss. This is checked by the
`phase1` command itself, which exits non-zero if it fails, and by a unit test.

### Figures

![Training loss per epoch](figures/phase1_loss_curves.png)

Mean training loss per epoch over ten runs, with a band of one standard
deviation. The two curves and their bands overlap along the whole trajectory.

![Final validation loss per run](figures/phase1_final_losses.png)

Final validation loss, one point per run, with the condition mean marked. The
distributions overlap substantially.

## 5. Interpretation

**H0 is not rejected** for final validation loss, validation accuracy, or
generalisation gap. It is rejected, decisively, for wall-clock time.

The failure to reject is not evidence of equivalence. With N = 10 per condition
the design has limited power: the observed effect size for final validation loss
is d = 0.44, and a two-sample comparison at alpha = 0.05 with N = 10 per group
has low power against an effect of that magnitude. In other words, an effect of
this size could easily be present and go undetected here. The correct statement
is that this experiment did not detect a difference, not that no difference
exists. Establishing equivalence would require an equivalence test against a
pre-specified margin, which was neither designed nor run.

The one clear result is the cost. The Lorenz generator took 1.117 seconds per
run against 0.371 for ChaCha8, three times slower, with an effect size so large
it is barely meaningful to quote. This is expected rather than surprising: each
variate costs five RK4 steps, that is fifteen evaluations of the Lorenz vector
field, against a few ARX rounds for ChaCha8. Whatever the case for a chaotic
generator, it will not be throughput.

The reproducibility property that motivated the comparison holds, but it is not
a discriminator: ChaCha8 reproduced bit for bit under exactly the same
conditions. Determinism from a seed is a property of any well-specified
pseudo-random generator, not a distinctive feature of the chaotic one.

## 6. Limitations

- **Sample size.** N = 10 per condition. Underpowered against small and moderate
  effects, as discussed above.
- **Scale.** A two-layer MLP with 2178 parameters on a two-dimensional synthetic
  task. Nothing here supports any claim about larger models, real data, or other
  architectures. Phase 2, which was intended to test that, was not executed.
- **One task, one architecture, one hyperparameter setting.** No sweep was run;
  a difference could exist in regions of the configuration space not sampled.
- **Single machine, single platform.** Bit-for-bit reproducibility was verified
  on one machine. Cross-platform reproducibility of the floating-point orbit is
  argued from the operations used, not measured on other hardware.
- **Statistical battery.** Basic tests only, as stated in Phase 0.
- **The pre-registered convergence threshold was badly chosen**, as described
  above.

## 7. Phase 2, not executed

The protocol called for a small character-level transformer on a public corpus,
with the same N = 10 per condition. It was not run, and no partial results are
reported for it.

The reason is compute. The machine available is a four-core AMD Ryzen 3 3200G
with integrated Vega 8 graphics and no CUDA. Phase 1, a network of roughly two
thousand parameters, took 1.1 seconds per run for the Lorenz condition. A
transformer of two to four layers at d_model 128 to 256 on a corpus of about one
megabyte is on the order of a million parameters and several orders of magnitude
more arithmetic per step; twenty runs of it on this CPU is not a matter of
minutes. Reporting a truncated version of that experiment, or a single run per
condition, would produce numbers that look like evidence without being evidence.

If Phase 2 is attempted, two things in this repository would need to change:
the hand-written backward pass would be replaced by an automatic
differentiation engine, and the randomness would need to remain externally
supplied so the injection points stay under experimental control.

## 8. Execution environment

- CPU: AMD Ryzen 3 3200G, 4 cores
- GPU: integrated Radeon Vega 8, not used
- Toolchain: rustc 1.97.1, release profile, opt-level 3, thin LTO, one codegen unit
- Date of the reported run: 2026-08-13
- Dependency versions pinned exactly in `Cargo.toml`

## 9. Reuse of existing work

The statistical tests and special functions in `crates/xstats` were implemented
from their published closed forms, as the protocol directed, to avoid a heavy
dependency. It is worth recording that overlapping functionality exists in
crates published by this project's author, in particular `yatrosci-stats`
(distributions and statistical functions), `yatrosci-special` (gamma, erf and
related), `yatrosci-random` (generation and sampling), `yatrosci-integrate` and
`yatrosci-ode` (RK4 and other solvers), and `yatrosci-ml-neural` (neural network
layers with autograd). None of them were used here and none were evaluated for
this purpose. `yatrosci-ml-neural` is the obvious candidate to examine first if
Phase 2 proceeds.

## 10. Conclusion

Within the scope of a small MLP on a synthetic task, with N = 10 runs per
condition, a Lorenz-driven generator was **not distinguishable** from ChaCha8 in
final validation loss, validation accuracy or generalisation gap, and was
**three times slower**. The experiment is a pilot. It does not establish
equivalence, and it says nothing about behaviour at scale.

Sections 11 and 12, added later, extend that with two further probes: the
extracted stream carries no detectable topological signature of the attractor
that produced it, and the training trajectories it drives have the same fractal
dimension as those driven by the control. Neither changes the conclusion above;
both narrow the space in which a difference could still be hiding.

---

## 11. Phase 0.5: topological fingerprint of the generators

The Phase 0 battery sees only first and second order structure: a histogram and
a correlation. A chaotic source could in principle carry the geometry of its
attractor into the extracted stream and still pass both. This phase looks for
that geometry directly.

The question was suggested by two pieces of prior work that apply the same
family of tools to learned representations and to optimisation trajectories
rather than to generators: Birdal, Lou, Guibas and Simsekli, "Intrinsic
Dimension, Persistent Homology and Generalization in Neural Networks", NeurIPS
2021 (DOI 10.48550/arXiv.2111.13171), and the Embedding-Manifold-Compression
work, which uses the Grassberger-Procaccia correlation dimension and Lyapunov
exponents on BERT embeddings. Neither is an antecedent of the specific question
asked here, but both are why it seemed worth asking.

**H0.** The total finite persistence in dimension one of the extracted stream,
under a Takens delay embedding, is not distinguishable from that of ChaCha8
against an empirical null of uniform noise, at alpha = 0.05.

### Method

Embedding parameters were chosen by the standard diagnostics rather than by
hand, on the Lorenz stream, and then applied unchanged to every condition so
that the comparison is not confounded by different embeddings.

- Delay, by the first local minimum of average mutual information (Fraser and
  Swinney, DOI 10.1103/PhysRevA.33.1134): **4**. The curve is essentially flat,
  from 0.0049 to 0.0062 nats across delays 1 to 12, which is itself the expected
  signature of a stream carrying almost no self-information at any lag.
- Embedding dimension, by the false nearest neighbour criterion (Kennel, Brown
  and Abarbanel, DOI 10.1103/PhysRevA.45.3403): **5**. The fractions fall
  0.9935, 0.6855, 0.1815, 0.0135, 0.0000 for dimensions one to five.

A caution about that second number. False nearest neighbours is a diagnostic for
deterministic signals; on genuine noise it does not have a well defined answer,
and a fraction reaching zero at dimension five should not be read as evidence
that the extracted stream has a five dimensional attractor. It is used here only
to fix a common embedding for all conditions.

Persistent homology was computed with the `tda` crate, version 0.1.0, declared
as an ordinary dependency. Before use it was validated against a case with a
known answer: thirty points on a unit circle give a single dominant
one dimensional feature of persistence 1.523, against the 1.52 predicted by the
birth at the point spacing and the death near the square root of three, while a
filled region of the same size gives a largest feature of 0.063. That test is
part of the suite. One anomaly is recorded: the implementation reports a number
of unpaired one simplices as infinite bars even on contractible clouds, so this
phase uses total **finite** persistence, which is unaffected and is what the
protocol asks for.

**Persistence carries the units of the data.** Multiplying every coordinate by a
constant multiplies every bar by the same constant. Totals are therefore
comparable only across clouds on a common scale: the fractional part, the mixed
output and the ChaCha8 control all live on the unit interval and are mutually
comparable and comparable to the null, whereas the raw attractor states and the
coordinate scaled by 2^28 are not.

### Results

Clouds of 120 points, Rips filtration at twice the median pairwise distance.

The empirical null was built from 30 uniform clouds of the same size and
dimension: mean total H1 of 3.9191, standard deviation 0.5401, range 2.8355 to
4.9321. P-values are two sided against that null, with the add-one correction of
Phipson and Smyth (DOI 10.2202/1544-6115.1585), so the smallest value this
design can express is 1/31 = 0.0323.

| Measurement | Scale | Total finite H1 | p |
|---|---|---|---|
| Positive control, raw attractor states (x, y, z) | native, tens | 37.94 | 0.0323 |
| Extraction stage 1, coordinate scaled by 2^28 | 2^28 | 9 088 642 394.96 | 0.0323 |
| Extraction stage 2, fractional part | unit interval | 3.8669 | 0.9032 |
| Extraction stage 3, after SplitMix64 | unit interval | 3.8453 | 0.9032 |
| ChaCha8 control | unit interval | 4.5255 | 0.3226 |

The positive control behaves as required: the attractor itself carries
substantial one dimensional structure, so the measurement pipeline does detect
geometry when geometry is present. Stage 1 is not comparable to anything else in
the table, for the reason given above; it is reported only for completeness.

The informative comparison is the last three rows, which share a scale. **The
structure is destroyed at stage 2, when the fractional part is taken.** Stage 2
at 3.8669 and stage 3 at 3.8453 are within half a percent of each other, so
SplitMix64 removes essentially nothing that survived the fractional part: the
mixing step is doing no work here, which is consistent with its stated role as a
bijection that cannot add entropy. Both sit close to the ChaCha8 control at
4.5255, on the same side of it.

**H0 is not rejected for either generator.** The extracted Lorenz stream sits at
p = 0.9032, almost exactly at the centre of the null: its total persistence of
3.8453 is well inside the null's range of 2.8355 to 4.9321 and within a fifth of
a standard deviation of the null mean. The ChaCha8 control sits at p = 0.3226.
Neither is distinguishable from uniform noise by this measurement.

The positive control reaches p = 0.0323, which is the floor: no null cloud came
close, so the observation is as extreme as thirty resamples can express. That is
the desired behaviour and it is worth stating plainly what it does and does not
show. It confirms the pipeline detects attractor geometry when that geometry is
present, which is what makes the null results above informative rather than
merely uninformative. It does not establish a small p-value in any precise
sense, because the resolution of the test bottoms out there; a tighter statement
would need a larger null.

Stage 1 also reaches the floor, but for a reason with no topological content:
its coordinates are 2^28 times larger than the null's, so its persistence is too,
and the comparison measures the scale difference rather than any structure. It
is listed for completeness and should not be read as a finding.

### Interpretation

H0 stands: the extracted stream carries no detectable topological signature, and
neither does the control. Localising where the signature disappears is the
useful result here, and it was only visible because the stages were measured
separately rather than end to end. Whatever geometry the Lorenz orbit carries does not survive the
discarding of the high order digits, and the mixer that follows is, on this
evidence, decorative with respect to topology.

Limitations: a single embedding, a single cloud size, one summary statistic, and
a diagnostic used outside the regime it was designed for. This phase does not
show that no topological signature exists, only that this measurement did not
find one in the extracted stream.

## 12. Phase 3: fractal dimension of the training trajectory

The trajectory of an optimiser through parameter space is a point cloud: one
vector of 2178 parameters per epoch, sixty of them per run. Birdal et al. show
that the persistent homology dimension of such a cloud correlates with the
generalisation gap, on networks much larger than this one. This phase measures
that dimension for the twenty runs of Phase 1 and asks two questions: whether it
differs between generators, and whether it tracks the gap already recorded.

### Method

The estimator follows Birdal et al. For a finite set X, E^alpha(X) is the sum
over the edges of the minimum spanning tree of the edge lengths raised to alpha,
and the dimension follows from the growth of that quantity with sample size,
d = alpha / (1 - m), where m is the slope of log E against log n. Alpha is fixed
at one, as in the paper, so E is the total tree length. The zero dimensional
persistence of a Rips filtration is exactly the minimum spanning tree, so the
tree is computed directly.

Subsample sizes 15, 20, 30, 40, 50 and 60, twenty draws each, drawn
deterministically so the estimate is reproducible.

**Determinism was verified before measuring anything.** The runs were repeated
with per epoch snapshots and every one of the twenty reproduced the parameter
hash published in Phase 1 exactly. This check initially reported five failures;
the cause was that the guard compared the bit pattern of a loss value that had
been through JSON, and a decimal round trip can move the last unit in the last
place. The parameter hashes were identical throughout, which is the
authoritative signal, and the guard now compares the loss with a tolerance and
says why. The discrepancy was in the check, not in the experiment.

**The estimator was calibrated at the sample size actually used.** It recovers
1.975 for a uniform square, where the truth is two, and 1.039 for a uniform
line, where the truth is one, from clouds of sixty points, with fits of
r squared 0.9997 and 0.9634. Without that calibration a number produced from
sixty points could not be trusted, since the method was validated in the
literature on far larger clouds.

### Results

| | Lorenz | ChaCha8 |
|---|---|---|
| PH-dim, mean and sd | 2.2779 ± 0.0781 | 2.3100 ± 0.0731 |
| Range | 2.160 to 2.378 | 2.171 to 2.379 |

Shapiro-Wilk did not reject normality for either sample (p = 0.4169 and
p = 0.0510), so Welch's test applies: t = -0.9468, p = 0.3563,
Cohen's d = -0.4234. **H0 is not rejected.**

The power law that the estimator assumes holds very well on these trajectories:
the log-log fits have r squared between 0.9984 and 0.9999, mean 0.9994, with
slopes between 0.5370 and 0.5797.

Correlation with the generalisation gap measured in Phase 1, all twenty runs
pooled:

| Coefficient | Value | p |
|---|---|---|
| Pearson r | 0.2140 | 0.3651 |
| Spearman rho | 0.0541 | 0.8207 |

### Interpretation

Neither question produced a positive result. The two generators give
trajectories of indistinguishable fractal dimension, and the dimension does not
track the generalisation gap in this data.

The second point deserves care, because Birdal et al. report such a correlation.
This is not a contradiction of that work and must not be read as one. Their
result is established on networks orders of magnitude larger, over trajectories
sampled per iteration rather than per epoch, and across a range of
generalisation gaps far wider than the one here: the twenty runs in this
experiment span gaps from 0.0519 to 0.0807, a range so narrow that a correlation
would have little room to express itself. With twenty points and that spread,
this experiment has almost no power to detect the effect they describe. The
honest summary is that the present design cannot test their claim, not that
their claim fails.

The consistency of the estimates is itself worth noting: every one of the twenty
trajectories, under either generator, has a dimension near 2.3 with a standard
deviation under 0.08. Whatever the optimiser is doing in a 2178 dimensional
parameter space, it is confined to something of very low effective dimension,
and the source of its randomness does not change that.

Limitations: sixty points per trajectory, one architecture, one task, twenty
runs, and per epoch rather than per iteration sampling. The estimator's
calibration at this sample size is good on synthetic clouds of known dimension,
which is evidence that it behaves, not proof that it behaves on trajectories.
