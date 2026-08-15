# Chaos-driven pseudo-randomness against ChaCha8 in neural network training

Status: Phases 0, 0.5, 1, 3, 4, 5, 6, 7, 8, 9, 10 and 11 complete. Phase 2 not executed; see
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
- **Scale.** A two-layer MLP with 1218 parameters on a two-dimensional synthetic
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
vector of 1218 parameters per epoch, sixty of them per run. Birdal et al. show
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
deviation under 0.08. Whatever the optimiser is doing in a 1218 dimensional
parameter space, it is confined to something of very low effective dimension,
and the source of its randomness does not change that.

Limitations: sixty points per trajectory, one architecture, one task, twenty
runs, and per epoch rather than per iteration sampling. The estimator's
calibration at this sample size is good on synthetic clouds of known dimension,
which is evidence that it behaves, not proof that it behaves on trajectories.

## 13. Phase 4: an iterated function system driven by the chaos game

A third family of generator, chosen to be unlike the other two. Lorenz is a
continuous flow integrated in time; ChaCha8 is a block construction with no
geometry at all. This one is a discrete attractor whose native geometry is a
fractal of exactly known, non-integer dimension. That last property is the point
of including it: it supplies a reference value against which the measurement
pipeline can be calibrated, which sections 11 and 12 lacked.

Motivation recorded plainly: the Embedding-Manifold-Compression work exploits the
fact that learned embeddings lie on a low-dimensional fractal manifold. Section
11 established where a continuous attractor's structure is destroyed during
extraction. This phase asks the same question of a source whose geometry is
fractal by construction.

REF: [Barnsley, 1988] "Fractals Everywhere", Academic Press, ISBN
978-0-12-079062-3, for the chaos game and its convergence to the attractor.

REF: [Grassberger and Procaccia, 1983] "Characterization of Strange Attractors",
Physical Review Letters 50(5), pp. 346-349, DOI 10.1103/PhysRevLett.50.346, for
the correlation dimension.

### 4a. The generator and its calibration

The Sierpinski triangle by chaos game: three fixed vertices at (0, 0), (1, 0)
and (1/2, sqrt(3)/2), an equilateral triangle of unit side; a point starting at
the centroid; at each step a vertex is chosen and the point moves half way to it.

The randomness that chooses the vertex is a parameter rather than a fixed
choice, exposed as two variants, one driven by the Lorenz generator and one by
ChaCha8. Without that separation, any structure found could not be attributed to
the chaos game rather than inherited from whatever drives it.

Burn-in is 100 iterations, determined for this system rather than copied from
Lorenz. Each step halves the distance to the attractor, so an initial offset of
at most one falls below 2^-60 after sixty steps, far under the resolution of a
double. A test confirms the reasoning numerically: two games driven by identical
vertex choices from different starting points merge to within 1e-15.

Extraction keeps the principle of the Lorenz extractor and changes two things,
both forced by the geometry. Kept: harvest digits far below the scale of the
attractor's motion by scaling and taking the fractional part, then mix through
the SplitMix64 finaliser. Changed: the scale is 2^36 rather than 2^28, because
these coordinates live in [0, 1] rather than in the tens; and two coordinates are
mixed rather than three. Decimation of four iterations per output was added after
the battery showed a lag-one autocorrelation of 0.0101, just over the threshold,
which is a real residual correlation: consecutive chaos game points share most of
their position.

**The blocking calibration.** The correlation dimension of a cloud of 12000
attractor points, against the theoretical log(3)/log(2):

| Variant | Correlation dimension | Theoretical | Error | Fit r squared |
|---|---|---|---|---|
| IFS over Lorenz | 1.582401 | 1.5849625 | 0.0026 | 0.999827 |
| IFS over ChaCha8 | 1.579283 | 1.5849625 | 0.0057 | 0.999842 |

Both reproduce the theoretical value to within a third of a percent.

**The calibration caught a defect in the estimator, not in the generator.** An
earlier version placed its radii between the fifth percentile and the median of
pairwise distances, and returned 1.751 for a uniform square whose true dimension
is 2, an underestimate of twelve percent, with a fit quality of r squared 0.9997.
A good fit over the wrong region: near the median the correlation sum approaches
saturation at one, which flattens the slope. Moving the band to between the 0.1th
and the 10th percentile removed the bias, and the estimator then recovers 2 for a
square and 1 for a line, which is what the tests now assert. Without an exact
reference to check against, that bias would have been invisible and would have
been reported as a property of the attractor.

**Qualification.** All four generators now pass the Phase 0 battery over one
million variates:

| Generator | Chi-squared | p | Mean | Variance | Largest abs. autocorrelation |
|---|---|---|---|---|---|
| lorenz | 1059.556 | 0.0896 | 0.500009 | 0.083316 | 0.00236 |
| chacha8 | 1014.616 | 0.3585 | 0.500281 | 0.083191 | 0.00203 |
| ifs-lorenz | 926.856 | 0.9495 | 0.500518 | 0.083237 | 0.00189 |
| ifs-chacha8 | 1059.186 | 0.0909 | 0.500107 | 0.083444 | 0.00166 |

The chaos game driven by Lorenz gives the strongest uniformity of the four.

### 4b. Topological fingerprint

The protocol of section 11, applied unchanged, with embedding parameters
recomputed rather than inherited: average mutual information gives a delay of 5,
against the 4 found for the Lorenz stream, and the false neighbour criterion
gives dimension 5. Recomputing mattered for the delay.

| Measurement | Total finite H1 | p against the uniform null |
|---|---|---|
| Positive control, raw chaos game points | 0.4723 | 0.0323 |
| Stage 1, coordinate scaled by 2^36 | 224 010 742 004.26 | 0.0323 |
| Stage 2, fractional part | 3.4104 | not comparable, see below |
| Stage 3, IFS over Lorenz | 4.0686 | 0.8710 |
| Stage 3, IFS over ChaCha8 | 3.9426 | 0.9032 |

Null over 30 uniform clouds: mean 3.8978, standard deviation 0.4106.

**H0 is not rejected for either variant.** Both extracted streams sit near the
centre of the null, at p = 0.8710 and p = 0.9032. The fractal geometry does not
survive extraction any better than the continuous attractor's did in section 11.

Two cautions about the table, both about scale. Persistence carries the units of
the data, so the raw control at 0.4723 cannot be ranked against the Lorenz raw
control at 37.94 from section 11: the Sierpinski points live in the unit square
while the Lorenz attractor spans tens of units. The difference is the scale, not
the amount of structure. For the same reason stage 1, scaled by 2^36, is not
comparable to anything else here, and its p-value at the floor of 0.0323 measures
that scale difference rather than any geometry. The raw control also sits in two
dimensions against a five-dimensional null, so its p-value is not a like-for-like
comparison either; the meaningful rows are the two stage-3 streams, which share
both the embedding and the scale of the null.

### 4c. PH-dim across four conditions

The protocol of section 12, with ten runs per variant at seeds 1000 to 1009 and
the same estimator.

| Condition | PH-dim | Shapiro-Wilk W | p |
|---|---|---|---|
| lorenz | 2.2779 ± 0.0781 | 0.9268 | 0.4169 |
| chacha8 | 2.3100 ± 0.0731 | 0.8452 | 0.0510 |
| ifs-lorenz | 2.3066 ± 0.0737 | 0.9797 | 0.9637 |
| ifs-chacha8 | 2.2980 ± 0.0780 | 0.9450 | 0.6099 |

Shapiro-Wilk rejected none of the four, so one-way analysis of variance applies:
F(3, 36) = 0.3595, p = 0.7826. **H0 is not rejected.** No pairwise comparisons
were run: performing them after a non-significant omnibus test would inflate the
error rate for no gain. Had the omnibus been significant, the pairwise tests were
prepared with Holm correction.

Note that the ChaCha8 sample sits at p = 0.0510, a hair above the threshold that
would have sent the whole comparison to Kruskal-Wallis.

Correlation of PH-dim with the generalisation gap, now over 40 runs rather than
20: Pearson r = 0.1954, p = 0.2269; Spearman rho = 0.0884, p = 0.5877. Still
absent, with double the sample.

### Interpretation

The falsifiable question of 4b was whether a natively fractal geometry, with a
dimension far from any integer, would resist extraction better than the Lorenz
attractor's, whose correlation dimension is close to 2. It does not. Both are
destroyed, and the stage-by-stage measurement puts the loss in the same place:
taking the fractional part.

Adding a third family changed nothing in 4c either. Four conditions, two of them
built on a completely different mathematical object, give training trajectories
of indistinguishable fractal dimension.

The most useful outcome of this phase is methodological rather than empirical.
Having a source with an exactly known dimension exposed a twelve percent bias in
the measurement instrument that no relative control would have revealed. That
bias would have propagated silently into any dimension reported here.

Limitations: one fractal, one dimension of embedding, thirty null resamples, ten
runs per condition, and the same single architecture and task as every previous
phase.

## 14. Phase 5: holographic (HRR) against non-holographic (MAP) binding of training trajectories

This phase does not add a generator. It asks whether the specific mathematics of
holographic binding, circular convolution in the frequency domain, preserves the
information in a training trajectory better than an equivalent binding that is
not holographic, when the stored trace is partially destroyed.

REF: [Plate, 1995] "Holographic Reduced Representations", IEEE Transactions on
Neural Networks 6(3), pp. 623-641, DOI 10.1109/72.377968. Binding by circular
convolution, computed through the Fourier transform, with unbinding by circular
correlation against the approximate inverse.

REF: [Gayler, 1998] "Multiplicative Binding, Representation Operators and
Analogy", https://arxiv.org/abs/cs/0412059. The MAP architecture: binding by
element-wise multiplication, bundling by addition, no transform.

The transform comes from `yatrosci-fft`, which delegates to a mixed-radix
implementation for lengths that are not powers of two. That is required here:
the vectors are 1218 wide, which factors as 2 * 3 * 7 * 29, so a radix-2
transform would pad, and padding turns circular convolution into linear
convolution, which breaks unbinding silently.

**A correction to section 12.** That section stated the network has 2178
parameters. It has 1218: 96 in the first layer, 1056 in the second, 66 in the
output. The figure has been corrected in place. The error was introduced when
section 12 was written and did not affect any measurement, since the code always
took the width from the data; it affected only the prose.

### 5a. Calibration, and two implementation defects it caught

Both schemes were first exercised in the regime they are analysed for:
independent items drawn from N(0, 1/d), bundled at increasing load.

| Bundled items | HRR fidelity | MAP fidelity |
|---|---|---|
| 5 | 0.4399 ± 0.0229 | 0.4501 ± 0.0187 |
| 15 | 0.2664 ± 0.0216 | 0.2620 ± 0.0320 |
| 30 | 0.1810 ± 0.0239 | 0.1909 ± 0.0286 |
| 60 | 0.1311 ± 0.0310 | 0.1293 ± 0.0392 |

The two are indistinguishable in the ideal regime, and both degrade with load as
crosstalk grows, which is the expected behaviour and the reason the comparison
in 5c is meaningful rather than a foregone conclusion.

Reaching that table required fixing two defects that the calibration exposed and
that would otherwise have produced a confident and wrong answer.

**The first was an unfair comparison rather than a coding error.** With Gaussian
keys, HRR unbinds by the involution, which is only an approximate inverse: the
round trip multiplies each frequency by |F(k)|^2, an exponential variable of
mean one rather than the constant one. The retrieval similarity therefore
converges to E[W]/sqrt(E[W^2]) with W exponential, that is 1/sqrt(2) = 0.7071,
and does not improve with width. Measured at 0.7272, 0.7196, 0.7051, 0.7037 and
0.7062 for widths 64, 256, 1024, 2178 and 8192: flat, and on the predicted
constant. MAP meanwhile unbinds by division, its exact inverse. Comparing an
approximate inverse against an exact one would have measured that asymmetry
rather than the property under study. Both schemes now use the key distribution
they are defined over, unitary for HRR and bipolar for MAP, so each unbinds
exactly in the noiseless case.

**The second was a genuine implementation error.** MAP with Gaussian keys and
division scored 0.0200, 0.0084, -0.0027 and 0.0025 across the four loads, which
is to say it retrieved nothing. The cause is that the crosstalk terms then carry
ratios of Gaussian variables, which are Cauchy distributed and have no finite
variance, so the noise has no scale to be small relative to. Bipolar keys, which
is what MAP is defined over, make multiplication its own exact inverse and bound
the crosstalk. The blocking gate did its job: without it, this phase would have
reported that holographic binding beats element-wise binding by an enormous
margin, and the finding would have been an artefact of using the wrong key
distribution for one of the two schemes.

### 5b. Real trajectories

Section 12 established that these trajectories have an effective dimension near
2.3 against 1218 nominal, and that consecutive epochs are strongly correlated
because the optimiser moves smoothly rather than jumping. That violates the
near-orthogonality premise under which both schemes are analysed, which is what
makes their behaviour here an open question.

Sixty epochs bundled into one trace, retrieval measured at epochs 1, 15, 30, 45
and 60, over the twenty runs of Phase 3.

| Scheme | Fidelity on real trajectories | Synthetic at 60 items |
|---|---|---|
| HRR | 0.1565 ± 0.0197 | 0.1311 ± 0.0310 |
| MAP | 0.1284 ± 0.0309 | 0.1293 ± 0.0392 |

The anticipated collapse did not happen. Fidelity on real trajectories is
comparable to the synthetic case at the same load, slightly higher for HRR and
essentially equal for MAP. Whatever the correlation between consecutive epochs
does to the trace, it does not cost more than independent items of the same
number. That is a null result against the expectation stated above, and it is
reported as such rather than quietly dropped.

### 5c. Degradation under corruption

The trace was damaged by erasing a growing fraction of its components, then
retrieval was attempted. Erasure was chosen over additive noise as the primary
model for two reasons: it represents loss of stored content, which is the
failure mode a distributed representation is supposed to tolerate, and it has no
free scale parameter, so the two schemes cannot be separated by an arbitrary
choice of noise magnitude.

Mean fidelity across twenty runs and five probe epochs:

| Erased | 10% | 30% | 50% | 70% | 90% |
|---|---|---|---|---|---|
| HRR | 0.1477 | 0.1304 | 0.1106 | 0.0848 | 0.0487 |
| MAP | 0.1220 | 0.1046 | 0.0876 | 0.0603 | 0.0273 |

Area under the degradation curve, one value per run, the pre-registered
statistic:

| Scheme | AUC |
|---|---|
| HRR | 0.084777 ± 0.012719 |
| MAP | 0.065439 ± 0.018957 |

Shapiro-Wilk did not reject normality for either sample (p = 0.0569 and
p = 0.8026), so Welch's test applies: t = 3.7884, p = 0.000606,
Cohen's d = 1.1980. **H0 is rejected** on the pre-registered statistic, in
favour of HRR.

**That result needs one further step before it can be read as a claim about
holographic binding.** HRR starts higher at zero corruption, 0.1565 against
0.1284, so part of the area under its curve is that offset rather than a slower
decay. Normalising each run's curve by its own uncorrupted fidelity separates
the two, and the picture changes:

| Scheme | Normalised AUC | Retention at 90% erased |
|---|---|---|
| HRR | 0.5413 ± 0.0356 | 0.3131 |
| MAP | 0.5094 ± 0.0769 | 0.2009 |

Welch on the normalised statistic: t = 1.6871, df = 26.77, p = 0.1032,
Cohen's d = 0.5335. **Not significant.** This analysis is post hoc: it was
performed after seeing that the two schemes differed at zero corruption, and it
is reported as exploratory rather than confirmatory.

### Interpretation

On the pre-registered comparison, holographic binding retains more of the
trajectory than element-wise binding under erasure, with a large effect. On the
follow-up that removes the head start, the difference in the shape of the decay
alone does not reach significance at twenty runs per scheme, though it points
the same way and the retention at ninety percent erasure differs substantially,
0.31 against 0.20.

The honest summary is narrower than the headline. HRR is better here, and most
of the measured advantage is that it retrieves better to begin with on these
trajectories, not that it decays more gracefully. Whether the residual
difference in decay is real would need more runs than this design has.

Nothing here supports general claims about holographic representations. It is
one binding task, on one kind of data, at one width, with one corruption model,
comparing two specific schemes each given the key distribution it was defined
over.

Limitations: twenty runs per scheme; a single width; a single corruption model,
with additive noise implemented but not reported here; probe epochs at five
fixed positions rather than all sixty; and trajectories from one architecture on
one task.

## 15. Phase 6: spectrum of the superposed HRR + IFS + TDA operator

Three matrices are derived from the same sixty per-epoch weight vectors of a
run, normalised by Frobenius norm, averaged into one operator, and its spectrum
compared against a null.

### The prototype, and why the null is not a Gaussian ensemble

A synthetic prototype run before this phase, at size 200, superposed a circulant
matrix, a block matrix of contractive affine maps, and a Euclidean distance
matrix over a Gaussian cloud. Against a superposition of three independent
Gaussian ensembles it showed a spectral gap of 0.246 against 0.003, with a
Kolmogorov-Smirnov test against the semicircle at p = 0.0003.

Most of that was an artefact. A distance matrix is non-negative by construction,
and any non-negative matrix carries a dominant eigenvalue of Perron-Frobenius
type that has nothing to do with geometry, topology or fractals. Substituting
the absolute value of a generic Gaussian matrix for the real distance matrix,
which shares the non-negativity and nothing else, reproduced most of the effect:
a gap of 0.193 against the original 0.246.

REF: [Perron, 1907] "Zur Theorie der Matrices", Mathematische Annalen 64(2),
pp. 248-263, DOI 10.1007/BF01449896.

REF: [Wigner, 1958] "On the Distribution of the Roots of Certain Symmetric
Matrices", Annals of Mathematics 67(2), pp. 325-327, DOI 10.2307/1970008.

The null used throughout this phase is therefore the corrected one: the same
circulant and affine terms, with the distance matrix replaced by the absolute
value of a Gaussian matrix with a vanishing diagonal. That isolates what a real
Euclidean distance matrix contributes beyond merely being non-negative, which is
the only question this design can answer. Reporting against a plain Gaussian
null would restate the artefact.

### 6a. Calibration at the real size

The prototype ran at size 200; the real matrices are 60 by 60, and the pattern
had to be shown to survive that reduction rather than assumed to.

| | Spectral gap | KS p against the semicircle |
|---|---|---|
| Real superposition | 0.6184 | 0.0273 |
| Naive null, three Gaussian ensembles | 0.0054 | 1.0000 |
| Corrected null, absolute Gaussian | 0.5217 | 0.9985 |

The real operator exceeds the naive null by a factor of 114.6 and the corrected
null by a factor of 1.19. The prototype's corresponding factors were 82 and
1.27. The pattern reproduces at a third of the size: an enormous effect against
the wrong null, and a small one against the right null.

The spectral tools were themselves calibrated first. A Gaussian ensemble is not
rejected against the law it obeys, and a non-negative matrix shows a leading
eigenvalue more than three times larger than its plain counterpart. Both are
tests in the suite. Without the first, any rejection reported here would be
uninterpretable.

### 6b. The three matrices

**D_TDA** is the matrix of pairwise Euclidean distances between the sixty epoch
vectors. It is checked to satisfy the metric axioms, including the triangle
inequality, which is the property the corrected null deliberately does not have.

**C_HRR** holds, at entry (i, j), the largest absolute component of the circular
correlation between the weight vectors of epochs i and j, computed through the
same transform used for unbinding in section 14. Reducing the full correlation
vector to its peak is the standard read-out in holographic representations:
retrieval asks where the correlation peaks and how high. Taking the value at lag
zero instead would collapse to an ordinary inner product and discard exactly the
shift structure that distinguishes circular correlation from a dot product.

**A_IFS** tiles twenty blocks along the diagonal, cycling through the three
affine maps of the Sierpinski chaos game in homogeneous form so that each block
carries both the contraction and the translation, then symmetrises the result.

**A_IFS is identical for all forty runs, and that is a finding rather than a
shortcut.** The design anticipated an asymmetry between the conditions that
involve a chaos game and those that do not, and expected the second group to use
a fixed reference operator. No such asymmetry arises, because the three maps are
fixed by the triangle: what the randomness selects at each step is which map to
apply, not what the maps are. The operator that drove an `ifs-lorenz` run is the
same one that drove an `ifs-chacha8` run and the same one used as a reference
for `lorenz` and `chacha8`. The consequence cuts the other way, though: this
term carries no run-level information at all and contributes a constant to every
superposition. Whatever the spectrum discriminates, it is not discriminating on
the strength of the IFS term.

### 6c. Spectral gap against the corrected null

Forty paired comparisons, each sharing its circulant and affine terms with its
own null by construction, which is why the test is paired.

| | Spectral gap |
|---|---|
| Real | 0.597332 ± 0.002576 |
| Corrected null | 0.546146 ± 0.003078 |
| Paired difference | 0.051187 ± 0.003522 |

Shapiro-Wilk on the differences did not reject normality (W = 0.9666,
p = 0.2784), so the paired t-test applies: t = 91.9249, df = 39, p below the
resolution of the arithmetic. **H0 is rejected.**

That p-value should not be read as a measure of importance. The paired design
shares two of three terms between each pair, so the differences have a standard
deviation of 0.0035 against a mean of 0.0512, and a t of ninety follows from
that consistency rather than from a large effect. In magnitude the real operator
exceeds its corrected null by about nine percent, against the eleven thousand
percent by which it exceeds the naive one. The artefact accounts for the great
majority of what the prototype found; what survives the correction is real,
highly consistent, and modest.

### 6d. Relation to generalisation

The question that motivated the phase: does the composite spectrum predict the
generalisation gap where the persistent-homology dimension of section 13 did not,
where it gave r = 0.195 at p = 0.227?

| Predictor | Pearson r | p | Spearman rho | p |
|---|---|---|---|---|
| Spectral gap of the real operator | 0.1608 | 0.3216 | 0.1480 | 0.3620 |
| Difference from the corrected null | 0.1927 | 0.2336 | 0.1529 | 0.3462 |

**No.** Neither correlates. The second row is worth a second look: r = 0.1927
against the 0.195 that PH-dim produced on the same forty runs, which is the same
value to within rounding. Building a composite operator out of three different
mathematical objects and extracting its leading spectral structure recovered
exactly as much about generalisation as the simpler measure did, which in both
cases is nothing detectable at this sample size.

### Interpretation

Two results, of unequal interest.

The one that survives scrutiny is narrow: a real Euclidean distance matrix
raises the spectral gap of the superposition about nine percent above what a
merely non-negative matrix does, consistently enough across forty runs to be
unmistakable. The most plausible reading is that the triangle inequality and the
embeddability that a genuine distance matrix carries add structure that
non-negativity alone does not. This phase does not test that reading; it only
excludes the non-negativity explanation.

The one that motivated the phase failed. The composite spectrum does not predict
generalisation, and does not improve on the simpler measure it was meant to
better.

The methodological lesson repeats the one from section 13. The headline of the
prototype, a hundredfold gap against a Gaussian null, was almost entirely an
artefact of a property so basic that it has a theorem named after it. It took a
control specifically designed to share that property and nothing else to find
out how much was left, and the answer was under a tenth of the original.

Limitations: forty runs from one architecture on one task; a single matrix size,
fixed by the number of epochs; one reduction of circular correlation to a scalar
among several defensible choices; an IFS term that is constant across all runs
and therefore contributes nothing to any contrast; and a null that controls for
non-negativity but not for other properties a distance matrix has, so the
residual nine percent is not attributed to any specific one of them.

## 16. Phase 7: formal applicability of Pesin-type formulas, by Horn resolution

### Isolation of the external repository

The `kirs` repository is consumed as a read-only reference, pinned at commit
`18a7276`. No file inside that checkout was edited, and none of its source is
copied into this project: the new crate `crates/kirs-pilot` depends on
`pirs-kirs` and `kirs-lab` by path. A change of pinned version would be a change
to this repository's manifest and never to that one.

Two checks establish that the coupling is clean rather than merely intended. The
fifteen tests of `pirs-kirs` were run before anything was written and again
afterwards, passing both times untouched, and `git status` in that checkout is
empty with its HEAD unmoved. The only file in this repository with `kirs` in its
path is the pilot crate itself.

### What this phase does and does not do

It computes no entropy, no Lyapunov exponent and no dimension. Those would be
numerical work of a different kind and are not attempted here.

What it does is ask a question the earlier phases never asked: given what is
actually known about each generator, is the machinery whose outputs sections 12,
13 and 15 measured formally applicable to it at all. The answer is a
classification, and it is produced by resolution over declared facts rather than
by assertion.

REF: [Pesin, 1977] "Characteristic Lyapunov exponents and smooth ergodic
theory", Russian Mathematical Surveys 32(4), pp. 55-114,
DOI 10.1070/RM1977v032n04ABEH001639. The classical formula requires a
diffeomorphism and an invariant measure absolutely continuous with respect to
Lebesgue.

REF: [Liu and Qian, 1995] "Smooth Ergodic Theory of Random Dynamical Systems",
Lecture Notes in Mathematics 1606, Springer, DOI 10.1007/BFb0094308, and
[Liu, 1998] "Random perturbations of Axiom A basic sets", Journal of Statistical
Physics 90, pp. 467-490, DOI 10.1023/A:1023280407906. The random extension
covers non-invertible systems that the classical statement excludes.

### 7a. The facts, and the ones that could not be declared

Every fact asserted carries its warrant. Properties with no result behind them
are simply not declared, and that absence is the honest answer rather than a
gap to be filled by assumption.

Declared from construction, needing no citation: the Lorenz flow is invertible
in time by uniqueness of solutions for an autonomous ODE; the ChaCha core is a
bijection on its state, being built from additions, rotations and exclusive-ors;
and the chaos game is not injective, since it applies one of three contractive
maps and keeps only the image, so two points can share a successor.

Declared from the literature: the Lorenz attractor supports an SRB measure,
absolutely continuous along unstable manifolds, established rigorously by
Tucker, "A Rigorous ODE Solver and Smale's 14th Problem", Foundations of
Computational Mathematics 2 (2002), pp. 53-117, DOI 10.1007/s002080010018.

**Not declared, and why.** No absolute-continuity fact is asserted for ChaCha8:
this project knows of no such result, and the question is arguably not well
posed for a finite-state permutation. None is asserted for the two chaos-game
families either, but for the opposite reason. Their attractor carries a
self-similar measure on a set of Hausdorff dimension log(3)/log(2), which has
zero Lebesgue measure in the plane, so that measure is singular rather than
absolutely continuous, as section 13 measured directly. Stating that negative
would require negation, which this engine does not have, so the fact is absent.
The absence therefore means "unknown" in one case and "known to be false" in the
other, and the engine cannot tell them apart. That is a real limitation of the
encoding, not of the mathematics.

### 7b. The rules, and the one that could not be written

Two rules are expressible and were written as clauses: the classical formula
applies to a generator that is invertible and has an absolutely continuous
invariant measure; the random extension applies to a generator that is not
invertible.

The third category, neither formula applying, is **not** a clause. The engine
accepts Horn clauses only, with no negation, as its own parser states: "No
operators, no cut — Horn clauses only". A rule of the form
`sin_formula(G) :- generador(G), not(...), not(...)` cannot be expressed and
would not parse. It was tempting to reshape the question so the design looked
more elegant than the engine allows; instead the two positive rules are queried
separately and the classification is made in Rust outside the engine. A test
asserts that the engine really does not resolve a negated goal, so that if a
future version gained negation, the test would fail and the logic could move
back inside the program where it belongs.

Every query runs through the bounded entry point with a budget of ten thousand
steps, and exhaustion is recorded per generator. No query exhausted its budget,
which matters because an exhausted query returns no answers and would otherwise
be indistinguishable from a genuine negative.

### 7c. Result

| Generator | Classical (Pesin 1977) | Random (Liu) | Neither | Budget exhausted |
|---|---|---|---|---|
| lorenz | yes | no | no | no |
| chacha8 | no | no | **yes** | no |
| ifs_lorenz | no | yes | no | no |
| ifs_chacha8 | no | yes | no | no |

### A retroactive limitation on sections 13 and 15

Sections 13 and 15 computed a persistent-homology dimension for all four
families and compared them with analysis of variance, treating them uniformly.
This phase shows they are not uniform in the relevant sense. One family has the
classical hypotheses satisfied, two fall under the random extension, and one,
ChaCha8, satisfies neither.

That last is the uncomfortable one, because ChaCha8 is the control against which
the whole project measures. It is a finite-state permutation with no attractor,
no invariant measure in the sense these theorems require, and no reason to
expect Pesin-type machinery to describe it at all.

This does not invalidate anything already reported. The PH-dim values are
measurements of the geometry of a point cloud in parameter space, and a point
cloud has a fractal dimension whether or not any ergodic theorem applies to what
produced it. The estimator was calibrated on synthetic clouds of known dimension
and reproduces them. What changes is the interpretation: those numbers were
never backed by a theoretical guarantee that the machinery applied to the
generators in the first place, and sections 13 and 15 did not say so because the
question had not been asked. It is being said now.

The comparison across four conditions remains legitimate as an empirical one.
Reading it as a comparison of dynamical invariants would not be.

### Limitations

This is a verification of formal applicability, not a computation of the
quantities themselves. Nothing here says what the entropy or the exponents are,
only which theorem could in principle be invoked.

The knowledge base is small and hand-authored, so it verifies the consistency of
what was declared and cannot discover a hypothesis nobody thought to encode.

Absence of a fact is ambiguous in this encoding, meaning "no known result" for
ChaCha8 and "known to be false" for the chaos-game families, and the engine
treats both identically for want of negation.

The random extension carries its own absolute-continuity condition on the random
measure, which is a hypothesis of Liu's theorem and is not verified here for
either chaos-game family. The row reading "yes" under the random extension
should therefore be read as "the invertibility obstruction does not apply",
which is weaker than "the theorem's hypotheses are met".

And the classification of the third category happens outside the engine, so it
is a claim about what the engine failed to prove rather than a proof of a
negative.

## 17. Phase 8: reservoir computing with the validated generators as fixed reservoir weights (not SGD)

This phase is a change of paradigm, not an eighth attempt at the question the
first seven closed. Phases 1 through 7 asked whether the source of randomness
matters inside stochastic gradient descent and answered no, from converging
directions: no empirical effect on learning, on trajectory dimension, on the
four-condition comparison or on the superposed spectrum, and no formal channel
at all for one of the generators. The reason that question kept coming back
negative is structural. Under backpropagation the drawn weights are a starting
point, and training walks away from them; whatever geometry the generator wrote
into the initial matrix is overwritten by the first few thousand updates.

Reservoir computing removes that escape. The recurrent dynamics are drawn once
and never trained, and only a linear readout is fitted, in closed form by ridge
regression. The drawn weights are not an initial condition here; they are the
computation, applied unchanged at every timestep for the life of the network. If
the structure of a generator's stream can matter anywhere in machine learning,
this is a paradigm where it has somewhere to matter.

REF: [Jaeger, 2001] "The 'echo state' approach to analysing and training
recurrent neural networks", GMD Report 148, German National Research Center for
Information Technology.
REF: [Maass, Natschlaeger and Markram, 2002] "Real-time computing without stable
states", Neural Computation 14(11), pp. 2531-2560,
DOI 10.1162/089976602760407955.
REF: [Jaeger and Haas, 2004] "Harnessing nonlinearity: predicting chaotic
systems and saving energy in wireless communication", Science 304(5667),
pp. 78-80, DOI 10.1126/science.1091277.

The architecture is the canonical one: `x(t+1) = tanh(W_res x(t) + W_in u(t))`
with a linear readout, a reservoir of 100 units, spectral radius 0.9, input
scaling 0.1, dense recurrent matrix, and a washout of 1000 steps. The matrix is
dense rather than sparse, and that is a decision worth stating: sparsity would
be a second variable, and it would also decide how much of the generator's
stream ever reached the matrix. Dense means every one of the ten thousand
entries comes from the stream in order, which gives whatever structure the
generator carries the largest possible surface. The input matrix is drawn from
the reference source in all five conditions, so the recurrent matrix is the only
thing that varies.

### 8a. The canonical reservoir, before any generator is used

Three predictions of the theory were checked first, as a blocking gate, on a
reservoir built entirely from the reference generator.

| Spectral radius | Memory capacity | MC/N |
|---|---|---|
| 0.50 | 14.639 | 0.146 |
| 0.90 | 32.642 | 0.326 |
| 0.99 | 36.169 | 0.362 |

Memory capacity stays well under the reservoir size, which is the theoretical
ceiling, and rises as the spectral radius approaches one, which is what slower
forgetting predicts. The ceiling check is not decorative: measuring the
correlations in sample rather than on held-out data inflates each of the two
hundred delays by roughly `(n+1)/T`, and the easiest way to produce a capacity
above the ceiling is exactly that mistake. The readouts here are evaluated on a
segment they never saw.

On NARMA-10, using the published recursion rather than any task invented here,
the canonical reservoir reaches a test NRMSE of 0.3527, inside the band the
reservoir literature reports for a hundred-unit reservoir. All three gates
passed, and only then were the generators introduced.

### 8b. The echo state property, for all five conditions

The property was verified numerically rather than inferred from the spectral
radius, which is a rule of thumb and neither necessary nor sufficient. Two
different initial states were driven by the same input and their separation
measured after two thousand steps.

Every condition holds it, in every one of the twenty instances, with a worst
final separation of 4.5e-16 across the whole experiment: the trajectories merge
to the last bit of double precision. No condition had to be excluded, so the
comparison that follows runs on all five.

The spectral radius of the raw fill, before rescaling, is worth recording as a
property of the sources in its own right. It sits at 6.07 for the reference and
between 6.03 and 6.13 for the four generators, each with a standard deviation
near 0.2, so all five agree within their own spread. The circular law puts the
asymptotic value at `sqrt(N/3) = 5.77` for entries uniform on this range, and
the measured excess is the finite-size behaviour expected at N = 100. No
generator produced a matrix with an anomalous spectrum.

### 8c. Comparison

Twenty reservoir instances per condition, each condition seeing an identical
driving input at each instance. Shapiro-Wilk passed on all five samples for both
metrics, so Welch was used throughout, with Holm across the four comparisons.

Memory capacity, omnibus one-way ANOVA F = 0.7337, p = 0.5712:

| Condition | Mean MC | Welch p | Holm p | d |
|---|---|---|---|---|
| standard-iid | 32.559 | | | |
| lorenz | 32.583 | 0.9660 | 0.9660 | -0.014 |
| chacha8 | 31.683 | 0.2373 | 0.9490 | 0.380 |
| ifs-lorenz | 31.763 | 0.3016 | 0.9490 | 0.332 |
| ifs-chacha8 | 31.904 | 0.3159 | 0.9490 | 0.321 |

NARMA-10 test NRMSE, omnibus one-way ANOVA F = 0.0390, p = 0.9971:

| Condition | Mean NRMSE | Welch p | Holm p | d |
|---|---|---|---|---|
| standard-iid | 0.3583 | | | |
| lorenz | 0.3567 | 0.8293 | 1.0000 | 0.069 |
| chacha8 | 0.3583 | 0.9991 | 1.0000 | -0.000 |
| ifs-lorenz | 0.3571 | 0.8807 | 1.0000 | 0.048 |
| ifs-chacha8 | 0.3593 | 0.9000 | 1.0000 | -0.040 |

H0 is not rejected on either metric, by any comparison, before or after
correction.

### What makes this null informative, and what does not

A p-value above a threshold is weak evidence on its own, so two things are
reported alongside it.

The first is the design's resolution. With twenty instances per condition this
comparison can detect a standardised effect of about 0.89 at eighty percent
power, and about 1.06 at the most stringent level the Holm correction imposes.
Effects smaller than that would not have been seen. The figure comes from the
normal approximation to the two-sample t-test, which errs in the direction that
flatters the design by roughly three percent, so the true resolution floor is a
little worse than stated. This experiment therefore rules out large effects, not
small ones.

The second is more useful. The chacha8 condition differs from the reference
baseline in nothing but the ChaCha round count, eight against twelve, so it is a
negative control: whatever it registers is what this design produces when
nothing of substance differs. On memory capacity it registered d = 0.380, the
**largest** of the four effects, larger than either chaos-game family and far
larger than Lorenz at d = -0.014. An effect of that size arose here from a
change that cannot matter. That is a within-experiment demonstration of the
noise scale, and it places all four generators at or below it.

On NARMA-10 the control happened to land at d = -0.000, the smallest of the
four. That is one realisation and calibrates nothing; it is reported so the two
metrics are not read as if the control behaved the same way in both.

### Limitations

One architecture, one reservoir size, one spectral radius, one input scaling.
Reservoir performance is known to depend on all of these, and the possibility
that a difference exists at some other operating point is untested.

Two tasks, both standard and both chosen for comparability with the literature,
neither chosen because a chaotic reservoir would be expected to do well on it.
A task designed to reward the specific structure of these generators was not
attempted, and would be a different kind of experiment.

The spectral radius is equalised across conditions and everything else about the
matrix, including its density of large entries and any correlation between
entries, is left as the generator produced it. That is the intended design, but
it means "the source does not matter" is established only after that one
normalisation. A comparison without it would be a comparison of spectral radii.

Twenty instances per condition, with the resolution floor stated above. Failing
to detect a difference is not the same as establishing equivalence, which is the
same caveat that has applied since Phase 1 and applies here unchanged.

## 18. Phase 9: predictive coding with precision modulated by the validated generators

**Scope first.** This is the perception and learning side only. The network
builds a hierarchy of value nodes and error nodes, each level predicting the one
below, relaxes the value nodes until they settle on an explanation of the input,
and then changes each synapse by the product of the two activities it already
connects. There is no action on the world, no policy selection and no expected
free energy. The full active inference framework is not implemented here and
nothing below should be read as implementing it. What is implemented is the
tractable subset.

REF: [Whittington and Bogacz, 2017] "An Approximation of the Error
Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian
Synaptic Plasticity", Neural Computation 29(5), pp. 1229-1262,
DOI 10.1162/NECO_a_00949. Title, journal, volume, issue, pages and authors were
checked against CrossRef rather than carried over from the specification.
REF: [Friston, 2013] "Life as we know it", Journal of The Royal Society
Interface 10(86), 20130475, DOI 10.1098/rsif.2013.0475. Cited for why precision
is the principled place to inject a modulating signal rather than an arbitrary
hook, not for anything implemented here. Also verified against CrossRef.

The architecture is three weight matrices over layers of 2, 32, 32 and 2 nodes,
the same widths as the Phase 1 network. The nonlinearity is `tanh`, chosen over a
rectifier because the inference loop needs the derivative at the current value of
every node and a rectifier's is zero over half its domain, where the relaxation
would stall. Sixteen relaxation steps per sample at an inference rate of 0.2,
then a weight step of 0.05, over thirty epochs in batches of 32. The inference
rate and the learning rate are separate quantities throughout.

The readout metric is the softmax cross-entropy Phase 1 reports, so the numbers
are the same quantity on the same data. The training objective is not the same:
predictive coding minimises squared prediction error, and Phase 1 trained on
cross-entropy with Adam. Values here are therefore comparable across the
conditions of this phase, and should not be set directly against Phase 1's.

### 9a. Does the local update approximate the backpropagation gradient?

The claim in the literature is that it does, approximately, once the inference
has settled. That is checkable rather than assumable, and checking it first is
what makes the rest of the phase meaningful: comparing generators on a network
that does not approximate what it is supposed to would measure nothing.

The exact backpropagation gradient of the squared output error is computed on
the same weights, the same input and the same target, and compared with the
accumulated local update. The gradient implementation is itself checked against
central finite differences, because a reference that is wrong certifies nothing.

| Relaxation steps | Correlation | Cosine |
|---|---|---|
| 1 | 0.7539 | 0.7551 |
| 2 | 0.7944 | 0.7958 |
| 4 | 0.8494 | 0.8507 |
| 8 | 0.8986 | 0.8996 |
| 16 | 0.9228 | 0.9237 |
| 32 | 0.9241 | 0.9251 |
| 64 | 0.9190 | 0.9200 |
| 128 | 0.9161 | 0.9171 |

Agreement climbs steeply from a single step to about sixteen, then flattens and
drifts very slightly down. The rise is the expected behaviour and the gate is
built on it. The small decline past the plateau is worth stating rather than
cropping: at that depth the relaxation has redistributed error into the interior
levels, and the resulting update is a step on the network's own energy rather
than on the output loss alone, so a little of the agreement with pure
backpropagation is given back. The approximation is a property of the settled
regime, not something that improves without limit.

The gate is passed. No exact match was expected, and one would have suggested the
inference loop had been short-circuited.

### 9b. Precision modulated by each generator

Precision enters as `pi_l`, the weight on each level's squared prediction error,
and is redrawn at every step of the inference loop rather than once per sample or
once per epoch, because that is where the theory says the weighting acts.

The map from a uniform variate to a precision is
`g(u) = 2 / (1 + exp(-2(2u - 1)))`. It is strictly positive, so the inference
stays a descent on a real energy; bounded above by two, so no step can be scaled
arbitrarily; and it sends the median of its input to exactly one. That last
property is what keeps the comparison fair, and it is verified as a test: a
modulation whose mean precision drifted from one would change the effective step
size, and any difference found would have been a difference of learning rate in a
theoretical costume.

Everything except precision is identical across conditions for a given seed: the
initial weights, the order of presentation, and the data. Six conditions, twenty
seeds each.

| Condition | Val loss | Accuracy | Gap |
|---|---|---|---|
| constant | 0.4607 | 0.8451 | 0.0174 |
| lorenz | 0.4612 | 0.8447 | 0.0173 |
| chacha8 | 0.4615 | 0.8439 | 0.0172 |
| ifs-lorenz | 0.4615 | 0.8449 | 0.0174 |
| ifs-chacha8 | 0.4611 | 0.8437 | 0.0174 |
| chacha12-control | 0.4613 | 0.8443 | 0.0175 |

Validation loss, omnibus Kruskal-Wallis H = 0.5549, p = 0.9900. Generalisation
gap, omnibus one-way ANOVA F = 0.2060, p = 0.9594. No pairwise comparison
approaches significance: the smallest raw p-value across both metrics is 0.5856,
and every Holm-adjusted value is 1.0000. All effect sizes are below 0.18 in
magnitude.

H0 is not rejected.

### The negative control, and what it does and does not settle

A sixth condition modulates precision with ChaCha12. It differs from the chacha8
condition only in the number of ChaCha rounds, and both are cryptographically
strong, so any difference between them is noise. This is the Phase 8 device, and
it was constructible here because the baseline is a modulated condition of the
same family rather than the unmodulated one.

On the generalisation gap it works as intended: the control registers d = 0.263,
larger than every genuine change of generator, the largest of which is 0.166.
Effects of that size arise here from a change that cannot matter.

On validation loss it does not. The control registers d = 0.035, smaller than
three of the four generator effects. That is one realisation of a noisy quantity
and it calibrates nothing on this metric, and reporting only the gap result would
have been picking the metric that flattered the conclusion. What can be said on
validation loss is the weaker statement: every effect is far below the 0.886 this
design can detect, and every p-value is above 0.58.

### Limitations

Twenty seeds resolve a standardised effect of about 0.886 at eighty percent
power, and the normal approximation behind that figure flatters the design by
roughly three percent. Large effects are ruled out; small ones are not.

One architecture, one relaxation depth, one inference rate, one learning rate,
one modulation function. The logistic map was chosen for the three properties
above rather than tuned, and a different shape, a different steepness or a
precision that varied per layer rather than independently at every level could
behave differently. None of that was explored.

The precision is drawn independently at each level and each step. A schedule
correlated across levels, or one whose autocorrelation matched the generator's
own, would be a different and arguably more faithful test of whether the
structure of a chaotic stream can matter; this design gives the stream's
dependence structure very little room to survive.

Failing to detect a difference is not establishing equivalence, unchanged since
Phase 1.

## 19. Phase 10: topological resilience and graded plasticity as design hypotheses (not prior validated results)

**Where these ideas come from, and how they are used here.** Both are taken from
drafts of the author's. What is reused is the mathematical shape of two
expressions and nothing else. Those drafts also contain experimental results
that cannot be verified and that carry the marks of fabrication, so no figure
from them appears anywhere in this section, and no claim below rests on anything
reported there. Every weight, threshold and steepness is a hyperparameter of
this experiment, swept and documented, not a value inherited as anyone's
optimum. The two expressions are treated as hypotheses being tested here for the
first time.

The idea behind the second does have a verifiable origin, and that is what is
cited for it.

REF: [Grossberg, 2013] "Adaptive Resonance Theory: How a brain learns to
consciously attend, learn, and recognize a changing world", Neural Networks 37,
pp. 1-47, DOI 10.1016/j.neunet.2012.09.017. Checked against CrossRef. Adaptive
resonance makes plasticity conditional rather than uniform; the expression used
here is one particular parameterisation of that.

### Two defects in the persistence implementation, found on the way

The topological signal is computed with the `tda` crate, the same one Phase 0.5
uses. Two defects surfaced, both established against inputs whose answer can be
worked out by hand.

Every finite dimension-zero bar is returned twice. On four points at 0, 1, 3 and
6, whose spanning tree has edges 1, 2 and 3, the crate returns `[1,1,2,2,3,3]`.
The cause is that `compute_persistence` computes dimension zero once by
union-find and again by reducing the edge boundary matrix, and keeps both. The
duplicates are exact, so this phase deduplicates the bars rather than halving
the sum, since halving a total would apply the per-feature threshold to phantom
features.

Homology is returned only up to one less than the dimension requested: asking
for two yields nothing in dimension two, silently and without error. This one
matters more, because without noticing it the `w_2` term of the weighting would
have been multiplying an empty set for the whole phase with nothing to signal
it.

Neither affects the published results of earlier phases. Phases 0.5 and 4b use
only dimension one, which is returned correctly, and the PH-dim of Phases 3 and
4c does not use this crate at all. A separate defect report has been written.

### 10a. Does the measure respond to structure at all?

`T(S) = sum_d w_d sum_i max(0, pers(f_i) - sigma_d)` over the correlation
structure of a layer's value nodes. Node activity is centred and scaled to unit
norm, so the Euclidean distance between two nodes is exactly `sqrt(2(1 - rho))`
and the geometry is a genuine metric rather than an ad hoc function of
correlation.

Four synthetic structures with known topology: modules driven by shared latent
signals, which have clusters but no cycle; a ring, which has a cycle; modules
arranged on a ring, which has both; and independent noise, which should have
neither.

| Weighting | modular | ring | modular-ring | noise | loops > noise |
|---|---|---|---|---|---|
| with-components | 3.3871 | 3.0112 | 3.2263 | 12.3309 | **no** |
| loops | 0.0000 | 1.2192 | 0.6062 | 0.0000 | yes |
| loops-and-voids | 0.0000 | 0.8535 | 0.4680 | 0.0000 | yes |
| voids-heavy | 0.0000 | 0.6096 | 0.3758 | 0.0000 | yes |
| loops-strict | 0.0000 | 1.0192 | 0.4062 | 0.0000 | yes |

**The first row is the finding, and it is a correction to the hypothesis rather
than to the code.** Any weighting that puts positive weight on dimension zero
ranks pure noise far above every structured case. The reason is not an
implementation error and not a matter of tuning: total dimension-zero
persistence is exactly the weight of the minimum spanning tree, which is largest
when every point is far from every other, and that is precisely what independent
noise produces. Structure clusters points, and clustering shortens those bars.
No positive `w_0` can make the expression rank structure first. That identity is
verified as a test against Prim's algorithm, and the failing configuration is
kept in the sweep so the failure is visible rather than quietly removed. It is
excluded from the comparison, for that reason.

With `w_0 = 0` the measure works, and the threshold is what makes it work.
Independent noise generates several hundred very short one-dimensional bars
whose sum exceeds a genuine loop's single long bar, so an unthresholded total
would also rank noise first. At a threshold of 0.10 every noise bar falls below
the cut and the noise column is exactly zero, while a real cycle survives. The
plain modular case also scores zero, which is correct rather than a failure: it
has clusters and no cycle, and a measure of loops should say so.

### 10b and 10c. The two mechanisms applied to the Phase 9 network

Precision modulated by `T(S)`, and a graded learning rate
`alpha_i = alpha_max / (1 + exp(-beta(l_i - l_threshold)))`, kept as separate
conditions and also combined. Twenty seeds per condition, the Phase 9 network
and the two-moons task throughout.

Two fairness constraints are enforced as tests, for the reason Phase 9's
precision map was built the same way. The plasticity multipliers are rescaled so
they average exactly one, and the topological signal is standardised against a
running mean and mean absolute deviation of itself before the logistic map, so
its median precision is one whatever the signal's units. Without either, a
condition would change the mean learning rate as well as its distribution, and
any difference would be a difference of step size in a theoretical costume.

The negative control is a permutation: each node's activity is shuffled
independently, which destroys the correlation between nodes while leaving every
node's own distribution exactly as it was. The signal keeps its scale and loses
its meaning.

**Validation loss.** The omnibus rejects at every setting. What moves it is
entirely the plasticity gate.

| Gate | Multipliers | Spread | d | Holm p |
|---|---|---|---|---|
| gentle-early | 0.275, 1.233, 1.492 | 1.217 | -0.759 | 0.0645 |
| gentle-midpoint | 0.095, 1.000, 1.905 | 1.810 | -1.230 | 0.0012 |
| sharp-midpoint | 0.005, 1.000, 1.995 | 1.990 | -1.469 | 0.0001 |

Graded plasticity makes the network **worse**, and the harm is monotone in how
widely the gate spreads the per-layer rates. The combined condition adds nothing
over the gate alone, at every setting, so the joint effect is entirely the gate's.

The mechanism is visible in the multipliers and should temper how the result is
read. At the sharp midpoint gate the first weight matrix receives 0.005 of the
base rate, so it barely trains at all. This is therefore not evidence against
graded plasticity as an idea; it is evidence that grading the rate across a
three-layer network of this size, at these settings, starves the early layer and
costs accuracy, and that the cost scales with how much it is starved. A gentler
grading costs less and is not significant after correction.

**Topological resilience does nothing.** Its effect on validation loss is
d = -0.268 at the light threshold and -0.159 at the strict one, against a
shuffled control at -0.170 and 0.000 respectively. At the strict threshold the
genuine signal and its own permutation control converge, which is what should
happen if what little the signal was moving came from short bars rather than
from structure. No comparison approaches significance.

**Generalisation gap.** Nothing, anywhere. The omnibus p-values run from 0.9023
to 0.9976 across the four settings, every Holm-adjusted comparison is 1.0000,
and no effect size exceeds 0.23 in magnitude.

### Sensitivity, as a result rather than a footnote

The topological conclusion is stable across both thresholds tested and is
identical across the three gate settings, as it must be since the gate does not
touch it. That invariance is a check on the harness as much as a result.

The plasticity conclusion is **not** stable, and that is the more informative
half. It is significant at both midpoint gates and falls to Holm p = 0.0645 at
the early gate, tracking the spread of the multipliers monotonically. A single
setting would have supported either "graded plasticity harms learning" or "no
effect", and reporting one of those alone would have been a claim about a
hyperparameter dressed as a claim about a mechanism.

### Limitations

The `loops-and-voids` weighting was calibrated but **not** carried into the
comparison, and the reason is cost. Carrying a dimension-two term into training
requires building the complex one dimension higher, which on thirty-two nodes
takes 3.49 seconds per evaluation against 47 milliseconds one dimension lower,
a seventy-five-fold wall that puts the sweep at roughly five hours. The
persistence reduction scans all previous columns inside a loop over columns, so
it is quadratic in the simplex count, and dimension three over thirty-two points
is close to thirty-six thousand tetrahedra. What is untested is whether the
comparison's conclusion would change if the signal counted voids as well as
loops. Two things bound that gap: the topological condition shows no effect at
either threshold, at or below its own control; and in the calibration the
dimension-two term barely moves, a single bar clearing the threshold in one of
four synthetic cases.

The signal is recomputed every thirty-two weight updates, a little under once
per epoch, on a fixed probe set. A schedule that tracked it continuously might
behave differently and would cost far more.

Twenty seeds resolve a standardised effect of about 0.886 at eighty percent
power, by the same slightly optimistic normal approximation used since Phase 8.
The plasticity effects clear that comfortably; the topological ones are far
below it, so small effects there are not excluded.

One network, one task, one depth. A three-layer hierarchy is a thin place to
test a hypothesis about grading across depth, and the starvation mechanism above
would be far less severe in a deeper network where adjacent layers differ less.

## 20. Phase 11: precision with shared trajectory and level offset (a structural revision of Phase 9)

**What changes and what does not.** The network, the two-moons task, the six
conditions including the ChaCha12 control, the logistic map from a variate to a
precision, the sixteen relaxation steps and every training hyperparameter are
exactly those of Phase 9. One thing changes: where the numbers come from.

Phase 9 drew each level's precision independently, at each relaxation step of
each sample. Nothing tied one draw to the next within a level, and nothing tied
one level to another. A generator's structure is a property of its orbit over
time, and that scheme gave it almost no way to survive as far as the place it
was supposed to act. Phase 9's report listed this as an open limitation rather
than a settled question, and this phase closes it.

Here all three levels read from a single continuous orbit. Level `l` reads the
value at trajectory step `t + l * delta`, and `t` advances by one at every
relaxation step without restarting between samples or between epochs. The
sixteen precisions a level sees while one sample relaxes are sixteen consecutive
states of that generator's own orbit, and the three levels are the same orbit at
three fixed offsets.

### 11a. Does the mechanism do what it claims?

Two blocking checks, both run before any comparison.

The offset must be real. The cross-correlation between level zero's precision
series and level one's must peak at the configured `delta`. An offset applied to
the wrong index, or applied to the relaxation counter instead of to the
trajectory, would still produce a plausible-looking modulation and every result
after it would be a measurement of something other than what is described.

| Offset | Peak lag, all five generators | Peak value | Median, level 0 | Median, level 2 |
|---|---|---|---|---|
| 0 | 0 | 1.0000 | 0.9932 to 1.0061 | 0.9932 to 1.0061 |
| 1 | 1 | 1.0000 | 0.9932 to 1.0061 | 0.9932 to 1.0061 |
| 4 | 4 | 1.0000 | 0.9932 to 1.0061 | 0.9932 to 1.0061 |
| 16 | 16 | 1.0000 | 0.9932 to 1.0061 | 0.9932 to 1.0060 |

Twenty combinations, four offsets by five generators, and every one peaks
exactly where it is configured to. The peak value is exactly one by
construction, since level one's value at time `t` is the identical float level
zero reads at `t + delta`, so the check establishes the *lag*, not the existence
of correlation.

The Phase 9 fairness property must also survive. Each level's median precision
must still be one, or the condition would differ from the baseline in the
effective step size and not only in how the precision varies. The logistic map
is unchanged so the marginal should be too, but correlation between levels is
new, and the property was checked rather than inherited. Every median sits
within 0.7 percent of one.

### 11b. The comparison

Twenty seeds per condition at each of the four offsets, the same pre-registered
test sequence as every earlier phase, on validation loss, accuracy and the
generalisation gap.

| Offset | Validation loss | Accuracy | Generalisation gap |
|---|---|---|---|
| 0 | p = 0.9669 | p = 0.9982 | p = 0.8257 |
| 1 | p = 0.9961 | p = 0.9996 | p = 0.9457 |
| 4 | p = 0.9994 | p = 0.9993 | p = 0.9794 |
| 16 | p = 0.9933 | p = 0.9970 | p = 0.9541 |

H0 is not rejected anywhere. Twelve omnibus tests, none below 0.82. Every one of
the sixty pairwise comparisons has a Holm-adjusted p-value of exactly 1.0000.
The largest effect anywhere in the phase is 0.263 in magnitude, against a design
that resolves 0.886.

There is no pattern in the offset. The result at `delta = 0`, where the three
levels are perfectly synchronised, is indistinguishable from the result at
`delta = 16`, where level one reads during one sample what level zero will read
during the next.

The negative control settles it. On the generalisation gap it registers 0.296,
0.247 and 0.243 at the first three offsets, **larger in each case than any
genuine change of generator at that offset**. A condition that differs from the
others only in the number of ChaCha rounds moved the metric more than swapping a
cryptographic stream for a chaotic attractor did.

### This is a stronger null than Phase 9's, and it should be read as one

Phase 9's result carried an objection it could not answer: the design destroyed
the very structure it was looking for, by drawing every value independently, so
a null there was partly a fact about the sampling scheme. That objection is now
gone. The generator's orbit reaches the precision weighting intact, its temporal
correlation is preserved by construction, the levels are coupled along the same
trajectory at a verified offset, and the offset was swept from perfect
synchrony to a full relaxation apart.

Nothing changed. Not on any metric, at any offset, for any generator.

This is not one more null on a pile. It is the null that the previous one was
not entitled to claim, obtained after removing the specific reason the previous
one could be doubted. Across eleven phases the finding is now consistent from
three independent directions: no formal channel exists for one of the four
generators at all, no empirical effect appears under gradient descent, under a
fixed reservoir, or under local Hebbian learning, and none appears when the
generator's own temporal structure is carried intact into the one quantity the
theory says modulates inference.

### Limitations

Twenty seeds resolve a standardised effect of about 0.886, by the same slightly
optimistic normal approximation used since Phase 8. Effects smaller than that
remain outside what this design can see, and that is unchanged from Phase 9.

Four offsets, one of them degenerate. A `delta` far larger than a relaxation
window, or an offset that varied during training rather than being fixed, was
not tried.

The trajectory is read at one value per relaxation step. A scheme where a level
consumed several trajectory steps per relaxation step, changing the effective
timescale of the modulation relative to the inference, is a different design and
is untested.

Failing to detect a difference is still not establishing equivalence. What has
changed is that the failure can no longer be attributed to the sampling.
