# tda 0.1.0: two defects in `compute_persistence`

Found while using the crate for a persistent homology computation over the
correlation structure of neural network activations. Both are reproducible from
inputs whose answer can be worked out by hand, and both are in
`src/persistent_homology.rs`.

Neither affects the *values* of the bars, which are correct. What is wrong is
how many bars come back, and which dimensions come back at all.

---

## Defect 1: every finite dimension-zero pair is reported twice

### Reproduction

Four points on a line at 0, 1, 3 and 6. The minimum spanning tree has edges of
length 1, 2 and 3, so dimension-zero persistence consists of exactly three
finite bars, of those lengths, plus one essential class.

```rust
use nalgebra::DMatrix;
use tda::{persistent_homology::compute_persistence, simplicial_complex::vietoris_rips_complex};

let pts = [0.0f64, 1.0, 3.0, 6.0];
let n = pts.len();
let mut d = DMatrix::zeros(n, n);
for i in 0..n {
    for j in 0..n {
        d[(i, j)] = (pts[i] - pts[j]).abs();
    }
}
let complex = vietoris_rips_complex(&d, 10.0, 2).unwrap();
let pairs = compute_persistence(&complex, 2).unwrap();
let finite: Vec<f64> = pairs
    .iter()
    .filter(|p| p.dimension == 0 && !p.is_infinite())
    .map(|p| p.persistence())
    .collect();
```

Expected `[1.0, 2.0, 3.0]`. Returned `[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]`.

The essential class count is correct: exactly one.

The same doubling appears on evenly spaced circles, independently of the point
count and of `max_dimension`:

| Points on a circle | Finite H0 bars expected | Returned |
|---|---|---|
| 12 | 11 | 22 |
| 16 | 15 | 30 |
| 20 | 19 | 38 |

### Cause

`compute_persistence` computes dimension zero along two paths and keeps both.

```rust
// Dimension 0 (connected components) using Union-Find
let pairs_0 = compute_persistence_dim0(complex)?;
pairs.extend(pairs_0);

// Higher dimensions using matrix reduction
for dim in 1..=max_dimension.min(complex.max_dimension) {
    let pairs_d = compute_persistence_dim_reduction(complex, dim)?;
    pairs.extend(pairs_d);
}
```

`compute_persistence_dim0` emits the dimension-zero pairs by union-find. Then
the loop runs `compute_persistence_dim_reduction(complex, 1)`, which reduces the
boundary matrix whose columns are edges and whose rows are vertices, and emits

```rust
pairs.push(PersistentPair::new(dimension - 1, birth_time, death_time));
```

With `dimension = 1` that label is zero. The reduction of the edge boundary
matrix *is* dimension-zero persistence, so the same pairs are produced a second
time, by a second correct method, and both sets are returned.

Higher dimensions are unaffected: dimension one comes only from `dim = 2`, and
is emitted once. That is consistent with what the reproduction shows, a circle
returning exactly one dimension-one class of nonzero length.

### Consequence for callers

Any statistic that counts bars or sums over them in dimension zero is exactly
twice what it should be. Total dimension-zero persistence, which equals the
weight of the minimum spanning tree, comes back doubled. Statistics restricted
to dimension one or above are correct.

### Suggested fix

Start the loop at `dim = 2`, so that the union-find result is the only source of
dimension zero:

```rust
for dim in 2..=max_dimension.min(complex.max_dimension) {
```

That leaves the union-find path, which is the faster and more numerically direct
of the two, as the single source for dimension zero. It also requires the
indexing change below, since the loop bound and the emitted dimension are
currently off by one relative to each other.

---

## Defect 2: `max_dimension` yields homology only up to `max_dimension - 1`

### Reproduction

Twelve points on a circle:

```
compute_persistence(complex, 1) -> finite pairs by dimension: {0: 22}
compute_persistence(complex, 2) -> finite pairs by dimension: {0: 22, 1: 55}
compute_persistence(complex, 3) -> finite pairs by dimension: {0: 22, 1: 55, 2: 165}
```

Asking for `max_dimension = 2` returns nothing in dimension two. To obtain
dimension-two homology the caller has to ask for three.

### Cause

The same line. The loop runs `dim` from 1 to `max_dimension`, and each iteration
emits pairs labelled `dim - 1`. The highest dimension actually produced is
therefore `max_dimension - 1`.

### Consequence for callers

Silent, which is what makes it worth reporting. A caller who asks for
`max_dimension = 2` and then filters for `dimension == 2` gets an empty set and
no error, and a weighted sum over dimensions has one of its terms quietly
multiplied by nothing. There is no signal that the requested dimension was never
computed.

### Suggested fix

Either iterate `dim` from 1 to `max_dimension + 1` while keeping the `dim - 1`
label, or keep the loop and emit `dimension` rather than `dimension - 1` while
passing `dimension + 1` to the reduction. The first is the smaller change:

```rust
for dim in 2..=(max_dimension + 1).min(complex.max_dimension) {
```

combined with the fix for defect 1. Whichever is chosen, the documented meaning
of `max_dimension` and the dimensions actually returned should be made to agree,
and a test that asks for a dimension and checks it is non-empty would catch a
regression.

---

## Note on what is not affected

The filtration values, the union-find deaths, the reduction itself and the
essential-class handling all appear correct on these inputs. A circle returns
exactly one dimension-one class, and its lifetime matches the geometry. Callers
that use dimension one or above, and that do not depend on the count of
dimension-zero bars, are getting right answers.

## Environment

`tda 0.1.0` from crates.io, `nalgebra 0.33.3`, Linux x86_64, stable Rust.
