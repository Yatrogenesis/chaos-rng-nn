# kirs-pilot

Phase 7. Decides, by Horn resolution, which Pesin-type formula has its
hypotheses satisfied for each of the four generator families. It computes no
entropy, no Lyapunov exponent and no dimension; it answers whether the machinery
the earlier phases measured is formally applicable in the first place.

## Why this crate is outside the workspace

It depends by path on a separate clone of
<https://github.com/Yatrogenesis/kirs>, which is not part of this repository and
is consumed strictly as a read-only reference:

- no file inside that checkout is edited by this project, not even a lockfile;
- no source from it is copied or vendored here;
- a change of pinned version is a change to this crate's manifest, never to that
  repository.

Because that clone is external, listing this crate as a workspace member would
make `cargo test --workspace` fail for anyone who has not fetched it. It is
excluded instead, so the rest of the repository builds from a fresh clone with
nothing else present, and this crate is built explicitly.

## Running it

The reference must be cloned as a sibling of this repository, at the pinned
commit:

```bash
git clone https://github.com/Yatrogenesis/kirs.git ../kirs-readonly
git -C ../kirs-readonly checkout 18a727649f455b1aa332f99057a2a2f747ce769c
```

Then, from the root of this repository, so that the output lands in `results/`:

```bash
cargo test  --release --manifest-path crates/kirs-pilot/Cargo.toml
cargo run   --release --manifest-path crates/kirs-pilot/Cargo.toml --bin classify
```

If the clone is absent the build fails rather than silently skipping. That is
intended: it makes the external dependency visible instead of stale.
