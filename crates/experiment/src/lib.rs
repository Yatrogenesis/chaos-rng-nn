// SPDX-License-Identifier: MIT
//! The parts of the experiment harness that other crates need.
//!
//! Only the dataset is exposed. Phase 9 runs on the same two-moons problem as
//! Phase 1, and it has to be the *same* data, not a second implementation that
//! agrees today and drifts later. Sharing the source is the only way to
//! guarantee that; the generator, the noise and the split are all deterministic
//! from their seeds, so both phases see identical points.
//!
//! Everything else stays private to the binary, because nothing outside needs
//! it and widening the surface would invite exactly the coupling this crate has
//! so far avoided.

pub mod dataset;
