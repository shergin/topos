//! The facade tier: convenience on the public surface.
//!
//! [`neural`] is layers, losses, optimizers, and initializers;
//! [`notebook`] is the inherent `to_html` and `evcxr_display` the
//! Evcxr feature adds. Both compose through the crate's public API
//! alone, with no privileged engine access, which
//! `tests/facade_surface.rs` enforces by scanning these sources.

pub(crate) mod neural;
#[cfg(feature = "evcxr")]
pub(crate) mod notebook;
