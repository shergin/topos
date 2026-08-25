//! The core tier: the spec and its named readings.
//!
//! What the crate exists to do. [`payload`] is the element seam,
//! [`function`] the closed instruction set, [`graph`] the spec and the
//! state beside it, [`engine`] the readings that run it -- interpret,
//! lower, catalog. The tier is a folder, not a type: everything real
//! lives in the children.

pub(crate) mod engine;
pub(crate) mod function;
pub(crate) mod graph;
pub(crate) mod payload;
