//! The derived tier: faster or foreign readings of the same spec.
//!
//! [`backend`] offers kernels and may decline, leaving the
//! interpreter as the truth; [`emission`] writes a lowered plan as
//! text. Neither belongs in the engine: the engine stays
//! backend-blind, and text is not a kernel.

pub(crate) mod backend;
pub(crate) mod emission;
