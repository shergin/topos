use super::{Differentiable, Elementary};

/// A number that can fill a [`Tensor`](super::Tensor): the open
/// payload seam.
///
/// The graph is always tensors — a scalar is a rank-0 tensor — so the
/// type every public phase is generic over is the element, not the
/// payload: `Tape<f32>` records tensors of `f32`. Plugging the seam
/// means implementing the element contracts ([`Differentiable`] for
/// arithmetic and the accumulator, [`Elementary`] for the maps and
/// the backend hooks) on a number type and declaring it here with an
/// empty `impl`; the tensor machinery, the derivative rules, and the
/// engine come along unchanged. A new element never reimplements
/// `unfold`.
///
/// The built-in elements are `f32`, `f64`, and
/// [`Bf16`](super::Bf16) — the same set the backend precision table,
/// StableHLO emission, and the notebook displays already speak.
pub trait Element: Differentiable + Elementary {}

impl Element for f32 {}

impl Element for f64 {}
