use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// The base element contract: the arithmetic of a number that can
/// fill a [`Tensor`](super::Tensor).
///
/// It never mentions [`Shape`](super::Shape) — shape belongs to the
/// tensor, and an element is exactly what is left when shape is taken
/// away: arithmetic, the identities, the accumulator, and the count
/// conversion behind size-derived constants. The built-in
/// implementations cover `f32`, `f64`, and [`Bf16`](super::Bf16).
///
/// Elements must be `Send + Sync` because networks can be shared and
/// evaluated across threads.
pub trait Differentiable:
    Clone
    + Debug
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// The type accumulating operations compute in before rounding
    /// back to `Self` once: matmul inner products, the sum
    /// reductions, `fold`, and the scatter adjoint promote every
    /// term, accumulate here, and demote the final total.
    ///
    /// `Self` for payloads that accumulate in their own precision;
    /// `f32` for `Bf16`, whose eight significand bits swamp once a
    /// total reaches 256 times a term. The choice is semantics, not
    /// an optimization: every representation and every path honors
    /// it — a constant operand accumulates exactly like a dense one —
    /// and StableHLO emission states it through
    /// `Emittable::ACCUMULATION`.
    type Accumulator: Clone
        + Debug
        + Send
        + Sync
        + Add<Output = Self::Accumulator>
        + Mul<Output = Self::Accumulator>;

    /// Returns this value in the accumulator type, exactly.
    fn promote(&self) -> Self::Accumulator;

    /// Returns an accumulated total rounded back into `Self`.
    fn demote(accumulated: Self::Accumulator) -> Self;

    /// Returns the additive identity: the zero one element of a
    /// tensor fill or a padding lane holds.
    fn zero() -> Self;

    /// Returns the multiplicative identity.
    fn one() -> Self;

    /// Returns `count` as an element, exactly.
    ///
    /// It is the element half of [`Tensor::counted`](super::Tensor::counted),
    /// the constructor behind size-derived constants: a composed
    /// formula that divides by an axis extent must mint that extent
    /// as a value. Counts convert exactly as long as the element type
    /// can represent them.
    fn from_count(count: usize) -> Self;

    /// Returns whether this element is exactly what
    /// [`from_count`](Differentiable::from_count) mints for `count`.
    ///
    /// It is the recognizer half of `from_count`: pattern matchers
    /// certify a recorded size-derived constant (the divisor of a
    /// composed mean) through it before raising the surrounding
    /// formula to a named target operation. The conservative default
    /// answers `false`, which only forgoes recognitions.
    fn is_count(&self, count: usize) -> bool {
        let _ = count;
        false
    }
}

impl Differentiable for f32 {
    type Accumulator = Self;

    fn promote(&self) -> Self {
        *self
    }

    fn demote(accumulated: Self) -> Self {
        accumulated
    }

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn from_count(count: usize) -> Self {
        count as f32
    }

    fn is_count(&self, count: usize) -> bool {
        *self == count as f32
    }
}

impl Differentiable for f64 {
    type Accumulator = Self;

    fn promote(&self) -> Self {
        *self
    }

    fn demote(accumulated: Self) -> Self {
        accumulated
    }

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn from_count(count: usize) -> Self {
        count as f64
    }

    fn is_count(&self, count: usize) -> bool {
        *self == count as f64
    }
}
