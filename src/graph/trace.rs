use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{Element, Shape, Tensor, Tensorial};

use super::Value;

/// A payload that records instead of computing: the second
/// interpretation of the derivative rules.
///
/// Every derivative rule is written against the recordable
/// vocabulary ([`Tensorial`]), not against a concrete payload.
/// `Trace` implements that trait by appending
/// the corresponding node to the tape and answering with a handle, so
/// running a rule with `Data = Trace` emits the rule's computation as
/// recorded graph — which is all
/// [`Tape::differentiate`](super::Tape::differentiate) does. The
/// rules cannot tell the difference, and that indistinguishability is
/// the design: derivative knowledge lives in exactly one place, and a
/// rule change reaches the engine's backward and the recorded gradient
/// alike, because both are the same code.
///
/// Public, the type hands the same trick to callers: an algorithm
/// written once against the recordable vocabulary gains a recording
/// interpretation — wrap recorded values with [`Trace::of`], run the
/// generic code, unwrap with [`Trace::value`] — beside its eager runs
/// over `f32` or [`Tensor`](crate::Tensor). What it does not open is
/// new scans over the crate's own derivative rules: the op set and
/// the graph walk stay crate-private, so new AD modes (forward mode,
/// checkpointed reverse) are in-crate transforms until a read surface
/// over the spec lands.
///
/// Every member of [`Tensorial`] records honestly: the trait was cut
/// along the recordable vocabulary, so nothing a trace implements
/// can panic by construction. Operations outside that vocabulary —
/// `max_along`, the fused executors, the `counted` constructor —
/// are inherent to [`Tensor`](crate::Tensor) and simply do not exist
/// here.
pub struct Trace<'tape, E> {
    value: Value<'tape, E>,
}

impl<'tape, E: Element> Trace<'tape, E> {
    /// Wraps a recorded value as a rule operand.
    pub fn of(value: Value<'tape, E>) -> Self {
        Self { value }
    }

    /// Returns the recorded value this trace stands for.
    pub fn value(&self) -> Value<'tape, E> {
        self.value
    }

    /// Records a literal leaf of `count` spread across this trace's
    /// recorded shape: the recording twin of `zero_like`/`one_like`.
    fn counted_like(&self, count: usize) -> Self {
        Self::of(
            self.value
                .literal(Tensor::counted(self.value.shape(), count)),
        )
    }
}

// Manual implementations avoid the `Data: Clone`/`Data: Copy` bounds a
// derive would demand; a trace is an index pair like the value it wraps.
impl<E> Clone for Trace<'_, E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E> Copy for Trace<'_, E> {}

impl<E> fmt::Debug for Trace<'_, E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Trace").finish_non_exhaustive()
    }
}

impl<'tape, E: Element> Add for Trace<'tape, E> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::of(self.value + rhs.value)
    }
}

impl<'tape, E: Element> Sub for Trace<'tape, E> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::of(self.value - rhs.value)
    }
}

impl<'tape, E: Element> Mul for Trace<'tape, E> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::of(self.value * rhs.value)
    }
}

impl<'tape, E: Element> Div for Trace<'tape, E> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::of(self.value / rhs.value)
    }
}

impl<'tape, E: Element> Neg for Trace<'tape, E> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::of(-self.value)
    }
}

/// The recording interpretation of the recordable vocabulary: every
/// operation appends its node and answers the handle, so a derivative
/// rule run over traces emits itself as graph.
impl<'tape, E: Element> Tensorial for Trace<'tape, E> {
    fn shape(&self) -> Shape {
        self.value.shape()
    }

    fn zero_like(&self) -> Self {
        self.counted_like(0)
    }

    fn one_like(&self) -> Self {
        self.counted_like(1)
    }

    fn exp(&self) -> Self {
        Self::of(self.value.exp())
    }

    fn ln(&self) -> Self {
        Self::of(self.value.ln())
    }

    fn sqrt(&self) -> Self {
        Self::of(self.value.sqrt())
    }

    fn tanh(&self) -> Self {
        Self::of(self.value.tanh())
    }

    fn powf(&self, exponent: Self) -> Self {
        Self::of(self.value.powf(exponent.value))
    }

    fn maximum(&self, other: &Self) -> Self {
        Self::of(self.value.maximum(other.value))
    }

    fn step(&self, threshold: &Self) -> Self {
        Self::of(self.value.step(threshold.value))
    }

    fn matmul(&self, rhs: &Self) -> Self {
        Self::of(self.value.matmul(rhs.value))
    }

    fn sum(&self) -> Self {
        Self::of(self.value.sum())
    }

    fn sum_along(&self, axis: usize) -> Self {
        Self::of(self.value.sum_along(axis))
    }

    fn broadcast(&self, shape: Shape) -> Self {
        Self::of(self.value.broadcast(shape))
    }

    fn broadcast_along(&self, axis: usize, extent: usize) -> Self {
        Self::of(self.value.broadcast_along(axis, extent))
    }

    fn reshape(&self, shape: Shape) -> Self {
        Self::of(self.value.reshape(shape))
    }

    fn permute(&self, order: &[usize]) -> Self {
        Self::of(self.value.permute(order.iter().copied()))
    }

    fn narrow(&self, axis: usize, start: usize, len: usize) -> Self {
        Self::of(self.value.narrow(axis, start, len))
    }

    fn pad(&self, axis: usize, start: usize, full_extent: usize) -> Self {
        Self::of(self.value.pad(axis, start, full_extent))
    }

    fn unfold(&self, axis: usize, size: usize, step: usize, dilation: usize) -> Self {
        Self::of(self.value.unfold(axis, size, step, dilation))
    }

    fn fold(&self, axis: usize, size: usize, step: usize, dilation: usize, extent: usize) -> Self {
        Self::of(self.value.fold(axis, size, step, dilation, extent))
    }

    fn gather(&self, selection: &Self) -> Self {
        Self::of(self.value.gather(selection.value))
    }

    fn scatter(&self, selection: &Self) -> Self {
        Self::of(self.value.scatter(selection.value))
    }
}
