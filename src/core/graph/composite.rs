//! Composite expressions over values: the second tier of the operation
//! surface.
//!
//! The first tier is `value.rs`, where every method is an opcode mnemonic
//! recording exactly one computed node. Each method here expands to a
//! formula over those opcodes — several computed nodes whose gradient the
//! chain rule pays with no dedicated backward rule. Everything in this
//! file compiles against the public operation surface alone: composites
//! need no privileged access to the engine, and once recorded they are
//! indistinguishable from hand-written primitives, so the tape stays a
//! uniform IR. The third tier is named formulas whose operands play
//! distinct roles (a loss's logits and targets); those are free functions
//! in domain modules such as the loss module.
//!
//! A formula belongs here only while composition expresses it faithfully;
//! it earns an `Op` variant the moment floating point breaks the
//! composed form, the way `log_softmax` did. `Sub` is the one
//! deliberate exception in the other direction: its composition
//! (`Add` of `Neg`) is bit-exact, yet the variant stays — a practical
//! decision for spec legibility and the oracle's one-pass cost,
//! documented on the operation itself.

use crate::{Element, Shape, Tensor};

use super::Value;

/// # Composites
///
/// Formulas that expand to several primitive nodes, paid by the
/// chain rule with no dedicated backward rule.
impl<'tape, E: Element> Value<'tape, E> {
    /// Records the absolute value of this value as the composition
    /// `self.maximum(-self)` and returns a proxy to it; the subgradient
    /// at zero is one, by `maximum`'s left-biased tie rule.
    pub fn abs(self) -> Self {
        self.maximum(-self)
    }

    /// Records the rectified linear unit of this value as the
    /// composition `self.maximum(zero)`, where the zero enters the
    /// graph as a [`counted`](crate::Tensor::counted) leaf of
    /// this value's shape — the same leaf a payload literal would
    /// record; the subgradient at zero is one, by `maximum`'s
    /// left-biased tie rule.
    ///
    /// The once-dedicated opcode was retired when the leaf failed to
    /// show up in a consumer-scale training step: the extra cost is
    /// one activation-sized zero buffer per occurrence, and the fused
    /// form never measured past it.
    pub fn relu(self) -> Self {
        self.maximum(self.literal(Tensor::counted(self.shape(), 0)))
    }
}

impl<'tape, E: Element> Value<'tape, E> {
    /// Records the softplus of this value, `ln(1 + e^x)`, as the
    /// stable split `self.relu() + log1p(exp(-|x|))` and returns a
    /// proxy to it.
    ///
    /// The naive composition overflows to infinity for large positive
    /// operands and answers zero long before the true value underflows
    /// for large negative ones; the split is finite and accurate over
    /// the whole line, riding the fused [`log1p`](Self::log1p) — the
    /// consumer that earned that opcode. The gradient is the chain
    /// rule over the parts, which analytically is the logistic
    /// sigmoid.
    pub fn softplus(self) -> Self {
        self.relu() + (-self.abs()).exp().log1p()
    }

    /// Records the exact Gaussian error linear unit of this value,
    /// `x * (1 + erf(x / sqrt(2))) / 2`, and returns a proxy to it:
    /// the consumer that earned the `Erf` opcode.
    ///
    /// Every constant is formula-pure: the 1 and 2 enter as
    /// [`counted`](crate::Tensor::counted) leaves and `sqrt(2)` is
    /// computed from the counted 2, so each element type rounds the
    /// formula at its own precision and the spec stores no decimal.
    /// The tanh approximation many models use instead is a caller
    /// composition over `tanh` (the gpt2 example records it); this is
    /// the exact form.
    pub fn gelu(self) -> Self {
        let one = self.literal(Tensor::counted(self.shape(), 1));
        let two = self.literal(Tensor::counted(self.shape(), 2));
        self * (one + (self / two.sqrt()).erf()) / two
    }

    /// Records the softmax probabilities of this value along `axis` as
    /// the composition `self.log_softmax(axis).exp()` and returns a proxy
    /// to it.
    ///
    /// Stability is inherited from the fused core: log-probabilities are
    /// at most zero, so the exponential cannot overflow — which is why
    /// softmax needs no fused form of its own.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank.
    pub fn softmax(self, axis: usize) -> Self {
        self.log_softmax(axis).exp()
    }

    /// Records the mean of this value along `axis` as the composition
    /// `self.sum_along(axis) / extent`, where the reduced axis's extent
    /// enters the graph as a [`counted`](crate::Tensor::counted)
    /// literal; like `sum_along`, the reduced axis is removed.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank.
    pub fn mean_along(self, axis: usize) -> Self {
        let shape = self.shape();
        assert!(axis < shape.rank(), "mean_along axis {axis} is out of rank");
        let extent = shape.axes()[axis];
        self.sum_along(axis) / Tensor::counted(shape.without_axis(axis), extent)
    }

    /// Records this single-value payload spread across `reference`'s
    /// shape: [`broadcast`](Self::broadcast) reading the reference for
    /// its shape alone, one node, no operand edge — a shape is static
    /// record-time data, not dataflow.
    ///
    /// # Panics
    /// Panics if this value's shape does not contain exactly one
    /// element.
    pub fn broadcast_like(self, reference: Self) -> Self {
        self.broadcast(reference.shape())
    }

    /// Records this value repeated along `axis` to match `reference`'s
    /// shape: [`broadcast_along`](Self::broadcast_along) reading the
    /// reference for its extent alone, one node, no operand edge.
    ///
    /// # Panics
    /// Panics if `axis` is out of `reference`'s rank or this value's
    /// shape differs from `reference`'s with that axis removed.
    pub fn broadcast_along_like(self, axis: usize, reference: Self) -> Self {
        let reference_shape = reference.shape();
        assert!(
            axis < reference_shape.rank(),
            "axis {axis} is out of rank for {reference_shape}"
        );
        assert_eq!(
            self.shape(),
            reference_shape.without_axis(axis),
            "broadcast along axis {axis} of {reference_shape} requires the remaining shape"
        );
        self.broadcast_along(axis, reference_shape.axes()[axis])
    }

    /// Records the transposition of this value — its axes reversed — as
    /// a `permute` of the reversed order, and returns a proxy to it.
    /// The once-dedicated opcode was retired as a strict special case:
    /// `Permute { [1, 0] }` is the same O(1) view with the same
    /// self-inverse gradient rule.
    ///
    /// # Panics
    /// Panics if this value's rank exceeds 2.
    pub fn transpose(self) -> Self {
        let rank = self.shape().rank();
        assert!(rank <= 2, "transpose supports rank 2 at most");
        self.permute((0..rank).rev())
    }

    /// Records this value with a new extent-1 axis inserted at `axis`:
    /// a `reshape` that leaves the elements unchanged.
    ///
    /// # Panics
    /// Panics if `axis` exceeds this value's rank.
    pub fn unsqueeze(self, axis: usize) -> Self {
        let mut axes: Vec<usize> = self.shape().axes().to_vec();
        assert!(axis <= axes.len(), "unsqueeze axis {axis} is out of rank");
        axes.insert(axis, 1);
        self.reshape(axes)
    }

    /// Records this value with the extent-1 axis at `axis` removed: a
    /// `reshape` that leaves the elements unchanged.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank or that axis is not extent 1.
    pub fn squeeze(self, axis: usize) -> Self {
        let mut axes: Vec<usize> = self.shape().axes().to_vec();
        assert!(axis < axes.len(), "squeeze axis {axis} is out of rank");
        assert_eq!(axes[axis], 1, "squeeze requires an extent-1 axis");
        axes.remove(axis);
        self.reshape(axes)
    }

    /// Records this value broadcast to `shape` under the right-aligned
    /// NumPy and TensorFlow rule, and returns a proxy to it.
    ///
    /// The two shapes align from the trailing axis: the target's rank must
    /// be at least this value's, and each source axis must either match its
    /// aligned target axis or have extent one, in which case it is repeated
    /// to the target extent. It composes the shape-changing primitives -- a
    /// right-aligning `reshape` that prepends the missing leading axes, then
    /// one `broadcast_along` per repeated axis, or a single `broadcast`
    /// when the source holds one element -- so the gradient is the chain
    /// rule over their adjoints: the incoming gradient summed back over
    /// every repeated axis.
    ///
    /// # Panics
    /// Panics if `shape`'s rank is smaller than this value's, or a source
    /// axis neither matches its aligned target axis nor has extent one.
    pub fn broadcast_to(self, shape: impl Into<Shape>) -> Self {
        let target = shape.into();
        let source = self.shape();
        if source == target {
            return self;
        }
        assert!(
            target.rank() >= source.rank(),
            "broadcast to {target} from {source} lowers the rank"
        );
        let offset = target.rank() - source.rank();
        for (axis, &extent) in source.axes().iter().enumerate() {
            let aligned = target.axes()[offset + axis];
            assert!(
                extent == aligned || extent == 1,
                "broadcast to {target} from {source} cannot align source axis \
                 {axis} of extent {extent} to extent {aligned}"
            );
        }
        // A single-element source reaches any shape in one node.
        if source.volume() == 1 {
            return self.broadcast(target);
        }
        // Right-align the source under the target by prepending unit axes, so
        // every axis is then either already matched or an extent-one axis to
        // repeat.
        let mut current = if offset == 0 {
            self
        } else {
            let mut axes = vec![1; offset];
            axes.extend_from_slice(source.axes());
            self.reshape(axes)
        };
        for axis in 0..target.rank() {
            let aligned = target.axes()[axis];
            if current.shape().axes()[axis] == aligned {
                continue;
            }
            // The only remaining mismatch is an extent-one axis; drop it and
            // repeat it to the target extent through the axis-wise adjoint.
            current = current.squeeze(axis).broadcast_along(axis, aligned);
        }
        current
    }
}

/// Records the concatenation of `values` along `axis` and returns a proxy
/// to it: each value is padded with zeros to the combined extent at its
/// running offset, and the pads are summed.
///
/// This is the designed route for sequence stacking and head
/// concatenation; a dedicated variadic opcode earns its node only if the
/// zero-padded intermediates ever measure. The gradient of each operand
/// is the incoming gradient narrowed back to its own window, through
/// `pad`'s adjoint.
///
/// # Panics
/// Panics if `values` is empty, the values belong to different networks,
/// `axis` is out of rank, or the shapes disagree anywhere but `axis`.
pub fn concat<'tape, E: Element>(values: &[Value<'tape, E>], axis: usize) -> Value<'tape, E> {
    let first = values.first().expect("concat requires at least one value");
    let reference = first.shape();
    assert!(
        axis < reference.rank(),
        "concat axis {axis} is out of rank for {reference}"
    );
    for value in &values[1..] {
        let shape = value.shape();
        assert_eq!(
            shape.without_axis(axis),
            reference.without_axis(axis),
            "concat along axis {axis} requires equal shapes off the axis, \
             got {shape} against {reference}"
        );
    }
    if values.len() == 1 {
        return *first;
    }
    let combined: usize = values.iter().map(|value| value.shape().axes()[axis]).sum();
    let mut offset = 0;
    let mut total: Option<Value<'tape, E>> = None;
    for &value in values {
        let padded = value.pad(axis, offset, combined);
        offset += value.shape().axes()[axis];
        total = Some(match total {
            Some(sum) => sum + padded,
            None => padded,
        });
    }
    total.expect("concat combines at least one value")
}

/// Records the stacking of `values` along a new axis at `axis` and returns
/// a proxy to it: each value gains an extent-1 axis there (`unsqueeze`)
/// and the lifted values concatenate.
///
/// # Panics
/// Panics if `values` is empty, the values belong to different networks,
/// `axis` exceeds the values' rank, or the shapes differ.
pub fn stack<'tape, E: Element>(values: &[Value<'tape, E>], axis: usize) -> Value<'tape, E> {
    let lifted: Vec<Value<'tape, E>> = values.iter().map(|&value| value.unsqueeze(axis)).collect();
    concat(&lifted, axis)
}

#[cfg(test)]
#[path = "tests/composite_tests.rs"]
mod tests;
