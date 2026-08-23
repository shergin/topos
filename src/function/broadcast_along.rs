use smallvec::smallvec;

use crate::{Element, Shape, Tensor, Tensorial};

use super::{Cotangents, Operation, Reads, binary};

/// The explicit repetition of a payload along one named axis of a
/// reference value's shape, with operands `[operand, like]`.
///
/// It is the axis-wise form of `Broadcast`, and `SumAlong` is its
/// adjoint: the operand's gradient is the incoming gradient summed
/// along the repeated axis. The axis is always named, so no shape
/// alignment is ever inferred; the reference contributes only its
/// shape, which is what its `None` cotangent states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BroadcastAlong {
    pub(crate) axis: usize,
}

impl BroadcastAlong {
    /// Returns the arity: two operands.
    pub(crate) fn arity(&self) -> usize {
        2
    }

    /// Returns the read set of the derivative rule below.
    /// It reads no payloads: the cotangent sums back along the axis.
    pub(crate) fn reads(&self) -> Reads {
        Reads::NOTHING
    }

    /// Infers the shape of the result: the reference's shape, reachable
    /// only from an operand shaped like the reference without the axis.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (operand, like) = binary(operands);
        assert_eq!(
            operand,
            &like.without_axis(self.axis),
            "broadcast along axis {} of {like} requires the remaining shape",
            self.axis
        );
        like.clone()
    }
}

impl BroadcastAlong {
    pub(crate) fn forward<E: Element>(&self, operands: &[&Tensor<E>]) -> Tensor<E> {
        let (&operand, &like) = binary(operands);
        operand.broadcast_along(self.axis, like)
    }
}

impl<Rule: Tensorial> Operation<Rule> for BroadcastAlong {
    fn backward(&self, _operands: &[&Rule], _output: &Rule, gradient: &Rule) -> Cotangents<Rule> {
        smallvec![Some(gradient.sum_along(self.axis)), None]
    }
}
