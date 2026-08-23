use smallvec::smallvec;

use crate::{Element, Shape, Tensor, Tensorial};

use super::{Cotangents, Operation, Reads, unary};

/// The explicit repetition of a payload along one new named axis of
/// `extent`, inserted at `axis`.
///
/// It is the axis-wise form of `Broadcast`, and `SumAlong` is its
/// exact adjoint: the operand's gradient is the incoming gradient
/// summed along the repeated axis. The axis and extent are recorded
/// parameters — never an operand, because a shape is static
/// record-time data, not dataflow — so no shape alignment is ever
/// inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BroadcastAlong {
    pub(crate) axis: usize,
    pub(crate) extent: usize,
}

impl BroadcastAlong {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rule below.
    /// It reads no payloads: the cotangent sums back along the axis.
    pub(crate) fn reads(&self) -> Reads {
        Reads::NOTHING
    }

    /// Infers the shape of the result: the operand's shape with an
    /// axis of `extent` inserted at `axis`.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let operand = unary(operands);
        assert!(
            self.axis <= operand.rank(),
            "broadcast axis {} is out of rank for {operand}",
            self.axis
        );
        assert!(self.extent > 0, "broadcast extent must be positive");
        let mut axes: Vec<usize> = operand.axes().to_vec();
        axes.insert(self.axis, self.extent);
        Shape::new(axes)
    }
}

impl BroadcastAlong {
    pub(crate) fn forward<E: Element>(&self, operands: &[&Tensor<E>]) -> Tensor<E> {
        unary(operands).broadcast_along(self.axis, self.extent)
    }
}

impl<Rule: Tensorial> Operation<Rule> for BroadcastAlong {
    fn backward(&self, _operands: &[&Rule], _output: &Rule, gradient: &Rule) -> Cotangents<Rule> {
        smallvec![Some(gradient.sum_along(self.axis))]
    }
}
