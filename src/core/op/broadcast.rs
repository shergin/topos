use smallvec::smallvec;

use crate::{Element, Shape, Tensor, Tensorial};

use super::{Cotangents, Operation, Reads, unary};

/// The explicit broadcast of a single-value payload across a target
/// shape carried by the operation itself.
///
/// It is the only shape-changing expansion in the engine, and it is
/// deliberately explicit: the target shape is a recorded parameter,
/// never an alignment rule — and never an operand, because a shape is
/// static record-time data, not dataflow. Broadcasting and summation
/// are adjoint, so the operand's gradient is the sum of the incoming
/// gradient, restored to the operand's own single-value shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Broadcast {
    pub(crate) shape: Shape,
}

impl Broadcast {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rule below.
    /// It reads its operand for shape only, which a placeholder answers.
    pub(crate) fn reads(&self) -> Reads {
        Reads::NOTHING
    }

    /// Infers the shape of the result: the carried target shape,
    /// reachable only from a single-value operand.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let operand = unary(operands);
        assert_eq!(
            operand.volume(),
            1,
            "broadcast requires a single-element operand, got {operand}"
        );
        self.shape.clone()
    }
}

impl Broadcast {
    pub(crate) fn forward<E: Element>(&self, operands: &[&Tensor<E>]) -> Tensor<E> {
        unary(operands).broadcast(self.shape.clone())
    }
}

impl<Rule: Tensorial> Operation<Rule> for Broadcast {
    fn backward(&self, operands: &[&Rule], _output: &Rule, gradient: &Rule) -> Cotangents<Rule> {
        let &operand = unary(operands);
        // The reduced gradient is rank 0, but the operand may be any
        // volume-1 shape (such as `[1]`); broadcasting the sum back to
        // the operand's own shape keeps the accumulation well-formed.
        smallvec![Some(gradient.sum().broadcast(operand.shape()))]
    }
}
