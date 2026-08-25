use smallvec::smallvec;

use crate::{Element, Shape, Tensor, Tensorial};

use super::{Cotangents, Operation, Reads, unary};

/// The sum of every value in a payload, reduced to a single value.
///
/// Summation and broadcasting are adjoint: the gradient of the operand is
/// the incoming single-value gradient spread back across the operand's
/// shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sum;

impl Sum {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rule below.
    /// It reads its operand for shape only, which a placeholder answers.
    pub(crate) fn reads(&self) -> Reads {
        Reads::NOTHING
    }

    /// Infers the shape of the result: a rank-0 single value.
    pub(crate) fn infer_shape(&self, _operands: &[Shape]) -> Shape {
        Shape::scalar()
    }
}

impl Sum {
    pub(crate) fn forward<E: Element>(&self, operands: &[&Tensor<E>]) -> Tensor<E> {
        unary(operands).sum()
    }
}

impl<Rule: Tensorial> Operation<Rule> for Sum {
    fn backward(&self, operands: &[&Rule], _output: &Rule, gradient: &Rule) -> Cotangents<Rule> {
        let &operand = unary(operands);
        smallvec![Some(gradient.broadcast(operand.shape()))]
    }
}
