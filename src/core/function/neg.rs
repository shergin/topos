use smallvec::smallvec;

use crate::{Element, Shape, Tensor, Tensorial};

use super::{Cotangents, Operation, Reads, unary};

/// The negation of a value.
///
/// The derivative with respect to the operand is minus one, so `backward`
/// hands the negated incoming gradient to the operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Neg;

impl Neg {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rule below.
    /// It reads no payloads.
    pub(crate) fn reads(&self) -> Reads {
        Reads::NOTHING
    }

    /// Infers the shape of the result: the operand's shape.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        unary(operands).clone()
    }
}

impl Neg {
    pub(crate) fn forward<E: Element>(&self, operands: &[&Tensor<E>]) -> Tensor<E> {
        let &operand = unary(operands);
        -operand.clone()
    }
}

impl<Rule: Tensorial> Operation<Rule> for Neg {
    fn backward(&self, _operands: &[&Rule], _output: &Rule, gradient: &Rule) -> Cotangents<Rule> {
        smallvec![Some(-gradient.clone())]
    }
}
