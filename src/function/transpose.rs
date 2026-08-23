use smallvec::smallvec;

use crate::{Element, Shape, Tensor, Tensorial};

use super::{Cotangents, Operation, Reads, unary};

/// The transposition of a value.
///
/// Transposition is linear and self-adjoint in shape: the gradient of the
/// operand is the transposed incoming gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Transpose;

impl Transpose {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rule below.
    /// It reads no payloads: the cotangent transposes back.
    pub(crate) fn reads(&self) -> Reads {
        Reads::NOTHING
    }

    /// Infers the shape of the result: the operand's axes reversed.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let operand = unary(operands);
        assert!(
            operand.rank() <= 2,
            "transpose supports rank 2 at most, got {operand}"
        );
        Shape::new(operand.axes().iter().rev().copied())
    }
}

impl Transpose {
    pub(crate) fn forward<E: Element>(&self, operands: &[&Tensor<E>]) -> Tensor<E> {
        unary(operands).transpose()
    }
}

impl<Rule: Tensorial> Operation<Rule> for Transpose {
    fn backward(&self, _operands: &[&Rule], _output: &Rule, gradient: &Rule) -> Cotangents<Rule> {
        smallvec![Some(gradient.transpose())]
    }
}
