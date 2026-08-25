use smallvec::smallvec;

use crate::{Element, Shape, Tensor, Tensorial};

use super::{Cotangents, Operation, Reads, binary};

/// The sum of two values, with operands `[left, right]`.
///
/// The derivative with respect to each operand is one, so `backward`
/// hands the incoming gradient to both operands unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Add;

impl Add {
    /// Returns the arity: two operands.
    pub(crate) fn arity(&self) -> usize {
        2
    }

    /// Returns the read set of the derivative rule below.
    /// It reads no payloads: both cotangents are the gradient itself.
    pub(crate) fn reads(&self) -> Reads {
        Reads::NOTHING
    }

    /// Infers the shape of the result, which both operands must share.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (left, right) = binary(operands);
        assert_eq!(left, right, "addition requires operands of equal shapes");
        left.clone()
    }
}

impl Add {
    pub(crate) fn forward<E: Element>(&self, operands: &[&Tensor<E>]) -> Tensor<E> {
        let (&left, &right) = binary(operands);
        left.clone() + right.clone()
    }
}

impl<Rule: Tensorial> Operation<Rule> for Add {
    fn backward(&self, _operands: &[&Rule], _output: &Rule, gradient: &Rule) -> Cotangents<Rule> {
        smallvec![Some(gradient.clone()), Some(gradient.clone())]
    }
}

#[cfg(test)]
#[path = "tests/add_tests.rs"]
mod tests;
