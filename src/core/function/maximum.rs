use smallvec::smallvec;

use crate::{Element, Shape, Tensor, Tensorial};

use super::{Cotangents, Operation, Reads, binary};

/// The elementwise maximum of two values, with operands `[left, right]`.
///
/// The gradient flows to whichever operand won each position, routed by
/// the 0/1 [`step`](Elementary::step) indicator. On a tie the left
/// operand takes the whole gradient rather than splitting it, so the two
/// cotangents always partition the incoming gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Maximum;

impl Maximum {
    /// Returns the arity: two operands.
    pub(crate) fn arity(&self) -> usize {
        2
    }

    /// Returns the read set of the derivative rule below.
    /// It reads both operands to mask the winners.
    pub(crate) fn reads(&self) -> Reads {
        Reads {
            operands: [true, true],
            output: false,
        }
    }

    /// Infers the shape of the result, which both operands must share.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (left, right) = binary(operands);
        assert_eq!(left, right, "maximum requires operands of equal shapes");
        left.clone()
    }
}

impl Maximum {
    pub(crate) fn forward<E: Element>(&self, operands: &[&Tensor<E>]) -> Tensor<E> {
        let (&left, &right) = binary(operands);
        left.maximum(right)
    }
}

impl<Rule: Tensorial> Operation<Rule> for Maximum {
    fn backward(&self, operands: &[&Rule], _output: &Rule, gradient: &Rule) -> Cotangents<Rule> {
        let (&left, &right) = binary(operands);
        let winners = left.step(right);
        let left_cotangent = gradient.clone() * winners.clone();
        let right_cotangent = gradient.clone() * (winners.one_like() - winners);
        smallvec![Some(left_cotangent), Some(right_cotangent)]
    }
}

#[cfg(test)]
#[path = "tests/maximum_tests.rs"]
mod tests;
