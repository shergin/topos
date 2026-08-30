use smallvec::smallvec;

use crate::{Element, Recordable, Shape, Tensor};

use super::{Cotangents, Operation, Reads, binary};

/// A value raised to the power of another, elementwise, with operands
/// `[base, exponent]`.
///
/// The base's gradient follows the power rule,
/// `exponent * base^(exponent - 1)`; the exponent's follows the
/// exponential rule, `output * ln(base)`, which is a number only on a
/// positive base — elsewhere the payload's `ln` semantics (`NaN` for
/// scalars) propagate, mirroring the mathematics: `x^y` has no exponent
/// derivative at a non-positive base.
///
/// This is the one operation kept ahead of its consumers, on record
/// (the mirror of `Sub`, the exception on the other membership
/// clause): nothing in the tree records a power yet, but the seat is
/// irreplaceable the day one does — `exp(exponent * ln(base))` is not
/// bit-faithful (the `ln` rounding is amplified by the exponent) and
/// does not exist at a negative base — and the expected consumers
/// (a learned pooling exponent, fractional robust-loss powers) make
/// the exponent a graph value no host-side constant can stand in
/// for. The op-set audit holds the roster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Powf;

impl Powf {
    /// Returns the arity: two operands.
    pub(crate) fn arity(&self) -> usize {
        2
    }

    /// Returns the read set of the derivative rule below.
    /// It reads both operands and its own output.
    pub(crate) fn reads(&self) -> Reads {
        Reads {
            operands: [true, true],
            output: true,
        }
    }

    /// Infers the shape of the result, which both operands must share.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (base, exponent) = binary(operands);
        assert_eq!(base, exponent, "powf requires operands of equal shapes");
        base.clone()
    }
}

impl Powf {
    pub(crate) fn forward<E: Element>(&self, operands: &[&Tensor<E>]) -> Tensor<E> {
        let (&base, &exponent) = binary(operands);
        base.powf(exponent.clone())
    }
}

impl<Rule: Recordable> Operation<Rule> for Powf {
    fn backward(&self, operands: &[&Rule], output: &Rule, gradient: &Rule) -> Cotangents<Rule> {
        let (&base, &exponent) = binary(operands);
        let lowered = exponent.clone() - exponent.one_like();
        let base_cotangent = gradient.clone() * exponent.clone() * base.powf(lowered);
        let exponent_cotangent = gradient.clone() * output.clone() * base.ln();
        smallvec![Some(base_cotangent), Some(exponent_cotangent)]
    }
}

#[cfg(test)]
#[path = "tests/powf_tests.rs"]
mod tests;
