use smallvec::smallvec;

use crate::{Element, MapOperation, Recordable, Shape, Tensor};

use super::{Cotangents, Operation, Reads, unary};

/// A unary elementwise transcendental of a value: one node kind
/// carrying the [`MapOperation`] it applies.
///
/// The IR and the backend map seam share this vocabulary on purpose:
/// `tanh` recorded here and `tanh` offered to the backend chain are
/// the same instruction, so adding a transcendental is a
/// `MapOperation` variant and the arms below — never a new
/// `Op` variant. Everything op-specific (the printed name, the
/// read set, the derivative) dispatches on `op`; the shape behavior
/// is shared, since a map always keeps its operand's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Map {
    pub(crate) op: MapOperation,
}

impl Map {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rules below, per
    /// operation: `Exp`, `Sqrt`, `Tanh`, and `Expm1` reuse their own
    /// output; `Ln`, `Sin`, `Cos`, `Log1p`, and `Erf` read their
    /// operand; `ErfDerivative` reads both. The set is deliberately
    /// not uniform — a shared one would retain buffers liveness does
    /// not need.
    pub(crate) fn reads(&self) -> Reads {
        match self.op {
            MapOperation::Exp | MapOperation::Sqrt | MapOperation::Tanh | MapOperation::Expm1 => {
                Reads {
                    operands: [false, false],
                    output: true,
                }
            }
            MapOperation::Ln
            | MapOperation::Sin
            | MapOperation::Cos
            | MapOperation::Log1p
            | MapOperation::Erf => Reads {
                operands: [true, false],
                output: false,
            },
            MapOperation::ErfDerivative => Reads {
                operands: [true, false],
                output: true,
            },
        }
    }

    /// Infers the shape of the result: the operand's shape.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        unary(operands).clone()
    }
}

impl Map {
    pub(crate) fn forward<E: Element>(&self, operands: &[&Tensor<E>]) -> Tensor<E> {
        let &operand = unary(operands);
        match self.op {
            MapOperation::Exp => operand.exp(),
            MapOperation::Ln => operand.ln(),
            MapOperation::Sqrt => operand.sqrt(),
            MapOperation::Tanh => operand.tanh(),
            MapOperation::Sin => operand.sin(),
            MapOperation::Cos => operand.cos(),
            MapOperation::Log1p => operand.log1p(),
            MapOperation::Expm1 => operand.expm1(),
            MapOperation::Erf => operand.erf(),
            MapOperation::ErfDerivative => operand.erf_derivative(),
        }
    }
}

impl<Rule: Recordable> Operation<Rule> for Map {
    fn backward(&self, operands: &[&Rule], output: &Rule, gradient: &Rule) -> Cotangents<Rule> {
        let cotangent = match self.op {
            // The derivative of `e^x` is `e^x` itself: the canonical
            // case of reusing the node's own output.
            MapOperation::Exp => gradient.clone() * output.clone(),
            // The derivative of `ln(x)` is `1 / x`; gradients inherit
            // the payload's logarithm and division semantics outside
            // the positive domain.
            MapOperation::Ln => {
                let &operand = unary(operands);
                gradient.clone() / operand.clone()
            }
            // The derivative of `sqrt(x)` is `1 / (2 * sqrt(x))` —
            // no generic literal `2` exists, so the doubling is
            // `output + output`.
            MapOperation::Sqrt => gradient.clone() / (output.clone() + output.clone()),
            // The derivative of `tanh(x)` is `1 - tanh(x)^2`: one
            // minus the square of the node's own output.
            MapOperation::Tanh => {
                gradient.clone() * (output.one_like() - output.clone() * output.clone())
            }
            // The derivative of `sin(x)` is `cos(x)`: the pair closes
            // over itself, which is why the two ship together.
            MapOperation::Sin => {
                let &operand = unary(operands);
                gradient.clone() * operand.cos()
            }
            // The derivative of `cos(x)` is `-sin(x)`.
            MapOperation::Cos => {
                let &operand = unary(operands);
                -(gradient.clone() * operand.sin())
            }
            // The derivative of `ln(1 + x)` is `1 / (1 + x)`; the
            // fused accuracy is a forward property, and the rule's
            // own `1 + x` is safe — no cancellation hides in an
            // addition this side of the logarithm.
            MapOperation::Log1p => {
                let &operand = unary(operands);
                gradient.clone() / (operand.one_like() + operand.clone())
            }
            // The derivative of `e^x - 1` is `e^x`: the node's own
            // output plus one, mirroring `Exp`'s output reuse.
            MapOperation::Expm1 => gradient.clone() * (output.clone() + output.one_like()),
            // The derivative of `erf(x)` is the scaled Gaussian — its
            // own operation, so the rule mints no constant.
            MapOperation::Erf => {
                let &operand = unary(operands);
                gradient.clone() * operand.erf_derivative()
            }
            // The derivative of the scaled Gaussian is `-2x` times
            // itself: the node's own output, doubled operand, negated.
            MapOperation::ErfDerivative => {
                let &operand = unary(operands);
                -(gradient.clone() * (operand.clone() + operand.clone()) * output.clone())
            }
        };
        smallvec![Some(cotangent)]
    }
}

#[cfg(test)]
#[path = "tests/map_tests.rs"]
mod tests;
