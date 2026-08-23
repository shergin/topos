use smallvec::smallvec;

use crate::{Element, MapOperation, Shape, Tensor, Tensorial};

use super::{Cotangents, Operation, Reads, unary};

/// A unary elementwise transcendental of a value: one node kind
/// carrying the [`MapOperation`] it applies.
///
/// The IR and the backend map seam share this vocabulary on purpose:
/// `tanh` recorded here and `tanh` offered to the backend chain are
/// the same instruction, so adding a transcendental is a
/// `MapOperation` variant and the arms below — never a new
/// `Function` variant. Everything op-specific (the printed name, the
/// read set, the derivative) dispatches on `op`; the shape behavior
/// is shared, since a map always keeps its operand's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Map {
    pub(crate) op: MapOperation,
}

impl Map {
    /// Returns the operation's display name — `"Tanh"`, not `"Map"`:
    /// the node kind is an internal grouping, and dumps stay in the
    /// mnemonics the user recorded.
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rules below, per
    /// operation: `Exp`, `Sqrt`, and `Tanh` reuse their own output,
    /// while `Ln` divides by its operand. The set is deliberately not
    /// uniform — a shared one would retain buffers liveness does not
    /// need.
    pub(crate) fn reads(&self) -> Reads {
        match self.op {
            MapOperation::Exp | MapOperation::Sqrt | MapOperation::Tanh => Reads {
                operands: [false, false],
                output: true,
            },
            MapOperation::Ln => Reads {
                operands: [true, false],
                output: false,
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
        }
    }
}

impl<Rule: Tensorial> Operation<Rule> for Map {
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
        };
        smallvec![Some(cotangent)]
    }
}

#[cfg(test)]
#[path = "tests/map_tests.rs"]
mod tests;
