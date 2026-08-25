use smallvec::smallvec;

use crate::{Element, Recordable, Shape, Tensor};

use super::{Cotangents, Operation, Reads, binary};

/// An embedding-style row gather with operands `[table, selection]`:
/// `output[i] = table[selection[i]]`, where the selection is a one-hot
/// `[count, vocab]` payload whose vocabulary is the table's first axis.
///
/// The gradient flows only to the table, `dtable[selection[i]] += grad[i]`
/// (a scatter-add that accumulates repeated rows). The selection is data,
/// not a differentiable value, so its cotangent is `None`: the
/// non-differentiability of the indices is a structural property of this
/// operation rather than a runtime flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Gather;

impl Gather {
    /// Returns the arity: two operands.
    pub(crate) fn arity(&self) -> usize {
        2
    }

    /// Returns the read set of the derivative rule below.
    /// It reads the selection payload (the scatter needs its indices);
    /// the table contributes nothing.
    pub(crate) fn reads(&self) -> Reads {
        Reads {
            operands: [false, true],
            output: false,
        }
    }

    /// Infers the result shape `[count, ...table.shape[1..]]`, requiring the
    /// selection to be rank 2 and its vocabulary to match the table's rows.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (table, selection) = binary(operands);
        assert_eq!(
            selection.rank(),
            2,
            "gather selection must be rank 2 [count, vocab], got {selection}"
        );
        assert!(
            table.rank() >= 1,
            "gather table needs at least one axis, got {table}"
        );
        assert_eq!(
            selection.axes()[1],
            table.axes()[0],
            "gather selection vocabulary {} does not match table rows {}",
            selection.axes()[1],
            table.axes()[0]
        );
        Shape::new(std::iter::once(selection.axes()[0]).chain(table.axes()[1..].iter().copied()))
    }
}

impl Gather {
    pub(crate) fn forward<E: Element>(&self, operands: &[&Tensor<E>]) -> Tensor<E> {
        let (&table, &selection) = binary(operands);
        table.gather(selection)
    }
}

impl<Rule: Recordable> Operation<Rule> for Gather {
    fn backward(&self, operands: &[&Rule], _output: &Rule, gradient: &Rule) -> Cotangents<Rule> {
        let (_, &selection) = binary(operands);
        smallvec![Some(gradient.scatter(selection)), None]
    }
}
