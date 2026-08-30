use std::marker::PhantomData;

use static_assertions::assert_impl_all;

use crate::{Element, Symbol, Tape, Tensor, Value};

use super::{Module, Visitor};

// Entry-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Embedding<f64>: Send, Sync);

/// The embedding lookup `table.gather(selection)`: one `[vocab, dim]`
/// parameter behind the gather that is the whole formula.
///
/// The input is a one-hot `[count, vocab]` selection — feed a
/// [`Tensor::selection`](crate::Tensor::selection) as a per-run input
/// so one recorded graph serves any batch of indices — and the output
/// is the selected `[count, dim]` rows. Layout beyond the lookup
/// (concatenating a context window, adding a position table) stays
/// with the caller; a position table is simply a second `Embedding`.
///
/// Tying goes through the exposed [`table`](Embedding::table) symbol,
/// like [`Linear::weights`](super::Linear::weights): a tied
/// language-model head is a matmul with the resolved table, not a
/// second module.
#[derive(Debug, Clone)]
pub struct Embedding<E> {
    table: Symbol,
    _marker: PhantomData<E>,
}

impl<E: Element> Embedding<E> {
    /// Allocates the `[vocab, dim]` table on `tape` from its initial
    /// payload and returns the module. Callers own initialization;
    /// the module records whatever it is given.
    ///
    /// # Panics
    /// Panics if `table` is not rank 2.
    pub fn new(tape: &Tape<E>, table: Tensor<E>) -> Self {
        let shape = table.shape();
        assert_eq!(
            shape.rank(),
            2,
            "an embedding table must be rank 2 [vocab, dim], got {shape}"
        );
        Self {
            table: tape.parameter(table).symbol(),
            _marker: PhantomData,
        }
    }

    /// Returns the symbol of the `[vocab, dim]` table.
    pub fn table(&self) -> Symbol {
        self.table
    }

    /// Returns the symbols of the module's parameters: the table.
    pub fn parameters(&self) -> impl Iterator<Item = Symbol> + '_ {
        super::parameters(self).into_iter()
    }
}

impl<E: Element> Module<E> for Embedding<E> {
    /// Records the lookup of the one-hot `[count, vocab]` `input`
    /// and returns the selected `[count, dim]` rows.
    ///
    /// # Panics
    /// Panics if the table is not allocated on the input's tape, or
    /// if the gather's shapes disagree.
    fn express<'tape>(&self, input: Value<'tape, E>) -> Value<'tape, E> {
        let table = input.tape().resolve(self.table);
        table.gather(input)
    }

    // The path segment is `weights`, like `Linear`'s: the in-repo
    // checkpoint mappings already spell embedding tables that way,
    // so migrating a table onto this module keeps its paths.
    fn visit(&self, visitor: &mut dyn Visitor) {
        visitor.parameter("weights", self.table);
    }
}

#[cfg(test)]
#[path = "tests/embedding_tests.rs"]
mod tests;
