use crate::{Element, Tape, Value};

use super::{Module, Segment, Visitor};

/// An ordered chain of modules: each stage's output feeds the next.
///
/// Stages are heterogeneous behind `dyn Module`, which is the
/// sanctioned record-time exception to the static-dispatch rule:
/// expression happens once per topology and its cost never reaches a
/// run, while the static alternative (tuple arities behind macros)
/// cannot hold a depth chosen at runtime. [`Sequential::then`] boxes
/// internally, so call sites never spell `Box`.
pub struct Sequential<E> {
    stages: Vec<Box<dyn Module<E>>>,
}

impl<E: Element> Sequential<E> {
    /// Creates an empty chain: the identity until stages arrive.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Appends `stage` to the chain and returns it, builder style.
    pub fn then(mut self, stage: impl Module<E> + 'static) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Returns the number of stages.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Returns `true` if the chain holds no stages.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

impl<E: Element> Default for Sequential<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Element> Module<E> for Sequential<E> {
    fn express<'tape>(&self, tape: &'tape Tape<E>, input: Value<'tape, E>) -> Value<'tape, E> {
        self.stages
            .iter()
            .fold(input, |value, stage| stage.express(tape, value))
    }

    fn visit(&self, visitor: &mut dyn Visitor) {
        for (index, stage) in self.stages.iter().enumerate() {
            visitor.enter(Segment::Index(index));
            stage.visit(visitor);
            visitor.leave();
        }
    }
}

#[cfg(test)]
#[path = "tests/sequential_tests.rs"]
mod tests;
