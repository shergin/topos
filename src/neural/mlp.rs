use static_assertions::assert_impl_all;

use crate::{Element, Shape, Symbol, Tape, Tensor, Value};

use super::{Activation, Linear, Module, Segment, Visitor};

// Request-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Mlp<f64>: Send, Sync);

/// A multilayer perceptron: affine stages chained by topology, the
/// convenience constructor over [`Linear`] and [`Activation`].
///
/// A topology such as `[3, 4, 4, 1]` defines three stages that map a
/// `[batch, 3]` input to a `[batch, 1]` output. Hidden stages apply
/// the caller's [`Activation`] after their affine transform; the
/// output stage is affine alone. The contained stages retain
/// parameter [`Symbol`]s, so the perceptron records in each
/// compatible generation.
#[derive(Debug, Clone)]
pub struct Mlp<E> {
    stages: Vec<Linear<E>>,
    activation: Activation,
}

impl<E: Element> Mlp<E> {
    /// Allocates the perceptron's stages on `tape` and returns it.
    ///
    /// `sizes` lists the value widths from the input width to the
    /// output width. `activation` is applied after every hidden
    /// stage's affine transform — caller-owned, like every
    /// hyperparameter, with no default. `initializer` produces the
    /// initial payload for each parameter from its shape — `[inputs,
    /// outputs]` weights and `[outputs]` biases, stage by stage. The
    /// initializer is responsible for returning payloads with the
    /// requested shapes, and callers control details such as fan-in
    /// scaling, randomness, and symmetry breaking.
    ///
    /// # Panics
    /// Panics if `sizes` has fewer than two entries. It also propagates
    /// [`Linear::new`] validation failures if initialized weights and
    /// biases do not form valid parameter shapes.
    pub fn new(
        tape: &Tape<E>,
        sizes: &[usize],
        activation: Activation,
        mut initializer: impl FnMut(&Shape) -> Tensor<E>,
    ) -> Self {
        assert!(
            sizes.len() >= 2,
            "an MLP topology needs an input and an output width"
        );
        let stages = sizes
            .windows(2)
            .map(|pair| {
                let weights = initializer(&Shape::new([pair[0], pair[1]]));
                let bias = initializer(&Shape::new([pair[1]]));
                Linear::new(tape, weights, bias)
            })
            .collect();
        Self { stages, activation }
    }

    /// Returns the symbols of all parameters, stage by stage: each
    /// stage's weights, then its bias.
    pub fn parameters(&self) -> impl Iterator<Item = Symbol> + '_ {
        super::parameters(self).into_iter()
    }
}

impl<E: Element> Module<E> for Mlp<E> {
    /// Records the perceptron's expression over the `[batch, inputs]`
    /// value `input` and returns the `[batch, outputs]` output value.
    ///
    /// # Panics
    /// Panics if the parameters or `input` are not allocated on the
    /// input's tape, or if `input` and the initialized stage shapes
    /// are incompatible.
    fn express<'tape>(&self, input: Value<'tape, E>) -> Value<'tape, E> {
        let last = self.stages.len() - 1;
        self.stages
            .iter()
            .enumerate()
            .fold(input, |value, (index, stage)| {
                let affine = stage.express(value);
                if index == last {
                    affine
                } else {
                    self.activation.express(affine)
                }
            })
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
#[path = "tests/mlp_tests.rs"]
mod tests;
