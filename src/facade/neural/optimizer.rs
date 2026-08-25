use crate::{Element, Parameters, Tensor};

/// A training-step strategy: how gradients and the current parameter
/// state become the next state.
///
/// It is the loop-land analogue of what [`Activation`](super::Activation)
/// is to a layer — a uniform slot the training loop can hand any
/// strategy — kept an *open* trait rather than a closed enum on
/// purpose: custom optimizers are ordinary implementations with the
/// same standing as the built-in ones, holding whatever state they
/// need as plain fields ([`Parameters`](crate::Parameters) algebra is
/// the designed state carrier: moments and velocities are
/// parameter-aligned tables). The trait is object-safe, so a
/// comparison loop can iterate `&mut dyn Optimizer` implementations
/// side by side.
///
/// The learning rate is a per-step argument, not optimizer state:
/// schedules stay caller-owned loop arithmetic, visible on the page
/// like every other training decision.
pub trait Optimizer<E: Element> {
    /// Returns `parameters` stepped by `gradients` at `learning_rate`,
    /// updating this optimizer's own state.
    ///
    /// The gradients arrive parameter-aligned:
    /// [`Run::recorded_gradients`](crate::Run::recorded_gradients)
    /// answers this grain directly, and an engine
    /// [`backward`](crate::Run::backward) projects through
    /// [`Field::parameters`](crate::Field::parameters).
    fn step(
        &mut self,
        parameters: &Parameters<E>,
        gradients: &Parameters<E>,
        learning_rate: &Tensor<E>,
    ) -> Parameters<E>;
}

/// Plain stochastic gradient descent: the strategy every example's
/// hand-written loop applies, as the trait's simplest implementation —
/// stateless, so the struct is a unit, and every richer optimizer is
/// this plus state.
///
/// The examples hand-roll this update on purpose and that is the
/// decision, on record: under the caller-owned doctrine the update
/// arithmetic is the pedagogy, so the copies across the examples are
/// the documented price, not an adoption gap. `Sgd`'s consumer of
/// record is the optimizer comparison loop (`mlp_adam`), where a
/// baseline must be a strategy the loop can iterate — exactly the
/// facade-tier bar: a hand-rolled equivalent behaves identically, and
/// this type exists for the call sites that need the *slot*, not the
/// arithmetic.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sgd;

impl<E: Element> Optimizer<E> for Sgd {
    fn step(
        &mut self,
        parameters: &Parameters<E>,
        gradients: &Parameters<E>,
        learning_rate: &Tensor<E>,
    ) -> Parameters<E> {
        parameters.step(gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        })
    }
}

/// Asserts that an optimizer hyperparameter holds exactly one value,
/// the contract every scalar factor spreads from.
pub(super) fn assert_single_value<E: Element>(payload: &Tensor<E>, name: &str) {
    assert_eq!(
        payload.shape().volume(),
        1,
        "optimizer {name} must hold a single value"
    );
}

#[cfg(test)]
#[path = "tests/optimizer_tests.rs"]
mod tests;
