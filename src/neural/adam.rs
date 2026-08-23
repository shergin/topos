use crate::{Element, Parameters, Symbol, Tensor};

use super::Optimizer;
use super::optimizer::assert_single_value;

/// The Adam optimizer: gradient descent through bias-corrected first
/// and second moment estimates (Kingma and Ba, 2015).
///
/// Hyperparameters are single-value payloads written by the caller
/// (`Tensor::new([], [0.9])`), like every learning rate in the
/// examples: `0.9` is not mintable from a generic payload, and the
/// values belong on the page. The moment tables are parameter-aligned
/// ([`Parameters`](crate::Parameters) algebra) and initialize lazily
/// from the first step's gradients, and the bias-correction powers
/// `beta^t` are carried as payloads multiplied once per step — exact
/// and deterministic, with no float-of-step-count conversion. Every
/// step is pure parameter-table algebra over pure payload arithmetic,
/// so two identical runs cannot differ.
#[derive(Debug, Clone)]
pub struct Adam<E> {
    beta1: Tensor<E>,
    beta2: Tensor<E>,
    epsilon: Tensor<E>,
    /// `1 - beta`, precomputed once: the incoming gradient's share of
    /// each moment update.
    first_share: Tensor<E>,
    second_share: Tensor<E>,
    first: Option<Parameters<E>>,
    second: Option<Parameters<E>>,
    /// The running `beta^t` each correction divides out of its moment.
    beta1_power: Tensor<E>,
    beta2_power: Tensor<E>,
}

impl<E: Element> Adam<E> {
    /// Creates the optimizer with the given decay rates and the
    /// denominator's stabilizer; the conventional values are `0.9`,
    /// `0.999`, and `1e-8`.
    ///
    /// # Panics
    /// Panics if any hyperparameter holds more than one value.
    pub fn new(beta1: Tensor<E>, beta2: Tensor<E>, epsilon: Tensor<E>) -> Self {
        assert_single_value(&beta1, "beta1");
        assert_single_value(&beta2, "beta2");
        assert_single_value(&epsilon, "epsilon");
        Self {
            first_share: beta1.one_like() - beta1.clone(),
            second_share: beta2.one_like() - beta2.clone(),
            beta1_power: beta1.one_like(),
            beta2_power: beta2.one_like(),
            beta1,
            beta2,
            epsilon,
            first: None,
            second: None,
        }
    }

    /// Advances the moments by `gradients` and returns the
    /// bias-corrected update direction
    /// `first / (sqrt(second) + epsilon)`: the shared core of `Adam`
    /// and [`AdamW`].
    fn direction(&mut self, gradients: &Parameters<E>) -> Parameters<E> {
        let zeros = || gradients.map(|gradient| gradient.zero_like());
        let first = self.first.take().unwrap_or_else(zeros);
        let second = self.second.take().unwrap_or_else(zeros);

        let first = first.scale(&self.beta1) + gradients.scale(&self.first_share);
        let squared = gradients.map(|gradient| gradient.clone() * gradient.clone());
        let second = second.scale(&self.beta2) + squared.scale(&self.second_share);

        self.beta1_power = self.beta1_power.clone() * self.beta1.clone();
        self.beta2_power = self.beta2_power.clone() * self.beta2.clone();
        let first_correction = self.beta1_power.one_like() - self.beta1_power.clone();
        let second_correction = self.beta2_power.one_like() - self.beta2_power.clone();

        let direction = first.zip(&second, |first, second| {
            let corrected_first = first.clone() / first_correction.broadcast_like(first);
            let corrected_second = second.clone() / second_correction.broadcast_like(second);
            corrected_first
                / (corrected_second.sqrt() + self.epsilon.broadcast_like(&corrected_second))
        });
        self.first = Some(first);
        self.second = Some(second);
        direction
    }
}

impl<E: Element> Optimizer<E> for Adam<E> {
    fn step(
        &mut self,
        parameters: &Parameters<E>,
        gradients: &Parameters<E>,
        learning_rate: &Tensor<E>,
    ) -> Parameters<E> {
        let direction = self.direction(gradients);
        parameters.step(&direction, |parameter, direction| {
            parameter.clone() - direction.clone() * learning_rate.broadcast_like(direction)
        })
    }
}

/// Adam with decoupled weight decay (Loshchilov and Hutter, 2019):
/// the same moment machinery, with `learning_rate * decay * parameter`
/// subtracted directly from the parameters the policy selects.
///
/// The default policy is structural, needing no registry: decay
/// applies to parameters of rank two and above (weights) and never to
/// rank-one parameters (biases, norm gains) — the standard convention,
/// decided per parameter from its payload's shape through the
/// identity-aware [`Parameters::step_each`]. Any other policy is a
/// caller predicate through [`AdamW::step_where`].
#[derive(Debug, Clone)]
pub struct AdamW<E> {
    adam: Adam<E>,
    decay: Tensor<E>,
}

impl<E: Element> AdamW<E> {
    /// Creates the optimizer; `decay` is the decoupled weight-decay
    /// factor applied per step alongside the conventional Adam rates.
    ///
    /// # Panics
    /// Panics if any hyperparameter holds more than one value.
    pub fn new(beta1: Tensor<E>, beta2: Tensor<E>, epsilon: Tensor<E>, decay: Tensor<E>) -> Self {
        assert_single_value(&decay, "decay");
        Self {
            adam: Adam::new(beta1, beta2, epsilon),
            decay,
        }
    }

    /// Steps like [`Optimizer::step`] with `policy` deciding, per
    /// parameter, whether decay applies — for conventions the default
    /// structural rule does not express (a symbol set works, since
    /// [`Symbol`] is `Eq + Hash`; the current payload carries the
    /// parameter's shape).
    pub fn step_where(
        &mut self,
        parameters: &Parameters<E>,
        gradients: &Parameters<E>,
        learning_rate: &Tensor<E>,
        mut policy: impl FnMut(Symbol, &Tensor<E>) -> bool,
    ) -> Parameters<E> {
        let direction = self.adam.direction(gradients);
        parameters.step_each(&direction, |symbol, current, direction| {
            let stepped =
                current.clone() - direction.clone() * learning_rate.broadcast_like(direction);
            if policy(symbol, current) {
                let decayed = current.clone()
                    * self.decay.broadcast_like(current)
                    * learning_rate.broadcast_like(current);
                stepped - decayed
            } else {
                stepped
            }
        })
    }
}

impl<E: Element> Optimizer<E> for AdamW<E> {
    fn step(
        &mut self,
        parameters: &Parameters<E>,
        gradients: &Parameters<E>,
        learning_rate: &Tensor<E>,
    ) -> Parameters<E> {
        self.step_where(parameters, gradients, learning_rate, |_, parameter| {
            parameter.shape().rank() >= 2
        })
    }
}

#[cfg(test)]
#[path = "tests/adam_tests.rs"]
mod adam_tests;
