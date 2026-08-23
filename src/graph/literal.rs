use std::ops::{Add, Div, Mul, Sub};

use crate::{Bf16, Element, Tensor};

use super::Value;

// Coherence forbids the generic reverse (`impl Mul<Value<E>> for E`
// leaves the `E` parameter uncovered), so the foreign element types
// get concrete implementations instead; each records a rank-0 leaf,
// like the forward-order element sugar.
macro_rules! literal_operand_for {
    ($($element:ty),*) => {$(
        impl<'tape> Add<Value<'tape, $element>> for $element {
            type Output = Value<'tape, $element>;

            fn add(self, rhs: Value<'tape, $element>) -> Self::Output {
                rhs.literal(Tensor::from(self)) + rhs
            }
        }

        impl<'tape> Sub<Value<'tape, $element>> for $element {
            type Output = Value<'tape, $element>;

            fn sub(self, rhs: Value<'tape, $element>) -> Self::Output {
                rhs.literal(Tensor::from(self)) - rhs
            }
        }

        impl<'tape> Mul<Value<'tape, $element>> for $element {
            type Output = Value<'tape, $element>;

            fn mul(self, rhs: Value<'tape, $element>) -> Self::Output {
                rhs.literal(Tensor::from(self)) * rhs
            }
        }

        impl<'tape> Div<Value<'tape, $element>> for $element {
            type Output = Value<'tape, $element>;

            fn div(self, rhs: Value<'tape, $element>) -> Self::Output {
                rhs.literal(Tensor::from(self)) / rhs
            }
        }
    )*};
}

literal_operand_for!(f32, f64, Bf16);

// `Tensor` is local, so its reversed literal operators can stay generic.
impl<'tape, E: Element> Add<Value<'tape, E>> for Tensor<E> {
    type Output = Value<'tape, E>;

    fn add(self, rhs: Value<'tape, E>) -> Self::Output {
        rhs.literal(self) + rhs
    }
}

impl<'tape, E: Element> Sub<Value<'tape, E>> for Tensor<E> {
    type Output = Value<'tape, E>;

    fn sub(self, rhs: Value<'tape, E>) -> Self::Output {
        rhs.literal(self) - rhs
    }
}

impl<'tape, E: Element> Mul<Value<'tape, E>> for Tensor<E> {
    type Output = Value<'tape, E>;

    fn mul(self, rhs: Value<'tape, E>) -> Self::Output {
        rhs.literal(self) * rhs
    }
}

impl<'tape, E: Element> Div<Value<'tape, E>> for Tensor<E> {
    type Output = Value<'tape, E>;

    fn div(self, rhs: Value<'tape, E>) -> Self::Output {
        rhs.literal(self) / rhs
    }
}
