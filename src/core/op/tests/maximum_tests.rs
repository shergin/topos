use smallvec::smallvec;

use crate::Tensor;

use super::{Cotangents, Maximum, Operation};

#[test]
fn rules_are_plain_math_without_a_network() {
    assert_eq!(Maximum.arity(), 2);
    let larger = Maximum.forward(&[&Tensor::from(2.0_f64), &Tensor::from(3.0)]);
    assert_eq!(larger.scalar(), 3.0);
}

#[test]
fn backward_hands_the_gradient_to_the_winner() {
    let operands = [Tensor::from(2.0_f64), Tensor::from(3.0)];
    let cotangents = Maximum.backward(
        &[&operands[0], &operands[1]],
        &Tensor::from(3.0),
        &Tensor::from(1.5),
    );
    let expected: Cotangents<Tensor<f64>> =
        smallvec![Some(Tensor::from(0.0)), Some(Tensor::from(1.5))];
    assert_eq!(cotangents, expected);
}

#[test]
fn backward_hands_ties_to_the_left_operand() {
    let operands = [Tensor::from(2.0_f64), Tensor::from(2.0)];
    let cotangents = Maximum.backward(
        &[&operands[0], &operands[1]],
        &Tensor::from(2.0),
        &Tensor::from(1.5),
    );
    let expected: Cotangents<Tensor<f64>> =
        smallvec![Some(Tensor::from(1.5)), Some(Tensor::from(0.0))];
    assert_eq!(cotangents, expected);
}
