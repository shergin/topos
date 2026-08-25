use smallvec::smallvec;

use crate::Tensor;

use super::{Add, Cotangents, Operation};

#[test]
fn rules_are_plain_math_without_a_network() {
    assert_eq!(Add.arity(), 2);
    let sum = Add.forward(&[&Tensor::from(2.0_f64), &Tensor::from(3.0)]);
    assert_eq!(sum.scalar(), 5.0);
}

#[test]
fn backward_hands_one_cotangent_per_operand() {
    let operands = [Tensor::from(2.0_f64), Tensor::from(3.0)];
    let cotangents = Add.backward(
        &[&operands[0], &operands[1]],
        &Tensor::from(5.0),
        &Tensor::from(1.5),
    );
    let expected: Cotangents<Tensor<f64>> =
        smallvec![Some(Tensor::from(1.5)), Some(Tensor::from(1.5))];
    assert_eq!(cotangents, expected);
}
