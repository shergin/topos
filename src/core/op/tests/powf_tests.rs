use smallvec::smallvec;

use crate::Tensor;

use super::{Cotangents, Operation, Powf};

#[test]
fn rules_are_plain_math_without_a_network() {
    assert_eq!(Powf.arity(), 2);
    let power = Powf.forward(&[&Tensor::from(2.0_f64), &Tensor::from(3.0)]);
    assert_eq!(power.scalar(), 8.0);
}

#[test]
fn backward_routes_the_power_and_exponential_rules() {
    // `d(x^y)/dx = y * x^(y-1) = 12`; `d(x^y)/dy = x^y * ln(x) = 8 ln 2`.
    let operands = [Tensor::from(2.0_f64), Tensor::from(3.0)];
    let cotangents = Powf.backward(
        &[&operands[0], &operands[1]],
        &Tensor::from(8.0),
        &Tensor::from(1.0),
    );
    let expected: Cotangents<Tensor<f64>> = smallvec![
        Some(Tensor::from(12.0)),
        Some(Tensor::from(8.0 * 2.0_f64.ln()))
    ];
    assert_eq!(cotangents, expected);
}

#[test]
fn exponent_gradient_is_undefined_for_negative_bases() {
    let operands = [Tensor::from(-2.0_f64), Tensor::from(2.0)];
    let cotangents = Powf.backward(
        &[&operands[0], &operands[1]],
        &Tensor::from(4.0),
        &Tensor::from(1.0),
    );
    assert_eq!(cotangents[0], Some(Tensor::from(-4.0)));
    assert!(cotangents[1].clone().unwrap().scalar().is_nan());
}
