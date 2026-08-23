use smallvec::smallvec;

use crate::Tensor;

use super::{Cotangents, Map, MapOperation, Operation};

#[test]
fn rules_are_plain_math_without_a_network() {
    let sqrt = Map {
        op: MapOperation::Sqrt,
    };
    assert_eq!(sqrt.arity(), 1);
    let root = sqrt.forward(&[&Tensor::from(9.0_f64)]);
    assert_eq!(root.scalar(), 3.0);

    let exp = Map {
        op: MapOperation::Exp,
    };
    let raised = exp.forward(&[&Tensor::from(0.0_f64)]);
    assert_eq!(raised.scalar(), 1.0);
}

#[test]
fn sqrt_backward_divides_by_twice_the_output() {
    // `d sqrt(x) / dx` at 9 is `1 / (2 * 3)`.
    let sqrt = Map {
        op: MapOperation::Sqrt,
    };
    let cotangents = sqrt.backward(
        &[&Tensor::from(9.0_f64)],
        &Tensor::from(3.0),
        &Tensor::from(6.0),
    );
    let expected: Cotangents<Tensor<f64>> = smallvec![Some(Tensor::from(1.0))];
    assert_eq!(cotangents, expected);
}

#[test]
fn exp_backward_reuses_the_output() {
    // `d e^x / dx` is the output itself.
    let exp = Map {
        op: MapOperation::Exp,
    };
    let cotangents = exp.backward(
        &[&Tensor::from(2.0_f64)],
        &Tensor::from(3.0),
        &Tensor::from(2.0),
    );
    let expected: Cotangents<Tensor<f64>> = smallvec![Some(Tensor::from(6.0))];
    assert_eq!(cotangents, expected);
}

#[test]
fn ln_backward_divides_by_the_operand() {
    // `d ln(x) / dx` at 4 is `1 / 4`.
    let ln = Map {
        op: MapOperation::Ln,
    };
    let cotangents = ln.backward(
        &[&Tensor::from(4.0_f64)],
        &Tensor::from(4.0_f64.ln()),
        &Tensor::from(1.0),
    );
    let expected: Cotangents<Tensor<f64>> = smallvec![Some(Tensor::from(0.25))];
    assert_eq!(cotangents, expected);
}

#[test]
fn tanh_backward_squares_the_output() {
    // `d tanh(x) / dx` at output 0.5 is `1 - 0.25`.
    let tanh = Map {
        op: MapOperation::Tanh,
    };
    let cotangents = tanh.backward(
        &[&Tensor::from(0.0_f64)],
        &Tensor::from(0.5),
        &Tensor::from(1.0),
    );
    let expected: Cotangents<Tensor<f64>> = smallvec![Some(Tensor::from(0.75))];
    assert_eq!(cotangents, expected);
}

#[test]
fn sin_backward_takes_the_cosine_of_the_operand() {
    // `d sin(x) / dx` at 0 is `cos(0) = 1`.
    let sin = Map {
        op: MapOperation::Sin,
    };
    let cotangents = sin.backward(
        &[&Tensor::from(0.0_f64)],
        &Tensor::from(0.0),
        &Tensor::from(2.0),
    );
    let expected: Cotangents<Tensor<f64>> = smallvec![Some(Tensor::from(2.0))];
    assert_eq!(cotangents, expected);
}

#[test]
fn cos_backward_negates_the_sine_of_the_operand() {
    // `d cos(x) / dx` at `pi / 2` is `-sin(pi / 2) = -1`.
    let cos = Map {
        op: MapOperation::Cos,
    };
    let half_pi = std::f64::consts::FRAC_PI_2;
    let cotangents = cos.backward(
        &[&Tensor::from(half_pi)],
        &Tensor::from(half_pi.cos()),
        &Tensor::from(3.0),
    );
    let expected: Cotangents<Tensor<f64>> = smallvec![Some(Tensor::from(-(3.0 * half_pi.sin())))];
    assert_eq!(cotangents, expected);
}

#[test]
fn reads_follow_the_operation() {
    // Output-reusing rules must not retain their operand; `Ln`,
    // `Sin`, and `Cos` must retain exactly their operand; liveness
    // depends on this.
    for op in [MapOperation::Exp, MapOperation::Sqrt, MapOperation::Tanh] {
        let reads = Map { op }.reads();
        assert!(reads.output);
        assert!(!reads.operands[0]);
    }
    for op in [MapOperation::Ln, MapOperation::Sin, MapOperation::Cos] {
        let reads = Map { op }.reads();
        assert!(!reads.output);
        assert!(reads.operands[0]);
    }
}

#[test]
fn names_print_the_operation_not_the_kind() {
    let names: Vec<&str> = [
        MapOperation::Exp,
        MapOperation::Ln,
        MapOperation::Sqrt,
        MapOperation::Tanh,
        MapOperation::Sin,
        MapOperation::Cos,
    ]
    .into_iter()
    .map(|operation| crate::Opcode::Map { operation }.name())
    .collect();
    assert_eq!(names, ["Exp", "Ln", "Sqrt", "Tanh", "Sin", "Cos"]);
}
