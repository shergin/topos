use smallvec::smallvec;

use super::{Cotangents, Map, MapOperation, Operation};

#[test]
fn rules_are_plain_math_without_a_network() {
    let sqrt = Map {
        op: MapOperation::Sqrt,
    };
    assert_eq!(sqrt.arity(), 1);
    let root: f64 = sqrt.forward(&[&9.0]);
    assert_eq!(root, 3.0);

    let exp = Map {
        op: MapOperation::Exp,
    };
    let raised: f64 = exp.forward(&[&0.0]);
    assert_eq!(raised, 1.0);
}

#[test]
fn sqrt_backward_divides_by_twice_the_output() {
    // `d sqrt(x) / dx` at 9 is `1 / (2 * 3)`.
    let sqrt = Map {
        op: MapOperation::Sqrt,
    };
    let cotangents = sqrt.backward(&[&9.0_f64], &3.0, &6.0);
    let expected: Cotangents<f64> = smallvec![Some(1.0)];
    assert_eq!(cotangents, expected);
}

#[test]
fn exp_backward_reuses_the_output() {
    // `d e^x / dx` is the output itself.
    let exp = Map {
        op: MapOperation::Exp,
    };
    let cotangents = exp.backward(&[&2.0_f64], &3.0, &2.0);
    let expected: Cotangents<f64> = smallvec![Some(6.0)];
    assert_eq!(cotangents, expected);
}

#[test]
fn ln_backward_divides_by_the_operand() {
    // `d ln(x) / dx` at 4 is `1 / 4`.
    let ln = Map {
        op: MapOperation::Ln,
    };
    let cotangents = ln.backward(&[&4.0_f64], &4.0_f64.ln(), &1.0);
    let expected: Cotangents<f64> = smallvec![Some(0.25)];
    assert_eq!(cotangents, expected);
}

#[test]
fn tanh_backward_squares_the_output() {
    // `d tanh(x) / dx` at output 0.5 is `1 - 0.25`.
    let tanh = Map {
        op: MapOperation::Tanh,
    };
    let cotangents = tanh.backward(&[&0.0_f64], &0.5, &1.0);
    let expected: Cotangents<f64> = smallvec![Some(0.75)];
    assert_eq!(cotangents, expected);
}

#[test]
fn reads_follow_the_operation() {
    // Output-reusing rules must not retain their operand, and `Ln`
    // must retain exactly its operand; liveness depends on this.
    for op in [MapOperation::Exp, MapOperation::Sqrt, MapOperation::Tanh] {
        let reads = Map { op }.reads();
        assert!(reads.output);
        assert!(!reads.operands[0]);
    }
    let reads = Map {
        op: MapOperation::Ln,
    }
    .reads();
    assert!(!reads.output);
    assert!(reads.operands[0]);
}

#[test]
fn names_print_the_operation_not_the_kind() {
    let names: Vec<&str> = [
        MapOperation::Exp,
        MapOperation::Ln,
        MapOperation::Sqrt,
        MapOperation::Tanh,
    ]
    .into_iter()
    .map(|op| Map { op }.name())
    .collect();
    assert_eq!(names, ["Exp", "Ln", "Sqrt", "Tanh"]);
}
