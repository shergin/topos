//! The dual element's crate-level gate: the same asserts the
//! example makes, so `cargo test` guards the seam without running
//! examples, plus the accumulator check the example keeps off its
//! page.
//!
//! The number type is the example's own, included by path — one
//! `Dual`, two consumers — and everything here goes through
//! `topos::` re-exports, so the out-of-tree claim is
//! compiler-checked.

// The example's `main` is its reading; here only the number is used.
#[allow(dead_code)]
#[path = "../examples/dual.rs"]
mod example;

use example::Dual;
use topos::{Detach, Tape, Tensor};

#[test]
fn the_dual_tangent_is_the_reverse_gradient_bitwise() {
    for (w0, x0, y0) in [(0.5, 0.25, 1.5), (-0.75, 1.25, 0.5), (2.0, -0.5, -1.25)] {
        let tape: Tape<f64> = Tape::new();
        let w = tape.parameter(w0);
        let x = tape.input(x0);
        let y = tape.input(y0);
        let error = w * x - y;
        let loss = error * error;
        let adjoints = tape.differentiate(loss, [w]);
        let (w, loss) = (w.symbol(), loss.symbol());
        let network = tape.into_network();
        let run = network.forward(&network.parameters(), []);
        let engine = run.backward(loss).of(w).scalar();
        let recorded = run.of(adjoints.pairs()[0].1).scalar();

        let (dual_network, [dual_loss]) = Tape::record(|tape| {
            let w = tape.parameter(Dual::var(w0));
            let x = tape.input(Dual::value(x0));
            let y = tape.input(Dual::value(y0));
            let error = w * x - y;
            [error * error].detach()
        });
        let dual_run = dual_network.forward(&dual_network.parameters(), []);
        let slope = dual_run.of(dual_loss).scalar();

        assert_eq!(slope.primal.to_bits(), run.of(loss).scalar().to_bits());
        assert_eq!(slope.tangent.to_bits(), engine.to_bits());
        assert_eq!(slope.tangent.to_bits(), recorded.to_bits());
    }
}

#[test]
fn a_product_tangent_matches_the_hand_derivative() {
    // One varying entry in a 2x2 product: the tangent of row 0 is
    // the corresponding row of the right operand (the accumulator
    // carries the product-rule terms through the inner sum), and
    // row 1 does not depend on the variable at all.
    let left = Tensor::new(
        [2, 2],
        vec![
            Dual::var(0.5),
            Dual::value(-0.25),
            Dual::value(1.5),
            Dual::value(0.75),
        ],
    );
    let right = Tensor::new(
        [2, 2],
        vec![
            Dual::value(2.0),
            Dual::value(-0.5),
            Dual::value(0.25),
            Dual::value(1.25),
        ],
    );
    let product = left.matmul(&right);
    let elements = product.to_vec();

    assert_eq!(elements[0].tangent.to_bits(), 2.0_f64.to_bits());
    assert_eq!(elements[1].tangent.to_bits(), (-0.5_f64).to_bits());
    assert_eq!(elements[2].tangent.to_bits(), 0.0_f64.to_bits());
    assert_eq!(elements[3].tangent.to_bits(), 0.0_f64.to_bits());

    // The primal half is the plain f64 product of the primal parts.
    let primal_left = Tensor::new([2, 2], vec![0.5_f64, -0.25, 1.5, 0.75]);
    let primal_right = Tensor::new([2, 2], vec![2.0_f64, -0.5, 0.25, 1.25]);
    let primal = primal_left.matmul(&primal_right).to_vec();
    for (dual, primal) in elements.iter().zip(&primal) {
        assert_eq!(dual.primal.to_bits(), primal.to_bits());
    }
}
