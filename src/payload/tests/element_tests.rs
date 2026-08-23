use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{Differentiable, Element, Elementary, Tape, Tensor};

/// A third element type implemented by delegation: the seam
/// demonstration. Everything the graph does — recording, forward,
/// backward — comes along from the element contracts alone.
#[derive(Debug, Clone, Copy, PartialEq)]
struct F64(f64);

impl Add for F64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        F64(self.0 + rhs.0)
    }
}

impl Sub for F64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        F64(self.0 - rhs.0)
    }
}

impl Mul for F64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        F64(self.0 * rhs.0)
    }
}

impl Div for F64 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        F64(self.0 / rhs.0)
    }
}

impl Neg for F64 {
    type Output = Self;
    fn neg(self) -> Self {
        F64(-self.0)
    }
}

impl Differentiable for F64 {
    type Accumulator = Self;

    fn promote(&self) -> Self {
        *self
    }

    fn demote(accumulated: Self) -> Self {
        accumulated
    }

    fn zero() -> Self {
        F64(0.0)
    }

    fn one() -> Self {
        F64(1.0)
    }

    fn from_count(count: usize) -> Self {
        F64(count as f64)
    }

    fn is_count(&self, count: usize) -> bool {
        self.0 == count as f64
    }
}

impl Elementary for F64 {
    fn exp(&self) -> Self {
        F64(Elementary::exp(&self.0))
    }

    fn ln(&self) -> Self {
        F64(Elementary::ln(&self.0))
    }

    fn sqrt(&self) -> Self {
        F64(Elementary::sqrt(&self.0))
    }

    fn tanh(&self) -> Self {
        F64(Elementary::tanh(&self.0))
    }

    fn sin(&self) -> Self {
        F64(Elementary::sin(&self.0))
    }

    fn cos(&self) -> Self {
        F64(Elementary::cos(&self.0))
    }

    fn log1p(&self) -> Self {
        F64(Elementary::log1p(&self.0))
    }

    fn expm1(&self) -> Self {
        F64(Elementary::expm1(&self.0))
    }

    fn erf(&self) -> Self {
        F64(Elementary::erf(&self.0))
    }

    fn erf_derivative(&self) -> Self {
        F64(Elementary::erf_derivative(&self.0))
    }

    fn powf(&self, exponent: Self) -> Self {
        F64(Elementary::powf(&self.0, exponent.0))
    }

    fn maximum(&self, other: &Self) -> Self {
        F64(self.0.max(other.0))
    }

    fn step(&self, threshold: &Self) -> Self {
        F64(if self.0 >= threshold.0 { 1.0 } else { 0.0 })
    }
}

impl Element for F64 {}

#[test]
fn a_delegating_element_differentiates_like_its_double() {
    // The same expression recorded over the new element and over the
    // built-in `f64` must produce bit-identical results everywhere:
    // the seam changes the number type, never the engine.
    let record = |probe: f64| {
        let tape: Tape<f64> = Tape::new();
        let x = tape.parameter(probe);
        let loss = ((x * x).tanh() + x.exp()).symbol();
        let x = x.symbol();
        let network = tape.into_network();
        let run = network.forward(&network.parameters(), []);
        (run.of(loss).scalar(), run.backward(loss).of(x).scalar())
    };
    let record_delegated = |probe: f64| {
        let tape: Tape<F64> = Tape::new();
        let x = tape.parameter(F64(probe));
        let loss = ((x * x).tanh() + x.exp()).symbol();
        let x = x.symbol();
        let network = tape.into_network();
        let run = network.forward(&network.parameters(), []);
        (run.of(loss).scalar().0, run.backward(loss).of(x).scalar().0)
    };

    for probe in [-1.5, -0.25, 0.0, 0.65, 2.0] {
        let (value, gradient) = record(probe);
        let (delegated_value, delegated_gradient) = record_delegated(probe);
        assert_eq!(value.to_bits(), delegated_value.to_bits());
        assert_eq!(gradient.to_bits(), delegated_gradient.to_bits());
    }
}

#[test]
fn a_delegating_element_matmuls_like_its_double() {
    let elements: Vec<f64> = (0..12).map(|at| (at as f64 - 5.0) / 3.0).collect();
    let left = Tensor::new([3, 4], elements.clone());
    let right = Tensor::new([4, 2], elements[..8].to_vec());
    let product = left.matmul(&right);

    let delegated_left = Tensor::new([3, 4], elements.iter().map(|&e| F64(e)).collect::<Vec<_>>());
    let delegated_right = Tensor::new(
        [4, 2],
        elements[..8].iter().map(|&e| F64(e)).collect::<Vec<_>>(),
    );
    let delegated = delegated_left.matmul(&delegated_right);

    let bits: Vec<u64> = product.to_vec().iter().map(|e| e.to_bits()).collect();
    let delegated_bits: Vec<u64> = delegated.to_vec().iter().map(|e| e.0.to_bits()).collect();
    assert_eq!(bits, delegated_bits);
}
