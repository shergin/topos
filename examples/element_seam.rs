//! Plugs the element seam from outside the crate: a third number type
//! joins the engine by implementing the element contracts, and nothing
//! else changes.
//!
//! The element here is `Audited`, an `f64` that counts how often the
//! backend chain is offered a matrix multiplication through its `gemm`
//! hook and answers it with the published reference kernel. The point
//! is what the example does *not* contain: no tensor code, no
//! derivative rules, no engine hooks — `unfold`, `backward`, plans,
//! and the notebook all come along from the contracts alone.
//!
//! Run with: `cargo run --example element_seam`

use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicUsize, Ordering};

use topos::{Differentiable, Element, Elementary, GemmTask, Numerics, Tape, Tensor, reference};

/// How many gemm tasks the seam has been offered, across all threads.
static OFFERED: AtomicUsize = AtomicUsize::new(0);

/// An `f64` in audit mode: arithmetic delegates, and the acceleration
/// seam counts its offers before answering with the reference kernel,
/// so the example can prove the hook was consulted.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Audited(f64);

impl Add for Audited {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Audited(self.0 + rhs.0)
    }
}

impl Sub for Audited {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Audited(self.0 - rhs.0)
    }
}

impl Mul for Audited {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Audited(self.0 * rhs.0)
    }
}

impl Div for Audited {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Audited(self.0 / rhs.0)
    }
}

impl Neg for Audited {
    type Output = Self;
    fn neg(self) -> Self {
        Audited(-self.0)
    }
}

impl Differentiable for Audited {
    type Accumulator = Self;

    fn promote(&self) -> Self {
        *self
    }

    fn demote(accumulated: Self) -> Self {
        accumulated
    }

    fn zero() -> Self {
        Audited(0.0)
    }

    fn one() -> Self {
        Audited(1.0)
    }

    fn from_count(count: usize) -> Self {
        Audited(count as f64)
    }

    fn is_count(&self, count: usize) -> bool {
        self.0 == count as f64
    }
}

impl Elementary for Audited {
    fn exp(&self) -> Self {
        Audited(self.0.exp())
    }

    fn ln(&self) -> Self {
        Audited(self.0.ln())
    }

    fn sqrt(&self) -> Self {
        Audited(self.0.sqrt())
    }

    fn tanh(&self) -> Self {
        Audited(self.0.tanh())
    }

    fn sin(&self) -> Self {
        Audited(self.0.sin())
    }

    fn cos(&self) -> Self {
        Audited(self.0.cos())
    }

    fn log1p(&self) -> Self {
        Audited(self.0.ln_1p())
    }

    fn expm1(&self) -> Self {
        Audited(self.0.exp_m1())
    }

    fn powf(&self, exponent: Self) -> Self {
        Audited(self.0.powf(exponent.0))
    }

    fn maximum(&self, other: &Self) -> Self {
        Audited(self.0.max(other.0))
    }

    fn step(&self, threshold: &Self) -> Self {
        Audited(if self.0 >= threshold.0 { 1.0 } else { 0.0 })
    }

    /// The acceleration seam: count the offer, then answer with the
    /// published reference kernel — the same bits the built-in path
    /// computes, which is what makes the differential test below
    /// meaningful.
    fn gemm(task: &GemmTask<'_, Self>) -> Option<Vec<Self>> {
        OFFERED.fetch_add(1, Ordering::Relaxed);
        Some(reference::multiply(task))
    }
}

impl Element for Audited {}

fn main() {
    // The differential test every in-crate element passes: the hook's
    // product must be bit-identical to the built-in slice path. The
    // `Exact` posture pins the built-in path to the reference bits, so
    // the comparison is against the oracle, not a backend.
    let elements: Vec<f64> = (0..64)
        .map(|at| ((at * 7 % 23) as f64 - 11.0) / 4.0)
        .collect();
    let audited_left = Tensor::new(
        [4, 16],
        elements.iter().map(|&e| Audited(e)).collect::<Vec<_>>(),
    );
    let audited_right = Tensor::new(
        [16, 4],
        elements
            .iter()
            .map(|&e| Audited(e * 0.5))
            .collect::<Vec<_>>(),
    );
    let product = audited_left.matmul(&audited_right);

    let left = Tensor::new([4, 16], elements.clone());
    let right = Tensor::new(
        [16, 4],
        elements.iter().map(|&e| e * 0.5).collect::<Vec<_>>(),
    );
    let oracle = Numerics::exactly(|| left.matmul(&right));

    let bits: Vec<u64> = product.to_vec().iter().map(|e| e.0.to_bits()).collect();
    let oracle_bits: Vec<u64> = oracle.to_vec().iter().map(|e| e.to_bits()).collect();
    assert_eq!(bits, oracle_bits);
    println!(
        "gemm hook consulted {} times; product matches the reference bit for bit",
        OFFERED.load(Ordering::Relaxed)
    );

    // The crate-root regression, recorded over the new element: the
    // whole training loop runs on `Audited` without one extra line.
    let tape: Tape<Audited> = Tape::new();
    let w = tape.parameter(Audited(0.0));
    let x = tape.input(Audited(0.0));
    let y = tape.input(Audited(0.0));
    let error = w * x - y;
    let loss = error * error;
    let (w, x, y, loss) = (w.symbol(), x.symbol(), y.symbol(), loss.symbol());
    let network = tape.into_network();
    let mut parameters = network.parameters();

    let samples = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
    for step in 0..100 {
        let (sample_x, sample_y) = samples[step % samples.len()];
        let run = network.forward(
            &parameters,
            [(x, Audited(sample_x).into()), (y, Audited(sample_y).into())],
        );
        let gradients = run.backward(loss).parameters(&parameters);
        parameters = parameters.step(&gradients, |w, g| {
            w.clone() - g.clone() * Tensor::from(Audited(0.02))
        });
    }
    let learned = parameters.of(w).scalar().0;
    assert!((learned - 2.0).abs() < 1e-6);
    println!("trained w = {learned:.6} on the audited element (target 2)");
}
