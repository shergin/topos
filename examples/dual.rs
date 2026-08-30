//! Dual numbers as a research element: forward mode by
//! reinterpreting the payloads, where `examples/forward_mode.rs` got
//! it by transforming the graph.
//!
//! This file is the element seam doing what it is for. `Dual` is a
//! number — a value and how fast it changes — implementing
//! [`Differentiable`] and [`Elementary`] from outside the crate, and
//! nothing else changes: `Tape<Dual>` records the same primal graph
//! `Tape<f64>` records, shapes infer identically, and the unchanged
//! interpreter computes a directional derivative in every payload
//! slot as a side effect of ordinary arithmetic. Where
//! `element_seam.rs` proves the seam's hook is consulted, this
//! example proves the seam carries new *semantics*.
//!
//! The grading is a triangle at dyadic values: the dual
//! interpreter's tangent, the engine reverse scan, and the recorded
//! gradient must agree bit for bit. (The fourth route — the
//! forward-mode example's JVP over recorded payloads — is welded to
//! the same reverse scan by its own asserts, so all four readings
//! agree transitively.)
//!
//! Deliberately not implemented: the backend hooks stay on their
//! `None` defaults (a dual GEMM kernel is a research question of its
//! own), and duals do not emit — no `Emittable`.

use std::ops::{Add, Div, Mul, Neg, Sub};

use topos::{Detach, Differentiable, Element, Elementary, Tape};

/// A dual number: the primal value and the tangent riding along —
/// how fast the value changes when the variable of differentiation
/// nudges. The chain rule is the arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    pub primal: f64,
    pub tangent: f64,
}

impl Dual {
    /// A constant carried unchanged through differentiation: zero
    /// tangent.
    pub fn value(primal: f64) -> Self {
        Self {
            primal,
            tangent: 0.0,
        }
    }

    /// The variable of differentiation: unit tangent.
    pub fn var(primal: f64) -> Self {
        Self {
            primal,
            tangent: 1.0,
        }
    }
}

impl Add for Dual {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            primal: self.primal + rhs.primal,
            tangent: self.tangent + rhs.tangent,
        }
    }
}

impl Sub for Dual {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            primal: self.primal - rhs.primal,
            tangent: self.tangent - rhs.tangent,
        }
    }
}

impl Mul for Dual {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            primal: self.primal * rhs.primal,
            tangent: self.primal * rhs.tangent + self.tangent * rhs.primal,
        }
    }
}

impl Div for Dual {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let primal = self.primal / rhs.primal;
        Self {
            tangent: (self.tangent - primal * rhs.tangent) / rhs.primal,
            primal,
        }
    }
}

impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            primal: -self.primal,
            tangent: -self.tangent,
        }
    }
}

impl Differentiable for Dual {
    // Duals accumulate in their own precision: promoting a term
    // keeps its tangent, so an inner product's tangent is the sum of
    // the product-rule terms — exactly the derivative of the sum.
    type Accumulator = Self;

    fn promote(&self) -> Self {
        *self
    }

    fn demote(accumulated: Self) -> Self {
        accumulated
    }

    fn zero() -> Self {
        Self::value(0.0)
    }

    fn one() -> Self {
        Self::value(1.0)
    }

    fn from_count(count: usize) -> Self {
        Self::value(count as f64)
    }

    fn is_count(&self, count: usize) -> bool {
        self.primal == count as f64 && self.tangent == 0.0
    }
}

/// Every map is the chain rule on the primal's own kernel, called
/// through `Elementary` explicitly — `f64`'s *inherent* `exp` is the
/// standard library's, whose last bits vary by platform, while the
/// trait's rides `libm` — so the primal half of a dual run is
/// bit-identical to the `f64` run of the same graph.
impl Elementary for Dual {
    fn exp(&self) -> Self {
        let primal = Elementary::exp(&self.primal);
        Self {
            tangent: primal * self.tangent,
            primal,
        }
    }

    fn ln(&self) -> Self {
        Self {
            primal: Elementary::ln(&self.primal),
            tangent: self.tangent / self.primal,
        }
    }

    fn sqrt(&self) -> Self {
        let primal = Elementary::sqrt(&self.primal);
        Self {
            tangent: self.tangent / (primal + primal),
            primal,
        }
    }

    fn tanh(&self) -> Self {
        let primal = Elementary::tanh(&self.primal);
        Self {
            tangent: self.tangent * (1.0 - primal * primal),
            primal,
        }
    }

    fn sin(&self) -> Self {
        Self {
            primal: Elementary::sin(&self.primal),
            tangent: self.tangent * Elementary::cos(&self.primal),
        }
    }

    fn cos(&self) -> Self {
        Self {
            primal: Elementary::cos(&self.primal),
            tangent: -(self.tangent * Elementary::sin(&self.primal)),
        }
    }

    fn log1p(&self) -> Self {
        Self {
            primal: Elementary::log1p(&self.primal),
            tangent: self.tangent / (1.0 + self.primal),
        }
    }

    fn expm1(&self) -> Self {
        let primal = Elementary::expm1(&self.primal);
        Self {
            tangent: self.tangent * (primal + 1.0),
            primal,
        }
    }

    fn erf(&self) -> Self {
        Self {
            primal: Elementary::erf(&self.primal),
            tangent: self.tangent * Elementary::erf_derivative(&self.primal),
        }
    }

    fn erf_derivative(&self) -> Self {
        let primal = Elementary::erf_derivative(&self.primal);
        Self {
            tangent: self.tangent * (-2.0 * self.primal) * primal,
            primal,
        }
    }

    /// The dual of `x^y` on a positive base; elsewhere the `ln`
    /// answers `NaN`, mirroring the operation's own documented
    /// domain.
    fn powf(&self, exponent: Self) -> Self {
        let primal = Elementary::powf(&self.primal, exponent.primal);
        Self {
            tangent: primal
                * (exponent.tangent * Elementary::ln(&self.primal)
                    + exponent.primal * self.tangent / self.primal),
            primal,
        }
    }

    /// The primal wins by the `f64` rule; the tangent follows the
    /// winner, ties to the left like `step`'s ties-answer-one.
    fn maximum(&self, other: &Self) -> Self {
        Self {
            primal: Elementary::maximum(&self.primal, &other.primal),
            tangent: if self.primal >= other.primal {
                self.tangent
            } else {
                other.tangent
            },
        }
    }

    fn step(&self, threshold: &Self) -> Self {
        // The indicator is piecewise constant: zero tangent.
        Self::value(Elementary::step(&self.primal, &threshold.primal))
    }
}

impl Element for Dual {}

fn main() {
    // The README scalar chain, `loss = (w * x - y)^2`, at three
    // dyadic points; the derivative with respect to `w` three ways.
    // The points keep the error nonzero: at a zero error the two
    // modes agree on the value but not the sign of zero (the
    // product rule multiplies into `-0.0` where the reverse
    // accumulation adds into `+0.0`), and the grading demands bits.
    for (w0, x0, y0) in [(0.5, 0.25, 1.5), (-0.75, 1.25, 0.5), (2.0, -0.5, -1.25)] {
        // The f64 twin carries both reverse-mode routes: the engine
        // scan and the recorded gradient.
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

        // The dual twin records the identical primal spelling; the
        // derivative is read off the loss payload's tangent after an
        // ordinary forward run.
        let (dual_network, [dual_loss]) = Tape::record(|tape| {
            let w = tape.parameter(Dual::var(w0));
            let x = tape.input(Dual::value(x0));
            let y = tape.input(Dual::value(y0));
            let error = w * x - y;
            [error * error].detach()
        });
        let dual_run = dual_network.forward(&dual_network.parameters(), []);
        let slope = dual_run.of(dual_loss).scalar();

        assert_eq!(
            slope.tangent.to_bits(),
            engine.to_bits(),
            "the dual tangent must be the engine gradient, bit for bit"
        );
        assert_eq!(
            slope.tangent.to_bits(),
            recorded.to_bits(),
            "the dual tangent must be the recorded gradient, bit for bit"
        );
        println!(
            "d/dw (w*{x0} - {y0})^2 at w = {w0}: {} on all three routes",
            slope.tangent
        );
    }
    println!("forward mode by payload, reverse mode by scan: one answer");
}
