//! Forward mode as an out-of-tree AD mode: the first consumer of
//! the spec read surface, and this file is the seam's proof.
//!
//! Nothing here is library machinery. `Dual` is an ordinary
//! `Recordable` written against the public surface — forward-mode
//! knowledge is dual *arithmetic*, a payload, never a second rule
//! body — and the walker below replays a frozen spec with
//! `Opcode::express`. Over `Dual<Tensor>` the walk computes a
//! directional derivative eagerly; over `Dual<Trace>` it records
//! the tangent computation as ordinary spec (sources resolve into
//! the same family, so parameters are shared; interiors re-record
//! beside their tangents, and a lowered entry's dead-node
//! elimination drops whichever twins it does not name).
//!
//! Three gradings, all bitwise at dyadic values:
//! - the eager JVP equals the recorded JVP run through the engine;
//! - on a rational spec, both equal the reverse gradient
//!   contracted with the seed;
//! - forward-over-reverse equals reverse-over-reverse on a
//!   Hessian-vector product, because the recorded gradient is just
//!   more spec.

use std::ops::{Add, Div, Mul, Neg, Sub};

use topos::{Detach, Network, Node, Numerics, Recordable, Shape, Symbol, Tape, Tensor, Trace};

/// A dual number over any recordable payload: the primal value and
/// the tangent riding along. The chain rule is the arithmetic.
#[derive(Clone, Debug)]
struct Dual<R> {
    primal: R,
    tangent: R,
}

impl<R: Recordable> Dual<R> {
    /// Wraps a value carried unchanged through differentiation: a
    /// constant's tangent is zero.
    fn constant(primal: R) -> Self {
        let tangent = primal.zero_like();
        Self { primal, tangent }
    }
}

impl<R: Recordable> Add for Dual<R> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            primal: self.primal + rhs.primal,
            tangent: self.tangent + rhs.tangent,
        }
    }
}

impl<R: Recordable> Sub for Dual<R> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            primal: self.primal - rhs.primal,
            tangent: self.tangent - rhs.tangent,
        }
    }
}

impl<R: Recordable> Mul for Dual<R> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            primal: self.primal.clone() * rhs.primal.clone(),
            tangent: self.primal * rhs.tangent + self.tangent * rhs.primal,
        }
    }
}

impl<R: Recordable> Div for Dual<R> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let primal = self.primal / rhs.primal.clone();
        let tangent = (self.tangent - primal.clone() * rhs.tangent) / rhs.primal;
        Self { primal, tangent }
    }
}

impl<R: Recordable> Neg for Dual<R> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            primal: -self.primal,
            tangent: -self.tangent,
        }
    }
}

impl<R: Recordable> Recordable for Dual<R> {
    fn shape(&self) -> Shape {
        self.primal.shape()
    }

    fn zero_like(&self) -> Self {
        Self::constant(self.primal.zero_like())
    }

    fn one_like(&self) -> Self {
        Self::constant(self.primal.one_like())
    }

    fn exp(&self) -> Self {
        let primal = self.primal.exp();
        let tangent = primal.clone() * self.tangent.clone();
        Self { primal, tangent }
    }

    fn ln(&self) -> Self {
        Self {
            primal: self.primal.ln(),
            tangent: self.tangent.clone() / self.primal.clone(),
        }
    }

    fn sqrt(&self) -> Self {
        let primal = self.primal.sqrt();
        let tangent = self.tangent.clone() / (primal.clone() + primal.clone());
        Self { primal, tangent }
    }

    fn tanh(&self) -> Self {
        let primal = self.primal.tanh();
        let tangent = self.tangent.clone() * (primal.one_like() - primal.clone() * primal.clone());
        Self { primal, tangent }
    }

    fn sin(&self) -> Self {
        Self {
            primal: self.primal.sin(),
            tangent: self.tangent.clone() * self.primal.cos(),
        }
    }

    fn cos(&self) -> Self {
        Self {
            primal: self.primal.cos(),
            tangent: -(self.tangent.clone() * self.primal.sin()),
        }
    }

    fn log1p(&self) -> Self {
        Self {
            primal: self.primal.log1p(),
            tangent: self.tangent.clone() / (self.primal.one_like() + self.primal.clone()),
        }
    }

    fn expm1(&self) -> Self {
        let primal = self.primal.expm1();
        let tangent = self.tangent.clone() * (primal.clone() + primal.one_like());
        Self { primal, tangent }
    }

    fn erf(&self) -> Self {
        Self {
            primal: self.primal.erf(),
            tangent: self.tangent.clone() * self.primal.erf_derivative(),
        }
    }

    fn erf_derivative(&self) -> Self {
        // The scaled Gaussian differentiates to `-2x` times itself.
        let primal = self.primal.erf_derivative();
        let tangent =
            -((self.primal.clone() + self.primal.clone()) * primal.clone() * self.tangent.clone());
        Self { primal, tangent }
    }

    fn powf(&self, exponent: Self) -> Self {
        let primal = self.primal.powf(exponent.primal.clone());
        let tangent = primal.clone()
            * (exponent.tangent * self.primal.ln()
                + exponent.primal * (self.tangent.clone() / self.primal.clone()));
        Self { primal, tangent }
    }

    fn maximum(&self, other: &Self) -> Self {
        let winners = self.primal.step(&other.primal);
        Self {
            primal: self.primal.maximum(&other.primal),
            tangent: winners.clone() * self.tangent.clone()
                + (winners.one_like() - winners) * other.tangent.clone(),
        }
    }

    fn step(&self, threshold: &Self) -> Self {
        Self::constant(self.primal.step(&threshold.primal))
    }

    fn matmul(&self, rhs: &Self) -> Self {
        Self {
            primal: self.primal.matmul(&rhs.primal),
            tangent: self.tangent.matmul(&rhs.primal) + self.primal.matmul(&rhs.tangent),
        }
    }

    fn sum(&self) -> Self {
        Self {
            primal: self.primal.sum(),
            tangent: self.tangent.sum(),
        }
    }

    fn sum_along(&self, axis: usize) -> Self {
        Self {
            primal: self.primal.sum_along(axis),
            tangent: self.tangent.sum_along(axis),
        }
    }

    fn logsumexp(&self, axis: usize) -> Self {
        let primal = self.primal.logsumexp(axis);
        let extent = self.primal.shape().axes()[axis];
        // The softmax weights, from the stable primal itself.
        let weights = (self.primal.clone() - primal.broadcast_along(axis, extent)).exp();
        Self {
            tangent: (weights * self.tangent.clone()).sum_along(axis),
            primal,
        }
    }

    fn log_softmax(&self, axis: usize) -> Self {
        let primal = self.primal.log_softmax(axis);
        let extent = self.primal.shape().axes()[axis];
        let mass = self.tangent.sum_along(axis).broadcast_along(axis, extent);
        Self {
            tangent: self.tangent.clone() - primal.exp() * mass,
            primal,
        }
    }

    fn broadcast(&self, shape: Shape) -> Self {
        Self {
            primal: self.primal.broadcast(shape.clone()),
            tangent: self.tangent.broadcast(shape),
        }
    }

    fn broadcast_along(&self, axis: usize, extent: usize) -> Self {
        Self {
            primal: self.primal.broadcast_along(axis, extent),
            tangent: self.tangent.broadcast_along(axis, extent),
        }
    }

    fn reshape(&self, shape: Shape) -> Self {
        Self {
            primal: self.primal.reshape(shape.clone()),
            tangent: self.tangent.reshape(shape),
        }
    }

    fn permute(&self, order: &[usize]) -> Self {
        Self {
            primal: self.primal.permute(order),
            tangent: self.tangent.permute(order),
        }
    }

    fn narrow(&self, axis: usize, start: usize, len: usize) -> Self {
        Self {
            primal: self.primal.narrow(axis, start, len),
            tangent: self.tangent.narrow(axis, start, len),
        }
    }

    fn pad(&self, axis: usize, start: usize, full_extent: usize) -> Self {
        Self {
            primal: self.primal.pad(axis, start, full_extent),
            tangent: self.tangent.pad(axis, start, full_extent),
        }
    }

    fn unfold(&self, axis: usize, size: usize, step: usize, dilation: usize) -> Self {
        Self {
            primal: self.primal.unfold(axis, size, step, dilation),
            tangent: self.tangent.unfold(axis, size, step, dilation),
        }
    }

    fn fold(&self, axis: usize, size: usize, step: usize, dilation: usize, extent: usize) -> Self {
        Self {
            primal: self.primal.fold(axis, size, step, dilation, extent),
            tangent: self.tangent.fold(axis, size, step, dilation, extent),
        }
    }

    fn gather(&self, selection: &Self) -> Self {
        // The selection is data, not a differentiable dependency.
        Self {
            primal: self.primal.gather(&selection.primal),
            tangent: self.tangent.gather(&selection.primal),
        }
    }

    fn scatter(&self, selection: &Self) -> Self {
        Self {
            primal: self.primal.scatter(&selection.primal),
            tangent: self.tangent.scatter(&selection.primal),
        }
    }
}

/// Replays every node of a spec over dual payloads: `source`
/// supplies the sources' duals, and every computed node is one
/// `Opcode::express` over its operands' duals.
fn jvp_walk<R: Recordable>(
    nodes: &[Node],
    mut source: impl FnMut(&Node) -> Dual<R>,
) -> Vec<Dual<R>> {
    let mut duals: Vec<Dual<R>> = Vec::new();
    for node in nodes {
        let dual = if node.is_source() {
            source(node)
        } else {
            let operands: Vec<&Dual<R>> = node
                .operands()
                .iter()
                .map(|symbol| &duals[symbol.index()])
                .collect();
            node.opcode().express(&operands)
        };
        duals.push(dual);
    }
    duals
}

/// The eager JVP: walks the spec over `Dual<Tensor>` and answers
/// the tangent at `read` for a `seed` planted at `wrt`.
fn jvp_eager(network: &Network<f64>, wrt: Symbol, seed: &Tensor<f64>, read: Symbol) -> Tensor<f64> {
    let nodes: Vec<Node> = network.nodes().collect();
    let duals = Numerics::exactly(|| {
        jvp_walk(&nodes, |node| {
            let primal = network
                .payload(node.symbol())
                .expect("sources hold payloads")
                .clone();
            let tangent = if node.symbol() == wrt {
                seed.clone()
            } else {
                primal.zero_like()
            };
            Dual { primal, tangent }
        })
    });
    duals[read.index()].tangent.clone()
}

fn assert_bitwise(expected: &Tensor<f64>, computed: &Tensor<f64>, subject: &str) {
    let expected = expected.to_vec();
    let computed = computed.to_vec();
    assert_eq!(expected.len(), computed.len(), "{subject}: length differs");
    for (expected, computed) in expected.iter().zip(&computed) {
        assert_eq!(
            expected.to_bits(),
            computed.to_bits(),
            "{subject}: {computed} differs from {expected}"
        );
    }
}

fn main() {
    // Dyadic values keep every product and short sum exact, so the
    // gradings below can demand bits, not tolerances (the
    // transcendental part compares two runs of the same arithmetic,
    // which is bitwise regardless).

    // Part one: a transcendental spec, eager against recorded. The
    // same dual arithmetic runs over `Dual<Tensor>` (computing) and
    // `Dual<Trace>` (recording tangent spec beside resolved
    // sources); the engine then runs the recorded form, and the two
    // answers must agree bit for bit — the closure contract of an
    // out-of-tree mode, exactly as `differentiate` is welded to
    // `Run::backward`.
    let (network, [weights, _inputs, loss]) = Tape::record(|tape| {
        let weights = tape.parameter(Tensor::new(
            [3, 2],
            vec![0.5, -0.25, 0.125, 0.75, -0.5, 0.25],
        ));
        let inputs = tape.input(Tensor::new([2, 3], vec![0.5, -1.0, 0.25, -0.75, 0.5, 1.25]));
        let scores = inputs.matmul(weights).tanh().log_softmax(1);
        let loss = (scores * scores).sum();
        [weights, inputs, loss].detach()
    });
    let seed = Tensor::new([3, 2], vec![0.25, -0.5, 1.0, 0.125, -0.25, 0.5]);

    let eager = jvp_eager(&network, weights, &seed, loss);

    let nodes: Vec<Node> = network.nodes().collect();
    let tape = network.into_tape();
    let duals = jvp_walk(&nodes, |node| {
        // Sources resolve into the same family — parameters are
        // shared, not duplicated — and their tangents are fresh
        // leaves: the seed at `wrt`, zero elsewhere.
        let primal = Trace::of(tape.resolve(node.symbol()));
        let payload = tape.payload(node.symbol()).expect("sources hold payloads");
        let tangent_payload = if node.symbol() == weights {
            seed.clone()
        } else {
            payload.zero_like()
        };
        let tangent = Trace::of(tape.leaf(tangent_payload));
        Dual { primal, tangent }
    });
    let tangent_symbol = duals[loss.index()].tangent.value().symbol();
    drop(duals);
    let jvp_network = tape.into_network();
    let recorded_run = jvp_network.forward(&jvp_network.parameters(), []);
    assert_bitwise(
        &eager,
        recorded_run.of(tangent_symbol),
        "recorded jvp against eager",
    );
    println!("eager and recorded jvp agree bitwise: {}", eager.scalar());

    // Part two: on a rational spec the directional derivative is
    // also the reverse gradient contracted with the seed, and at
    // dyadic values every route computes the same rational number —
    // so forward mode and reverse mode agree bit for bit.
    let (cubic, [x, value]) = Tape::record(|tape| {
        let x = tape.parameter(Tensor::new([3], vec![0.5, -0.25, 1.5]));
        let value = (x * x * x).sum();
        [x, value].detach()
    });
    let direction = Tensor::new([3], vec![0.25, 1.0, -0.5]);
    let forward_slope = jvp_eager(&cubic, x, &direction, value);
    let run = cubic.forward(&cubic.parameters(), []);
    let gradients = run.backward(value);
    let reverse_slope = Numerics::exactly(|| (gradients.of(x).clone() * direction.clone()).sum());
    assert_bitwise(&forward_slope, &reverse_slope, "slope against the gradient");
    println!(
        "forward slope equals <gradient, seed> bitwise: {}",
        forward_slope.scalar()
    );

    // Part three: the recorded gradient is just more spec, so
    // forward mode walks straight over it — forward-over-reverse
    // and reverse-over-reverse compute the same Hessian-vector
    // product, bit for bit at dyadic values.
    let tape: Tape<f64> = Tape::new();
    let x = tape.parameter(Tensor::new([3], vec![0.5, -0.25, 1.5]));
    let value = (x * x * x).sum();
    let adjoints = tape.differentiate(value, [x]);
    let &(_, gradient_symbol) = &adjoints.pairs()[0];
    let direction_leaf = tape.leaf(direction.clone());
    let hessian_adjoints = tape.vjp(gradient_symbol, direction_leaf, [x]);
    let &(_, reverse_hvp_symbol) = &hessian_adjoints.pairs()[0];
    let x = x.symbol();
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    let reverse_hvp = run.of(reverse_hvp_symbol);
    let forward_hvp = jvp_eager(&network, x, &direction, gradient_symbol);
    assert_bitwise(
        reverse_hvp,
        &forward_hvp,
        "forward-over-reverse against reverse-over-reverse",
    );
    println!(
        "hessian-vector product agrees across modes: {:?}",
        forward_hvp.to_vec()
    );
}
