//! `topos` is a tiny autograd engine for the GPU-poor.
//!
//! Expressions record a static computation graph onto a [`Tape`];
//! sealing it yields an immutable [`Network`] (the spec) and a
//! caller-owned [`Parameters`] value (the state). `forward`
//! materializes every value, `backward` differentiates one scalar
//! target, and `step` is a pure data transformation of the
//! parameters — training never touches the graph:
//!
//! ```
//! use topos::{Tape, Tensor};
//!
//! let tape = Tape::new();
//! let w = tape.parameter(0.0_f64);
//! let x = tape.input(0.0);
//! let y = tape.input(0.0);
//!
//! // Operators record the graph; values are `Copy` and never consumed.
//! // A scalar is a rank-0 tensor: the graph is always tensors, and the
//! // element type (`f64` here) is the open seam.
//! let error = w * x - y;
//! let loss = error * error;
//!
//! // Symbols are the detached names every later phase speaks.
//! let (w, x, y, loss) = (w.symbol(), x.symbol(), y.symbol(), loss.symbol());
//! let network = tape.into_network();
//! let mut parameters = network.parameters();
//!
//! // The graph is recorded once; every step feeds one sample of the line
//! // `y = 2 * x` and steps the parameters, leaving the network untouched.
//! let samples = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
//! for step in 0..100 {
//!     let (sample_x, sample_y) = samples[step % samples.len()];
//!     let run = network.forward(&parameters, [(x, sample_x.into()), (y, sample_y.into())]);
//!     let gradients = run.backward(loss).parameters(&parameters);
//!     parameters = parameters.step(&gradients, |w, g| {
//!         w.clone() - g.clone() * Tensor::from(0.02)
//!     });
//! }
//!
//! let learned = parameters.of(w).scalar();
//! assert!((learned - 2.0).abs() < 1e-6);
//! ```
//!
//! Differentiation comes in a hierarchy of three, in this order of
//! recommendation. [`Tape::differentiate`] records the chain rule as
//! ordinary nodes and answers [`Adjoints`] — the derivative as spec:
//! compile a forward-only plan over `adjoints.roots()` and fusion and
//! liveness apply to the chain rule itself, with
//! [`Run::recorded_gradients`] bridging to `step`. [`Run::backward`]
//! (the loop above) is the interpreter applying the same rules
//! without recording — the oracle the transform is proven against
//! bitwise, shipped forever. [`Request::backward`] is neither: a
//! memory posture that retains what the engine scan reads, so a plan
//! that did not record its derivative can still answer `backward`.
// The default build forbids `unsafe` outright. A backend feature
// drops `forbid` but keeps the crate-wide `deny`, so `unsafe`
// outside a scope-allowed backend module stays a compile error.
#![cfg_attr(
    not(any(
        feature = "accelerate",
        feature = "metal",
        feature = "simd",
        feature = "cuda"
    )),
    forbid(unsafe_code)
)]
#![deny(unsafe_code)]

mod backend;
mod emission;
mod engine;
mod function;
mod graph;
mod neural;
#[cfg(feature = "evcxr")]
mod notebook;
mod payload;
pub mod reference;

pub use backend::{
    Backend, BackendUnavailable, Coverage, Dispatch, Fidelity, Formula, MapTask, Numerics,
    Precision,
};
pub use emission::{EmitError, Emittable};
pub use engine::{Plan, Request, Run};
pub use graph::{
    Adjoints, Field, Gradients, Network, Parameters, Symbol, Tape, Trace, Value, concat, stack,
};
pub use neural::{
    Activation, Adam, AdamW, BatchNorm, Conv2d, Dropout, LayerNorm, Linear, Mlp, Module,
    Normalization, Optimizer, Path, RmsNorm, Segment, Sequential, Sgd, Visitor, checkpoint, conv2d,
    cross_entropy, init, max_pool, named_parameters, parameters,
};
pub use payload::{
    BatchNormTask, Bf16, Differentiable, Element, Elementary, GemmTask, MapOperation, Normalized,
    Shape, Tensor, Tensorial,
};
