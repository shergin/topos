//! `topos` is an autodiff compiler stack. Record a graph, inspect
//! it, differentiate it, compile it, emit it. The spec is an
//! immutable [`Network`]; the state is a caller-owned
//! [`Parameters`].
//!
//! Expressions record onto a [`Tape`]. Sealing yields the network.
//! `forward` materializes every value, `backward` differentiates one
//! scalar target, and `step` is a pure data transform of the
//! parameters. Training never touches the graph:
//!
//! ```
//! use topos::{Keep, Tape, Tensor};
//!
//! // Record the graph in one closure; the return value is the
//! // keep-set, detached to symbols in one call. Operators record as
//! // they run; values are `Copy` and never consumed. A scalar is a
//! // rank-0 tensor: the graph is always tensors, and the element
//! // type (`f64` here) is the open seam.
//! let (network, [w, x, y, loss]) = Tape::record(|tape| {
//!     let w = tape.parameter(0.0_f64);
//!     let x = tape.input(0.0);
//!     let y = tape.input(0.0);
//!     let error = w * x - y;
//!     [w, x, y, error * error].keep()
//! });
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
//! lower a forward-only entry over `adjoints.roots()` and fusion and
//! liveness apply to the chain rule itself, with
//! [`Run::recorded_gradients`] bridging to `step`. [`Run::backward`]
//! (the loop above) is the interpreter applying the same rules
//! without recording — the oracle the transform is proven against
//! bitwise, shipped forever. [`Entry::backward`] is neither: a
//! memory posture that retains what the engine scan reads, so a plan
//! that did not record its derivative can still answer `backward`.
//!
//! # The stack: one spec, named interpretations
//!
//! The tape is the spec; everything after it is a derived
//! interpretation of the same columns, each with a printable
//! artifact, and the whole compiler is this list:
//!
//! ```text
//! spec       Tape / Network        record, describe
//! shape      inferred at record    panics at the recording expression
//! value      BoundEntry::interpret the oracle; Network::forward is the whole-spec form
//! cotangent  Run::backward         the engine reverse scan, oracle of reverse mode
//! trace      Tape::differentiate   the same rules recording themselves (Trace)
//! schedule   BoundEntry::lower     Plan: keep-set, liveness, election; describe
//! catalog    Plan::patterns        elected offers as data, never rewrites
//! text       Plan::emit_stablehlo  the interchange boundary
//! ```
//!
//! The value and cotangent rows compute over [`Tensor`]; the trace
//! row records over [`Trace`] — one derivative-rule body, two
//! interpretations of the recordable vocabulary ([`Recordable`]).
//! A new idea plugs in at a named seam, costed like an opcode: an
//! element type at [`Element`], a transcendental at [`MapOperation`],
//! a fusion as a pattern plus matcher, an AD mode as a recording
//! interpretation proven against [`Run::backward`], an industrial
//! target as an emission sibling consuming [`Plan`]. The core stays
//! closed; the table is how the crate refuses a pass manager and
//! still says yes to research.
//!
//! # Two surfaces, one crate
//!
//! Two audiences read this crate, and each has a map — rustdoc
//! modules that only re-export, so `use topos::Tape` keeps working
//! and nothing moves:
//!
//! - [`model`] — write a network, train it, checkpoint it: the
//!   recording and run types, the neural facades, the optimizers.
//! - [`compiler`] — inspect, lower, emit, extend: the printable IR
//!   ([`Opcode`], [`Node`], `describe`), the catalog as data, the
//!   recording interpretation ([`Trace`]), the element seam, the
//!   backend interrogation types, and the [`reference`](mod@reference)
//!   kernels.
//!
//! ```no_run
//! # use topos::{Keep, Tape};
//! # let (network, [loss]) = Tape::record(|tape| {
//! #     let w = tape.parameter(1.0_f64);
//! #     [w * w].keep()
//! # });
//! // The compiler surface in three lines: print the spec, lower an
//! // entry, emit the schedule.
//! println!("{}", network.describe());
//! let plan = network.entry([loss]).lower();
//! println!("{}", plan.emit_stablehlo().expect("every operation lowers"));
//! ```
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

// The tiers are folders; the modules keep their flat crate paths, so
// `crate::graph` reads the same wherever the files sit. `core` is
// never re-exported publicly: `topos::Tape` is the only spelling.

// Core: the spec and its named readings. What the crate exists to
// do -- record a spec, then read it.
mod core;
pub(crate) use core::{engine, graph, op, payload};

// Derived: faster or foreign readings of the same spec. A backend may
// decline and the interpreter remains the truth; emission writes the
// plan as text and is a sibling of `describe`, not a second compiler.
mod derived;
pub(crate) use derived::{backend, emission};

// Facades: convenience on the public surface, composed through it
// alone, with no privileged engine access.
mod facade;
pub(crate) use facade::neural;
// `notebook` adds inherent methods to types the crate already owns, so
// nothing ever names its path; declaring it inside the tier is enough.

// Tools: the bitwise references a new element is graded against.
pub mod reference;

pub use backend::{
    Backend, BackendUnavailable, Coverage, Dispatch, Fidelity, Formula, MapTask, Numerics,
    Precision,
};
pub use emission::{EmitError, Emittable};
pub use engine::{BoundEntry, Entry, PatternKind, PatternMatch, Plan, Run};
pub use graph::{
    Adjoints, Field, Gradients, Keep, Network, Node, Opcode, Parameters, Symbol, Tape, Trace,
    Value, concat, stack,
};
pub use neural::{
    Activation, Adam, AdamW, BatchNorm, Conv2d, Dropout, LayerNorm, Linear, Mlp, Module,
    Normalization, Optimizer, Path, RmsNorm, Segment, Sequential, Sgd, Visitor, checkpoint, conv2d,
    cross_entropy, init, max_pool, named_parameters, parameters,
};
pub use payload::{
    BatchNormTask, Bf16, Differentiable, Element, Elementary, GemmTask, MapOperation, Normalized,
    Recordable, Shape, Tensor,
};

/// The model surface: write a network, train it, checkpoint it.
///
/// Everything here re-exports the crate root — `use topos::model::*`
/// is enough to record, run, and train, and flat `use topos::Tape`
/// imports keep working unchanged.
pub mod model {
    pub use crate::{
        Activation, Adam, AdamW, Adjoints, BatchNorm, BoundEntry, Conv2d, Dropout, Entry, Field,
        Gradients, Keep, LayerNorm, Linear, Mlp, Module, Network, Normalization, Optimizer,
        Parameters, Path, Plan, RmsNorm, Run, Segment, Sequential, Sgd, Shape, Symbol, Tape,
        Tensor, Value, Visitor, checkpoint, concat, conv2d, cross_entropy, init, max_pool,
        named_parameters, parameters, stack,
    };
}

/// The compiler surface: inspect, lower, emit, and extend the stack.
///
/// The closed IR view, the catalog as data, the recording
/// interpretation, the element seam, and the backend interrogation
/// types — everything a research consumer reads that a training loop
/// never mentions. All re-exports of the crate root; the reference
/// kernels live in [`crate::reference`].
pub mod compiler {
    pub use crate::{
        Adjoints, Backend, BackendUnavailable, BatchNormTask, Bf16, BoundEntry, Coverage,
        Differentiable, Dispatch, Element, Elementary, EmitError, Emittable, Entry, Fidelity,
        Formula, GemmTask, Keep, MapOperation, MapTask, Network, Node, Normalized, Numerics,
        Opcode, Parameters, PatternKind, PatternMatch, Plan, Precision, Recordable, Run, Shape,
        Symbol, Tape, Tensor, Trace,
    };
}
