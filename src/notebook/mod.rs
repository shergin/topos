//! Rich display for Evcxr notebooks and the Evcxr REPL.
//!
//! The module adds no types and no vocabulary. It implements
//! `to_html` and `evcxr_display` on the types the crate already
//! has. Setup, training loops, and the cell-output table live in
//! [`docs/notebooks.md`](../../docs/notebooks.md).
//!
//! These methods exist only with the `evcxr` feature: they need
//! `malevich` for charts, and the default build does not take that
//! dependency. That is a deliberate exception to "features change
//! behavior, not the surface."
//!
//! # Persisting across cells
//!
//! Evcxr compiles every cell as its own crate and keeps the
//! variables between them.
//!
//! **A persisted variable cannot borrow another one.** A
//! [`Value`](crate::Value) borrows its tape, so it dies with the
//! cell; [`Symbol`](crate::Symbol) is the cross-cell name. End a
//! recording cell with `.symbol()` and reenter through
//! [`Tape::resolve`](crate::Tape::resolve).
//!
//! **A persisted variable needs an explicit type.** Evcxr infers a
//! cell from that cell alone. Annotate every binding that must
//! survive — the same as `let v = vec![1.0_f64];`.

mod adjoints;
mod entry;
mod field;
mod html;
mod network;
mod parameters;
mod plan;
mod render;
mod tape;
mod tensor;
mod value;
