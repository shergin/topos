//! The executor tier: the interpreting [`Run`], the compiled [`Plan`]
//! with its compile [`Entry`], and the forward entry points on the
//! sealed spec. Everything here reads the graph tier; nothing below
//! it depends on it.

mod entry;
mod pattern;
mod plan;
mod run;

pub use entry::{BoundEntry, Entry};
pub use plan::Plan;
pub use run::Run;

pub(crate) use pattern::{BatchNormalization, Catalog, Pattern, ReduceWindow, WindowProduct};
pub(crate) use run::Posture;
