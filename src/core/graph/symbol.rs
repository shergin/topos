use static_assertions::assert_impl_all;

use super::{Origin, ValueId};

// Entry-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Symbol: Send, Sync, Copy);

/// A detached, `Copy` name for a recorded value: the currency of every
/// phase after recording.
///
/// Unlike [`Value`](crate::Value), a symbol carries no tape borrow and
/// no payload; it is an origin plus a node position, valid across
/// threads, notebook cells, checkpoints, and
/// [`Network::into_tape`](crate::Network::into_tape) round trips —
/// linear extension never moves a recorded node. Take one with
/// [`Value::symbol`](crate::Value::symbol) before sealing the tape;
/// read through it with [`Parameters::of`](crate::Parameters::of),
/// [`Run::of`](crate::Run::of), or [`Field::of`](crate::Field::of),
/// and turn it back into a proxy with
/// [`Tape::resolve`](crate::Tape::resolve) when a network reopens.
///
/// The origin participates in equality and hashing, so equally
/// positioned nodes from unrelated graphs do not compare equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol {
    pub(crate) origin: Origin,
    pub(crate) id: ValueId,
}

impl Symbol {
    /// Returns the position of the named node on its tape: the
    /// number [`Network::describe`](crate::Network::describe) prints
    /// for the node and its operands. Nodes never move, so the
    /// position is stable for the life of the family.
    ///
    /// The position is meaningful only within one family. Symbol
    /// equality, which includes the origin, is identity; comparing
    /// indices across networks means nothing.
    pub fn index(self) -> usize {
        self.id.index()
    }
}
