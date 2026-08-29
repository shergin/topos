use super::{Origin, Symbol};

/// The kinship check every detached carrier makes when it meets a
/// name: same family, and a position this carrier covers.
///
/// Origin plus coverage is one concept, spelled here once. Each
/// carrier builds the check from its own origin and covered length
/// and supplies its own coverage wording, so a panic still says
/// which carrier rejected and why; the family message is shared
/// verbatim. Slot-grained carriers use [`Kinship::family`] alone
/// and keep their own coverage lookup.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Kinship {
    origin: Origin,
    length: usize,
}

impl Kinship {
    /// Returns the check of a carrier covering `length` nodes of the
    /// `origin` family.
    pub(crate) fn over(origin: Origin, length: usize) -> Self {
        Self { origin, length }
    }

    /// Asserts the family half alone: `symbol` names a node of this
    /// carrier's network.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network.
    pub(crate) fn family(&self, symbol: Symbol) {
        assert!(
            symbol.origin == self.origin,
            "symbol belongs to a different network"
        );
    }

    /// Locates `symbol` under the whole check, answering its
    /// position.
    ///
    /// # Panics
    /// Panics with the shared family message if `symbol` belongs to
    /// a different network, and with `coverage` if its position lies
    /// past this carrier's length.
    pub(crate) fn locate(&self, symbol: Symbol, coverage: &str) -> usize {
        self.family(symbol);
        assert!(symbol.id.index() < self.length, "{coverage}");
        symbol.id.index()
    }
}
