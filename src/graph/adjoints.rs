use super::Symbol;

/// The recorded reverse-mode result: one gradient symbol per `wrt`
/// entry, in `wrt` order, paired with the entry it differentiates.
///
/// [`Tape::differentiate`](super::Tape::differentiate) and
/// [`Tape::vjp`](super::Tape::vjp) return this carrier instead of bare
/// symbols because their product exists to be paired — each gradient
/// with its `wrt` entry for
/// [`Run::recorded_gradients`](crate::Run::recorded_gradients), and
/// all of them with the target for a training request's roots.
/// Holding the pairs makes misordered pairs unrepresentable: no
/// consumer rebuilds the pairing by parallel-vector discipline.
///
/// The carrier is plain data — detached symbols, no tape borrow — so
/// it survives sealing the tape and crosses threads like any
/// [`Symbol`].
#[derive(Debug, Clone)]
pub struct Adjoints {
    target: Symbol,
    pairs: Vec<(Symbol, Symbol)>,
}

impl Adjoints {
    /// Wraps a transform's product: the differentiated `target` and
    /// its `(wrt, gradient)` pairs in `wrt` order.
    pub(crate) fn new(target: Symbol, pairs: Vec<(Symbol, Symbol)>) -> Self {
        Self { target, pairs }
    }

    /// Returns the differentiated value: the scalar loss for
    /// [`Tape::differentiate`](super::Tape::differentiate), the seeded
    /// value — any shape — for [`Tape::vjp`](super::Tape::vjp).
    pub fn target(&self) -> Symbol {
        self.target
    }

    /// Returns the `(wrt, gradient)` pairs in `wrt` order.
    pub fn pairs(&self) -> &[(Symbol, Symbol)] {
        &self.pairs
    }

    /// Returns the `wrt` entries in their original order.
    pub fn wrt(&self) -> impl Iterator<Item = Symbol> + '_ {
        self.pairs.iter().map(|&(wrt, _)| wrt)
    }

    /// Returns the gradient symbols in `wrt` order.
    pub fn gradients(&self) -> impl Iterator<Item = Symbol> + '_ {
        self.pairs.iter().map(|&(_, gradient)| gradient)
    }

    /// Returns the gradient recorded for `wrt`.
    ///
    /// # Panics
    /// Panics if `wrt` was not an entry of the transform that produced
    /// this carrier.
    pub fn of(&self, wrt: Symbol) -> Symbol {
        self.pairs
            .iter()
            .find(|&&(entry, _)| entry == wrt)
            .map(|&(_, gradient)| gradient)
            .expect("no gradient was recorded for this symbol; it was not a `wrt` entry")
    }

    /// Returns the training roots: the target, then every gradient in
    /// `wrt` order — the exact root list a compiled training plan
    /// wants, so `Request::roots(adjoints.roots())` replaces the
    /// hand-chained loss-plus-gradients idiom.
    pub fn roots(&self) -> impl Iterator<Item = Symbol> + '_ {
        std::iter::once(self.target).chain(self.gradients())
    }

    /// Returns a new carrier with each gradient symbol rewritten by
    /// `rewrite`, pairing and target preserved.
    ///
    /// The emission consumers use this to substitute same-shape
    /// reshape aliases for the raw gradient nodes — pinning the
    /// emitted result order — without ever holding gradients as a
    /// bare list.
    pub fn map_gradients(&self, mut rewrite: impl FnMut(Symbol) -> Symbol) -> Self {
        Self {
            target: self.target,
            pairs: self
                .pairs
                .iter()
                .map(|&(wrt, gradient)| (wrt, rewrite(gradient)))
                .collect(),
        }
    }
}

#[cfg(test)]
#[path = "tests/adjoints_tests.rs"]
mod tests;
