use std::sync::Arc;

use static_assertions::assert_impl_all;

use crate::{Element, Tensor};

use crate::op::Op;

use super::{Node, Origin, Parameters, SlotStore, Structure, Symbol, Tape, ValueId};

// Entry-time thread-safety contract. `Differentiable` already requires
// `Data: Send + Sync`, so only a structural change (an `Rc`, a `RefCell`, a
// raw pointer) could break sharing across threads; a single concrete anchor
// is enough to catch that.
assert_impl_all!(Network<f64>: Send, Sync);

/// The sealed phase of a recording: an immutable computation-graph
/// spec.
///
/// A network holds structure, shapes, parameter initials, and input
/// defaults — the whole spec, runnable standalone — and no live state:
/// parameter payloads are the caller's [`Parameters`], fed inputs are
/// per-run overlays. Nothing mutates a network, so it is `Send + Sync`
/// with no lock, and any number of threads can run one shared network
/// concurrently through `&Network` or `Arc<Network>`.
///
/// A network is only ever born from a tape ([`Tape::into_network`]),
/// and [`Network::into_tape`] consumes it to reopen recording — the
/// consuming pair keeps one origin's history linear by ownership, so
/// symbols and plans stay valid across every round trip. It is
/// deliberately not `Clone`: a second sealed copy could be reopened
/// into a divergent future, which is exactly what the ownership rule
/// exists to make unrepresentable.
#[derive(Debug)]
pub struct Network<E> {
    origin: Origin,
    structure: Structure<Tensor<E>>,
    initials: SlotStore<Tensor<E>>,
    inputs: Arc<SlotStore<Tensor<E>>>,
}

impl<E: Element> Network<E> {
    /// Seals the recorded columns and stores under `origin`: the body
    /// of [`Tape::into_network`].
    pub(super) fn seal(
        origin: Origin,
        structure: Structure<Tensor<E>>,
        initials: SlotStore<Tensor<E>>,
        inputs: SlotStore<Tensor<E>>,
    ) -> Self {
        Self {
            origin,
            structure,
            initials,
            inputs: Arc::new(inputs),
        }
    }

    /// Reopens the network for further recording, consuming it: the
    /// inverse of [`Tape::into_network`].
    ///
    /// The tape keeps the same origin, so every existing [`Symbol`]
    /// keeps naming its node, and extension is linear: a consumed
    /// network cannot also stay sealed, which is what makes divergent
    /// histories unconstructible. State carried in a
    /// [`Parameters`] value survives the round trip through
    /// [`Parameters::carried`].
    pub fn into_tape(self) -> Tape<E> {
        Tape::reopen(self.origin, self)
    }

    /// Hands the stores back for [`Tape::reopen`], unsharing the input
    /// defaults if a plan still holds them.
    pub(super) fn into_stores(
        self,
    ) -> (
        Structure<Tensor<E>>,
        SlotStore<Tensor<E>>,
        SlotStore<Tensor<E>>,
    ) {
        (
            self.structure,
            self.initials,
            Arc::unwrap_or_clone(self.inputs),
        )
    }

    /// Materializes the record-site initials into a fresh caller-owned
    /// [`Parameters`] value.
    ///
    /// Every call answers a new value, so initialization stays visible
    /// at the record site and what-if states are independent from
    /// birth.
    pub fn parameters(&self) -> Parameters<E> {
        Parameters::new(self.origin, self.initials.clone())
    }

    /// Returns the number of recorded nodes.
    pub fn len(&self) -> usize {
        self.structure.len()
    }

    /// Returns `true` if it holds no nodes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the origin token of this network's family.
    pub(crate) fn origin(&self) -> Origin {
        self.origin
    }

    /// Returns the recorded node columns.
    pub(crate) fn structure(&self) -> &Structure<Tensor<E>> {
        &self.structure
    }

    /// Returns the input-default store, shared for plan freezes.
    pub(crate) fn inputs(&self) -> &Arc<SlotStore<Tensor<E>>> {
        &self.inputs
    }

    /// Returns the number of recorded parameter slots.
    pub(crate) fn parameters_len(&self) -> usize {
        self.initials.len()
    }

    /// Returns the public snapshot of the node `symbol` names:
    /// opcode, operands, and recorded shape, detached from the
    /// network.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network or is not
    /// allocated in it.
    pub fn node(&self, symbol: Symbol) -> Node {
        let id = self.locate(symbol);
        self.structure.node_at(self.origin, id.index())
    }

    /// Returns every recorded node in allocation order, as public
    /// snapshots.
    pub fn nodes(&self) -> impl Iterator<Item = Node> + '_ {
        (0..self.len()).map(|index| self.structure.node_at(self.origin, index))
    }

    /// Returns the stored payload of the node `symbol` names: a
    /// leaf's constant, a parameter's record-site initial, or an
    /// input's default — `None` for computed nodes.
    ///
    /// It is the sealed form of [`Value::payload`](crate::Value::payload).
    /// Live parameter payloads are the caller's
    /// [`Parameters`](crate::Parameters), read by
    /// [`Parameters::of`](crate::Parameters::of); run results are
    /// read from a [`Run`](crate::Run).
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network or is not
    /// allocated in it.
    pub fn payload(&self, symbol: Symbol) -> Option<&Tensor<E>> {
        let id = self.locate(symbol);
        match self
            .structure
            .ops
            .get(id.index())
            .expect("`locate` checked the bounds")
        {
            Op::Leaf(leaf) => Some(&leaf.0),
            Op::Parameter(parameter) => Some(&self.initials.payloads()[parameter.0.index()]),
            Op::Input(input) => Some(&self.inputs.payloads()[input.0.index()]),
            _ => None,
        }
    }

    /// Renders the spec as text: one line per node in allocation
    /// order — index, opcode, operand indices, parameters, shape —
    /// then a summary. The IR dump, and the same line format
    /// [`Plan::describe`](crate::Plan::describe) uses for its
    /// scheduled subset.
    pub fn describe(&self) -> String {
        use std::fmt::Write;

        let mut lines = String::new();
        for node in self.nodes() {
            writeln!(lines, "{}", node.spec_line()).expect("writing to a string cannot fail");
        }
        let nodes = self.len();
        let parameters = self.initials.len();
        let inputs = self.inputs.len();
        writeln!(
            lines,
            "network: {nodes} node{}, {parameters} parameter{}, {inputs} input{}",
            if nodes == 1 { "" } else { "s" },
            if parameters == 1 { "" } else { "s" },
            if inputs == 1 { "" } else { "s" },
        )
        .expect("writing to a string cannot fail");
        lines
    }

    /// Locates the node `symbol` names on this network.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network or is not
    /// allocated in it.
    pub(crate) fn locate(&self, symbol: Symbol) -> ValueId {
        assert!(
            symbol.origin == self.origin,
            "symbol belongs to a different network"
        );
        assert!(
            symbol.id.index() < self.structure.len(),
            "symbol is not allocated in this network"
        );
        symbol.id
    }
}

#[cfg(test)]
#[path = "tests/network_tests.rs"]
mod tests;
