use std::sync::{Mutex, MutexGuard};

use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::op::Op;
use crate::{Element, Shape, Tensor, Tensorial};

use super::trace::Trace;
use super::{
    Adjoints, Keep, Network, Node, Operands, Origin, SlotStore, Structure, Symbol, Value, ValueId,
};

// Entry-time thread-safety contract; the anchor rationale is documented
// in `network.rs`. The tape is the root every other guarantee rests on.
assert_impl_all!(Tape<f64>: Send, Sync);

/// The recorded columns and stores, guarded together by one lock.
///
/// Parameters and inputs share the same store type; they stay separate
/// fields because their roles differ: parameter initials seed
/// [`Parameters`](crate::Parameters), input defaults are spec that
/// feeds overlay per run.
#[derive(Debug)]
struct TapeInner<E> {
    structure: Structure<Tensor<E>>,
    initials: SlotStore<Tensor<E>>,
    inputs: SlotStore<Tensor<E>>,
}

/// The construction phase of a network: an append-only record of every
/// node of one computation graph.
///
/// It is the engine's take on the classic autograd tape (a Wengert
/// list): expressions record `Op` nodes onto it as they are built
/// — each with its `Shape`, inferred and validated at record time — so
/// invalid expressions panic at the expression that records them,
/// before anything runs. Recording happens through [`Value`] proxies
/// and their operators; the tape is the only synchronization point in
/// the crate, a single `Mutex` taken briefly per recording, quarantined
/// to this phase.
///
/// [`Tape::into_network`] consumes the tape and seals the recording
/// into an immutable [`Network`]; [`Network::into_tape`] consumes the
/// network to reopen it. Both conversions are consuming, so one
/// origin's history is linear by ownership: two divergent futures of
/// one recording cannot be constructed. Recorded nodes are never
/// mutated or removed, and linear extension never moves one, so a
/// [`Symbol`] stays valid across every round trip.
#[derive(Debug)]
pub struct Tape<E> {
    origin: Origin,
    inner: Mutex<TapeInner<E>>,
}

impl<E: Element> Tape<E> {
    /// Records a whole graph in one closure and seals it: the
    /// default construction path, whose return value *is* the
    /// construction keep-set.
    ///
    /// The closure builds on a fresh tape and returns its keep-set
    /// in detached form — one [`Keep::keep`] call turns any array,
    /// `Vec`, or tuple of values into symbols — and the seal follows
    /// immediately, so no proxy can escape the phase and the names
    /// later phases read are a value, not a pile of `.symbol()`
    /// locals. Reopen, twins, and piecewise recording keep the
    /// explicit [`Tape::new`] / [`Tape::into_network`] pair.
    ///
    /// # Examples
    /// ```
    /// use topos::{Keep, Tape};
    ///
    /// let (network, [w, loss]) = Tape::record(|tape| {
    ///     let w = tape.parameter(3.0_f64);
    ///     let loss = w * w;
    ///     [w, loss].keep()
    /// });
    /// assert_eq!(network.parameters().of(w).scalar(), 3.0);
    /// # let _ = loss;
    /// ```
    pub fn record<Out: Keep>(build: impl FnOnce(&Self) -> Out) -> (Network<E>, Out::Kept) {
        let tape = Self::new();
        let kept = build(&tape).keep();
        (tape.into_network(), kept)
    }

    /// Creates an empty `Tape`.
    pub fn new() -> Self {
        Self {
            origin: Origin::new(),
            inner: Mutex::new(TapeInner {
                structure: Structure::new(),
                initials: SlotStore::new(),
                inputs: SlotStore::new(),
            }),
        }
    }

    /// Reopens `network` for further recording: the inverse of
    /// [`Tape::into_network`], with the same origin, so every existing
    /// [`Symbol`] keeps naming its node.
    pub(super) fn reopen(origin: Origin, network: Network<E>) -> Self {
        let (structure, initials, inputs) = network.into_stores();
        Self {
            origin,
            inner: Mutex::new(TapeInner {
                structure,
                initials,
                inputs,
            }),
        }
    }

    /// Seals the recording into an immutable [`Network`], consuming the
    /// tape.
    ///
    /// The conversion is infallible and moves the recorded columns and
    /// stores without copying. Take [`Value::symbol`] for every value
    /// the later phases will name first: proxies borrow the tape, so
    /// the borrow checker rejects one outliving this call — the phase
    /// boundary is a compile error, not a runtime check.
    pub fn into_network(self) -> Network<E> {
        let inner = self
            .inner
            .into_inner()
            .expect("tape is poisoned: a recording panicked earlier on this tape");
        Network::seal(self.origin, inner.structure, inner.initials, inner.inputs)
    }

    /// Returns the origin token of this tape's family.
    pub(crate) fn origin(&self) -> Origin {
        self.origin
    }

    /// Allocates a constant leaf and returns a proxy to it.
    ///
    /// Constants are fixed at recording time; see `parameter` for
    /// trainable leaves and `input` for leaves fed per run.
    pub fn leaf(&self, data: impl Into<Tensor<E>>) -> Value<'_, E> {
        let id = self.record_node(Op::leaf(data.into()), &[]);
        Value::bind(self, id)
    }

    /// Allocates a learnable parameter and returns a proxy to it.
    ///
    /// `data` is the parameter's record-site initial: the payload a
    /// fresh [`Network::parameters`](crate::Network::parameters)
    /// starts from. Live payloads are caller-owned
    /// [`Parameters`](crate::Parameters) state; training never touches
    /// the recorded node.
    pub fn parameter(&self, data: impl Into<Tensor<E>>) -> Value<'_, E> {
        let data = data.into();
        let shape = data.shape();
        let id = {
            let mut guard = self.lock();
            let inner = &mut *guard;
            // Disjoint fields: the store borrow and the structure push
            // in `install`'s closure are simultaneous without conflict.
            let structure = &mut inner.structure;
            inner.initials.install(data, |slot| {
                structure.push(Op::parameter(slot), Operands::none(), shape)
            })
        };
        Value::bind(self, id)
    }

    /// Allocates a declared per-run input and returns a proxy to it.
    ///
    /// `initial` supplies the input's recorded shape and its default
    /// payload — part of the spec, so a network with its defaults is
    /// runnable standalone; feeds overlay the defaults per run.
    pub fn input(&self, initial: impl Into<Tensor<E>>) -> Value<'_, E> {
        let initial = initial.into();
        let shape = initial.shape();
        let id = {
            let mut guard = self.lock();
            let inner = &mut *guard;
            let structure = &mut inner.structure;
            inner.inputs.install(initial, |slot| {
                structure.push(Op::input(slot), Operands::none(), shape)
            })
        };
        Value::bind(self, id)
    }

    /// Resolves `symbol` back into a proxy on this tape: the reopen
    /// flow's bridge from the eternal name to the recording phase.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network or is not
    /// allocated on this tape.
    pub fn resolve(&self, symbol: Symbol) -> Value<'_, E> {
        assert!(
            symbol.origin == self.origin,
            "symbol belongs to a different network"
        );
        assert!(
            symbol.id.index() < self.len(),
            "symbol is not allocated on this tape"
        );
        Value::bind(self, symbol.id)
    }

    /// Returns the number of recorded nodes.
    pub fn len(&self) -> usize {
        self.lock().structure.len()
    }

    /// Returns the public snapshot of the node `symbol` names:
    /// opcode, operands, and recorded shape, detached from the tape,
    /// so it outlives the lock.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network or is not
    /// allocated on this tape.
    pub fn node(&self, symbol: Symbol) -> Node {
        assert!(
            symbol.origin == self.origin,
            "symbol belongs to a different network"
        );
        let inner = self.lock();
        assert!(
            symbol.id.index() < inner.structure.len(),
            "symbol is not allocated on this tape"
        );
        inner.structure.node_at(self.origin, symbol.id.index())
    }

    /// Returns every node recorded so far, in allocation order, as a
    /// snapshot taken under the tape lock.
    pub fn nodes(&self) -> Vec<Node> {
        let inner = self.lock();
        (0..inner.structure.len())
            .map(|index| inner.structure.node_at(self.origin, index))
            .collect()
    }

    /// Returns a clone of the stored payload of the node `symbol`
    /// names: a leaf's constant, a parameter's record-site initial,
    /// or an input's default — `None` for computed nodes.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network or is not
    /// allocated on this tape.
    pub fn payload(&self, symbol: Symbol) -> Option<Tensor<E>> {
        self.resolve(symbol).payload()
    }

    /// Renders the recording so far as text: one line per node in
    /// allocation order, then a summary — the open-phase twin of
    /// [`Network::describe`](crate::Network::describe).
    pub fn describe(&self) -> String {
        use std::fmt::Write;

        let inner = self.lock();
        let mut lines = String::new();
        for index in 0..inner.structure.len() {
            writeln!(
                lines,
                "{}",
                inner.structure.node_at(self.origin, index).spec_line()
            )
            .expect("writing to a string cannot fail");
        }
        let nodes = inner.structure.len();
        let parameters = inner.initials.len();
        let inputs = inner.inputs.len();
        writeln!(
            lines,
            "tape: {nodes} node{}, {parameters} parameter{}, {inputs} input{}",
            if nodes == 1 { "" } else { "s" },
            if parameters == 1 { "" } else { "s" },
            if inputs == 1 { "" } else { "s" },
        )
        .expect("writing to a string cannot fail");
        lines
    }

    /// Returns `true` if it holds no nodes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Records `op` with its positional `operands` and returns
    /// its handle.
    ///
    /// It infers and stores the result's shape on the way in, so shape
    /// mismatches panic at the expression that records them, before
    /// anything runs.
    ///
    /// # Panics
    /// Panics if `operands` does not match the op's arity or
    /// references a node that is not recorded on this tape, or if the
    /// operands' shapes are incompatible.
    pub(crate) fn record_node(&self, op: Op<Tensor<E>>, operands: &[ValueId]) -> ValueId {
        assert_eq!(
            operands.len(),
            op.arity(),
            "operand count must match the operation's arity"
        );
        let mut inner = self.lock();
        for operand in operands {
            assert!(
                operand.index() < inner.structure.len(),
                "operand is out of bounds for its tape"
            );
        }
        let shape = {
            let shapes = &inner.structure.shapes;
            let operand_shapes: SmallVec<[Shape; 2]> = operands
                .iter()
                .map(|operand| {
                    shapes
                        .get(operand.index())
                        .expect("operand shape is recorded")
                        .clone()
                })
                .collect();
            op.infer_shape(&operand_shapes)
        };
        inner
            .structure
            .push(op, Operands::from_slice(operands), shape)
    }

    /// Returns a clone of the payload behind `id`: a leaf's embedded
    /// payload, a parameter's record-site initial, or an input's
    /// default, or `None` for computed values.
    ///
    /// # Panics
    /// Panics if `id` is not recorded on this tape.
    pub(crate) fn payload_of(&self, id: ValueId) -> Option<Tensor<E>> {
        let inner = self.lock();
        let op = inner
            .structure
            .ops
            .get(id.index())
            .expect("`ValueId` is out of bounds for its tape");
        match op {
            Op::Leaf(leaf) => Some(leaf.0.clone()),
            Op::Parameter(parameter) => {
                Some(inner.initials.payloads()[parameter.0.index()].clone())
            }
            Op::Input(input) => Some(inner.inputs.payloads()[input.0.index()].clone()),
            _ => None,
        }
    }

    /// Returns the shape inferred for `id` when it was recorded.
    ///
    /// # Panics
    /// Panics if `id` is not recorded on this tape.
    pub(crate) fn shape(&self, id: ValueId) -> Shape {
        self.lock()
            .structure
            .shapes
            .get(id.index())
            .expect("`ValueId` is out of bounds for its tape")
            .clone()
    }

    /// Returns a clone of the operand links recorded for `id`.
    ///
    /// # Panics
    /// Panics if `id` is not recorded on this tape.
    #[cfg(test)]
    pub(crate) fn operands_of(&self, id: ValueId) -> Operands {
        self.lock()
            .structure
            .operands
            .get(id.index())
            .expect("`ValueId` is out of bounds for its tape")
            .clone()
    }

    /// Runs `reader` over the node behind `id` while holding the tape lock.
    ///
    /// # Panics
    /// Panics if `id` is not recorded on this tape.
    #[cfg(test)]
    pub(crate) fn with_node<Output>(
        &self,
        id: ValueId,
        reader: impl FnOnce(&Op<Tensor<E>>) -> Output,
    ) -> Output {
        let inner = self.lock();
        let op = inner
            .structure
            .ops
            .get(id.index())
            .expect("`ValueId` is out of bounds for its tape");
        reader(op)
    }

    /// Returns an O(1) freeze of the recorded columns, so a scan can
    /// read them without holding the lock while new nodes record.
    fn structure_freeze(&self) -> Structure<Tensor<E>> {
        self.lock().structure.clone()
    }

    /// Locks the tape's columns.
    ///
    /// A poisoned lock stays fatal on purpose: it means a recording
    /// panicked on this tape earlier, the panic was caught, and the
    /// program kept going — a state this crate's panics-mean-bugs
    /// contract does not support. The message names that cause so the
    /// debugging trail leads to the original panic.
    fn lock(&self) -> MutexGuard<'_, TapeInner<E>> {
        self.inner
            .lock()
            .expect("tape is poisoned: a recording panicked earlier on this tape")
    }
}

impl<E: Element> Tape<E> {
    /// Records the reverse-mode gradient of `loss` with respect to each
    /// `wrt` entry as ordinary computed nodes on this tape, and returns
    /// the [`Adjoints`] pairing each entry with its gradient.
    ///
    /// It is `backward` as a tape-to-tape transform: the same reverse
    /// scan the engine runs over payload buffers runs here over
    /// recording `Trace` handles, applying the very same derivative
    /// rules — so the recorded gradient and the engine's are one body
    /// of knowledge, and a compiled plan over the adjoints' roots
    /// reproduces [`Run::backward`](crate::Run::backward) bitwise
    /// (same seed, same accumulation order). Gradients become
    /// first-class values: compilable, emittable, readable, and
    /// differentiable again for higher-order derivatives.
    ///
    /// A `wrt` value that is not an ancestor of the loss answers a
    /// recorded zero of its own shape, exactly as
    /// [`Gradients`](crate::Gradients) would. The transform reads
    /// graph structure only, never payloads; recording appends to the
    /// tape and leaves every existing node untouched.
    ///
    /// It is [`Tape::vjp`] with a recorded ones seed: the wrapper
    /// mints the seed leaf and delegates, so the two share one scan.
    ///
    /// # Panics
    /// Panics if `loss` is not a recorded scalar (reduce with `sum`
    /// first) or any symbol belongs to a different network.
    pub fn differentiate(
        &self,
        loss: impl Into<Symbol>,
        wrt: impl IntoIterator<Item = impl Into<Symbol>>,
    ) -> Adjoints {
        let loss_value = self.resolve(loss.into());
        assert_eq!(
            loss_value.shape().rank(),
            0,
            "differentiate requires a scalar loss; reduce it with `sum` first"
        );
        let seed = loss_value.literal(Tensor::counted(loss_value.shape(), 1));
        self.vjp(loss_value.symbol(), seed.symbol(), wrt)
    }

    /// Records the vector-Jacobian product of `target` with respect to
    /// each `wrt` entry — reverse mode with an explicit `seed` planted
    /// at `target` instead of [`Tape::differentiate`]'s ones — and
    /// returns the [`Adjoints`] pairing each entry with its gradient.
    ///
    /// The explicit seed is what makes a non-scalar `target` honest:
    /// the seed supplies the contraction weights a scalar loss would
    /// have supplied implicitly, so the scalar rule (never sum
    /// implicitly) stays intact while `J^T seed` becomes recordable
    /// directly. A seed may itself be a computed value — a gradient
    /// node from an earlier `differentiate` — which is how
    /// Hessian-vector products and reverse-over-reverse stay ordinary
    /// recording: `vjp(adjoints.of(x), vector, [x])`.
    ///
    /// The seed enters as the initial cotangent payload, not as a
    /// graph edge: the transform treats it as a constant weight and
    /// never differentiates through it.
    ///
    /// # Panics
    /// Panics if `seed`'s recorded shape differs from `target`'s or
    /// any symbol belongs to a different network.
    pub fn vjp(
        &self,
        target: impl Into<Symbol>,
        seed: impl Into<Symbol>,
        wrt: impl IntoIterator<Item = impl Into<Symbol>>,
    ) -> Adjoints {
        let target_value = self.resolve(target.into());
        let seed_value = self.resolve(seed.into());
        assert_eq!(
            seed_value.shape(),
            target_value.shape(),
            "a vjp seed must have the target's shape"
        );
        let output_index = target_value.id().index();
        let structure = self.structure_freeze();
        let trace = |index: usize| Trace::of(Value::bind(self, ValueId(index)));

        // The scan mirrors `Run::backward` deliberately and
        // exactly — the seed planting, the ancestor marking through
        // `Some` cotangents, the zero-seeded accumulation in reverse
        // scan order — because the bitwise parity contract welds the
        // two: any change to either scan's arithmetic must reach both.
        // It stays a twin rather than one parameterized body because
        // the two live in different phases with different asserts
        // (posture and numerics here have no recording analogue);
        // the closure suite is the weld.
        let mut cotangents: Vec<Option<Trace<'_, E>>> = vec![None; output_index + 1];
        cotangents[output_index] = Some(Trace::of(seed_value));
        let mut ancestors = vec![false; output_index + 1];
        ancestors[output_index] = true;
        for index in (0..=output_index).rev() {
            if !ancestors[index] {
                continue;
            }
            let links = structure
                .operands
                .get(index)
                .expect("the freeze cannot shrink")
                .as_slice();
            if links.is_empty() {
                // Sources: leaves, parameters, and inputs, where
                // gradients stop and get read out below.
                continue;
            }
            let op = structure.ops.get(index).expect("the freeze cannot shrink");
            let operand_traces: SmallVec<[Trace<'_, E>; 2]> =
                links.iter().map(|link| trace(link.index())).collect();
            let operands: SmallVec<[&Trace<'_, E>; 2]> = operand_traces.iter().collect();
            let gradient = cotangents[index].expect("ancestors carry cotangents");
            let recorded = op.backward(&operands, &trace(index), &gradient);
            debug_assert_eq!(recorded.len(), links.len());
            for (&link, cotangent) in links.iter().zip(recorded) {
                if let Some(contribution) = cotangent {
                    let slot = link.index();
                    ancestors[slot] = true;
                    let seeded = match cotangents[slot] {
                        Some(existing) => existing,
                        None => trace(slot).zero_like(),
                    };
                    cotangents[slot] = Some(seeded + contribution);
                }
            }
        }

        let pairs = wrt
            .into_iter()
            .map(|entry| {
                let value = self.resolve(entry.into());
                let gradient = match cotangents.get(value.id().index()).copied().flatten() {
                    Some(gradient) => gradient.value().symbol(),
                    // A non-ancestor's gradient is a recorded zero of
                    // its own shape, the tape twin of the zeros a
                    // gradient field holds there.
                    None => value.literal(Tensor::counted(value.shape(), 0)).symbol(),
                };
                (value.symbol(), gradient)
            })
            .collect();
        Adjoints::new(target_value.symbol(), pairs)
    }
}

impl<E: Element> Default for Tape<E> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests/tape_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/differentiate_tests.rs"]
mod differentiate_tests;
