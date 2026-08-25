use std::collections::HashMap;
use std::sync::Arc;

use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::{Element, Numerics, Tensor};

use crate::backend::NumericsScope;
use crate::graph::{
    Adjoints, Field, Gradients, Network, Origin, Parameters, SlotStore, Structure, Symbol, ValueId,
};
use crate::op::{Op, SlotId};

use super::Entry;

// Entry-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Run<f64>: Send, Sync);

/// The producer-specific shape of one run: which slots answer reads,
/// and whether `backward` may differentiate it.
///
/// Every forward path yields the same `Run`, but the four producers
/// leave it in genuinely different states; the posture names that
/// state as one explicit sum, so an impossible combination — remat
/// recipes on a run that refuses `backward` — cannot be represented.
/// Masked slots hold shape-correct zero placeholders that reads must
/// never answer with, so `of` and `backward` consult the posture
/// first.
#[derive(Debug)]
pub(crate) enum Provenance {
    /// Full interpreter run: every slot is genuine.
    Complete,
    /// Target-sliced interpreter run: the ancestor closure of the
    /// declared targets was computed; every slot outside it holds a
    /// placeholder.
    Sliced { computed: Vec<bool> },
    /// Forward-only plan run: only the keep-set answers reads, and
    /// `backward` is refused — the liveness pass freed the buffers
    /// it would need.
    Observed { readable: Arc<Vec<bool>> },
    /// Engine-backward plan run: only the keep-set answers reads,
    /// and the run retains every forward value `backward` reads.
    Training { readable: Arc<Vec<bool>> },
}

impl Provenance {
    /// Returns the mask of slots that answer reads, `None` for a
    /// complete run where every slot does.
    fn mask(&self) -> Option<&[bool]> {
        match self {
            Provenance::Complete => None,
            Provenance::Sliced { computed } => Some(computed),
            Provenance::Observed { readable } | Provenance::Training { readable, .. } => {
                Some(readable)
            }
        }
    }

    /// Returns whether runs of this posture may differentiate: only a
    /// forward-only plan run refuses, because its liveness pass freed
    /// the forward values the derivative rules read.
    fn differentiable(&self) -> bool {
        !matches!(self, Provenance::Observed { .. })
    }
}

/// The materialized payloads of one forward run.
///
/// A run is immutable, per-run state: the graph structure frozen at
/// the start of the run and the payloads that run produced. It borrows
/// nothing — kinship is the same origin-and-coverage check every
/// detached carrier makes — so runs can be stashed, moved, or
/// differentiated concurrently without pinning a [`Network`](crate::Network),
/// and a reopened tape recording new nodes does not change its values
/// or the operations differentiated by [`Run::backward`].
#[derive(Debug)]
pub struct Run<E> {
    /// Frozen node columns for this run: ops, operands, and the
    /// shapes inferred at record time.
    structure: Structure<Tensor<E>>,
    field: Field<E>,
    provenance: Provenance,
    /// The numerics posture the forward producer executed under;
    /// `backward` re-enters it so gradients follow the same paths.
    numerics: Numerics,
}

impl<E: Element> Run<E> {
    pub(crate) fn new(
        structure: Structure<Tensor<E>>,
        origin: Origin,
        values: Vec<Tensor<E>>,
        provenance: Provenance,
        numerics: Numerics,
    ) -> Self {
        debug_assert_eq!(structure.len(), values.len());
        if let Some(mask) = provenance.mask() {
            debug_assert_eq!(structure.len(), mask.len());
        }
        Self {
            structure,
            field: Field::new(origin, values),
            provenance,
            numerics,
        }
    }

    /// Returns whether this run computed the slot at `index` as a
    /// readable value, as opposed to leaving a placeholder there.
    fn computed(&self, index: usize) -> bool {
        match self.provenance.mask() {
            Some(mask) => mask[index],
            None => true,
        }
    }

    /// Locates `symbol` in this run's slots.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network or was
    /// allocated after this run.
    fn locate(&self, symbol: Symbol) -> usize {
        assert!(
            symbol.origin == self.field.origin(),
            "symbol belongs to a different network"
        );
        assert!(
            symbol.id.index() < self.field.len(),
            "symbol was allocated after this run"
        );
        symbol.id.index()
    }

    /// Returns the computed payload of the value named by `symbol`.
    ///
    /// It is the shared read-back accessor of every position-indexed
    /// buffer: runs, gradients, and fields all answer `of(symbol)`.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network, was
    /// allocated after this run, or was skipped by a target-sliced run
    /// (see [`BoundEntry::interpret`](crate::BoundEntry::interpret)): a
    /// placeholder must never read as a result.
    pub fn of(&self, symbol: Symbol) -> &Tensor<E> {
        let index = self.locate(symbol);
        assert!(
            self.computed(index),
            "value was not computed by this target-sliced run; add it to the targets"
        );
        &self.field.payloads()[index]
    }

    /// Returns the run's values as a field, for the displays and
    /// analyses that walk a whole pass rather than read one value out
    /// of it.
    ///
    /// Slots this run did not compute — the ones a target-sliced
    /// interpretation skipped — hold placeholders, not results;
    /// [`Run::of`] is the asserting read that refuses them.
    pub fn field(&self) -> &Field<E> {
        &self.field
    }

    /// Assembles the parameter-aligned gradients of this run: each of
    /// the adjoints' `(wrt, gradient)` pairs copies the gradient
    /// node's payload into the `wrt` parameter's slot, with a zero of
    /// the parameter's shape in every slot the adjoints do not name —
    /// exactly the entries [`Run::backward`] would produce for those
    /// parameters, when the gradients were recorded by
    /// [`Tape::differentiate`](crate::Tape::differentiate) instead of
    /// computed by the engine.
    ///
    /// The result feeds [`Parameters::step`](crate::Parameters::step)
    /// directly: one forward run of a plan compiled over the adjoints'
    /// roots yields the update direction with no backward pass and no
    /// node-aligned buffer at all, and the closure suite pins the two
    /// routes bitwise.
    ///
    /// # Panics
    /// Panics as [`Run::of`] panics for either half of a pair,
    /// if a `wrt` entry is not a parameter, or if a gradient's
    /// payload shape differs from its parameter's recorded shape.
    pub fn recorded_gradients(&self, adjoints: &Adjoints) -> Parameters<E> {
        let mut recorded: HashMap<usize, Tensor<E>> = HashMap::new();
        for &(parameter, gradient) in adjoints.pairs() {
            let index = self.locate(parameter);
            assert!(
                matches!(self.structure.ops.get(index), Some(Op::Parameter(_))),
                "recorded gradients fill parameter slots; a `wrt` entry of \
                 these adjoints is not a parameter"
            );
            let payload = self.of(gradient).clone();
            assert_eq!(
                payload.shape(),
                self.structure
                    .shapes
                    .get(index)
                    .expect("shapes cover the run")
                    .clone(),
                "recorded gradient shape does not match its parameter's"
            );
            recorded.insert(index, payload);
        }
        // Parameter nodes appear in recording order, which is slot
        // order; unnamed slots answer a zero of the parameter's shape,
        // like the non-ancestor entries of an engine backward.
        let values = self.field.payloads();
        let rows = self
            .structure
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| matches!(op, Op::Parameter(_)))
            .map(|(index, _)| {
                let payload = recorded
                    .remove(&index)
                    .unwrap_or_else(|| values[index].zero_like());
                (ValueId(index), payload)
            });
        Parameters::from_rows(self.field.origin(), rows)
    }
}

impl<E: Element> Run<E> {
    /// Propagates gradients backward from `output`, returning the
    /// gradient of `output` with respect to every value of this run.
    ///
    /// It is the oracle of reverse mode: the interpreter applying the
    /// same derivative rules
    /// [`Tape::differentiate`](crate::Tape::differentiate) records,
    /// without recording — the transform is proven against this scan
    /// bitwise, and this scan ships forever. For compiled training,
    /// prefer the recorded route: a forward-only plan over the
    /// adjoints' roots, where fusion and liveness apply to the chain
    /// rule itself.
    ///
    /// The target must be a scalar (rank 0): a gradient is always of one
    /// chosen scalar, so a non-scalar value is reduced explicitly with
    /// `sum` before differentiation, never summed implicitly.
    ///
    /// It seeds the output gradient with `one_like` and accumulates into
    /// a fresh buffer initialized with `zero_like`, scanning this
    /// run's own structure in reverse allocation order. Only the
    /// ancestors of `output` execute their derivative rules: every other
    /// value's gradient is exactly zero, and expressions the target does
    /// not depend on — including singular ones such as a division by
    /// zero, even when the target uses them purely as a shape or index
    /// reference — cannot disturb the result. The run borrows nothing,
    /// so any number of threads can differentiate one shared run for
    /// their own targets at once. Values recorded after this run are
    /// absent from the result, exactly as they are absent from `of`.
    ///
    /// # Panics
    /// Panics if `output` is not a scalar, belongs to a different
    /// network, was allocated after this run, or was skipped by a
    /// target-sliced run.
    pub fn backward(&self, output: Symbol) -> Gradients<E> {
        let output_index = self.locate(output);
        let values = self.field.payloads();
        // A sliced run evaluates the whole ancestor closure of its
        // targets, so any computed output has every operand its
        // backward needs.
        assert!(
            self.computed(output_index),
            "value was not computed by this target-sliced run; add it to the targets"
        );
        assert!(
            self.provenance.differentiable(),
            "this run came from a forward-only plan, whose liveness pass freed \
             the buffers backward reads; compile with `Entry::backward` to differentiate"
        );
        assert_eq!(
            values[output_index].shape().rank(),
            0,
            "backward requires a scalar target; reduce it with `sum` first"
        );
        // Both views of the target's shape must agree: the payload above
        // and the recorded column here, so a payload that ignored a
        // recorded movement cannot smuggle a non-scalar target through.
        assert_eq!(
            self.structure
                .shapes
                .get(output_index)
                .expect("shapes cover the run")
                .rank(),
            0,
            "backward requires a scalar target; reduce it with `sum` first"
        );

        // Gradients follow the forward pass's numerics posture, so an
        // exact run differentiates exactly.
        let _numerics = NumericsScope::enter(self.numerics);
        let mut gradients: Vec<Tensor<E>> = values.iter().map(|value| value.zero_like()).collect();
        gradients[output_index] = values[output_index].one_like();
        // The single reverse scan doubles as reachability marking: every
        // consumer lives at a higher index than its operands, so when the
        // scan reaches a node it is already marked exactly when it is an
        // ancestor of the target. Skipping non-ancestors is a correctness
        // measure, not an optimization: their derivative rules must not
        // run, because a singular disconnected expression (`x / x` at
        // zero) would poison genuine gradients with NaN even through a
        // zero cotangent.
        let mut ancestors = vec![false; output_index + 1];
        ancestors[output_index] = true;
        for index in (0..=output_index).rev() {
            if !ancestors[index] {
                continue;
            }
            let op = self
                .structure
                .ops
                .get(index)
                .expect("the freeze cannot shrink");
            let links = self
                .structure
                .operands
                .get(index)
                .expect("the freeze cannot shrink")
                .as_slice();
            // Every payload a derivative rule reads is present:
            // interpreter runs hold everything, and engine-backward
            // plan runs retain what the read contract names.
            let operands: SmallVec<[&Tensor<E>; 2]> =
                links.iter().map(|link| &values[link.index()]).collect();
            let gradient = gradients[index].clone();
            let cotangents = op.backward(&operands, &values[index], &gradient);
            debug_assert_eq!(cotangents.len(), links.len());
            // Accumulation is the multivariate chain rule: when a value
            // feeds several consumers, its gradient is the sum of the
            // cotangents arriving along every path. Only a `Some`
            // cotangent marks its operand as an ancestor: `None` declares
            // the operand data rather than a differentiable dependency
            // (a broadcast's reference, a gather's selection), so its
            // producers stay outside the scan — a singular expression
            // behind a shape-only edge must not leak NaN into genuine
            // gradients. `Some(zero)` is still an edge and still marks.
            for (&link, cotangent) in links.iter().zip(cotangents) {
                if let Some(contribution) = cotangent {
                    let slot = link.index();
                    ancestors[slot] = true;
                    gradients[slot] = gradients[slot].clone() + contribution;
                }
            }
        }
        Field::new(self.field.origin(), gradients)
    }
}

#[cfg(test)]
#[path = "tests/run_tests.rs"]
mod tests;

// The forward entry points live here rather than on the spec's own
// file for the same reason `compile` lives in `plan.rs`: running is
// the executor's business, and the graph tier must not depend on it.
impl<E: Element> Network<E> {
    /// Evaluates every node in allocation order, materializing the
    /// payload of each value into a fresh [`Run`], reading parameter
    /// payloads from `parameters` and binding `feeds` to declared
    /// inputs for this run only.
    ///
    /// Feeds are run-local state: they overlay the input defaults
    /// without touching the spec, so any number of threads can forward
    /// one shared network on different batches — or different
    /// [`Parameters`] — concurrently. Unfed inputs use their defaults.
    /// Allocation order is dependency order by construction, which is
    /// what makes the single forward scan sufficient. The returned run
    /// owns its values, so [`Run::backward`] needs no network borrow.
    ///
    /// # Panics
    /// Panics if `parameters` belongs to a different network or does
    /// not cover this one, if a fed symbol does not resolve here or
    /// names a node that is not an input, or if a fed payload's shape
    /// differs from the input's recorded shape.
    pub fn forward(
        &self,
        parameters: &Parameters<E>,
        feeds: impl IntoIterator<Item = (Symbol, Tensor<E>)>,
    ) -> Run<E> {
        self.run(parameters, None, feeds)
    }

    /// Panics unless `parameters` was born from this network's exact
    /// extent: the run-side kinship check.
    fn assert_covering(&self, parameters: &Parameters<E>) {
        assert!(
            parameters.origin() == self.origin(),
            "parameters belong to a different network"
        );
        assert_eq!(
            parameters.len(),
            self.parameters_len(),
            "parameters do not cover this network's parameter slots; \
             carry them across a reopen with `Parameters::carried`"
        );
    }

    /// Evaluates only the ancestors of an entry's declared results —
    /// the body of [`BoundEntry::interpret`](crate::BoundEntry::interpret).
    ///
    /// It is `forward` restricted to what the caller declared:
    /// reachability over the operand links selects the results'
    /// ancestor closure, and every node outside it is skipped, its slot
    /// holding an O(1) zero placeholder of the recorded shape. Reads
    /// stay loud: [`Run::of`] and [`Run::backward`] panic on a skipped
    /// value instead of answering with a placeholder. With several
    /// expressions recorded on one tape (the training and evaluation
    /// twins of the examples), one entry's closure skips the other
    /// expression entirely — the first rung of the lowering ladder,
    /// applied without any plan object.
    pub(crate) fn interpret_entry(
        &self,
        entry: &Entry,
        parameters: &Parameters<E>,
        feeds: impl IntoIterator<Item = (Symbol, Tensor<E>)>,
    ) -> Run<E> {
        let targets: Vec<ValueId> = entry
            .roots
            .iter()
            .chain(&entry.observe)
            .map(|&target| self.locate(target))
            .collect();
        self.run(parameters, Some(targets), feeds)
    }

    /// Returns the input slot behind `id`, or `None` if the node is
    /// not an input.
    fn input_slot(&self, id: ValueId) -> Option<SlotId> {
        match self
            .structure()
            .ops
            .get(id.index())
            .expect("`ValueId` is in bounds for its network")
        {
            Op::Input(input) => Some(input.0),
            _ => None,
        }
    }

    /// Replays the recording: the shared body of `forward` (every
    /// node) and `forward_for` (the targets' ancestor closure).
    fn run(
        &self,
        parameters: &Parameters<E>,
        targets: Option<Vec<ValueId>>,
        feeds: impl IntoIterator<Item = (Symbol, Tensor<E>)>,
    ) -> Run<E> {
        self.assert_covering(parameters);
        let mut bindings = Vec::new();
        for (symbol, payload) in feeds {
            let id = self.locate(symbol);
            let slot = self.input_slot(id).expect("only inputs can be fed");
            let declared = self
                .structure()
                .shapes
                .get(id.index())
                .expect("shapes cover the network");
            assert_eq!(
                &payload.shape(),
                declared,
                "fed payload must match the input's recorded shape"
            );
            bindings.push((slot, payload));
        }
        let inputs = SlotStore::overlaid(self.inputs(), bindings);

        let structure = self.structure();
        let computed = targets.map(|targets| structure.ancestors(targets));
        let mut values = Vec::with_capacity(structure.len());
        for (index, (op, links)) in structure
            .ops
            .iter()
            .zip(structure.operands.iter())
            .enumerate()
        {
            let skipped = matches!(&computed, Some(wanted) if !wanted[index]);
            let value = if skipped {
                // A shape-correct, non-allocating zero: never read back
                // (`of` checks the computed set), but shaped so that
                // gradient buffers stay coherent.
                let shape = structure
                    .shapes
                    .get(index)
                    .expect("shapes cover the network")
                    .clone();
                Tensor::counted(shape, 0)
            } else {
                let operands: SmallVec<[&Tensor<E>; 2]> = links
                    .as_slice()
                    .iter()
                    .map(|link| &values[link.index()])
                    .collect();
                let value = op.forward(&operands, parameters.payloads(), inputs.payloads());
                // The recorded shape is the type of this node; a payload
                // whose rule answers a different shape has broken the
                // operation contract at exactly this producing node.
                debug_assert_eq!(
                    value.shape(),
                    *structure
                        .shapes
                        .get(index)
                        .expect("shapes cover the network"),
                    "operation output shape disagrees with the recorded shape at node {index}"
                );
                value
            };
            values.push(value);
        }
        let provenance = match computed {
            Some(computed) => Provenance::Sliced { computed },
            None => Provenance::Complete,
        };
        // Interpreter runs execute under the ambient default posture;
        // a chosen posture is a plan affair, carried by the request.
        Run::new(
            structure.clone(),
            self.origin(),
            values,
            provenance,
            Numerics::Fast,
        )
    }
}
