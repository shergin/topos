use smallvec::SmallVec;

use crate::Tensorial;

/// One cotangent per operand, in the operation's positional order.
///
/// `None` marks an operand that is data rather than a differentiable
/// value (a broadcast's reference, a gather's selection), so
/// non-differentiability is structural in the rule's signature rather
/// than implicit by omission.
pub(crate) type Cotangents<Data> = SmallVec<[Option<Data>; 2]>;

/// Which values a derivative rule reads when it runs: the per-operand
/// payloads and the node's own output.
///
/// Shape-only reads (a `reshape` backward reading its operand's shape,
/// a reduction broadcasting over a reference) need no read set,
/// because a freed slot holds a shape-correct placeholder. Reads
/// therefore names exactly the payloads whose *values* a rule reads,
/// and a training plan may free everything else once its forward
/// consumers finish. Each `reads` sits beside the `backward` it
/// describes; keeping the two in step is part of changing a rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Reads {
    /// Whether the rule reads the operand payload at each position.
    pub(crate) operands: [bool; 2],
    /// Whether the rule reads the node's own output payload.
    pub(crate) output: bool,
}

impl Reads {
    /// A rule that reads no payload values at all, or shapes only.
    pub(crate) const NOTHING: Reads = Reads {
        operands: [false, false],
        output: false,
    };
}

/// A derivative rule: the cotangent an operation hands back to each
/// operand, written against the recordable vocabulary
/// ([`Tensorial`]) so one body serves two interpretations — the
/// engine computes it over payload buffers, and `differentiate`
/// records it through `Trace` handles.
///
/// It is implemented by each computed `Op` variant and dispatched
/// through the enum with a plain `match`, so implementations stay
/// statically sized and the trait never needs to be object safe. Leaves
/// and parameters do not implement it: they are supplied, not computed,
/// and the enum's dispatch handles them directly. The rules are pure:
/// operands arrive as a positional slice of references gathered by the
/// engine, results are returned rather than written, and no rule ever
/// sees the tape, a `ValueId`, or a run buffer. Gradient
/// accumulation — the multivariate chain rule — is the engine's job,
/// stated once in `Run::backward`. Each operation's `forward` is a
/// sibling inherent method over `Tensor<E>`: computing a payload is
/// engine business, not part of the rule, which is why the recording
/// interpretation never needs it.
pub(crate) trait Operation<Rule: Tensorial> {
    /// Computes one cotangent per operand, given this node's computed
    /// `output` and its own `gradient`.
    fn backward(&self, operands: &[&Rule], output: &Rule, gradient: &Rule) -> Cotangents<Rule>;
}

/// Splits the positional operand list of a unary operation.
///
/// It is generic so the same helper serves operand payloads and operand
/// shapes.
///
/// # Panics
/// Panics if `operands` does not hold exactly one entry; recording
/// checks every node against its `arity`, so a mismatch is an engine
/// bug.
pub(crate) fn unary<T>(operands: &[T]) -> &T {
    let [operand] = operands else {
        panic!("unary operation expects exactly one operand");
    };
    operand
}

/// Splits the positional operand list of a binary operation.
///
/// It is generic so the same helper serves operand payloads and operand
/// shapes.
///
/// # Panics
/// Panics if `operands` does not hold exactly two entries; recording
/// checks every node against its `arity`, so a mismatch is an engine
/// bug.
pub(crate) fn binary<T>(operands: &[T]) -> (&T, &T) {
    let [left, right] = operands else {
        panic!("binary operation expects exactly two operands");
    };
    (left, right)
}
